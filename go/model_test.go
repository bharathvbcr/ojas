package ojas

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/bharathvbcr/gusset"
)

// roundTrip trains 4 steps straight, and separately 2 steps, saves, frees,
// resumes into a new id and trains 2 more: every step's loss, gradient norm
// and learning rate are bit-identical, and both models then sample the same
// ids.
func roundTrip(t *testing.T, opts LoadOptions) {
	t.Helper()
	ctx := context.Background()
	straight := trainedModel(t, opts)
	want := make([]StepResult, 0, 4)
	for i := 0; i < 4; i++ {
		r, err := TrainStep(ctx, straight)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, r)
	}
	first := trainedModel(t, opts)
	got := make([]StepResult, 0, 4)
	for i := 0; i < 2; i++ {
		r, err := TrainStep(ctx, first)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, r)
	}
	if err := SaveCheckpoint(ctx, first, "ckpt"); err != nil {
		t.Fatal(err)
	}
	if err := Free(ctx, first); err != nil {
		t.Fatal(err)
	}
	resumed, err := Resume(ctx, "ckpt", opts, testConfig())
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		r, err := TrainStep(ctx, resumed)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, r)
	}
	for i := range want {
		a, b := want[i], got[i]
		if math.Float32bits(a.Loss) != math.Float32bits(b.Loss) || math.Float32bits(a.GradNorm) != math.Float32bits(b.GradNorm) ||
			a.Step != b.Step || math.Float64bits(a.MatrixLR) != math.Float64bits(b.MatrixLR) || a.Tokens != b.Tokens {
			t.Fatalf("step %d: straight %+v, resumed %+v", i, a, b)
		}
	}
	opt := SampleOptions{Temperature: 0.8, TopK: 20, Seed: 5, MaxNewTokens: 6}
	x, err := GenerateIDs(ctx, straight, []uint32{9, 8, 7}, opt)
	if err != nil {
		t.Fatal(err)
	}
	y, err := GenerateIDs(ctx, resumed, []uint32{9, 8, 7}, opt)
	if err != nil || fmt.Sprint(x) != fmt.Sprint(y) {
		t.Fatalf("samples: straight %v, resumed %v (%v)", x, y, err)
	}
	// Resume refuses another run's config and creates nothing.
	other := testConfig()
	other.Seed++
	if id, err := Resume(ctx, "ckpt", opts, other); err == nil || !strings.Contains(err.Error(), "differs") {
		t.Fatalf("resume with another seed: id=%d err=%v", id, err)
	}
	for _, id := range []uint64{straight, resumed} {
		if err := Free(ctx, id); err != nil {
			t.Fatal(err)
		}
	}
}

func TestTrainSaveResumeGenerateRoundTripCPU(t *testing.T) {
	harness(t)
	roundTrip(t, LoadOptions{Numerics: NumericsExact})
}

func TestTrainSaveResumeGenerateRoundTripMetal(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal needs macOS")
	}
	harness(t)
	skipWithoutMetal4(t)
	roundTrip(t, LoadOptions{Device: DeviceMetal})
}

func TestTrainSaveResumeGenerateRoundTripWgpu(t *testing.T) {
	harness(t)
	probe, err := LoadModel(context.Background(), "model.safetensors", LoadOptions{Device: DeviceWgpu})
	if err != nil {
		if !strings.HasPrefix(engineMessage(err), "wgpu:") {
			t.Fatal(err)
		}
		t.Skipf("no wgpu adapter: %v", err)
	}
	if err := Free(context.Background(), probe); err != nil {
		t.Fatal(err)
	}
	roundTrip(t, LoadOptions{Device: DeviceWgpu})
}

// With two workers, two goroutines stepping one id either both succeed (one
// waited for the pool) or exactly one gets ErrBusy; never both. Every
// success is committed once: the step count is the successes plus one.
func TestTwoCallsOnOneIDAreBusyOrSerialized(t *testing.T) {
	if err := Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := SetPoolSize(2); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = Close(context.Background())
		if err := SetPoolSize(1); err != nil {
			t.Error(err)
		}
	})
	harness(t)
	id := trainedModel(t, LoadOptions{})
	x := [][]uint32{tokenRows(64)}
	var successes, busy int
	finished := make(chan struct{})
	var failure error
	go func() {
		defer close(finished)
		for round := 0; round < 40; round++ {
			var wg sync.WaitGroup
			errs := make([]error, 2)
			start := make(chan struct{})
			for g := 0; g < 2; g++ {
				wg.Add(1)
				go func(g int) {
					defer wg.Done()
					<-start
					_, errs[g] = TrainStepTokens(context.Background(), id, testSeq, x, x)
				}(g)
			}
			close(start)
			wg.Wait()
			ok := 0
			for _, err := range errs {
				switch {
				case err == nil:
					ok++
				case errors.Is(err, ErrBusy):
					busy++
				default:
					failure = fmt.Errorf("round %d: %v", round, err)
					return
				}
			}
			if ok == 0 {
				failure = fmt.Errorf("round %d: both calls were busy: %v", round, errs)
				return
			}
			successes += ok
		}
	}()
	select {
	case <-finished:
	case <-time.After(2 * time.Minute):
		t.Fatal("watchdog: two goroutines on one id did not finish within 2m")
	}
	if failure != nil {
		t.Fatal(failure)
	}
	next, err := TrainStepTokens(context.Background(), id, testSeq, x, x)
	if err != nil {
		t.Fatal(err)
	}
	if next.Step != uint64(successes)+1 {
		t.Fatalf("step %d after %d successful steps (%d busy)", next.Step, successes, busy)
	}
	t.Logf("%d steps, %d busy refusals", successes, busy)
}

// A context cancelled mid-step returns context.Canceled at once. The engine
// stops at its next op before the optimizer, so nothing is committed: the
// next step (queued behind the stopping job on a pool of 1) is step 1 and
// has the loss a fresh model's first step has.
func TestACancelMidStepCommitsNothing(t *testing.T) {
	harness(t)
	cfg := testConfig()
	open := func() uint64 {
		id, err := LoadModel(context.Background(), "model.safetensors", LoadOptions{Numerics: NumericsExact})
		if err != nil {
			t.Fatal(err)
		}
		if err := OpenTrainer(context.Background(), id, cfg); err != nil {
			t.Fatal(err)
		}
		return id
	}
	// Grow the step until it is long enough to cancel inside, whatever the
	// engine's build profile: a fresh model's first step at that size is the
	// reference.
	var (
		want StepResult
		took time.Duration
	)
	for cfg.Accum = 64; ; cfg.Accum *= 4 {
		ref := open()
		began := time.Now()
		r, err := TrainStep(context.Background(), ref)
		if err != nil {
			t.Fatal(err)
		}
		want, took = r, time.Since(began)
		if err := Free(context.Background(), ref); err != nil {
			t.Fatal(err)
		}
		if took >= 200*time.Millisecond {
			break
		}
		if cfg.Accum >= 1<<16 {
			t.Fatalf("a %d-micro-batch step took only %s", cfg.Accum, took)
		}
	}
	t.Logf("one %d-micro-batch step took %s", cfg.Accum, took)

	id := open()
	ctx, cancel := context.WithCancel(context.Background())
	timer := time.AfterFunc(took/5, cancel)
	defer timer.Stop()
	if _, err := TrainStep(ctx, id); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled step: %v", err)
	}
	done := make(chan struct{})
	var (
		next StepResult
		err  error
	)
	go func() {
		defer close(done)
		next, err = TrainStep(context.Background(), id)
	}()
	select {
	case <-done:
	case <-time.After(time.Minute):
		t.Fatal("watchdog: the step after a cancel did not return")
	}
	if err != nil {
		t.Fatal(err)
	}
	if next.Step != 1 || math.Float32bits(next.Loss) != math.Float32bits(want.Loss) {
		t.Fatalf("after the cancel: %+v, a fresh first step: %+v", next, want)
	}
}

// F10: an in-band kind selects a sentinel only at the start of the engine's
// message. A user path that spells one is echoed in a "missing file" error
// and selects nothing.
func TestAUserPathNeverSelectsAnErrorKind(t *testing.T) {
	harness(t)
	sentinels := kindSentinels()
	for _, name := range []string{
		"ojas:E_BUSY: x.safetensors",
		"ojas:E_CAPACITY: x.safetensors",
		"ojas:E_NONFINITE: x.safetensors",
		"ojas:E_DEVICE_LOST: x.safetensors",
		"ojas:E_POISONED: x.safetensors",
		"ojas:E_PRESSURE: x.safetensors",
	} {
		_, err := load(name)
		if err == nil || !strings.Contains(err.Error(), "missing file") || !strings.Contains(err.Error(), name) {
			t.Fatalf("%s: %v", name, err)
		}
		for _, s := range sentinels {
			if errors.Is(err, s) {
				t.Fatalf("%s selected %v: %v", name, s, err)
			}
		}
	}
}

func TestInBandKindsMustLeadTheEngineMessage(t *testing.T) {
	cases := []struct {
		msg  string
		want error
	}{
		{"ojas:E_CAPACITY: capacity exceeded", ErrCapacity},
		{"ojas:E_DEVICE_LOST: train_step: device lost", ErrDeviceLost},
		{"ojas:E_BUSY: model 3 is busy", ErrBusy},
		{"ojas:E_NONFINITE: step: non-finite value", ErrNonFinite},
		{"ojas:E_POISONED: train_step: poisoned", ErrPoisoned},
		{"ojas:E_PRESSURE: memory pressure: opcode 1 refused", ErrPressure},
		{"load: ojas:E_PRESSURE: inside", nil},
		{"missing file: /root/ojas:E_BUSY: x", nil},
		{"load: ojas:E_CAPACITY: inside", nil},
		{" ojas:E_NONFINITE: leading space", nil},
	}
	for _, c := range cases {
		err := annotateEngineError(&gusset.Error{Code: 1, Msg: c.msg})
		for _, s := range kindSentinels() {
			if errors.Is(err, s) != (s == c.want) {
				t.Fatalf("%q: errors.Is(%v) = %v", c.msg, s, errors.Is(err, s))
			}
		}
		var ge *gusset.Error
		if !errors.As(err, &ge) || ge.Msg != c.msg {
			t.Fatalf("%q: the engine error left the chain: %v", c.msg, err)
		}
	}
	// A message made from ojas_take_last_error is read the same way.
	if err := annotateEngineError(errors.New("ojas:E_CAPACITY: root")); !errors.Is(err, ErrCapacity) {
		t.Fatalf("plain error: %v", err)
	}
}

// NewModel is the same init the fixture was written from (seed 1337), and a
// spec the engine cannot run is refused.
func TestNewModelIsTheFixturesInit(t *testing.T) {
	harness(t)
	ctx := context.Background()
	fresh, err := NewModel(ctx, nanoSpec(), 1337, LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := load("model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	opt := SampleOptions{Temperature: 1, Seed: 3, MaxNewTokens: 10}
	a, err := GenerateIDs(ctx, fresh, []uint32{4, 2}, opt)
	if err != nil {
		t.Fatal(err)
	}
	b, err := GenerateIDs(ctx, loaded, []uint32{4, 2}, opt)
	if err != nil || fmt.Sprint(a) != fmt.Sprint(b) {
		t.Fatalf("new %v, loaded %v (%v)", a, b, err)
	}
	bad := nanoSpec()
	bad.HeadDim = 7
	if _, err := NewModel(ctx, bad, 1, LoadOptions{}); err == nil || !strings.Contains(err.Error(), "head_dim") {
		t.Fatalf("odd head_dim: %v", err)
	}
	if _, err := NewModel(ctx, nanoSpec(), 1, LoadOptions{BudgetBytes: 1024}); !errors.Is(err, ErrCapacity) {
		t.Fatalf("tiny budget: %v", err)
	}
}

// gpt2ByteRunes is GPT-2's bytes_to_unicode table.
func gpt2ByteRunes() [256]rune {
	var out [256]rune
	n := 0
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			out[b] = rune(b)
		} else {
			out[b] = rune(256 + n)
			n++
		}
	}
	return out
}

func TestTokenizerRoundTrip(t *testing.T) {
	dir := harness(t)
	var vocab strings.Builder
	vocab.WriteString("{")
	for i, r := range gpt2ByteRunes() {
		piece := string(r)
		switch r {
		case '"':
			piece = `\"`
		case '\\':
			piece = `\\`
		}
		vocab.WriteString(`"` + piece + `":` + fmt.Sprint(i) + ",")
	}
	vocab.WriteString(`"he":256,"ll":257}`)
	if err := os.WriteFile(filepath.Join(dir, "vocab.json"), []byte(vocab.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "merges.txt"), []byte("#version: 0.2\nh e\nl l\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	id, err := load("model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Tokenize(ctx, id, "hello"); err == nil || !strings.Contains(err.Error(), "no tokenizer") {
		t.Fatalf("before LoadTokenizer: %v", err)
	}
	if err := LoadTokenizer(ctx, id, "missing.json", "merges.txt"); err == nil || !strings.Contains(err.Error(), "missing file") {
		t.Fatalf("missing vocab: %v", err)
	}
	if err := LoadTokenizer(ctx, id, "vocab.json", "merges.txt"); err != nil {
		t.Fatal(err)
	}
	ids, err := Tokenize(ctx, id, "hello héllo")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < 3 || ids[0] != 256 || ids[1] != 257 || ids[2] != 'o' {
		t.Fatalf("ids %v", ids)
	}
	text, err := Detokenize(ctx, id, ids)
	if err != nil || text != "hello héllo" {
		t.Fatalf("detokenize: %q %v", text, err)
	}
	if _, err := Detokenize(ctx, id, []uint32{999}); err == nil || !strings.Contains(err.Error(), "outside the vocabulary") {
		t.Fatalf("id past the vocabulary: %v", err)
	}
}
