package ojas

import (
	"context"
	"encoding/binary"
	"encoding/json"
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

// The checked-in nanolab model ojas-capi's tests use: 2 layers, d 16,
// 2 x 8 heads, vocab 64, block 32, init seed 1337.
const (
	fixturePath = "../ojas-capi/tests/fixtures/nano.safetensors"
	testSeq     = 8
	testVocab   = 64
)

func nanoSpec() ModelSpec {
	return ModelSpec{Vocab: 64, NEmbd: 16, NLayer: 2, NHead: 2, NKVHead: 2, HeadDim: 8, Hidden: 64, MaxSeq: 32, RopeBase: 10000, RMSEps: 1e-6}
}

// harness closes the handle, drops every model, and sets a new root that
// holds the fixture as model.safetensors and a token bin as tokens.bin.
func harness(t *testing.T) string {
	t.Helper()
	if err := Close(context.Background()); err != nil {
		t.Fatalf("close: %v", err)
	}
	engineReset()
	dir := t.TempDir()
	if err := SetModelRoot(context.Background(), dir); err != nil {
		t.Fatal(err)
	}
	installModel(t, dir, "model.safetensors")
	writeBin(t, dir, "tokens.bin")
	t.Cleanup(func() { _ = Close(context.Background()) })
	return dir
}

func installModel(t *testing.T, dir, name string) {
	t.Helper()
	body, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), body, 0o644); err != nil {
		t.Fatal(err)
	}
}

// installNaNModel writes the fixture with blocks.0.norm1.weight[0] = NaN.
func installNaNModel(t *testing.T, dir, name string) {
	t.Helper()
	body, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	n := binary.LittleEndian.Uint64(body[:8])
	var header map[string]json.RawMessage
	if err := json.Unmarshal(body[8:8+n], &header); err != nil {
		t.Fatal(err)
	}
	var info struct {
		DataOffsets [2]uint64 `json:"data_offsets"`
	}
	if err := json.Unmarshal(header["blocks.0.norm1.weight"], &info); err != nil {
		t.Fatal(err)
	}
	at := 8 + n + info.DataOffsets[0]
	binary.LittleEndian.PutUint32(body[at:], math.Float32bits(float32(math.NaN())))
	if err := os.WriteFile(filepath.Join(dir, name), body, 0o644); err != nil {
		t.Fatal(err)
	}
}

// writeTensor is a valid one-tensor safetensors file with no spec.
func writeTensor(t *testing.T, dir, name string) string {
	t.Helper()
	header := []byte(`{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`)
	body := make([]byte, 8+len(header)+4)
	body[0] = byte(len(header))
	copy(body[8:], header)
	if err := os.WriteFile(filepath.Join(dir, name), body, 0o644); err != nil {
		t.Fatal(err)
	}
	return name
}

// writeBin is 4096 headerless uint16 ids below the fixture's vocabulary.
func writeBin(t *testing.T, dir, name string) {
	t.Helper()
	body := make([]byte, 2*4096)
	for i := 0; i < 4096; i++ {
		binary.LittleEndian.PutUint16(body[2*i:], uint16((i*7+i/64)%64))
	}
	if err := os.WriteFile(filepath.Join(dir, name), body, 0o644); err != nil {
		t.Fatal(err)
	}
}

func testConfig() TrainConfig {
	return NanolabTrainConfig("tokens.bin", 2, testSeq, 2, 7, Schedule{Kind: ScheduleCosine, Warmup: 2, Total: 40})
}

func trainedModel(t *testing.T, opts LoadOptions) uint64 {
	t.Helper()
	id, err := LoadModel(context.Background(), "model.safetensors", opts)
	if err != nil {
		t.Fatal(err)
	}
	if err := OpenTrainer(context.Background(), id, testConfig()); err != nil {
		t.Fatal(err)
	}
	return id
}

// tokenRows is rows x testSeq ids below the vocabulary.
func tokenRows(rows int) []uint32 {
	x := make([]uint32, rows*testSeq)
	for i := range x {
		x[i] = uint32((i*5 + 1) % testVocab)
	}
	return x
}

func load(path string) (uint64, error) {
	return LoadModel(context.Background(), path, LoadOptions{})
}

func TestPathEscapeAndMissingFile(t *testing.T) {
	harness(t)
	for _, path := range []string{"../secret.safetensors", "a/../../etc/passwd", "/etc/passwd"} {
		if _, err := load(path); err == nil || !(strings.Contains(err.Error(), "..") || strings.Contains(err.Error(), "relative") || strings.Contains(err.Error(), "escapes")) {
			t.Fatalf("LoadModel(%q) = %v", path, err)
		}
		if _, err := Inspect(context.Background(), path); err == nil || !(strings.Contains(err.Error(), "..") || strings.Contains(err.Error(), "relative") || strings.Contains(err.Error(), "escapes")) {
			t.Fatalf("Inspect(%q) = %v", path, err)
		}
	}
	if _, err := load("missing.safetensors"); err == nil || !strings.Contains(err.Error(), "missing file") {
		t.Fatalf("missing file: %v", err)
	}
	if _, err := TrainStep(context.Background(), 0); err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Fatalf("unknown id: %v", err)
	}
}

// Inspect counts tensors from the header; LoadModel reads a real model and
// refuses a file without a spec.
func TestInspectAndLoadModel(t *testing.T) {
	dir := harness(t)
	writeTensor(t, dir, "one.safetensors")
	if n, err := Inspect(context.Background(), "one.safetensors"); err != nil || n != 1 {
		t.Fatalf("inspect one: %d %v", n, err)
	}
	n, err := Inspect(context.Background(), "model.safetensors")
	// 2 + 14 per layer (ojas_model::param_count).
	if err != nil || n != 2+14*2 {
		t.Fatalf("inspect model: %d %v", n, err)
	}
	if _, err := load("one.safetensors"); err == nil || !strings.Contains(err.Error(), "ojas.spec") {
		t.Fatalf("load without a spec: %v", err)
	}
	id, err := load("model.safetensors")
	if err != nil || id == 0 {
		t.Fatalf("load: %d %v", id, err)
	}
}

func TestDoubleFree(t *testing.T) {
	harness(t)
	id, err := load("model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if err := Free(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if err := Free(context.Background(), id); err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Fatalf("double free: %v", err)
	}
	if err := Free(context.Background(), 0); err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Fatalf("unknown id: %v", err)
	}
}

func TestSessionCap(t *testing.T) {
	harness(t)
	ids := make([]uint64, 0, 64)
	for i := 0; i < 64; i++ {
		id, err := load("model.safetensors")
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, id)
	}
	if _, err := load("model.safetensors"); !errors.Is(err, ErrCapacity) || !strings.Contains(err.Error(), "capacity exceeded") {
		t.Fatalf("cap: %v", err)
	}
	for _, id := range ids {
		if err := Free(context.Background(), id); err != nil {
			t.Fatal(err)
		}
	}
}

// The process memory ceiling caps every model's budget: a BudgetBytes above
// it is ErrCapacity at load. SetMemoryCeiling refuses 0 and refuses while a
// model is open; once the model is freed it takes effect, and the larger
// budget then loads.
func TestMemoryCeilingRoundTripsAndRefusesWhileAModelIsOpen(t *testing.T) {
	harness(t)
	ctx := context.Background()
	t.Cleanup(func() {
		if err := Close(ctx); err != nil {
			t.Errorf("close: %v", err)
		}
		if err := SetMemoryCeiling(ctx, DefaultMemoryCeiling); err != nil {
			t.Errorf("restore the default ceiling: %v", err)
		}
	})
	const big = 2 << 30
	if err := SetMemoryCeiling(ctx, 0); err == nil || !strings.Contains(err.Error(), "0 bytes") {
		t.Fatalf("zero ceiling: %v", err)
	}
	if _, err := LoadModel(ctx, "model.safetensors", LoadOptions{BudgetBytes: big}); !errors.Is(err, ErrCapacity) || !strings.Contains(err.Error(), "process ceiling") {
		t.Fatalf("budget above the default ceiling: %v", err)
	}
	id, err := load("model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if err := SetMemoryCeiling(ctx, big); err == nil || !strings.Contains(err.Error(), "free them first") {
		t.Fatalf("ceiling change with a model open: %v", err)
	}
	if err := Free(ctx, id); err != nil {
		t.Fatal(err)
	}
	if err := SetMemoryCeiling(ctx, big); err != nil {
		t.Fatalf("ceiling change with no model open: %v", err)
	}
	id, err = LoadModel(ctx, "model.safetensors", LoadOptions{BudgetBytes: big})
	if err != nil {
		t.Fatalf("budget under the raised ceiling: %v", err)
	}
	if err := Free(ctx, id); err != nil {
		t.Fatal(err)
	}
	if err := SetMemoryCeiling(ctx, DefaultMemoryCeiling); err != nil {
		t.Fatalf("restore: %v", err)
	}
}

func TestCloseEmpty(t *testing.T) {
	if err := Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	harness(t)
	id, err := load("model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if err := Free(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if err := Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// A step returns the model's loss and moves the step count; another batch
// gives another loss; a step before OpenTrainer or with the wrong seq is
// refused and the model stays.
func TestTrainStepLossAndShape(t *testing.T) {
	harness(t)
	id, err := load("model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := TrainStep(context.Background(), id); err == nil || !strings.Contains(err.Error(), "no trainer") {
		t.Fatalf("step before OpenTrainer: %v", err)
	}
	if err := OpenTrainer(context.Background(), id, testConfig()); err != nil {
		t.Fatal(err)
	}
	got, err := TrainStep(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if got.Loss <= 0 || math.IsNaN(float64(got.Loss)) || math.IsInf(float64(got.Loss), 0) || got.Step != 1 || got.Tokens != 2*2*testSeq {
		t.Fatalf("step = %+v", got)
	}
	if got.MatrixLR <= 0 || got.MatrixLR > 0.025 || got.AdamLR <= 0 || got.AdamLR > 6e-4 {
		t.Fatalf("lrs = %+v", got)
	}
	x := tokenRows(2)
	y := append([]uint32(nil), x...)
	tok, err := TrainStepTokens(context.Background(), id, testSeq, [][]uint32{x}, [][]uint32{y})
	if err != nil {
		t.Fatal(err)
	}
	for i := range y {
		y[i] = (y[i] + 3) % testVocab
	}
	other, err := TrainStepTokens(context.Background(), id, testSeq, [][]uint32{x}, [][]uint32{y})
	if err != nil {
		t.Fatal(err)
	}
	if tok.Loss == other.Loss || other.Step != 3 {
		t.Fatalf("tokens: %+v then %+v", tok, other)
	}
	if _, err := TrainStepTokens(context.Background(), id, testSeq-1, [][]uint32{x[:14]}, [][]uint32{y[:14]}); err == nil || !strings.Contains(err.Error(), "shape") {
		t.Fatalf("wrong seq: %v", err)
	}
	if err := Free(context.Background(), id); err != nil {
		t.Fatal(err)
	}
}

// Go refuses token batches it cannot encode before calling the engine.
func TestTrainStepTokensRejectsRaggedBatches(t *testing.T) {
	harness(t)
	id := trainedModel(t, LoadOptions{})
	x := tokenRows(1)
	for name, c := range map[string]struct {
		seq  uint32
		x, y [][]uint32
	}{
		"seq 0":         {0, [][]uint32{x}, [][]uint32{x}},
		"no batches":    {testSeq, nil, nil},
		"x and y count": {testSeq, [][]uint32{x, x}, [][]uint32{x}},
		"x and y len":   {testSeq, [][]uint32{x}, [][]uint32{x[:7]}},
		"not seq rows":  {testSeq, [][]uint32{x[:7]}, [][]uint32{x[:7]}},
		"empty batch":   {testSeq, [][]uint32{{}}, [][]uint32{{}}},
	} {
		if _, err := TrainStepTokens(context.Background(), id, c.seq, c.x, c.y); err == nil || !strings.HasPrefix(err.Error(), "ojas: ") {
			t.Errorf("%s: %v", name, err)
		}
	}
	if err := Free(context.Background(), id); err != nil {
		t.Fatal(err)
	}
}

// TestConcurrentLoadStepGenerateFreeOneHandle hammers the one gusset
// handle this package opens (pool size 1) from many goroutines. Expected
// contention is unknown-model and capacity errors, not a deadlock.
func TestConcurrentLoadStepGenerateFreeOneHandle(t *testing.T) {
	const (
		goroutines = 32
		rounds     = 8
	)
	harness(t)
	seed := trainedModel(t, LoadOptions{})
	stepStarted := time.Now()
	if _, err := TrainStep(context.Background(), seed); err != nil {
		t.Fatal(err)
	}
	t.Logf("one TrainStep took %s", time.Since(stepStarted))
	x := [][]uint32{tokenRows(1)}

	var (
		wg     sync.WaitGroup
		issued sync.Map
		errs   = make(chan error, goroutines)
	)
	issued.Store(seed, -1)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			fail := func(format string, args ...any) {
				errs <- fmt.Errorf("goroutine %d: "+format, append([]any{g}, args...)...)
			}
			tolerated := func(err error) bool {
				return err != nil && (strings.Contains(err.Error(), "unknown model") || strings.Contains(err.Error(), "capacity exceeded"))
			}
			for r := 0; r < rounds; r++ {
				id, err := load("model.safetensors")
				if err != nil {
					if !tolerated(err) {
						fail("load: %v", err)
						return
					}
				} else {
					if _, dup := issued.LoadOrStore(id, struct{}{}); dup {
						fail("id %d reused", id)
						return
					}
					if err := OpenTrainer(context.Background(), id, testConfig()); err != nil {
						fail("open trainer: %v", err)
						return
					}
					if _, err := TrainStepTokens(context.Background(), id, testSeq, x, x); err != nil {
						fail("step: %v", err)
						return
					}
					if tok, err := Generate(context.Background(), id, []float32{0, 2, 1}); err != nil || tok != 1 {
						fail("generate: tok=%d err=%v", tok, err)
						return
					}
					if _, err := GenerateGreedy(context.Background(), id, []uint32{1, 2}); err != nil {
						fail("greedy: %v", err)
						return
					}
					if err := Free(context.Background(), id); err != nil {
						fail("free: %v", err)
						return
					}
				}
				if _, err := TrainStepTokens(context.Background(), seed, testSeq, x, x); err != nil && !tolerated(err) {
					fail("seed step: %v", err)
					return
				}
				if _, err := Generate(context.Background(), seed, []float32{3, 1}); err != nil && !tolerated(err) {
					fail("seed generate: %v", err)
					return
				}
				if g%4 == 0 {
					if err := Free(context.Background(), seed); err != nil && !tolerated(err) {
						fail("seed free: %v", err)
						return
					}
				}
			}
		}(g)
	}
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(60 * time.Second):
		t.Fatal("deadlock: LoadModel, TrainStep, Generate, and Free on one pool-size-1 handle did not finish within 60s")
	}
	close(errs)
	for err := range errs {
		t.Error(err)
	}

	id := trainedModel(t, LoadOptions{})
	if _, err := TrainStep(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if tok, err := Generate(context.Background(), id, []float32{0, 2, 1}); err != nil || tok != 1 {
		t.Fatalf("generate after hammer: tok=%d err=%v", tok, err)
	}
	if err := Free(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if err := Free(context.Background(), seed); err != nil && !strings.Contains(err.Error(), "unknown model") {
		t.Fatal(err)
	}
}

func TestConcurrentSessionStress(t *testing.T) {
	const goroutines = 64
	const rounds = 25
	if err := Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := SetPoolSize(1); err != nil {
		t.Fatal(err)
	}
	if poolSize.Load() != 1 {
		t.Fatalf("stress pool size %d, want 1", poolSize.Load())
	}
	harness(t)
	// 80 rows of 8 ids, twice: past the inline limit, so the buffer path.
	big := [][]uint32{tokenRows(80)}
	small := [][]uint32{tokenRows(1)}
	var (
		wg     sync.WaitGroup
		issued sync.Map
		errs   = make(chan error, goroutines)
	)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			fail := func(format string, args ...any) {
				errs <- fmt.Errorf("goroutine %d: "+format, append([]any{g}, args...)...)
			}
			gone := func(err error) bool {
				if err == nil {
					return false
				}
				msg := err.Error()
				return strings.Contains(msg, "handle is closed") || strings.Contains(msg, "unknown model")
			}
			for r := 0; r < rounds; r++ {
				id, err := load("model.safetensors")
				if err != nil {
					if strings.Contains(err.Error(), "capacity exceeded") || gone(err) {
						continue
					}
					fail("load: %v", err)
					return
				}
				if _, dup := issued.LoadOrStore(id, g); dup {
					fail("id %d reused", id)
					return
				}
				if err := OpenTrainer(context.Background(), id, testConfig()); err != nil {
					if gone(err) {
						continue
					}
					fail("open trainer: %v", err)
					return
				}
				if _, err := TrainStepTokens(context.Background(), id, testSeq, small, small); err != nil {
					if gone(err) {
						continue
					}
					fail("step: %v", err)
					return
				}
				if r%5 == 0 {
					if _, err := TrainStepTokens(context.Background(), id, testSeq, big, big); err != nil {
						if gone(err) {
							continue
						}
						fail("buffer step: %v", err)
						return
					}
				}
				if tok, err := Generate(context.Background(), id, []float32{0, 2, 1}); err != nil || tok != 1 {
					if gone(err) {
						continue
					}
					fail("generate: %d %v", tok, err)
					return
				}
				if _, err := GenerateGreedy(context.Background(), id, []uint32{0, 1}); err != nil {
					if gone(err) {
						continue
					}
					fail("greedy: %v", err)
					return
				}
				if (g+r)%7 == 0 {
					if err := Close(context.Background()); err != nil {
						fail("close: %v", err)
						return
					}
					if err := Close(context.Background()); err != nil {
						fail("double close: %v", err)
						return
					}
				}
				if err := Free(context.Background(), id); err != nil {
					// Close already dropped every session. A second owner must
					// not still count toward the cap.
					if gone(err) {
						continue
					}
					fail("free: %v", err)
					return
				}
				if err := Free(context.Background(), id); err == nil || !gone(err) {
					fail("double free: %v", err)
					return
				}
				if _, err := Generate(context.Background(), id, []float32{1}); err == nil || !strings.Contains(err.Error(), "unknown model") {
					if gone(err) {
						continue
					}
					fail("generate after free: %v", err)
					return
				}
			}
		}(g)
	}
	finished := make(chan struct{})
	go func() {
		wg.Wait()
		close(finished)
	}()
	select {
	case <-finished:
	case <-time.After(4 * time.Minute):
		t.Fatal("watchdog: 64 goroutines on a pool of size 1 did not finish within 4m")
	}
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	// Every session was freed, so the whole cap is available again.
	ids := make([]uint64, 0, 64)
	for i := 0; i < 64; i++ {
		id, err := load("model.safetensors")
		if err != nil {
			t.Fatalf("load %d after stress: %v", i, err)
		}
		ids = append(ids, id)
	}
	if _, err := load("model.safetensors"); err == nil || !strings.Contains(err.Error(), "capacity exceeded") {
		t.Fatalf("cap after stress: %v", err)
	}
	for _, id := range ids {
		if err := Free(context.Background(), id); err != nil {
			t.Fatal(err)
		}
	}
	// Close during Generate must not leave a session counted after it returns.
	freed, err := load("model.safetensors")
	if err != nil {
		t.Fatalf("cap stayed occupied after Close: %v", err)
	}
	if err := Free(context.Background(), freed); err != nil {
		t.Fatal(err)
	}
}

func TestMalformedInputsAreErrors(t *testing.T) {
	dir := harness(t)
	id := trainedModel(t, LoadOptions{})
	for _, path := range []string{"", "a\x00b", "\xff\xfe", "model.safetensors/", strings.Repeat("a", 4096), strings.Repeat("a", 4097)} {
		if _, err := load(path); err == nil {
			t.Errorf("LoadModel(%q) succeeded", path)
		}
		if _, err := Inspect(context.Background(), path); err == nil {
			t.Errorf("Inspect(%q) succeeded", path)
		}
	}
	x := tokenRows(1)
	bad := append([]uint32(nil), x...)
	bad[2] = testVocab
	steps := map[string][2][][]uint32{
		"id past the vocabulary":     {{bad}, {x}},
		"target past the vocabulary": {{x}, {bad}},
	}
	for name, xy := range steps {
		if _, err := TrainStepTokens(context.Background(), id, testSeq, xy[0], xy[1]); err == nil {
			t.Errorf("%s: step succeeded", name)
		}
	}
	samples := map[string]SampleOptions{
		"nan temperature":      {Temperature: float32(math.NaN()), MaxNewTokens: 1},
		"negative temperature": {Temperature: -1, MaxNewTokens: 1},
		"top_p past 1":         {Temperature: 1, TopP: 2, MaxNewTokens: 1},
		"stop past vocabulary": {MaxNewTokens: 1, Stop: []uint32{testVocab}},
		"past the context":     {MaxNewTokens: 40},
	}
	for name, opts := range samples {
		if _, err := GenerateIDs(context.Background(), id, []uint32{1}, opts); err == nil {
			t.Errorf("%s: sample succeeded", name)
		}
	}
	if _, err := Generate(context.Background(), id, nil); err == nil {
		t.Error("empty logits returned a token")
	}
	if _, err := Generate(context.Background(), id, []float32{float32(math.Inf(-1)), 0}); !errors.Is(err, ErrNonFinite) {
		t.Errorf("inf logit: %v", err)
	}
	if _, err := GenerateGreedy(context.Background(), id, nil); err == nil {
		t.Error("empty prompt returned a token")
	}
	if _, err := TrainStep(context.Background(), math.MaxUint64); err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Errorf("unknown id step: %v", err)
	}
	if err := Free(context.Background(), math.MaxUint64); err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Errorf("unknown id free: %v", err)
	}
	if err := SetModelRoot(context.Background(), ""); err == nil {
		t.Error("empty model root accepted")
	}
	if err := SetModelRoot(context.Background(), dir+"\x00x"); err == nil {
		t.Error("NUL model root accepted")
	}
	if err := SetModelRoot(context.Background(), dir); err != nil {
		t.Fatal(err)
	}
	if err := Free(context.Background(), id); err != nil {
		t.Fatal(err)
	}
}

// Caller-logits argmax is unchanged; greedy generation reads the model, so
// its token is in the vocabulary and an id outside it is refused.
func TestGenerateNaNAndGreedy(t *testing.T) {
	harness(t)
	id, err := load("model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	token, err := Generate(context.Background(), id, []float32{0.1, 2.5, 0.2})
	if err != nil {
		t.Fatal(err)
	}
	if token != 1 {
		t.Fatalf("argmax = %d", token)
	}
	if _, err := Generate(context.Background(), id, []float32{float32(math.NaN()), float32(math.NaN())}); !errors.Is(err, ErrNonFinite) || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("nan logits: %v", err)
	}
	next, err := GenerateGreedy(context.Background(), id, []uint32{0})
	if err != nil {
		t.Fatal(err)
	}
	if next >= testVocab {
		t.Fatalf("greedy token %d", next)
	}
	ids, err := GenerateIDs(context.Background(), id, []uint32{0}, SampleOptions{MaxNewTokens: 3})
	if err != nil || len(ids) != 3 || ids[0] != next {
		t.Fatalf("greedy ids %v %v, first %d", ids, err, next)
	}
	if _, err := GenerateGreedy(context.Background(), id, []uint32{testVocab}); err == nil {
		t.Fatal("out of range prompt returned a token")
	}
}

func TestPoisonDropsSession(t *testing.T) {
	dir := harness(t)
	id, err := load("model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	// Close calls engineReset before it returns, so a Free after Close
	// passes even when this reset left the session in the table.
	engineReset()
	if err := Free(context.Background(), id); err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Fatalf("reset dropped nothing: %v", err)
	}
	id = trainedModel(t, LoadOptions{})
	err = poisonHandle(context.Background())
	if !errors.Is(err, gusset.ErrPanic) {
		t.Fatalf("panic: %v", err)
	}
	if _, err := TrainStep(context.Background(), id); !errors.Is(err, gusset.ErrPoisoned) {
		t.Fatalf("after panic: %v", err)
	}
	if err := Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := SetModelRoot(context.Background(), dir); err != nil {
		t.Fatal(err)
	}
	if err := Free(context.Background(), id); err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Fatalf("freed session survived poison: %v", err)
	}
}

func TestOverflowAndPoolSizeAreRefused(t *testing.T) {
	if err := fitUint32(int(math.MaxUint32) + 1); err == nil || !strings.Contains(err.Error(), "uint32") {
		t.Fatalf("len: %v", err)
	}
	if _, err := encodeTokens(0, 0, [][]uint32{{1}}, [][]uint32{{1}}); err == nil {
		t.Fatal("encode accepted seq 0")
	}
	if _, err := encodeTokens(0, 3, [][]uint32{{1, 2}}, [][]uint32{{1, 2}}); err == nil {
		t.Fatal("encode accepted a batch that is not whole rows")
	}
	if err := SetPoolSize(0); err == nil {
		t.Fatal("zero pool size")
	}
	if err := SetPoolSize(1); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadModel(nil, "model.safetensors", LoadOptions{}); err == nil || !strings.Contains(err.Error(), "nil context") {
		t.Fatalf("nil context: %v", err)
	}
}

// deviceLeg trains and samples on a device id and compares with a CPU id:
// each step's loss within 1e-4 relative, greedy ids equal. A NaN weight is
// ErrNonFinite from the step that hit it, and the next call on that model
// (SaveCheckpoint, which synchronizes) does not inherit it.
func deviceLeg(t *testing.T, dir string, device uint32) {
	ctx := context.Background()
	dev := trainedModel(t, LoadOptions{Device: device})
	cpu := trainedModel(t, LoadOptions{})
	near := func(a, b float32) bool {
		return math.Abs(float64(a-b)) <= 1e-4*math.Max(math.Abs(float64(b)), math.SmallestNonzeroFloat32)
	}
	for i := 0; i < 3; i++ {
		want, err := TrainStep(ctx, cpu)
		if err != nil {
			t.Fatalf("cpu step %d: %v", i, err)
		}
		got, err := TrainStep(ctx, dev)
		if err != nil {
			t.Fatalf("device step %d: %v", i, err)
		}
		if !near(got.Loss, want.Loss) || !near(got.GradNorm, want.GradNorm) || got.Step != want.Step || got.MatrixLR != want.MatrixLR {
			t.Fatalf("step %d: device %+v, cpu %+v", i, got, want)
		}
	}
	for _, prompt := range [][]uint32{{0}, {1}, {0, 1, 1}} {
		want, err := GenerateIDs(ctx, cpu, prompt, SampleOptions{MaxNewTokens: 3})
		if err != nil {
			t.Fatal(err)
		}
		got, err := GenerateIDs(ctx, dev, prompt, SampleOptions{MaxNewTokens: 3})
		if err != nil || fmt.Sprint(got) != fmt.Sprint(want) {
			t.Fatalf("greedy %v: device %v (%v), cpu %v", prompt, got, err, want)
		}
	}
	if tok, err := Generate(ctx, dev, []float32{0, 2, 1}); err != nil || tok != 1 {
		t.Fatalf("Generate on the device = %d, %v", tok, err)
	}
	installNaNModel(t, dir, "nan.safetensors")
	nan, err := LoadModel(ctx, "nan.safetensors", LoadOptions{Device: device})
	if err != nil {
		t.Fatal(err)
	}
	if err := OpenTrainer(ctx, nan, testConfig()); err != nil {
		t.Fatal(err)
	}
	if _, err := TrainStep(ctx, nan); !errors.Is(err, ErrNonFinite) {
		t.Fatalf("NaN step on the device: %v", err)
	}
	if err := SaveCheckpoint(ctx, nan, "nan-ckpt"); err != nil {
		t.Fatalf("the call after a NaN step inherited it: %v", err)
	}
	for _, id := range []uint64{dev, cpu, nan} {
		if err := Free(ctx, id); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := TrainStep(ctx, dev); err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Fatalf("step after free: %v", err)
	}
}

// DeviceWgpu trains and samples like a CPU model. With no wgpu adapter the
// load is a "wgpu:" error and leaves no session behind.
func TestWgpuSessionMatchesCPU(t *testing.T) {
	dir := harness(t)
	ctx := context.Background()
	probe, err := LoadModel(ctx, "model.safetensors", LoadOptions{Device: DeviceWgpu})
	if err != nil {
		if !strings.HasPrefix(err.Error(), "wgpu:") {
			t.Fatalf("wgpu load: %v", err)
		}
		t.Logf("no wgpu adapter; checking the load failed closed: %v", err)
		ids := make([]uint64, 0, 64)
		for i := 0; i < 64; i++ {
			id, err := load("model.safetensors")
			if err != nil {
				t.Fatalf("load %d after a wgpu refusal: %v", i, err)
			}
			ids = append(ids, id)
		}
		for _, id := range ids {
			if err := Free(ctx, id); err != nil {
				t.Fatal(err)
			}
		}
		return
	}
	if err := Free(ctx, probe); err != nil {
		t.Fatal(err)
	}
	deviceLeg(t, dir, DeviceWgpu)
}

// On macOS DeviceMetal trains and samples like a CPU model. Off macOS the
// load is a "metal:" error.
func TestMetalSessionMatchesCPU(t *testing.T) {
	dir := harness(t)
	if runtime.GOOS != "darwin" {
		id, err := LoadModel(context.Background(), "model.safetensors", LoadOptions{Device: DeviceMetal})
		if err == nil || !strings.HasPrefix(err.Error(), "metal:") {
			t.Fatalf("Metal off macOS: id=%d err=%v", id, err)
		}
		return
	}
	deviceLeg(t, dir, DeviceMetal)
}

func TestCPUParallelThreadCountIsBounded(t *testing.T) {
	harness(t)
	for _, threads := range []uint32{1, 4, MaxCPUThreads} {
		id := trainedModel(t, LoadOptions{Device: DeviceCPUParallel, Threads: threads})
		if _, err := TrainStep(context.Background(), id); err != nil {
			t.Fatalf("threads %d step: %v", threads, err)
		}
		if err := Free(context.Background(), id); err != nil {
			t.Fatal(err)
		}
	}
	for _, threads := range []uint32{MaxCPUThreads + 1, 1 << 20, math.MaxUint32} {
		if id, err := LoadModel(context.Background(), "model.safetensors", LoadOptions{Device: DeviceCPUParallel, Threads: threads}); err == nil || !strings.Contains(err.Error(), "exceeds") {
			t.Fatalf("threads %d: id=%d err=%v", threads, id, err)
		}
	}
	if _, err := LoadModel(context.Background(), "model.safetensors", LoadOptions{Device: DeviceCPUParallel}); err == nil || !strings.Contains(err.Error(), "thread count is 0") {
		t.Fatalf("zero threads: %v", err)
	}
	if _, err := LoadModel(context.Background(), "model.safetensors", LoadOptions{Device: DeviceMetal, Numerics: NumericsExact}); err == nil || !strings.Contains(err.Error(), "fixed by") {
		t.Fatalf("numerics on Metal: %v", err)
	}
}

func TestCancelledContextIsAContextError(t *testing.T) {
	harness(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if id, err := LoadModel(ctx, "model.safetensors", LoadOptions{}); err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("LoadModel on a cancelled context: id=%d err=%v", id, err)
	}
	expired, stop := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer stop()
	if _, err := Generate(expired, 1, []float32{0, 1}); err == nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Generate past its deadline: %v", err)
	}
	// A refused call did not leave a session behind.
	id, err := load("model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if err := Free(context.Background(), id); err != nil {
		t.Fatal(err)
	}
}

func TestPkgconfigNamesAppleFrameworks(t *testing.T) {
	for _, path := range []string{"gusset.pc", "release/gusset.pc"} {
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		text := string(body)
		for _, fragment := range []string{"-framework Metal", "-framework Foundation", "-framework QuartzCore", "-framework CoreFoundation", "-framework CoreGraphics", "-framework Accelerate", "-lobjc"} {
			if !strings.Contains(text, fragment) {
				t.Fatalf("%s missing %s", path, fragment)
			}
		}
	}
	debug, err := os.ReadFile("gusset.pc")
	if err != nil {
		t.Fatal(err)
	}
	release, err := os.ReadFile("release/gusset.pc")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(debug), "target/debug") || !strings.Contains(string(release), "target/release") {
		t.Fatal("pc files do not name distinct profiles")
	}
	linux, err := os.ReadFile("linux/gusset.pc")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(linux), "-framework") || !strings.Contains(string(linux), "target/debug") || !strings.Contains(string(linux), "-lgusset") {
		t.Fatalf("linux/gusset.pc must link target/debug libgusset without Apple frameworks:\n%s", linux)
	}
}
