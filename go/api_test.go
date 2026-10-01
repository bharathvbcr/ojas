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
	t.Cleanup(func() { _ = Close(context.Background()) })
	return dir
}

func writeTensor(t *testing.T, dir, name string) string {
	t.Helper()
	header := []byte(`{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`)
	body := make([]byte, 8+len(header)+4)
	body[0] = byte(len(header))
	copy(body[8:], header)
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	return name
}

func TestPathEscapeAndMissingFile(t *testing.T) {
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	for _, path := range []string{"../secret.safetensors", "a/../../etc/passwd", "/etc/passwd"} {
		if _, err := Load(context.Background(), path); err == nil || !(strings.Contains(err.Error(), "..") || strings.Contains(err.Error(), "relative") || strings.Contains(err.Error(), "escapes")) {
			t.Fatalf("Load(context.Background(), %q) = %v", path, err)
		}
	}
	if _, err := Load(context.Background(), "missing.safetensors"); err == nil || !strings.Contains(err.Error(), "missing file") {
		t.Fatalf("missing file: %v", err)
	}
}

func TestDoubleFree(t *testing.T) {
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	id, err := Load(context.Background(), "model.safetensors")
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
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	ids := make([]uint64, 0, 64)
	for i := 0; i < 64; i++ {
		id, err := Load(context.Background(), "model.safetensors")
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, id)
	}
	if _, err := Load(context.Background(), "model.safetensors"); err == nil || !strings.Contains(err.Error(), "capacity exceeded") {
		t.Fatalf("cap: %v", err)
	}
	for _, id := range ids {
		if err := Free(context.Background(), id); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCloseEmpty(t *testing.T) {
	if err := Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	id, err := Load(context.Background(), "model.safetensors")
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

func TestStepLossAndShape(t *testing.T) {
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	id, err := Load(context.Background(), "model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	got, err := Step(context.Background(), id, StepRequest{
		Batch: 1, Seq: 1, Step: 0, Lr: 1e-3,
		Logits:  []float32{0, 0},
		Targets: []uint32{0},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Loss == 0 || math.IsNaN(float64(got.Loss)) || math.IsInf(float64(got.Loss), 0) {
		t.Fatalf("loss = %v", got.Loss)
	}
	if got.Lr != 1e-3 {
		t.Fatalf("lr = %v", got.Lr)
	}
	other, err := Step(context.Background(), id, StepRequest{
		Batch: 1, Seq: 1, Step: 0, Lr: 1e-3,
		Logits:  []float32{4, -3},
		Targets: []uint32{1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Loss == other.Loss {
		t.Fatalf("loss did not depend on the payload: %v", got.Loss)
	}
	if _, err := Step(context.Background(), id, StepRequest{Batch: 1, Seq: 1, Step: 0}); err == nil || !strings.Contains(err.Error(), "missing logits") {
		t.Fatalf("header-only: %v", err)
	}
	tok, err := Step(context.Background(), id, StepRequest{
		Batch: 1, Seq: 1, Step: 1, Lr: 1e-3,
		Tokens:       []uint16{3},
		TokenTargets: []uint16{0},
	})
	if err != nil {
		t.Fatal(err)
	}
	if tok.Loss == 0 || math.IsNaN(float64(tok.Loss)) {
		t.Fatalf("token loss = %v", tok.Loss)
	}
	if err := Free(context.Background(), id); err != nil {
		t.Fatal(err)
	}
}

func TestStepRejectsMixedLogitAndTokenFields(t *testing.T) {
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	id, err := Load(context.Background(), "model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	for name, req := range map[string]StepRequest{
		"logits+token targets": {Batch: 1, Seq: 1, Lr: 1e-3, Logits: []float32{0, 0}, Targets: []uint32{0}, TokenTargets: []uint16{1}},
		"tokens+targets":       {Batch: 1, Seq: 1, Lr: 1e-3, Tokens: []uint16{3}, TokenTargets: []uint16{0}, Targets: []uint32{0}},
		"targets+tokens only":  {Batch: 1, Seq: 1, Lr: 1e-3, Targets: []uint32{0}, Tokens: []uint16{3}},
	} {
		if _, err := Step(context.Background(), id, req); err == nil || !strings.Contains(err.Error(), "both logits and tokens") {
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
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	seed, err := Load(context.Background(), "model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	stepStarted := time.Now()
	if _, err := Step(context.Background(), seed, StepRequest{
		Batch: 1, Seq: 1, Lr: 1e-3,
		Logits: []float32{0, 1}, Targets: []uint32{1},
	}); err != nil {
		t.Fatal(err)
	}
	t.Logf("one Step took %s", time.Since(stepStarted))

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
				id, err := Load(context.Background(), "model.safetensors")
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
					if _, err := Step(context.Background(), id, StepRequest{
						Batch: 1, Seq: 1, Lr: 1e-3,
						Logits: []float32{0, 1}, Targets: []uint32{1},
					}); err != nil {
						fail("step: %v", err)
						return
					}
					if tok, err := Generate(context.Background(), id, []float32{0, 2, 1}); err != nil || tok != 1 {
						fail("generate: tok=%d err=%v", tok, err)
						return
					}
					if err := Free(context.Background(), id); err != nil {
						fail("free: %v", err)
						return
					}
				}
				if _, err := Step(context.Background(), seed, StepRequest{
					Batch: 1, Seq: 1, Lr: 1e-3,
					Logits: []float32{1, 0}, Targets: []uint32{0},
				}); err != nil && !tolerated(err) {
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
	case <-time.After(30 * time.Second):
		t.Fatal("deadlock: Load, Step, Generate, and Free on one pool-size-1 handle did not finish within 30s")
	}
	close(errs)
	for err := range errs {
		t.Error(err)
	}

	id, err := Load(context.Background(), "model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Step(context.Background(), id, StepRequest{
		Batch: 1, Seq: 1, Lr: 1e-3,
		Logits: []float32{0, 1}, Targets: []uint32{1},
	}); err != nil {
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
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	rows := 1100
	big := StepRequest{Batch: 1, Seq: uint32(rows), Lr: 1e-3, Logits: make([]float32, rows*2), Targets: make([]uint32, rows)}
	for i := range big.Targets {
		big.Logits[2*i+i%2] = 1
		big.Targets[i] = uint32(i % 2)
	}
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
				id, err := Load(context.Background(), "model.safetensors")
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
				if _, err := Step(context.Background(), id, StepRequest{Batch: 1, Seq: 1, Lr: 1e-3, Logits: []float32{0, 1}, Targets: []uint32{1}}); err != nil {
					if gone(err) {
						continue
					}
					fail("step: %v", err)
					return
				}
				if _, err := Step(context.Background(), id, StepRequest{Batch: 1, Seq: 1, Lr: 1e-3, Tokens: []uint16{3}, TokenTargets: []uint16{0}}); err != nil {
					if gone(err) {
						continue
					}
					fail("token step: %v", err)
					return
				}
				if r%5 == 0 {
					if _, err := Step(context.Background(), id, big); err != nil {
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
		id, err := Load(context.Background(), "model.safetensors")
		if err != nil {
			t.Fatalf("load %d after stress: %v", i, err)
		}
		ids = append(ids, id)
	}
	if _, err := Load(context.Background(), "model.safetensors"); err == nil || !strings.Contains(err.Error(), "capacity exceeded") {
		t.Fatalf("cap after stress: %v", err)
	}
	for _, id := range ids {
		if err := Free(context.Background(), id); err != nil {
			t.Fatal(err)
		}
	}
	// Close during Generate must not leave a session counted after it returns.
	freed, err := Load(context.Background(), "model.safetensors")
	if err != nil {
		t.Fatalf("cap stayed occupied after Close: %v", err)
	}
	if err := Free(context.Background(), freed); err != nil {
		t.Fatal(err)
	}
}

func TestMalformedInputsAreErrors(t *testing.T) {
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	id, err := Load(context.Background(), "model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"", "a\x00b", "\xff\xfe", "model.safetensors/", strings.Repeat("a", 4096), strings.Repeat("a", 4097)} {
		if _, err := Load(context.Background(), path); err == nil {
			t.Errorf("Load(context.Background(), %q) succeeded", path)
		}
	}
	steps := map[string]StepRequest{
		"batch*seq overflow": {Batch: math.MaxUint32, Seq: math.MaxUint32, Lr: 1e-3, Logits: []float32{0, 0}, Targets: []uint32{0}},
		"zero batch":         {Batch: 0, Seq: 1, Lr: 1e-3, Logits: []float32{0, 0}, Targets: []uint32{0}},
		"ragged logits":      {Batch: 1, Seq: 1, Lr: 1e-3, Logits: []float32{0, 0, 0}, Targets: []uint32{0, 0}},
		"target >= classes":  {Batch: 1, Seq: 1, Lr: 1e-3, Logits: []float32{0, 0}, Targets: []uint32{9}},
		"nan lr":             {Batch: 1, Seq: 1, Lr: float32(math.NaN()), Logits: []float32{0, 0}, Targets: []uint32{0}},
		"negative lr":        {Batch: 1, Seq: 1, Lr: -1, Logits: []float32{0, 0}, Targets: []uint32{0}},
		"inf logit":          {Batch: 1, Seq: 1, Lr: 1e-3, Logits: []float32{float32(math.Inf(1)), 0}, Targets: []uint32{0}},
		"tokens no targets":  {Batch: 1, Seq: 1, Lr: 1e-3, Tokens: []uint16{3}},
		"short tokens":       {Batch: 2, Seq: 1, Lr: 1e-3, Tokens: []uint16{3}, TokenTargets: []uint16{0}},
	}
	for name, req := range steps {
		if _, err := Step(context.Background(), id, req); err == nil {
			t.Errorf("%s: step succeeded", name)
		}
	}
	if _, err := Generate(context.Background(), id, nil); err == nil {
		t.Error("empty logits returned a token")
	}
	if _, err := Generate(context.Background(), id, []float32{float32(math.Inf(-1)), 0}); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Errorf("inf logit: %v", err)
	}
	if _, err := GenerateGreedy(context.Background(), id, nil); err == nil {
		t.Error("empty prompt returned a token")
	}
	if _, err := Step(context.Background(), math.MaxUint64, StepRequest{Batch: 1, Seq: 1, Lr: 1e-3, Logits: []float32{0, 0}, Targets: []uint32{0}}); err == nil || !strings.Contains(err.Error(), "unknown model") {
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

func TestGenerateNaN(t *testing.T) {
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	id, err := Load(context.Background(), "model.safetensors")
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
	if _, err := Generate(context.Background(), id, []float32{float32(math.NaN()), float32(math.NaN())}); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("nan logits: %v", err)
	}
	next, err := GenerateGreedy(context.Background(), id, []uint32{0})
	if err != nil {
		t.Fatal(err)
	}
	if next > 1 {
		t.Fatalf("greedy token %d", next)
	}
	if _, err := GenerateGreedy(context.Background(), id, []uint32{7}); err == nil {
		t.Fatal("out of range prompt returned a token")
	}
}

func TestPoisonDropsSession(t *testing.T) {
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	id, err := Load(context.Background(), "model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	// Close calls engineReset before it returns, so a Free after Close
	// passes even when this reset left the session in the table.
	engineReset()
	if err := Free(context.Background(), id); err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Fatalf("reset dropped nothing: %v", err)
	}
	id, err = Load(context.Background(), "model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	err = poisonHandle(context.Background())
	if !errors.Is(err, gusset.ErrPanic) {
		t.Fatalf("panic: %v", err)
	}
	if _, err := Step(context.Background(), id, StepRequest{
		Batch: 1, Seq: 1, Step: 0, Lr: 1e-3,
		Logits: []float32{0, 0}, Targets: []uint32{0},
	}); !errors.Is(err, gusset.ErrPoisoned) {
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
	if _, err := mulRows(math.MaxUint32, math.MaxUint32); err == nil || !strings.Contains(err.Error(), "overflows") {
		t.Fatalf("rows: %v", err)
	}
	if err := fitUint32(int(math.MaxUint32) + 1); err == nil || !strings.Contains(err.Error(), "uint32") {
		t.Fatalf("len: %v", err)
	}
	if _, err := encodeStep(0, StepRequest{Batch: math.MaxUint32, Seq: math.MaxUint32}); err == nil {
		t.Fatal("encode accepted an overflowing batch*seq")
	}
	if err := SetPoolSize(0); err == nil {
		t.Fatal("zero pool size")
	}
	if err := SetPoolSize(1); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(nil, "model.safetensors"); err == nil || !strings.Contains(err.Error(), "nil context") {
		t.Fatalf("nil context: %v", err)
	}
}

// DeviceWgpu returns a model whose Step and GenerateGreedy match a CPU model
// of the same request (loss and grad norm within 1e-4 relative). That the
// wgpu path does not fall back to the CPU is asserted in Rust, where device
// readbacks are visible. With no wgpu adapter the load is a "wgpu:" error
// and leaves no session behind.
func TestWgpuSessionMatchesCPU(t *testing.T) {
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	ctx := context.Background()
	wgpu, err := LoadOn(ctx, DeviceWgpu, 1, "model.safetensors")
	if err != nil {
		if !strings.HasPrefix(err.Error(), "wgpu:") {
			t.Fatalf("wgpu load: %v", err)
		}
		t.Logf("no wgpu adapter; checking the load failed closed: %v", err)
		ids := make([]uint64, 0, 64)
		for i := 0; i < 64; i++ {
			id, err := Load(ctx, "model.safetensors")
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
	cpu, err := Load(ctx, "model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	logits := make([]float32, 30)
	for i := range logits {
		logits[i] = (float32(i*7%11) - 5) * 0.37
	}
	near := func(a, b float32) bool {
		return math.Abs(float64(a-b)) <= 1e-4*math.Max(math.Abs(float64(b)), math.SmallestNonzeroFloat32)
	}
	for name, req := range map[string]StepRequest{
		"logits": {Batch: 1, Seq: 6, Step: 1, Lr: 1e-3, Logits: logits, Targets: []uint32{0, 4, 2, 1, 3, 4}},
		"tokens": {Batch: 1, Seq: 5, Step: 1, Lr: 1e-3, Tokens: []uint16{3, 1, 4, 1, 5}, TokenTargets: []uint16{5, 0, 2, 4, 1}},
	} {
		want, err := Step(ctx, cpu, req)
		if err != nil {
			t.Fatalf("%s cpu step: %v", name, err)
		}
		got, err := Step(ctx, wgpu, req)
		if err != nil {
			t.Fatalf("%s wgpu step: %v", name, err)
		}
		if !near(got.Loss, want.Loss) || !near(got.GradNorm, want.GradNorm) || got.Lr != want.Lr {
			t.Fatalf("%s: wgpu %+v, cpu %+v", name, got, want)
		}
	}
	if _, err := Step(ctx, wgpu, StepRequest{Batch: 1, Seq: 1, Lr: 1e-3, Logits: []float32{float32(math.NaN()), 0}, Targets: []uint32{0}}); !errors.Is(err, ErrNonFinite) {
		t.Fatalf("NaN step on wgpu: %v", err)
	}
	for _, prompt := range [][]uint32{{0}, {1}, {0, 1, 1}} {
		want, err := GenerateGreedy(ctx, cpu, prompt)
		if err != nil {
			t.Fatal(err)
		}
		got, err := GenerateGreedy(ctx, wgpu, prompt)
		if err != nil || got != want {
			t.Fatalf("greedy %v: wgpu %d (%v), cpu %d", prompt, got, err, want)
		}
	}
	if tok, err := Generate(ctx, wgpu, []float32{0, 2, 1}); err != nil || tok != 1 {
		t.Fatalf("Generate on wgpu = %d, %v", tok, err)
	}
	for _, id := range []uint64{wgpu, cpu} {
		if err := Free(ctx, id); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := Step(ctx, wgpu, StepRequest{Batch: 1, Seq: 1, Lr: 1e-3, Logits: []float32{0, 1}, Targets: []uint32{1}}); err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Fatalf("step after free: %v", err)
	}
}

// On macOS DeviceMetal returns a model whose Step and GenerateGreedy match a
// CPU model of the same request (loss and grad norm within 1e-4 relative).
// That the Metal path does not fall back to the CPU is asserted in Rust,
// where device readbacks are visible. Off macOS the load is a "metal:" error.
func TestMetalSessionMatchesCPU(t *testing.T) {
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	ctx := context.Background()
	metal, err := LoadOn(ctx, DeviceMetal, 1, "model.safetensors")
	if runtime.GOOS != "darwin" {
		if err == nil || !strings.HasPrefix(err.Error(), "metal:") {
			t.Fatalf("Metal off macOS: id=%d err=%v", metal, err)
		}
		return
	}
	if err != nil {
		t.Fatalf("Metal load: %v", err)
	}
	cpu, err := Load(ctx, "model.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	logits := make([]float32, 30)
	for i := range logits {
		logits[i] = (float32(i*7%11) - 5) * 0.37
	}
	near := func(a, b float32) bool {
		return math.Abs(float64(a-b)) <= 1e-4*math.Max(math.Abs(float64(b)), math.SmallestNonzeroFloat32)
	}
	for name, req := range map[string]StepRequest{
		"logits": {Batch: 1, Seq: 6, Step: 1, Lr: 1e-3, Logits: logits, Targets: []uint32{0, 4, 2, 1, 3, 4}},
		"tokens": {Batch: 1, Seq: 5, Step: 1, Lr: 1e-3, Tokens: []uint16{3, 1, 4, 1, 5}, TokenTargets: []uint16{5, 0, 2, 4, 1}},
	} {
		want, err := Step(ctx, cpu, req)
		if err != nil {
			t.Fatalf("%s cpu step: %v", name, err)
		}
		got, err := Step(ctx, metal, req)
		if err != nil {
			t.Fatalf("%s metal step: %v", name, err)
		}
		if !near(got.Loss, want.Loss) || !near(got.GradNorm, want.GradNorm) || got.Lr != want.Lr {
			t.Fatalf("%s: metal %+v, cpu %+v", name, got, want)
		}
	}
	for _, prompt := range [][]uint32{{0}, {1}, {0, 1, 1}} {
		want, err := GenerateGreedy(ctx, cpu, prompt)
		if err != nil {
			t.Fatal(err)
		}
		got, err := GenerateGreedy(ctx, metal, prompt)
		if err != nil || got != want {
			t.Fatalf("greedy %v: metal %d (%v), cpu %d", prompt, got, err, want)
		}
	}
	if tok, err := Generate(ctx, metal, []float32{0, 2, 1}); err != nil || tok != 1 {
		t.Fatalf("Generate on Metal = %d, %v", tok, err)
	}
	for _, id := range []uint64{metal, cpu} {
		if err := Free(ctx, id); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := Step(ctx, metal, StepRequest{Batch: 1, Seq: 1, Lr: 1e-3, Logits: []float32{0, 1}, Targets: []uint32{1}}); err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Fatalf("step after free: %v", err)
	}
}

func TestCPUParallelThreadCountIsBounded(t *testing.T) {
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	for _, threads := range []uint32{1, 4, MaxCPUThreads} {
		id, err := LoadOn(context.Background(), DeviceCPUParallel, threads, "model.safetensors")
		if err != nil {
			t.Fatalf("threads %d: %v", threads, err)
		}
		if _, err := Step(context.Background(), id, StepRequest{Batch: 1, Seq: 1, Lr: 1e-3, Logits: []float32{0, 1}, Targets: []uint32{1}}); err != nil {
			t.Fatalf("threads %d step: %v", threads, err)
		}
		if err := Free(context.Background(), id); err != nil {
			t.Fatal(err)
		}
	}
	for _, threads := range []uint32{MaxCPUThreads + 1, 1 << 20, math.MaxUint32} {
		if id, err := LoadOn(context.Background(), DeviceCPUParallel, threads, "model.safetensors"); err == nil || !strings.Contains(err.Error(), "exceeds") {
			t.Fatalf("threads %d: id=%d err=%v", threads, id, err)
		}
	}
	if _, err := LoadOn(context.Background(), DeviceCPUParallel, 0, "model.safetensors"); err == nil || !strings.Contains(err.Error(), "thread count is 0") {
		t.Fatalf("zero threads: %v", err)
	}
}

func TestCancelledContextIsAContextError(t *testing.T) {
	dir := harness(t)
	writeTensor(t, dir, "model.safetensors")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if id, err := Load(ctx, "model.safetensors"); err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("Load on a cancelled context: id=%d err=%v", id, err)
	}
	expired, stop := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer stop()
	if _, err := Generate(expired, 1, []float32{0, 1}); err == nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Generate past its deadline: %v", err)
	}
	// A refused call did not leave a session behind.
	id, err := Load(context.Background(), "model.safetensors")
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
