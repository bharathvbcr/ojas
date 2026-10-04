package ojas

import (
	"context"
	"errors"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
	"time"

	"github.com/bharathvbcr/gusset"
)

// kindSentinels is every in-band sentinel, from the one table the decoder
// reads.
func kindSentinels() []error {
	out := make([]error, 0, len(inBandKinds))
	for _, k := range inBandKinds {
		out = append(out, k.sentinel)
	}
	return out
}

// The table matches ojas-capi's ErrorKind::prefix list (a Rust test checks
// each Rust prefix appears in ffi.go), every sentinel is distinct, and no
// prefix is a prefix of another, so the decoder's order cannot matter.
func TestInBandKindsTable(t *testing.T) {
	want := []string{
		"ojas:E_CAPACITY:", "ojas:E_DEVICE_LOST:", "ojas:E_BUSY:",
		"ojas:E_NONFINITE:", "ojas:E_POISONED:", "ojas:E_PRESSURE:",
	}
	if len(inBandKinds) != len(want) {
		t.Fatalf("%d kinds, want %d", len(inBandKinds), len(want))
	}
	for i, k := range inBandKinds {
		if k.prefix != want[i] {
			t.Fatalf("kind %d is %q, want %q", i, k.prefix, want[i])
		}
		for j, o := range inBandKinds {
			if i == j {
				continue
			}
			if errors.Is(k.sentinel, o.sentinel) {
				t.Fatalf("%q and %q share a sentinel", k.prefix, o.prefix)
			}
			if strings.HasPrefix(k.prefix, o.prefix) {
				t.Fatalf("%q starts with %q", k.prefix, o.prefix)
			}
		}
	}
}

func TestInBandErrorPrefixes(t *testing.T) {
	cases := []struct {
		err  error
		msg  string
		want error
	}{
		{errors.New("ojas:E_CAPACITY: session cap"), "ojas:E_CAPACITY: session cap", ErrCapacity},
		// A gusset engine error: the kind leads its Msg, after gusset's head.
		{&gusset.Error{Code: 1, Msg: "ojas:E_DEVICE_LOST: gpu reset"}, "gusset error [1]: ojas:E_DEVICE_LOST: gpu reset", ErrDeviceLost},
		{errors.New("ojas:E_NONFINITE: logit"), "ojas:E_NONFINITE: logit", ErrNonFinite},
		{errors.New("ojas:E_POISONED: train_step: poisoned"), "ojas:E_POISONED: train_step: poisoned", ErrPoisoned},
		// Critical pressure is its own kind: transient, never ErrCapacity.
		{&gusset.Error{Code: 1, Msg: "ojas:E_PRESSURE: memory pressure: opcode 8 refused"}, "ojas:E_PRESSURE: memory pressure: opcode 8 refused", ErrPressure},
		// F10: a kind that does not lead the engine's message, such as one
		// inside an echoed user path, selects nothing. This case expected
		// ErrBusy while the match was a substring search.
		{errors.New("prefix ojas:E_BUSY: try later"), "prefix ojas:E_BUSY: try later", nil},
	}
	for _, tc := range cases {
		err := annotateEngineError(tc.err)
		for _, s := range kindSentinels() {
			if errors.Is(err, s) != (s == tc.want) {
				t.Errorf("%q: errors.Is(%v) = %v", tc.msg, s, errors.Is(err, s))
			}
		}
		if err == nil || !strings.Contains(err.Error(), tc.msg) {
			t.Errorf("%q: message dropped: %v", tc.msg, err)
		}
	}
	plain := errors.New("missing file")
	if got := annotateEngineError(plain); got != plain {
		t.Fatalf("plain error rewritten: %v", got)
	}
	if annotateEngineError(nil) != nil {
		t.Fatal("nil became an error")
	}
}

func TestMemoryGovernorIsOptIn(t *testing.T) {
	if memoryGovernorTotal.Load() != 0 {
		t.Fatal("governor is on before anyone opts in")
	}
	if err := WithMemoryGovernor(0); err == nil {
		t.Fatal("accepted a zero total")
	}
	if err := WithMemoryGovernor(-1); err == nil {
		t.Fatal("accepted a negative total")
	}
	if memoryGovernorTotal.Load() != 0 {
		t.Fatal("a refused governor stored a total")
	}
}

func TestMemoryGovernorCapturesTotalAndRateLimitsRefresh(t *testing.T) {
	prev := gusset.AdviseMemoryLimit(-1)
	t.Cleanup(func() {
		memoryGovernorTotal.Store(0)
		memoryGovernorLast.Store(0)
		debug.SetMemoryLimit(prev)
	})
	const total int64 = 1 << 30
	if err := WithMemoryGovernor(total); err != nil {
		t.Fatal(err)
	}
	if memoryGovernorTotal.Load() != total {
		t.Fatalf("captured %d", memoryGovernorTotal.Load())
	}
	stamped := memoryGovernorLast.Load()
	refreshMemoryGovernor()
	if memoryGovernorLast.Load() != stamped {
		t.Fatal("AdviseMemoryLimit ran again inside the interval")
	}
	memoryGovernorLast.Store(time.Now().Add(-2 * memoryGovernorInterval).UnixNano())
	refreshMemoryGovernor()
	if time.Since(time.Unix(0, memoryGovernorLast.Load())) > time.Second {
		t.Fatal("a stale governor stamp was not refreshed")
	}
}

func TestDefaultBufferBudgetIsSharedAndBelowOneMaxBuffer(t *testing.T) {
	if defaultBufferBudget != 64<<20 {
		t.Fatalf("production default is %d, want 64 MiB", defaultBufferBudget)
	}
	if defaultBufferBudget >= int64(gusset.MaxBufferBytes) {
		t.Fatalf("default budget %d is not below MaxBufferBytes", defaultBufferBudget)
	}
}

func TestBufferBudgetRefusesAQueuedNewBuffer(t *testing.T) {
	if err := Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := SetBufferBudget(32); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = Close(context.Background())
		if err := SetBufferBudget(defaultBufferBudget); err != nil {
			t.Error(err)
		}
	})
	dir := t.TempDir()
	if err := SetModelRoot(context.Background(), dir); err != nil {
		t.Fatal(err)
	}
	installModel(t, dir, "model.safetensors")
	id, err := LoadModel(context.Background(), "model.safetensors", LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	rows := 2000
	x := [][]uint32{make([]uint32, rows*testSeq)}
	_, err = TrainStepTokens(context.Background(), id, testSeq, x, x)
	if !errors.Is(err, gusset.ErrBufferBudget) {
		t.Fatalf("step over the buffer budget: %v", err)
	}
}

func TestWithDerivedPoolSizeFollowsGOMAXPROCS(t *testing.T) {
	if err := Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = Close(context.Background())
		if err := SetPoolSize(1); err != nil {
			t.Error(err)
		}
	})
	if poolSize.Load() != 1 {
		t.Fatalf("default pool size is %d, tests require 1", poolSize.Load())
	}
	if err := WithDerivedPoolSize(); err != nil {
		t.Fatal(err)
	}
	want := runtime.GOMAXPROCS(0)
	if want < 1 {
		want = 1
	}
	if want > gusset.MaxPoolSize {
		want = gusset.MaxPoolSize
	}
	if int(poolSize.Load()) != want {
		t.Fatalf("derived pool %d, GOMAXPROCS cap %d", poolSize.Load(), want)
	}
}

func TestCloseHonorsACancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := Close(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("Close: %v", err)
	}
}

func TestDrainHonorsDeadline(t *testing.T) {
	if err := calls.enter(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		calls.leave()
		calls.reopen()
	})
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	errc := make(chan error, 1)
	go func() {
		errc <- calls.drain(ctx)
	}()
	select {
	case err := <-errc:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("drain: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("drain ignored the deadline")
	}
}

func TestCloseRacingAnInFlightStepLeavesOnePool(t *testing.T) {
	if err := Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := SetPoolSize(1); err != nil {
		t.Fatal(err)
	}
	harness(t)
	id := trainedModel(t, LoadOptions{})
	// 64 rows of 8 ids, twice: past the inline limit, so the buffer path.
	x := [][]uint32{tokenRows(64)}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	entered := make(chan struct{})
	stepErr := make(chan error, 1)
	go func() {
		close(entered)
		_, err := TrainStepTokens(ctx, id, testSeq, x, x)
		stepErr <- err
	}()
	select {
	case <-entered:
	case <-ctx.Done():
		t.Fatal("step never started")
	}
	during := make(chan error, 1)
	go func() {
		_, err := LoadModel(ctx, "model.safetensors", LoadOptions{})
		during <- err
	}()
	closeErr := Close(ctx)

	var stepped error
	select {
	case stepped = <-stepErr:
	case <-time.After(30 * time.Second):
		t.Fatal("watchdog: in-flight step did not return")
	}
	var loaded error
	select {
	case loaded = <-during:
	case <-time.After(30 * time.Second):
		t.Fatal("watchdog: load during close did not return")
	}

	if closeErr != nil && strings.Contains(closeErr.Error(), "workers still running") {
		if loaded == nil || !strings.Contains(loaded.Error(), "workers still running") {
			t.Fatalf("close left workers running and a load still opened a pool: close=%v load=%v step=%v", closeErr, loaded, stepped)
		}
		return
	}
	if closeErr != nil && !errors.Is(closeErr, context.DeadlineExceeded) && !errors.Is(closeErr, context.Canceled) {
		t.Fatalf("close: %v (step %v)", closeErr, stepped)
	}
	if stepped != nil {
		msg := stepped.Error()
		if !strings.Contains(msg, "handle is closed") && !strings.Contains(msg, "unknown model") && !errors.Is(stepped, context.DeadlineExceeded) && !errors.Is(stepped, context.Canceled) {
			t.Fatalf("in-flight step: %v", stepped)
		}
	}
	if loaded != nil {
		msg := loaded.Error()
		if !strings.Contains(msg, "handle is closed") && !strings.Contains(msg, "close is still joining") && !strings.Contains(msg, "workers still running") && !errors.Is(loaded, context.DeadlineExceeded) && !errors.Is(loaded, context.Canceled) {
			t.Fatalf("load during close: %v", loaded)
		}
	}
	fresh, err := LoadModel(context.Background(), "model.safetensors", LoadOptions{})
	if err != nil {
		t.Fatalf("load after the raced close: %v", err)
	}
	if err := Free(context.Background(), fresh); err != nil {
		t.Fatal(err)
	}
}

func TestCloseKeepsTheGateShutWhenWorkersRemain(t *testing.T) {
	t.Cleanup(func() {
		closeFailed.Store(nil)
		closeJoining.Store(false)
		calls.reopen()
	})
	if err := calls.drain(context.Background()); err != nil {
		t.Fatal(err)
	}
	err := errors.New("gusset error [1]: close: workers still running after 30s; the pool stays alive until they exit")
	if got := releaseAfterClose(err); got != err {
		t.Fatalf("release: %v", got)
	}
	if enterErr := calls.enter(); enterErr == nil || !strings.Contains(enterErr.Error(), "workers still running") {
		t.Fatalf("enter after a stuck close: %v", enterErr)
	}
	if _, openErr := ensureHandle(); openErr == nil || !strings.Contains(openErr.Error(), "workers still running") {
		t.Fatalf("ensureHandle after a stuck close: %v", openErr)
	}
}
