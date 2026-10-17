package ojas

import (
	"context"
	"errors"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// Two tests of ErrCapacity on DeviceMetal, both the BUDGET path, not the
// device path. A Metal session reserves every tensor in its Budget before the
// device is asked for memory (ojas-metal/src/backend.rs), and the budget is
// cut to the device's room (ojas-device/src/plan.rs), so no request Go can
// make reaches tessl's allocator refusal: ops index with u32 dims, which stay
// under the device's buffer limit. The device-path proof (tessl's real
// allocator refusing, classified as CapacityExceeded) is Rust-side:
// a_real_oversize_allocation_is_capacity_exceeded_without_a_retry and
// classify_alloc_reads_tessls_failures in ojas-metal/src/device.rs.

// paramBytes reads the nano model's parameter size from the preflight's own
// refusal under a 1-byte budget on the CPU device.
func paramBytes(t *testing.T) uint64 {
	t.Helper()
	_, err := LoadModel(context.Background(), "model.safetensors", LoadOptions{BudgetBytes: 1})
	m := regexp.MustCompile(`parameters take (\d+) bytes`).FindStringSubmatch(engineMessage(err))
	if m == nil {
		t.Fatalf("no parameter size in %v", err)
	}
	n, perr := strconv.ParseUint(m[1], 10, 64)
	if perr != nil {
		t.Fatal(perr)
	}
	return n
}

// A budget under the model's parameters is refused by ojas-capi's load
// preflight (ojas-capi/src/load.rs preflight, spec-only: before the device
// opens or a weight is read), so it never reaches the Metal Budget and is
// the same on every device (that is why this test does not discriminate
// Metal). The process still loads normally afterwards.
func TestMetalLoadUnderParameterBytesIsRefusedByPreflight(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal needs macOS")
	}
	harness(t)
	skipWithoutMetal4(t)
	ctx := context.Background()
	_, err := LoadModel(ctx, "model.safetensors", LoadOptions{Device: DeviceMetal, BudgetBytes: 1024})
	if !errors.Is(err, ErrCapacity) || !strings.Contains(engineMessage(err), "parameters take") {
		t.Fatalf("metal load under a 1 KiB budget: %v, want the preflight's ErrCapacity", err)
	}
	id, err := LoadModel(ctx, "model.safetensors", LoadOptions{Device: DeviceMetal})
	if err != nil {
		t.Fatalf("metal load after a refused one: %v", err)
	}
	if err := Free(ctx, id); err != nil {
		t.Fatal(err)
	}
}

// A budget that passes the preflight (it covers the parameters) but not what
// the session then reserves is refused by the Metal session's Budget, with
// the budget's figures: the load's device uploads (budget = parameter bytes,
// each upload charged before the device is asked), and a trainer's optimizer
// state on a model that did load. The message is the budget's
// ("requested N bytes, cap C, live L"), not the preflight's.
func TestMetalSessionBudgetRefusalIsErrCapacity(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal needs macOS")
	}
	harness(t)
	skipWithoutMetal4(t)
	ctx := context.Background()
	n := paramBytes(t)
	budgetRefusal := regexp.MustCompile(`requested \d+ bytes, cap \d+, live \d+`)

	_, err := LoadModel(ctx, "model.safetensors", LoadOptions{Device: DeviceMetal, BudgetBytes: n})
	msg := engineMessage(err)
	if !errors.Is(err, ErrCapacity) || strings.Contains(msg, "parameters take") || !budgetRefusal.MatchString(msg) {
		t.Fatalf("load at the parameter bytes: %v, want the Metal budget's ErrCapacity", err)
	}

	id, err := LoadModel(ctx, "model.safetensors", LoadOptions{Device: DeviceMetal, BudgetBytes: n + n/4})
	if err != nil {
		t.Fatalf("load with a quarter more: %v", err)
	}
	err = OpenTrainer(ctx, id, testConfig())
	msg = engineMessage(err)
	if !errors.Is(err, ErrCapacity) || !strings.Contains(msg, "train_open") || !budgetRefusal.MatchString(msg) {
		t.Fatalf("trainer under that budget: %v, want the Metal budget's ErrCapacity", err)
	}
	if err := Free(ctx, id); err != nil {
		t.Fatal(err)
	}
}
