package ojas

import (
	"context"
	"errors"
	"runtime"
	"testing"
)

// The BUDGET path to ErrCapacity on DeviceMetal, not the device path. A
// Metal session reserves every tensor in its Budget before the device is
// asked for memory (ojas-metal/src/backend.rs), and the budget is cut to the
// device's room (ojas-device/src/plan.rs), so no request that Go can make
// reaches tessl's allocator refusal: ops index with u32 dims, which stay
// under the device's buffer limit. The device-path proof (tessl's real
// allocator refusing, classified as CapacityExceeded) is Rust-side:
// a_real_oversize_allocation_is_capacity_exceeded_without_a_retry and
// classify_alloc_reads_tessls_failures in ojas-metal/src/device.rs.
//
// A budget the nano model's weights do not fit is ErrCapacity from the
// load, and the same process then loads the model under a budget that fits.
func TestMetalBudgetRefusalIsErrCapacity(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal needs macOS")
	}
	harness(t)
	skipWithoutMetal4(t)
	ctx := context.Background()
	if _, err := LoadModel(ctx, "model.safetensors", LoadOptions{Device: DeviceMetal, BudgetBytes: 1024}); !errors.Is(err, ErrCapacity) {
		t.Fatalf("metal load under a 1 KiB budget: %v, want ErrCapacity", err)
	}
	id, err := LoadModel(ctx, "model.safetensors", LoadOptions{Device: DeviceMetal})
	if err != nil {
		t.Fatalf("metal load after a refused one: %v", err)
	}
	if err := Free(ctx, id); err != nil {
		t.Fatal(err)
	}
}
