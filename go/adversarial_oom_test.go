package ojas

import (
	"context"
	"errors"
	"runtime"
	"strings"
	"testing"
)

// Adversarial test for card ft-cab0fdf1, criterion 4. TestMetalBudgetRefusalIsErrCapacity
// refuses a load under a 1 KiB budget. That refusal is made by
// ojas-capi/src/load.rs preflight (spec-only: "before the device opens or a
// byte of weights is read"), so it is the same on every device. If Metal's
// budget or allocator were not involved at all, the same load on the CPU
// device must refuse identically; this test shows it does, so the Metal test
// does not discriminate Metal.
func TestBudgetRefusalIsDeviceIndependent(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal needs macOS")
	}
	harness(t)
	skipWithoutMetal4(t)
	ctx := context.Background()
	for _, budget := range []uint64{1, 1024, 4096} {
		for _, dev := range []uint32{DeviceCPU, DeviceMetal} {
			_, err := LoadModel(ctx, "model.safetensors", LoadOptions{Device: dev, BudgetBytes: budget})
			if !errors.Is(err, ErrCapacity) {
				t.Fatalf("device %d budget %d: %v, want ErrCapacity", dev, budget, err)
			}
			if msg := engineMessage(err); !(strings.Contains(msg, "parameters take") && strings.Contains(msg, "session budget")) {
				t.Logf("device %d budget %d refused outside preflight: %s", dev, budget, msg)
			}
		}
	}
}
