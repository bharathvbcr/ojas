package ojas

import (
	"context"
	"errors"
	"testing"

	"github.com/bharathvbcr/gusset"
)

// wideSpec's token embedding is 8192 x 256 f32 = 8 MiB, past the 1 MiB
// request floor of the heap ceiling; every other tensor is under it.
func wideSpec() ModelSpec {
	return ModelSpec{Vocab: 8192, NEmbd: 256, NLayer: 1, NHead: 4, NKVHead: 4, HeadDim: 64, Hidden: 512, MaxSeq: 32, RopeBase: 10000, RMSEps: 1e-6}
}

// A heap ceiling the engine's large allocations run into is ErrCapacity
// from the call that made them, not an abort of this process: the same
// process then builds small models under the ceiling and the wide one once
// it is lifted. An infallible model-sized allocation on these paths would
// kill the test binary here instead of failing an assertion.
func TestHeapCeilingIsACleanError(t *testing.T) {
	harness(t)
	ctx := context.Background()
	t.Cleanup(func() { SetHeapCeiling(NoHeapCeiling) })

	// Without a ceiling the wide model builds and trains.
	trained, err := NewModel(ctx, wideSpec(), 1, LoadOptions{})
	if err != nil {
		t.Fatalf("wide model without a ceiling: %v", err)
	}
	if err := Free(ctx, trained); err != nil {
		t.Fatal(err)
	}
	opened, err := NewModel(ctx, wideSpec(), 2, LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Room for 4 MiB more than is live now: under the 8 MiB embedding.
	ceiling := gusset.Stats().LiveBytes + 4<<20
	if prev := SetHeapCeiling(ceiling); prev != NoHeapCeiling {
		t.Fatalf("previous ceiling %d, want none", prev)
	}

	// NEW: the embedding's init values.
	if _, err := NewModel(ctx, wideSpec(), 3, LoadOptions{}); !errors.Is(err, ErrCapacity) {
		t.Fatalf("wide model under the ceiling: %v, want ErrCapacity", err)
	}
	// TRAIN_OPEN: the embedding's AdamW moments, on a model built before.
	if err := OpenTrainer(ctx, opened, testConfig()); !errors.Is(err, ErrCapacity) {
		t.Fatalf("trainer under the ceiling: %v, want ErrCapacity", err)
	}

	// Small allocations are not refused: a nano model builds, trains a step
	// and is freed under the same ceiling.
	small, err := NewModel(ctx, nanoSpec(), 4, LoadOptions{})
	if err != nil {
		t.Fatalf("nano model under the ceiling: %v", err)
	}
	if err := OpenTrainer(ctx, small, testConfig()); err != nil {
		t.Fatalf("nano trainer under the ceiling: %v", err)
	}
	if _, err := TrainStep(ctx, small); err != nil {
		t.Fatalf("nano step under the ceiling: %v", err)
	}
	if err := Free(ctx, small); err != nil {
		t.Fatal(err)
	}

	// Lifted, the refused calls go through on the same models and process.
	if prev := SetHeapCeiling(NoHeapCeiling); prev != ceiling {
		t.Fatalf("previous ceiling %d, want %d", prev, ceiling)
	}
	if err := OpenTrainer(ctx, opened, testConfig()); err != nil {
		t.Fatalf("trainer once the ceiling is lifted: %v", err)
	}
	if err := Free(ctx, opened); err != nil {
		t.Fatal(err)
	}
	id, err := NewModel(ctx, wideSpec(), 3, LoadOptions{})
	if err != nil {
		t.Fatalf("wide model once the ceiling is lifted: %v", err)
	}
	if err := Free(ctx, id); err != nil {
		t.Fatal(err)
	}
}
