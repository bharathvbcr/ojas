package ojas

import (
	"context"
	"encoding/binary"
	"runtime"
	"strings"
	"testing"
)

// profileRecord builds a profile record: version, count, then count entries.
func profileRecord(version, count uint32, entries func(i int) (byte, uint64)) []byte {
	out := binary.LittleEndian.AppendUint32(nil, version)
	out = binary.LittleEndian.AppendUint32(out, count)
	for i := 0; i < int(count); i++ {
		k, v := entries(i)
		out = append(out, k)
		out = binary.LittleEndian.AppendUint64(out, v)
	}
	return out
}

// countClaims returns rec with its count field replaced by n, entries
// unchanged: a header that claims more fields than the bytes hold.
func countClaims(n uint32, rec []byte) []byte {
	out := append([]byte{}, rec...)
	binary.LittleEndian.PutUint32(out[4:8], n)
	return out
}

func allKnown(i int) (byte, uint64) {
	if i == 14 {
		return 1, 1
	}
	return 1, uint64(100 + i)
}

func TestDecodeProfileRefusesWhatItCannotReadWhole(t *testing.T) {
	good := profileRecord(1, profileFields, allKnown)
	p, err := decodeProfile(good)
	if err != nil {
		t.Fatal(err)
	}
	if p.BudgetBytes != 100 || p.Pressure != (Reading{115, true}) || p.UnifiedMemory != (Reading{1, true}) {
		t.Fatalf("%+v", p)
	}
	// A later version's appended fields are skipped.
	if _, err := decodeProfile(profileRecord(1, allProfileFields+3, deviceKnown)); err != nil {
		t.Fatalf("appended fields: %v", err)
	}
	discrete := profileRecord(1, profileFields, func(i int) (byte, uint64) {
		if i == 14 {
			return 1, 2
		}
		return allKnown(i)
	})
	if p, err := decodeProfile(discrete); err != nil || p.UnifiedMemory != (Reading{0, true}) {
		t.Fatalf("discrete: %+v %v", p, err)
	}
	cases := map[string][]byte{
		"short":           good[:7],
		"version":         profileRecord(2, profileFields, allKnown),
		"too few fields":  profileRecord(1, profileFields-1, allKnown),
		"truncated":       good[:len(good)-1],
		"trailing":        append(append([]byte{}, good...), 0),
		"count overflows": countClaims(0xffffffff, good),
		"known byte 2": profileRecord(1, profileFields, func(i int) (byte, uint64) {
			if i == 3 {
				return 2, 0
			}
			return allKnown(i)
		}),
		"unknown with a value": profileRecord(1, profileFields, func(i int) (byte, uint64) {
			if i == 5 {
				return 0, 7
			}
			return allKnown(i)
		}),
		"no budget": profileRecord(1, profileFields, func(i int) (byte, uint64) {
			if i == 0 {
				return 0, 0
			}
			return allKnown(i)
		}),
		"bad architecture": profileRecord(1, profileFields, func(i int) (byte, uint64) {
			if i == 14 {
				return 1, 9
			}
			return allKnown(i)
		}),
	}
	for name, b := range cases {
		if _, err := decodeProfile(b); err == nil {
			t.Errorf("%s: decoded", name)
		}
	}
	if _, err := decodeProfile(nil); err == nil || !strings.Contains(err.Error(), "short") {
		t.Fatalf("nil: %v", err)
	}
}

// deviceKnown is allKnown with the device fields a Metal probe sends.
func deviceKnown(i int) (byte, uint64) {
	switch i {
	case 20:
		return 1, uint64(DeviceMetal)
	case 25, 26:
		return 1, 1
	}
	return allKnown(i)
}

// The device fields are read when the record carries them, unknown when it
// stops at the host fields, and refused when a flag or the device is out of
// range.
func TestDecodeProfileReadsTheDeviceFields(t *testing.T) {
	p, err := decodeProfile(profileRecord(1, allProfileFields, deviceKnown))
	if err != nil {
		t.Fatal(err)
	}
	if p.ProbeDevice != (Reading{uint64(DeviceMetal), true}) || p.DeviceRoomBytes != (Reading{124, true}) ||
		p.SharedBudget != (Reading{1, true}) || p.DeviceBudgetBytes != (Reading{127, true}) ||
		p.WgpuDropsTimedOut != (Reading{129, true}) {
		t.Fatalf("%+v", p)
	}
	for _, n := range []uint32{profileFields + 1, 25, allProfileFields - 1} {
		if _, err := decodeProfile(profileRecord(1, n, deviceKnown)); err == nil {
			t.Errorf("%d fields: half a device block decoded", n)
		}
	}
	host, err := decodeProfile(profileRecord(1, profileFields, allKnown))
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range []Reading{host.ProbeDevice, host.DeviceRoomBytes, host.SharedBudget, host.WgpuDropsParked} {
		if r.Known {
			t.Fatalf("a host-only record reported a device field: %+v", host)
		}
	}
	for name, field := range map[string]int{"shares host": 25, "shared budget": 26, "probe device": 20} {
		bad := profileRecord(1, allProfileFields, func(i int) (byte, uint64) {
			if i == field {
				return 1, 9
			}
			return deviceKnown(i)
		})
		if _, err := decodeProfile(bad); err == nil {
			t.Errorf("%s 9: decoded", name)
		}
	}
}

// FuzzDecodeProfile: no input panics, and an accepted record always has a
// known budget, a 0/1 architecture, and zero values for unknown fields.
func FuzzDecodeProfile(f *testing.F) {
	f.Add(profileRecord(1, profileFields, allKnown))
	f.Add(profileRecord(1, profileFields+1, allKnown))
	f.Add(profileRecord(1, profileFields, func(int) (byte, uint64) { return 0, 0 }))
	f.Add([]byte{})
	f.Add(countClaims(0xffffffff, profileRecord(1, profileFields, allKnown)))
	f.Fuzz(func(t *testing.T, b []byte) {
		p, err := decodeProfile(b)
		if err != nil {
			return
		}
		if p.UnifiedMemory.Known && p.UnifiedMemory.Value > 1 {
			t.Fatalf("architecture %d", p.UnifiedMemory.Value)
		}
		for _, r := range []Reading{p.TotalBytes, p.AvailableBytes, p.ThreadCeiling, p.Pressure} {
			if !r.Known && r.Value != 0 {
				t.Fatalf("unknown with a value: %+v", r)
			}
		}
	})
}

func TestSystemProfileFromTheEngine(t *testing.T) {
	harness(t)
	ctx := context.Background()
	host, err := SystemProfile(ctx, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if !host.ThreadCeiling.Known || host.ThreadCeiling.Value == 0 {
		t.Fatalf("thread ceiling: %+v", host.ThreadCeiling)
	}
	if host.FastThreads.Known && host.FastThreads.Value > host.ThreadCeiling.Value {
		t.Fatalf("fast %d > ceiling %d", host.FastThreads.Value, host.ThreadCeiling.Value)
	}
	if host.AvailableBytes.Known && host.BudgetBytes > host.AvailableBytes.Value {
		t.Fatalf("budget %d > available %d", host.BudgetBytes, host.AvailableBytes.Value)
	}
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" && host.UnifiedMemory != (Reading{1, true}) {
		t.Fatalf("Apple silicon not unified: %+v", host.UnifiedMemory)
	}
	small, err := SystemProfile(ctx, 1<<20, false)
	if err != nil {
		t.Fatal(err)
	}
	if small.BudgetBytes > 1<<20 {
		t.Fatalf("budget widened: %d", small.BudgetBytes)
	}
	if host.SingleBandwidth.Known || host.MemoryBoundThreads.Known {
		t.Fatalf("bandwidth reported without being asked: %+v", host)
	}
	bw, err := SystemProfile(ctx, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	switch {
	case bw.SingleBandwidth.Known && bw.MultiBandwidth.Known && bw.MemoryBoundThreads.Known:
	case bw.AvailableBytes.Known && bw.AvailableBytes.Value < 16<<20:
		// The engine refuses to measure when 1/8 of available memory is
		// under 2 MiB; a tight CI container can hit that. Anything roomier
		// must measure.
		t.Skipf("only %d bytes available; bandwidth not measured", bw.AvailableBytes.Value)
	default:
		t.Fatalf("bandwidth not measured: %+v", bw)
	}
	if m := bw.MemoryBoundThreads.Value; m == 0 || m > bw.BandwidthThreads.Value || m > bw.ThreadCeiling.Value {
		t.Fatalf("memory-bound threads %d of %d (ceiling %d)", m, bw.BandwidthThreads.Value, bw.ThreadCeiling.Value)
	}
	// The profile drives a load: threads from the plan, clamped to the
	// engine's own cap.
	threads := uint32(min(host.ThreadCeiling.Value, uint64(MaxCPUThreads)))
	id := trainedModel(t, LoadOptions{Device: DeviceCPUParallel, Threads: threads})
	if _, err := TrainStep(ctx, id); err != nil {
		t.Fatalf("step on %d threads: %v", threads, err)
	}
	if err := Free(ctx, id); err != nil {
		t.Fatal(err)
	}
}

// DeviceProfile names the probe it planned against and the room it read.
// On Apple silicon the Metal device shares host memory, so its room is at
// most the host budget and the plan takes the shared budget path.
func TestDeviceProfileShowsTheMetalProbeAndItsRoom(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal needs macOS")
	}
	harness(t)
	skipWithoutMetal4(t)
	ctx := context.Background()
	if _, err := DeviceProfile(ctx, 0, DeviceCPU); err == nil {
		t.Fatal("a CPU device profile was accepted")
	}
	if host, err := SystemProfile(ctx, 0, false); err != nil || host.ProbeDevice.Known {
		t.Fatalf("a host profile named a probe: %+v %v", host.ProbeDevice, err)
	}
	p, err := DeviceProfile(ctx, 1<<30, DeviceMetal)
	if err != nil {
		t.Fatal(err)
	}
	if p.ProbeDevice != (Reading{uint64(DeviceMetal), true}) || !p.DeviceMemoryBytes.Known ||
		!p.DeviceRoomBytes.Known || !p.DeviceBudgetBytes.Known || !p.DevicePoolCacheBytes.Known {
		t.Fatalf("metal probe fields: %+v", p)
	}
	if p.DeviceBudgetBytes.Value > 1<<30 || p.DeviceBudgetBytes.Value > p.DeviceRoomBytes.Value {
		t.Fatalf("device budget %d past the caller or the room %d", p.DeviceBudgetBytes.Value, p.DeviceRoomBytes.Value)
	}
	if runtime.GOARCH == "arm64" {
		if p.DeviceSharesHost != (Reading{1, true}) || p.SharedBudget != (Reading{1, true}) {
			t.Fatalf("Apple silicon Metal not on the shared budget: %+v", p)
		}
		if p.DeviceRoomBytes.Value > p.BudgetBytes {
			t.Fatalf("shared room %d past the host budget %d", p.DeviceRoomBytes.Value, p.BudgetBytes)
		}
	}
}

// wgpu reports no memory size: its room stays unknown, never a default,
// and the session budget is the caller's cut only by what is known.
func TestDeviceProfileShowsTheWgpuProbe(t *testing.T) {
	harness(t)
	w, err := DeviceProfile(context.Background(), 1<<30, DeviceWgpu)
	if err != nil {
		if !strings.HasPrefix(engineMessage(err), "wgpu:") {
			t.Fatal(err)
		}
		t.Skipf("no wgpu adapter: %v", err)
	}
	if w.ProbeDevice != (Reading{uint64(DeviceWgpu), true}) || w.DeviceMemoryBytes.Known || w.DeviceRoomBytes.Known {
		t.Fatalf("wgpu probe fields: %+v", w)
	}
	if !w.DeviceBudgetBytes.Known || w.DeviceBudgetBytes.Value > 1<<30 || !w.WgpuDropsParked.Known {
		t.Fatalf("wgpu plan: %+v", w)
	}
}
