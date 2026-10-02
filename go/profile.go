package ojas

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
)

// Reading is one probed figure. Known is false when the engine's probe
// could not read it; Value is then 0 and must not be used as a number.
type Reading struct {
	Value uint64
	Known bool
}

// Profile is what the engine read about this machine and process, and the
// plan it derived for a caller budget (ojas_device::ResourcePlan). It
// changes nothing: SetMemoryCeiling and LoadOptions.Threads are still set
// by the caller, who may take them from here.
type Profile struct {
	// BudgetBytes is the caller budget cut by total RAM, available RAM,
	// the cgroup limit and the cgroup room (its working set: page cache does
	// not count). Always known, and 0 on a machine with nothing to spare;
	// SetMemoryCeiling refuses 0, so check before forwarding it. The ceiling
	// counts the logical bytes of tensors: a Metal model's resident memory can
	// be up to twice that (tessl rounds buffers up to a power of two) plus its
	// pool cache, so leave headroom rather than setting the ceiling to all of
	// AvailableBytes.
	BudgetBytes    uint64
	TotalBytes     Reading
	AvailableBytes Reading
	// CgroupLimitBytes is the tightest memory limit over the cgroup and
	// its ancestors (Linux).
	CgroupLimitBytes Reading
	// ThreadCeiling is the usable CPU count, at most the engine's ceiling
	// (1024). Clamp it to MaxCPUThreads before passing it as
	// LoadOptions.Threads for a compute-bound DeviceCPUParallel load.
	ThreadCeiling Reading
	// FastThreads is the CPU count of the fastest core cluster the OS
	// reports (Intel P-cores, the top arm64 capacity, or Apple's perflevel0,
	// which on an M5 Pro is 6 of 18 performance-class cores), at most
	// ThreadCeiling. On a machine with one kind of core it equals
	// ThreadCeiling.
	FastThreads    Reading
	LogicalCPUs    Reading
	PhysicalCPUs   Reading
	CPUQuotaMillis Reading
	// Cache sizes every core can rely on: the smallest over the clusters.
	L1dBytes       Reading
	L2PerCoreBytes Reading
	// L3Bytes is known only when the OS reports an L3. Apple silicon
	// reports none (its system-level cache is not exposed); the L2 is never
	// reported in its place.
	L3Bytes        Reading
	CacheLineBytes Reading
	PageBytes      Reading
	// UnifiedMemory.Value is 1 when the CPU and GPU share one memory (Apple
	// silicon) and 0 when the GPU has its own. On unified memory a Metal
	// model's device memory comes out of the same RAM as BudgetBytes;
	// budget the two together.
	UnifiedMemory Reading
	// Pressure is 1 normal, 2 warning, 3 critical (macOS).
	Pressure Reading
	// Copy bandwidth, read plus write bytes per second, on one thread and
	// on BandwidthThreads threads. Known only when SystemProfile was asked to
	// measure and the engine could (it refuses when 1/8 of available memory
	// is under 2 MiB); a figure taken on a busy machine is a lower bound.
	SingleBandwidth  Reading
	MultiBandwidth   Reading
	BandwidthThreads Reading
	// MemoryBoundThreads is ceil(MultiBandwidth / SingleBandwidth), at most
	// ThreadCeiling: past it a memory-bound op stops scaling. An estimate
	// from two points, not a sweep.
	MemoryBoundThreads Reading
}

const (
	profileVersion = 1
	profileFields  = 20
	// flagBandwidth asks the engine to measure (or reuse) copy bandwidth.
	flagBandwidth uint32 = 1
	profileEntry         = 9
)

// SystemProfile reads the host profile. budget is the caller budget the
// plan cuts; 0 asks for the host limits alone (no caller cap). With
// measureBandwidth the engine times a bounded memory copy (about half a
// second the first time in a process; later calls reuse that figure) and
// fills the bandwidth fields. A measurement in progress is not cancelled by
// ctx; it is bounded to about half a second, and concurrent callers wait
// for it.
func SystemProfile(ctx context.Context, budget uint64, measureBandwidth bool) (Profile, error) {
	var payload []byte
	switch {
	case measureBandwidth:
		if budget == 0 {
			budget = ^uint64(0)
		}
		payload = binary.LittleEndian.AppendUint64(nil, budget)
		payload = binary.LittleEndian.AppendUint32(payload, flagBandwidth)
	case budget != 0:
		payload = binary.LittleEndian.AppendUint64(nil, budget)
	}
	out, err := callEngine(ctx, opSystemProfile, payload)
	if err != nil {
		return Profile{}, err
	}
	return decodeProfile(out)
}

// decodeProfile refuses a record it cannot read whole: a wrong version, a
// count below the fields this package knows, a length that disagrees with
// the count, a known byte other than 0 or 1, or an unknown entry with a
// non-zero value. Entries past the known fields are a later version's
// appends and are skipped.
func decodeProfile(out []byte) (Profile, error) {
	if len(out) < 8 {
		return Profile{}, errors.New("system profile: short result")
	}
	if v := binary.LittleEndian.Uint32(out[0:4]); v != profileVersion {
		return Profile{}, fmt.Errorf("system profile: version %d, want %d", v, profileVersion)
	}
	count := uint64(binary.LittleEndian.Uint32(out[4:8]))
	if count < profileFields {
		return Profile{}, fmt.Errorf("system profile: %d fields, want at least %d", count, profileFields)
	}
	if uint64(len(out)-8) != count*profileEntry {
		return Profile{}, fmt.Errorf("system profile: %d bytes for %d fields", len(out)-8, count)
	}
	r := make([]Reading, profileFields)
	for i := range r {
		e := out[8+i*profileEntry : 8+(i+1)*profileEntry]
		v := binary.LittleEndian.Uint64(e[1:])
		switch e[0] {
		case 0:
			if v != 0 {
				return Profile{}, fmt.Errorf("system profile: field %d unknown with value %d", i, v)
			}
		case 1:
			r[i] = Reading{Value: v, Known: true}
		default:
			return Profile{}, fmt.Errorf("system profile: field %d known byte %d", i, e[0])
		}
	}
	if !r[0].Known {
		return Profile{}, errors.New("system profile: budget not reported")
	}
	// The engine sends 1 unified, 2 discrete.
	unified := r[14]
	if unified.Known {
		switch unified.Value {
		case 1:
		case 2:
			unified.Value = 0
		default:
			return Profile{}, fmt.Errorf("system profile: architecture %d", r[14].Value)
		}
	}
	return Profile{
		BudgetBytes:        r[0].Value,
		TotalBytes:         r[1],
		AvailableBytes:     r[2],
		CgroupLimitBytes:   r[3],
		ThreadCeiling:      r[4],
		FastThreads:        r[5],
		LogicalCPUs:        r[6],
		PhysicalCPUs:       r[7],
		CPUQuotaMillis:     r[8],
		L1dBytes:           r[9],
		L2PerCoreBytes:     r[10],
		L3Bytes:            r[11],
		CacheLineBytes:     r[12],
		PageBytes:          r[13],
		UnifiedMemory:      unified,
		Pressure:           r[15],
		SingleBandwidth:    r[16],
		MultiBandwidth:     r[17],
		BandwidthThreads:   r[18],
		MemoryBoundThreads: r[19],
	}, nil
}
