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
	// not count). 0 on a machine with nothing to spare, under critical
	// memory pressure, and when the call passed no budget and the engine
	// could read no limit (an unbounded budget is never reported as a
	// number); SetMemoryCeiling refuses 0, so check before forwarding it. The ceiling
	// counts the logical bytes of tensors: a Metal model's resident memory can
	// pass that by tessl's rounding and its pool cache (SetMemoryCeiling has
	// the bounds), so leave headroom rather than setting the ceiling to all of
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
	// silicon, a Linux host whose GPUs are all integrated, or one with no
	// GPU at all, where every device draws on host RAM) and 0 when the GPU
	// has its own. On unified memory a Metal
	// model's device memory comes out of the same RAM as BudgetBytes;
	// budget the two together.
	UnifiedMemory Reading
	// Pressure is 1 normal, 2 warning, 3 critical: macOS's own level, or on
	// Linux the worst pressure stall reading of the system and this
	// process's cgroups (warning at 10% of the last 10 s with some task
	// stalled on memory, critical at 10% with every task stalled or 50%
	// with some).
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

	// The device fields are known only from DeviceProfile, which opens the
	// device and plans against what it reports, as a load on that device
	// does. From SystemProfile they are all unknown.
	//
	// ProbeDevice is the device whose probe the plan used: DeviceMetal or
	// DeviceWgpu.
	ProbeDevice Reading
	// DeviceMemoryBytes is the memory the device says it may use: Metal's
	// recommended working set. wgpu reports no device memory size, so on
	// wgpu it is unknown, and the plan does not stand a number in for it.
	DeviceMemoryBytes Reading
	// DeviceResidentBytes is what this process already holds on the
	// device: Metal's allocated size (every backend and pool in the
	// process), or wgpu's allocator report where its HAL keeps one (Vulkan,
	// DX12; not Metal).
	DeviceResidentBytes Reading
	// DevicePoolCacheBytes is the most freed buffers the runtime keeps for
	// reuse on a session opened with this budget. That memory is not
	// charged to the session budget, so the room below already sets it
	// aside.
	DevicePoolCacheBytes Reading
	// DeviceRoomBytes is DeviceMemoryBytes less DeviceResidentBytes and
	// DevicePoolCacheBytes, and on shared memory at most BudgetBytes.
	// Unknown when the device reported no memory.
	DeviceRoomBytes Reading
	// DeviceSharesHost.Value is 1 when the device draws on host RAM (Apple
	// silicon, an integrated or CPU adapter), 0 when it has its own.
	DeviceSharesHost Reading
	// SharedBudget.Value is 1 when the device's memory and BudgetBytes are
	// one pool: budget the CPU and the device together.
	SharedBudget Reading
	// DeviceBudgetBytes is the session budget a load on ProbeDevice with
	// this budget gets: the budget cut to DeviceRoomBytes when that is
	// known, and to BudgetBytes on shared memory.
	DeviceBudgetBytes Reading
	// WgpuDropsParked counts wgpu contexts this process freed whose GPU
	// work had not finished when the free returned: each still holds its
	// device, its memory and one thread, and a new wgpu load is refused
	// while 4 are parked. WgpuDropsTimedOut counts every such free, ever.
	// Both are known on every profile.
	WgpuDropsParked   Reading
	WgpuDropsTimedOut Reading
}

const (
	profileVersion = 1
	// profileFields is the least a record carries; allProfileFields is
	// every field this package reads.
	profileFields    = 20
	allProfileFields = 30
	// flagBandwidth asks the engine to measure (or reuse) copy bandwidth.
	flagBandwidth uint32 = 1
	// flagDeviceMetal and flagDeviceWgpu ask the engine to open that device
	// and plan against its probe.
	flagDeviceMetal uint32 = 2
	flagDeviceWgpu  uint32 = 4
	profileEntry           = 9
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

// DeviceProfile is SystemProfile planned against a device's own memory
// probe, as a load on that device plans its session budget: device is
// DeviceMetal or DeviceWgpu. The engine opens the device on a short-lived
// thread (ctx cancels the wait), reads what it reports, and closes it; the
// device fields say which probe was used and the room it reported. budget 0
// asks for the host and device limits alone. A device that does not open
// is its error ("metal:" or "wgpu:"), never a CPU profile.
func DeviceProfile(ctx context.Context, budget uint64, device uint32) (Profile, error) {
	var flag uint32
	switch device {
	case DeviceMetal:
		flag = flagDeviceMetal
	case DeviceWgpu:
		flag = flagDeviceWgpu
	default:
		return Profile{}, fmt.Errorf("device profile: device %d has no memory probe; use DeviceMetal or DeviceWgpu", device)
	}
	if budget == 0 {
		budget = ^uint64(0)
	}
	payload := binary.LittleEndian.AppendUint64(nil, budget)
	payload = binary.LittleEndian.AppendUint32(payload, flag)
	out, err := callEngine(ctx, opSystemProfile, payload)
	if err != nil {
		return Profile{}, err
	}
	return decodeProfile(out)
}

// decodeProfile refuses a record it cannot read whole: a wrong version, a
// count below the fields this package knows, a length that disagrees with
// the count, a known byte other than 0 or 1, or an unknown entry with a
// non-zero value. The device fields are read when the record carries them
// and are unknown otherwise. Entries past every known field are a later
// version's appends and are skipped.
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
	// The device fields arrive together or not at all: half a block would
	// read as a probe with no room or no budget.
	if count > profileFields && count < allProfileFields {
		return Profile{}, fmt.Errorf("system profile: %d fields cut the device block (%d..%d)", count, profileFields, allProfileFields)
	}
	if uint64(len(out)-8) != count*profileEntry {
		return Profile{}, fmt.Errorf("system profile: %d bytes for %d fields", len(out)-8, count)
	}
	r := make([]Reading, allProfileFields)
	for i := range r[:min(count, allProfileFields)] {
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
	for _, i := range []int{25, 26} {
		if r[i].Known && r[i].Value > 1 {
			return Profile{}, fmt.Errorf("system profile: field %d is a flag, got %d", i, r[i].Value)
		}
	}
	if p := r[20]; p.Known && p.Value != uint64(DeviceMetal) && p.Value != uint64(DeviceWgpu) {
		return Profile{}, fmt.Errorf("system profile: probe device %d", p.Value)
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

		ProbeDevice:          r[20],
		DeviceMemoryBytes:    r[21],
		DeviceResidentBytes:  r[22],
		DevicePoolCacheBytes: r[23],
		DeviceRoomBytes:      r[24],
		DeviceSharesHost:     r[25],
		SharedBudget:         r[26],
		DeviceBudgetBytes:    r[27],
		WgpuDropsParked:      r[28],
		WgpuDropsTimedOut:    r[29],
	}, nil
}
