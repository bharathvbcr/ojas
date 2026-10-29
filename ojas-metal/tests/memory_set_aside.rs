//! What the resource plan sets aside for a Metal backend's uncharged memory
//! (ft-8ce3348b). The Budget charges live tensors only. The device also holds
//! the pool cache (`MetalMemory::pool_cache_cap`), dropped buffers that wait
//! for a waited commit (under one memory cap), and an op's temporaries that
//! wait for one (under another), so the probe reports the pool cap plus two
//! memory caps, and the plan takes that out of the device's room. The memory
//! cap moves with `set_mem_cap`: the figure is the one in force.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::Budget;
use ojas_device::{
    Device, HostMemory, MemoryProbe, MemoryReport, ResourcePlan, ResourcePolicy, SystemProfile,
};
use ojas_metal::{MetalBackend, MetalMemory};

const MIB: u64 = 1 << 20;

/// A 64 MiB budget opens with a pool cap and a memory cap of a quarter of
/// it, 16 MiB each.
fn open() -> MetalBackend {
    ok("Metal", MetalBackend::new(Budget::new(64 * MIB)))
}

/// The plan for a Metal device probed as `mem`, with host memory and a
/// caller budget so large that the device's own room is the bound.
fn plan_for(mem: MetalMemory) -> ResourcePlan {
    let mut host = HostMemory::all_unknown();
    host.available_bytes = MemoryReport::Known(1 << 60);
    let profile = SystemProfile::from_memory(host);
    let mut policy = ResourcePolicy::new(1 << 60);
    policy.devices = vec![Device::Metal, Device::Cpu];
    ResourcePlan::derive(&policy, &profile, &[mem])
}

#[test]
fn the_probe_sets_aside_the_pool_cache_and_two_memory_caps_in_force() {
    let m = open();
    let mem = ok("memory", m.memory());
    let cap = ok("mem_cap", m.mem_cap());
    assert_eq!(cap, 16 * MIB, "a quarter of the 64 MiB budget");
    assert_eq!(mem.pool_cache_cap, 16 * MIB);
    assert_eq!(mem.mem_cap, cap, "the reading carries the cap in force");
    assert_eq!(
        mem.pool_cache_bytes(),
        MemoryReport::Known(mem.pool_cache_cap + 2 * cap),
        "pool cache 16 MiB plus two 16 MiB caps: {mem:?}"
    );

    // The figure follows the setter; the pool cap does not move with it.
    ok("set_mem_cap", m.set_mem_cap(MIB));
    let lowered = ok("memory", m.memory());
    assert_eq!(lowered.pool_cache_cap, 16 * MIB);
    assert_eq!(lowered.mem_cap, MIB);
    assert_eq!(
        lowered.pool_cache_bytes(),
        MemoryReport::Known(16 * MIB + 2 * MIB),
        "{lowered:?}"
    );

    // A with_budget handle shares the device thread, so its probe reads the cap in force.
    let planned = m.with_budget(Budget::new(8 * MIB));
    assert_eq!(
        ok("memory", planned.memory()).pool_cache_bytes(),
        MemoryReport::Known(16 * MIB + 2 * MIB)
    );

    // The budget itself is the largest cap accepted.
    ok("set_mem_cap", m.set_mem_cap(64 * MIB));
    assert_eq!(
        ok("memory", m.memory()).pool_cache_bytes(),
        MemoryReport::Known(16 * MIB + 2 * 64 * MIB)
    );
}

#[test]
fn the_plans_device_room_subtracts_the_memory_caps_and_moves_with_the_setter() {
    let m = open();
    for (set, cap) in [
        (None, 16 * MIB),
        (Some(MIB), MIB),
        (Some(64 * MIB), 64 * MIB),
    ] {
        if let Some(bytes) = set {
            ok("set_mem_cap", m.set_mem_cap(bytes));
        }
        assert_eq!(ok("mem_cap", m.mem_cap()), cap);
        let mem = ok("memory", m.memory());
        if mem.recommended_working_set == 0 {
            eprintln!("SKIP: the device reports no working set, so its room is unknown");
            return;
        }
        let plan = plan_for(mem);
        assert_eq!(
            plan.device_pool_cache[0],
            MemoryReport::Known(mem.pool_cache_cap + 2 * cap),
            "cap {cap}"
        );
        let room = match plan.device_room[0] {
            MemoryReport::Known(room) => room,
            MemoryReport::Unknown => panic!("a reporting device has a room: {plan:?}"),
        };
        // The room is the working set less what the process holds, the pool
        // cache, and a memory cap each of frees and temporaries.
        assert_eq!(
            room + mem.allocated + mem.pool_cache_cap + 2 * cap,
            mem.recommended_working_set,
            "cap {cap}: {mem:?}"
        );
    }
}
