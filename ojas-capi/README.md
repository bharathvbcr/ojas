# ojas-capi

`ojas-capi` provides the C-ABI engine dispatch layer, session store management, device routing, and panic isolation boundaries for foreign language runtimes (such as Go via `gusset` and C/C++ host applications).

---

## C-ABI Dispatch Architecture & Device Routing

```mermaid
flowchart TD
    subgraph Host["Foreign Host Process (Go / C / Gusset)"]
        Caller["Worker Thread Request"]
    end

    subgraph FFI["ojas-capi::engine::dispatch"]
        OpcodeSwitch{"Opcode Switch"}
    end

    subgraph Opcodes["Supported Opcodes"]
        OP_LOAD["1: OP_LOAD\n(Resolve path & validate safetensors header)"]
        OP_STEP["2: OP_STEP\n(Forward, backward, optimizer step)"]
        OP_GENERATE["3: OP_GENERATE\n(Argmax of caller logits, or a fixed\ntwo-token greedy demo; weights not read)"]
        OP_FREE["4: OP_FREE\n(Deallocate session from store)"]
        OP_PANIC["5: OP_PANIC\n(Diagnostic induced panic for red-teaming)"]
    end

    subgraph DeviceRouting["Device Backend Assignment (At Load Time)"]
        DeviceKind{"Session Device Kind"}
        CPU["Kind 0: DeviceCPU (1 thread)"]
        CPUP["Kind 1: DeviceCPUParallel (1..=256 threads)"]
        Metal["Kind 2: DeviceMetal (MetalBackend)"]
        WGPU["Kind 3: DeviceWgpu (WgpuBackend)"]
    end

    Caller --> OpcodeSwitch
    OpcodeSwitch -->|1| OP_LOAD
    OpcodeSwitch -->|2| OP_STEP
    OpcodeSwitch -->|3| OP_GENERATE
    OpcodeSwitch -->|4| OP_FREE
    OpcodeSwitch -->|5| OP_PANIC

    OP_LOAD --> DeviceKind
    DeviceKind --> CPU
    DeviceKind --> CPUP
    DeviceKind --> Metal
    DeviceKind --> WGPU
```

---

## Panic Isolation & Poisoning Lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant Host as Go Application (go/api.go)
    participant Gusset as Worker Pool (libgusset.a)
    participant CAPI as FFI Dispatcher (ojas-capi)
    participant Session as Session Store (session.rs)

    Host->>Gusset: ojas.Step(sessionID, req)
    Gusset->>CAPI: dispatch(OP_STEP, payload)
    
    alt Normal Execution
        CAPI->>Session: Execute Step on Backend
        Session-->>CAPI: StepStats{Loss, GradNorm, Lr}
        CAPI-->>Gusset: Return Success
        Gusset-->>Host: StepStats, nil
    else Rust Panic Occurs
        Note over CAPI: Panic caught by worker catch_unwind
        CAPI-->>Gusset: FFI_PANIC Status Code
        Note over Gusset: Poison handle (gusset.ErrPoisoned)
        Gusset-->>Host: error (handle poisoned)
        Note over Session: If panic held session lock, next lock drops ALL sessions
        Host->>Host: Subsequent calls fail with ErrPoisoned until Close()
    end
```

---

## Defensive Countermeasures & Security Invariants

> [!IMPORTANT]
> 1. **Zero Silent Fallback on Device Failure:** If a requested GPU backend (e.g. Metal or wgpu) fails to open, `OP_LOAD` returns an explicit error (the backend's open error, as a `metal: ...` or `wgpu: ...` string) immediately. **It will never silently substitute a CPU session.**
> 2. **Panic Boundary Isolation:** Rust panics are caught by `catch_unwind` on the worker thread, preventing an FFI crash of the host Go runtime. A panic poisons the handle, returning `gusset.ErrPoisoned` to all subsequent calls until `Close()`.
> 3. **Path Traversal Defense:** Model file paths must reside strictly within the designated `model_root`. Paths attempting directory traversal (e.g. `../../etc/passwd`) are rejected by `resolve_under_root` with a path error string (`path must not contain ..`, `path escapes model root`, `path must be relative`).
> 4. **Session Cap Enforcement:** The active session table is strictly bounded to 64 sessions (`SESSION_CAP = 64`) to prevent memory exhaustion from unclosed sessions.
> 5. **Safetensors Header-Only Load:** `OP_LOAD` parses and validates only the JSON header of safetensors files to confirm schema and tensor counts; it does not read parameter weights into host memory during load.
> 6. **Thread Count Clamping:** `DeviceCPUParallel` accepts thread counts strictly within $1 \le \text{threads} \le 256$. Zero or counts greater than 256 are refused.
> 7. **Typed Error Kinds:** The `ojas:E_CAPACITY:`, `ojas:E_NONFINITE:` and `ojas:E_DEVICE_LOST:` prefixes are chosen from the typed error where it is produced (`kind_of` in `src/lib.rs`: `CapacityExceeded`, `NonFinite`, a backend's own device-lost detail, `DeviceError::Capacity` on a device open), never from message text, which carries user paths. A missing `busy_model.safetensors` is a plain load error. Rust never emits `ojas:E_BUSY:`.

> [!WARNING]
> While `catch_unwind` catches standard Rust panics, it **cannot catch process aborts**, out-of-memory aborts from infallible system allocators (`vec!`, `format!`), or stack overflows.
