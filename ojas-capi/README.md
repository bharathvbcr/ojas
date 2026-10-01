# ojas-capi

`ojas-capi` provides the C-ABI engine dispatch layer, session management, and panic isolation boundaries for foreign language runtimes (such as Go via `gusset`).

---

## C-ABI Dispatch Architecture

```mermaid
flowchart TD
    subgraph HostCaller["Foreign Runtime (Go / C / Gusset)"]
        Caller["Host Process Worker"]
    end

    subgraph FFIEntryPoint["ojas-capi::engine::dispatch"]
        CatchUnwind["std::panic::catch_unwind Boundary"]
        OpcodeSwitch{"Opcode Switch"}
    end

    subgraph Operations["Supported Opcodes"]
        OP_LOAD["1: OP_LOAD\n(Resolve path & load safetensors)"]
        OP_STEP["2: OP_STEP\n(Execute train step: loss, grad norm, lr)"]
        OP_GENERATE["3: OP_GENERATE\n(Argmax or greedy token decode)"]
        OP_FREE["4: OP_FREE\n(Deallocate session from store)"]
        OP_PANIC["5: OP_PANIC\n(Diagnostic induced panic for red-teaming)"]
    end

    subgraph SessionManager["Session Store"]
        Store["Global Session Map\n- Max 64 concurrent sessions\n- Safe atomic index allocation"]
    end

    Caller --> CatchUnwind
    CatchUnwind --> OpcodeSwitch
    OpcodeSwitch -->|1| OP_LOAD
    OpcodeSwitch -->|2| OP_STEP
    OpcodeSwitch -->|3| OP_GENERATE
    OpcodeSwitch -->|4| OP_FREE
    OpcodeSwitch -->|5| OP_PANIC
    
    OP_LOAD --> Store
    OP_STEP --> Store
    OP_GENERATE --> Store
    OP_FREE --> Store
```

---

## Invariants & Defect Defenses

1. **Panic Boundary (`catch_unwind`):** Every foreign dispatch is protected against Rust panics. If a panic occurs, `ojas-capi` catches it, drops the associated session to prevent reusing torn internal state, and writes a descriptive error payload to the host caller.
2. **Session Capacity Cap:** Sessions are capped at 64 (`SESSION_CAP = 64`). Exceeding this limit returns an error instead of permitting memory exhaustion.
3. **Safe Path Resolution:** Model file paths are verified to reside strictly within `model_root`. Directory traversal attacks (e.g. `../../etc/passwd`) are rejected.
4. **Double Free Detection:** Attempting to free an already-freed or non-existent session returns an error rather than inducing undefined behavior.
