# ojas-device

`ojas-device` provides hardware device enumeration, host resource probing, memory planning (`ResourcePolicy`), and strict error propagation for hardware accelerators across **ojas**.

---

## Device Enumeration & Fallback Policy

```mermaid
flowchart TD
    subgraph Enumeration["Hardware Device Kinds"]
        CPU["Device::Cpu"]
        Metal["Device::Metal"]
        Vulkan["Device::Vulkan"]
        CUDA["Device::Cuda"]
        HIP["Device::Hip"]
    end

    subgraph ErrorTaxonomy["DeviceError (Zero Silent Fallbacks!)"]
        NotCompiled["DeviceError::NotCompiled\n(Requested accelerator feature disabled at compile time)"]
        NoDevice["DeviceError::NoDevice\n(Target hardware absent on host machine)"]
        Mismatch["DeviceError::DeviceMismatch\n(Operation dispatched across incompatible device buffers)"]
    end

    CUDA -.->|Feature cuda disabled| NotCompiled
    HIP -.->|Feature hip disabled| NotCompiled
    Metal -.->|Non-Apple hardware| NoDevice
```

---

## Host Probing & Resource Planning

`ojas-device` probes host CPU core topology and physical memory capacity to plan training and inference resource allocations:

```mermaid
flowchart LR
    subgraph Probe["host::probe()"]
        Cores["Logical / Physical CPU Cores"]
        RAM["Physical RAM Extents"]
    end

    subgraph Policy["ResourcePolicy & ResourcePlan"]
        Plan["Compute Maximum Micro-Batch Shape\nand Buffer Budgets based on Available RAM"]
    end

    Probe --> Policy
```

---

## The "Never Fall Back" Guarantee

> [!IMPORTANT]
> **No Silent CPU Fallbacks:** Most legacy machine learning frameworks silently route missing accelerator workloads onto the host CPU when a GPU is misconfigured or missing. This masks operational deployment errors and produces catastrophic latency spikes.
> 
> In `ojas`, selecting `Device::Cuda` or `Device::Hip` when that runtime is disabled returns `Err(DeviceError::NotCompiled)` immediately. Calling an unsupported operation on a device returns `Err(DeviceError::DeviceMismatch)`.

> [!NOTE]
> `ojas-device` defines device kinds and memory policies; it does **not** open GPU contexts or implement compute backends. Driver initialization and backend kernels reside in `ojas-metal`, `ojas-wgpu`, `ojas-cuda`, and `ojas-hip`.
