# ojas-device

`ojas-device` provides hardware device identification, hardware probing facilities, and strict error propagation for non-available backends.

---

## Hardware Hierarchy

```mermaid
flowchart TD
    subgraph DeviceEnum["Device Enumeration"]
        CPU["Device::Cpu"]
        Metal["Device::Metal"]
        Vulkan["Device::Vulkan"]
        Cuda["Device::Cuda"]
        Hip["Device::Hip"]
    end

    subgraph ErrorDomain["DeviceError (No Fallbacks!)"]
        NotCompiled["DeviceError::NotCompiled\n(Optional feature not enabled at build time)"]
        NoDevice["DeviceError::NoDevice\n(Target hardware absent on host)"]
        DeviceMismatch["DeviceError::DeviceMismatch\n(Operation dispatched to incompatible device)"]
    end

    Metal -.->|Absent on Linux| NoDevice
    Cuda -.->|Feature cuda off| NotCompiled
    Hip -.->|Feature hip off| NotCompiled
```

---

## Core Philosophy: Never Fall Back

Most deep learning frameworks silently route missing accelerator workloads onto the host CPU. This masks deployment errors and produces unpredictable performance cliffs.

In `ojas`, selecting `Device::Cuda` when the `cuda` feature is disabled returns `Err(DeviceError::NotCompiled)` immediately. Calling an unsupported operation on a device returns `Err(DeviceError::DeviceMismatch)`.
