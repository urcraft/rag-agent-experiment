---
name: hardware-profiler
description: >
  Probe local hardware (CPU, RAM, GPU, disk) and produce a structured JSON
  hardware profile. Use at the start of any local ML/RAG system setup.
---

# Hardware Profiler — Phase 1(a)

Detect this machine's hardware capabilities and save a structured profile.

## Steps

1. **CPU:** Run `lscpu` and extract: architecture, model name, core count, thread
   count. On ARM, also note the specific core types if available.

2. **RAM:** Run `free -h` and parse total/available memory. Also check
   `/proc/meminfo` for MemTotal and MemAvailable in bytes (more precise).

3. **GPU:** Try `nvidia-smi` first.
   - If it reports GPU name, VRAM, driver version, CUDA version — record all.
   - If VRAM shows as N/A or 0 (common on unified memory architectures like
     NVIDIA GB10), the GPU shares system RAM. Record GPU name and note
     "unified memory — use system RAM total as GPU memory budget."
   - If `nvidia-smi` is not found, check for other GPUs (`lspci | grep -i vga`),
     or note "no discrete GPU detected."
   - For CUDA compute capability, check `nvidia-smi --query-gpu=compute_cap
     --format=csv,noheader` or parse from GPU model name.

4. **Disk:** Run `df -h` on the home directory or wherever models will be stored.
   Record total and available space.

5. **Save output** to `results/phase1/hardware_profile.json` with this structure:

```json
{
  "timestamp": "ISO-8601",
  "cpu": {
    "architecture": "x86_64 | aarch64 | ...",
    "model": "...",
    "cores": N,
    "threads": N
  },
  "ram": {
    "total_gb": N,
    "available_gb": N
  },
  "gpu": {
    "detected": true|false,
    "name": "...",
    "vram_gb": N | null,
    "unified_memory": true|false,
    "cuda_version": "...",
    "compute_capability": "...",
    "driver_version": "..."
  },
  "disk": {
    "total_gb": N,
    "available_gb": N,
    "mount_point": "..."
  }
}
```

6. **Print a human-readable summary** to the terminal after saving.
