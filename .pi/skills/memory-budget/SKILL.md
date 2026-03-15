---
name: memory-budget
description: >
  Calculate which LLM and embedding models fit in available memory at each
  quantization level. Requires a hardware profile from hardware-profiler.
  Use after hardware profiling to determine feasible model sizes.
---

# Memory Budget Calculator — Phase 1(b)

Compute feasible (model_size, quantization) pairs based on available memory.

## Prerequisites

Read `results/phase1/hardware_profile.json`. If it doesn't exist, run the
hardware-profiler skill first.

## Formula

```
required_memory_gb = (parameters_billions × bits_per_weight) / 8 + overhead
```

Where:
- `parameters_billions` = model parameter count (e.g., 7, 13, 34, 70, 235)
- `bits_per_weight` = quantization bits (4 for Q4, 5 for Q5, 6 for Q6, 8 for Q8,
  16 for FP16)
- `overhead` = 15% of the raw weight size (covers KV-cache, activations, framework)

## Memory budget

- **Total available for models:** system RAM (or VRAM if discrete GPU) minus
  reserves.
- **Reserve 4–6 GB** for OS, framework overhead, and system processes.
- **Reserve 0.5–1.5 GB** for the embedding model (will be selected separately).
- **Effective budget** = total_available - os_reserve - embedding_reserve

## Steps

1. Read the hardware profile.
2. Determine effective memory budget.
3. For each common model size (7B, 13B, 14B, 27B, 32B, 70B, 72B, 120B, 235B),
   calculate required memory at Q4, Q5, Q6, Q8, and FP16.
4. Mark each combination as FITS or DOES NOT FIT against the effective budget.
5. For MoE models, note that the *total* parameter count determines memory for
   weights, but only *active* parameters hit memory bandwidth during inference.
   This means MoE models are faster than dense models at the same total size on
   bandwidth-limited hardware.
6. Save results to `results/phase1/memory_budget.json` and print as a table.
