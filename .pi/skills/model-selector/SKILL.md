---
name: model-selector
description: >
  Select the best generation model, embedding model, inference backend, and
  quantization level based on memory budget and current leaderboard rankings.
  Requires memory budget from memory-budget skill. Use after memory budget
  calculation.
---

# Model & Backend Selector — Phase 1(c–e)

Select optimal models and backend using leaderboard rankings filtered by hardware
constraints.

## Prerequisites

Read `results/phase1/memory_budget.json` and `results/phase1/hardware_profile.json`.
Also read `config/decisions.json` for pre-made preferences.

## Step (c): Generation model selection

1. **Check current rankings.** Look up the latest open-weight model rankings from:
   - LMArena / Chatbot Arena (human preference, Bradley-Terry scores)
   - Artificial Analysis (composite Intelligence Index)
   Use web search or known leaderboard URLs to get current top models.

2. **Filter to feasible models.** From the memory budget, determine which models
   fit. Consider:
   - Dense models: full parameter count must fit at the selected quantization.
   - MoE models: full parameter count must fit (all experts are stored in memory),
     but active parameters determine inference speed. **Prefer MoE on bandwidth-
     limited hardware** (< 500 GB/s) because only active params hit the bandwidth
     bottleneck.

3. **Select the top-ranked feasible model.** If LMArena and Artificial Analysis
   disagree, use average rank to break ties.

4. **Check GGUF availability.** The selected model must be available in GGUF format
   for llama.cpp. Search HuggingFace for GGUF quantized versions. If no GGUF
   exists, move to the next-ranked model.

## Step (d): Embedding model selection

1. Compute residual memory after the generation model.
2. Check MTEB *retrieval-task* rankings (NOT the overall MTEB average — that blends
   irrelevant tasks like classification and STS).
3. Select the top-ranked embedding model that fits in residual memory.
4. Check `config/decisions.json` for candidate preferences.

## Step (e): Backend + quantization selection

Deterministic rules:
- No GPU detected → `llama.cpp` with CPU-only (GGUF format)
- GPU detected with enough memory → `llama.cpp` with CUDA (preferred for GGUF),
  or `vLLM` / `Ollama` if the model is in a supported format
- Select the highest quantization that fits: Q8 > Q6 > Q5 > Q4

## Output

Save complete selection to `results/phase1/selected_config.json`:

```json
{
  "generation_model": {
    "name": "...",
    "parameters": "...",
    "quantization": "...",
    "format": "GGUF",
    "gguf_source": "HuggingFace URL",
    "memory_required_gb": N,
    "moe": true|false,
    "active_parameters": "...",
    "leaderboard_rank": { "lmarena": N, "artificial_analysis": N }
  },
  "embedding_model": {
    "name": "...",
    "memory_required_gb": N,
    "mteb_retrieval_rank": N
  },
  "backend": {
    "name": "llama.cpp",
    "cuda": true|false,
    "build_flags": "..."
  },
  "memory_summary": {
    "total_gb": N,
    "generation_model_gb": N,
    "embedding_model_gb": N,
    "os_reserve_gb": N,
    "headroom_gb": N
  }
}
```

**IMPORTANT: After saving, STOP and print the full selection summary. Wait for human
approval before downloading models or proceeding to Phase 2.**
