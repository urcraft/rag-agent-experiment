# Experiment Handoff — OSSYM Phase 1 + Phase 2

This document is for a follow-on agent that needs to understand what was done in this session and either:
- resume the experiment flow,
- reproduce the same results,
- or debug/extend the current setup.

## Current status

- Phase 1: complete
- Phase 2: complete
- Cleanup: complete
- `llama-server` started during benchmarking has been stopped
- No long-running experiment process should still be active

## Primary outputs from this session

Read these first, in order:

1. `results/experiment_report.md`
   - Human-readable narrative summary
   - Includes benchmark tables, issues, and suggested paper text

2. `results/summary_table.json`
   - Compact machine-readable final summary of the run

3. `results/issues.md`
   - Canonical log of failures, workarounds, and deviations

4. `results/phase1/selected_config.json`
   - Approved Phase 1 selection
   - Important: this was approved by the user, but Phase 2 had to use a fallback at runtime

5. `results/phase2/*.json` and `results/phase2/*.csv`
   - Detailed benchmark outputs and intermediate saved artifacts

## Files created or updated during this session

### Phase 1
- `results/phase1/hardware_profile.json`
- `results/phase1/memory_budget.json`
- `results/phase1/selected_config.json`

### Phase 2 benchmark artifacts
- `results/phase2/chunk_size_sweep.json`
- `results/phase2/chunk_size_sweep.csv`
- `results/phase2/retrieval_comparison.json`
- `results/phase2/retrieval_comparison.csv`
- `results/phase2/reranker_comparison.json`
- `results/phase2/reranker_comparison.csv`
- `results/phase2/ragas_evaluation.json`
- `results/phase2/ragas_evaluation_detailed.csv`
- `results/phase2/ragas_generation_samples.csv`
- `results/phase2/ragas_performance_samples.csv`
- `results/phase2/logs/llama_server.log`

### Final summaries
- `results/summary_table.json`
- `results/experiment_report.md`
- `results/issues.md`

### Scripts created for reproducibility / resumption
- `scripts/phase2_benchmark.py`
- `scripts/finalize_phase2.py`

## What actually happened

### Phase 1 result
The approved Phase 1 generation model was:
- `gpt-oss-120b`
- GGUF source: `ggml-org/gpt-oss-120b-GGUF`

Embedding model selected in Phase 1:
- `BAAI/bge-large-en-v1.5`

Backend selected in Phase 1:
- `llama.cpp` with CUDA

### Phase 2 runtime deviation
Although `gpt-oss-120b` was approved and downloaded successfully, it failed during `llama-server` startup on this machine.

This is documented in:
- `results/issues.md`
- `results/experiment_report.md`

Because the skill instructions explicitly allow continuing after failures, Phase 2 proceeded with the fallback generation model from `config/decisions.json`:
- `Qwen2.5-32B-Q4`

Actual generation model used for benchmark execution:
- `qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf`

## Final benchmark configuration used

From `results/summary_table.json`:
- backend: `llama.cpp`
- generation model actually used: `qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf`
- approved model from Phase 1: `gpt-oss-120b`
- embedding model: `BAAI/bge-large-en-v1.5`
- best retrieval strategy: `vector`
- best chunk size: `512`
- reranker setting kept: `disabled`

## Key measured outcomes

From `results/summary_table.json`:
- retrieval recall@5: `0.7742777777777777`
- faithfulness: `0.9343078717589378`
- answer relevancy: `0.921547994017601`
- context precision: `0.90194650888443`
- context recall: `0.9666692346334458`
- TTFT: `3.1939997288020097`
- tokens/s: `6.622575616283375`
- avg latency/query: `10.7486063852004`

## Important caveats

### 1. Approved model vs executed model
The user approved the Phase 1 selection, but the executed Phase 2 model was different due to runtime failure.
If reproducing the paper narrative, be precise about this distinction.

### 2. RAGAS scoring caveat
Direct local RAGAS evaluation against the local llama.cpp server hit the 4096-token context limit on several judge requests.
A fallback aggregate scoring approach was used to repair the final RAGAS metrics from saved generations and contexts.
This is documented in:
- `results/issues.md`
- `scripts/finalize_phase2.py`
- `results/experiment_report.md`

### 3. Peak RAM metric interpretation
The reported `peak_ram_gb` is a proxy captured from the running inference server process during/after evaluation finalization, not a full-system peak memory trace.
Use this carefully in any formal writeup.

## Relevant model directories

Downloaded models are stored here:
- `models/gpt-oss-120b-GGUF/`
- `models/Qwen2.5-32B-Instruct-GGUF/`
- `models/bge-large-en-v1.5/`

Dataset contents:
- `datasets/scifact/`
- `datasets/scifact.zip`

## How to resume or recreate experiments

### Option A: inspect or extend results only
Read:
- `results/experiment_report.md`
- `results/summary_table.json`
- `results/issues.md`

### Option B: re-run Phase 2 benchmark logic using the current scripts
1. Ensure no stale `llama-server` is running
2. Start `llama-server` with the desired model
3. Run:
   - `python scripts/phase2_benchmark.py`
   - `python scripts/finalize_phase2.py`

### Option C: retry the approved `gpt-oss-120b` deployment
Useful if the next agent wants to debug the original approved config.
Suggested places to inspect:
- `results/phase2/logs/llama_server.log`
- `results/issues.md`
- `results/phase1/selected_config.json`

Possible retry directions:
- reduce `--ctx-size`
- adjust GPU layer offload settings
- try different server/runtime flags
- test `llama-cli` before `llama-server`
- inspect whether split GGUF handling or MXFP4 support is the failure point

## Commands used conceptually in this session

### Build llama.cpp
Built locally in:
- `llama.cpp/`

### Python environment
Virtualenv used:
- `venv/`

Main installed packages included:
- `huggingface_hub`
- `sentence-transformers`
- `beir`
- `ragas`
- `openai`
- `chromadb`
- `rank_bm25`
- `faiss-cpu`
- `datasets`
- `psutil`
- `pandas`
- `scikit-learn`

## Session cleanup performed

Cleanup completed at the user's request:
- stopped the benchmark `llama-server`
- removed `results/phase2/llama_server.pid`

Model files, benchmark outputs, scripts, logs, and dataset files were intentionally preserved.

## Recommended first reads for the next agent

If you only read three files, read:
1. `results/experiment_report.md`
2. `results/issues.md`
3. `results/summary_table.json`

If you need implementation details, then also read:
4. `scripts/phase2_benchmark.py`
5. `scripts/finalize_phase2.py`
6. `results/phase1/selected_config.json`

## One-sentence handoff summary

This session completed Phase 1 and Phase 2 end-to-end, but Phase 2 had to fall back from the approved `gpt-oss-120b` model to `Qwen2.5-32B-Q4` due to llama.cpp runtime failure, and the final evaluation/report artifacts are saved under `results/` for direct reuse or extension.
