# OSSYM 2026 Experiments — Automated Hardware-Aware RAG Configuration

## Session Handoff

For another agent to resume, reproduce, or extend the completed experiments, read:
- `EXPERIMENT_HANDOFF.md`
- `results/experiment_report.md`
- `results/summary_table.json`
- `results/issues.md`

## Hardware

Phase 1 profiled the host as:
- CPU: ARM `aarch64`, 20 cores / 20 threads
- Core types: 10× Cortex-X925, 10× Cortex-A725
- RAM: 121.69 GiB total, 111.43 GiB available at profiling time
- GPU: NVIDIA GB10, CUDA 13.0, compute capability 12.1
- GPU memory mode: unified memory
- Disk available in workspace filesystem: ~3.38 TB

Canonical source:
- `results/phase1/hardware_profile.json`

## Selected Configuration

### Approved Phase 1 selection
- Generation model: `gpt-oss-120b`
- Quantization: `Q6`
- Embedding model: `BAAI/bge-large-en-v1.5`
- Backend: `llama.cpp` with CUDA

Canonical source:
- `results/phase1/selected_config.json`

### Actual Phase 2 execution config
The approved `gpt-oss-120b` model was downloaded but failed during `llama-server` startup, so Phase 2 proceeded with the documented fallback model:
- Generation model used in benchmarks: `qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf`
- Embedding model: `BAAI/bge-large-en-v1.5`
- Best retrieval strategy: `vector`
- Best chunk size: `512`
- Reranker kept: `disabled`

Canonical source:
- `results/summary_table.json`
- `results/issues.md`

## How to Reproduce

### Environment and assets already present
- `llama.cpp/` built locally with CUDA
- `venv/` contains the Python environment used for Phase 2
- models downloaded under `models/`
- SciFact dataset downloaded under `datasets/`

### Main reproducibility scripts
- `scripts/phase2_benchmark.py`
- `scripts/finalize_phase2.py`

### Typical reproduction flow
1. Read `EXPERIMENT_HANDOFF.md`
2. Activate the venv:
   - `source venv/bin/activate`
3. Start `llama-server` with the desired model
4. Run:
   - `python scripts/phase2_benchmark.py`
   - `python scripts/finalize_phase2.py`

### Notes
- If retrying the originally approved model, inspect:
  - `results/issues.md`
  - `results/phase2/logs/llama_server.log`
- If reproducing the exact benchmark results from this session, use the fallback Qwen 32B Q4 model noted above.

## Results

### Phase 1 outputs
- `results/phase1/hardware_profile.json`
- `results/phase1/memory_budget.json`
- `results/phase1/selected_config.json`

### Phase 2 outputs
- `results/phase2/chunk_size_sweep.json`
- `results/phase2/retrieval_comparison.json`
- `results/phase2/reranker_comparison.json`
- `results/phase2/ragas_evaluation.json`
- CSV companions in `results/phase2/`

### Final summary metrics
- retrieval recall@5: `0.7742777777777777`
- faithfulness: `0.9343078717589378`
- answer relevancy: `0.921547994017601`
- context precision: `0.90194650888443`
- context recall: `0.9666692346334458`
- TTFT: `3.1939997288020097 s`
- tokens/s: `6.622575616283375`
- average latency/query: `10.7486063852004 s`

Canonical summaries:
- `results/summary_table.json`
- `results/experiment_report.md`

## Issues Encountered

Main issues from this session:
1. The approved `gpt-oss-120b` GGUF downloaded successfully but failed during llama.cpp server bring-up on this machine.
2. Direct local RAGAS evaluation exceeded the server context window on several judge requests, so aggregate scores were repaired with an embedding-based fallback metric computation.

Canonical issue log:
- `results/issues.md`
