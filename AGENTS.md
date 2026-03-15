# OSSYM 2026 — Automated Hardware-Aware RAG Configuration

## What this project is

Experiments for a research paper on automated local RAG system configuration.
The paper proposes a two-phase agent-based approach:
- Phase 1: Leaderboard-driven infrastructure selection (hardware → model → backend)
- Phase 2: Empirical RAG pipeline optimization (chunking, retrieval, reranking)

You (the agent) are the execution layer described in the paper. Your job is to
run both phases autonomously on this machine, producing structured results.

**Bootstrapping note:** You (the agent) are running on a cloud LLM (GPT-5.4).
This is the paper's acknowledged bootstrapping solution — a cloud API powers the
one-time configuration, after which the locally configured RAG system operates
independently. This is a feature, not a limitation.

## Project layout

- `config/decisions.json` — pre-made decisions (dataset, model preferences, etc.)
- `results/phase1/` — save Phase 1 outputs here (JSON)
- `results/phase2/` — save Phase 2 outputs here (JSON + CSV)
- `scripts/` — save any scripts you create here
- `README.md` — document what you did and how to reproduce it

## Rules

- You are powered by GPT-5.4 via an OpenAI Codex subscription. If you hit rate
  limits, pause and notify the user — they can switch providers with /model.
- Save all work to this project directory. Do not install things globally unless
  necessary (prefer venvs for Python).
- Save intermediate results frequently. A partial result set is better than none.
- If any single step takes more than 45 minutes of debugging, document the issue
  in `results/issues.md` and move on to the next step.
- Read `config/decisions.json` before starting — some choices are pre-made.
- The machine-agnostic algorithm is in the skills. Use /skill:hardware-profiler,
  /skill:memory-budget, /skill:model-selector, and /skill:rag-benchmarker.
- After completing all experiments, write a final experiment report to
  `results/experiment_report.md` — see Step 8 of the rag-benchmarker skill.

## Container usage

Podman is available (rootless, `ai-agent` namespace). Use containers for isolation
where practical, but don't let container setup block progress.

- **Phase 1 (hardware profiling):** Run directly on the host. `lscpu`, `nvidia-smi`,
  `free -h`, and `df -h` must report real hardware values, not container limits.
- **Phase 2 (llama.cpp, RAG benchmarks):** Prefer containers for workloads:
  - llama.cpp build + server → use a CUDA container with GPU passthrough
  - Python RAG benchmarks → use `python:3.12-slim` or a CUDA container

Container rules:
- Always use `--name` and `--label agent=pi` on every container
- Use `--rm` for ephemeral tests (omit only if you need to inspect logs after exit)
- Mount the project directory: `-v ~/workspace/ossym-experiments:/workspace:Z`
- GPU passthrough: `--device nvidia.com/gpu=all`
- Clean up containers when done: `podman rm -f <name>`

Example — llama.cpp server in a container:
```bash
podman run -d \
  --name llama-server \
  --label agent=pi \
  --label purpose="llama.cpp inference server" \
  --device nvidia.com/gpu=all \
  -v ~/workspace/ossym-experiments:/workspace:Z \
  -p 8080:8080 \
  nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 \
  bash -c "cd /workspace/llama.cpp && ./build/bin/llama-server \
    --model /workspace/models/<model>.gguf \
    --ctx-size 4096 --n-gpu-layers 999 --no-mmap --flash-attn \
    --host 0.0.0.0 --port 8080"
```

**Fallback:** If container setup takes more than 15 minutes of debugging, run
directly on the host instead. Results are what matter, not the isolation method.

## Known issues on ARM64 + Blackwell hardware (if relevant to this machine)

These are documented from prior work on similar hardware. Consult if you hit issues:

- PyTorch stable does not support Blackwell (sm_120+). Use nightly from cu128 index.
- PyTorch nightly (cu128) vs CUDA 13.0: `cpp_extension.py` raises RuntimeError on
  major version mismatch. Patch it to a warning. CUDA 13.0 is backward compatible.
- No prebuilt aarch64 wheels for flash-attention, nvdiffrast, etc. Build from source
  with `TORCH_CUDA_ARCH_LIST="12.0+PTX"`.
- `nvidia-smi` does not display memory info on unified memory (GB10). Use `free -h`.
- mmap is slow on DGX Spark. Use `--no-mmap` with llama.cpp.
- pillow-simd is x86 only. Standard Pillow works fine.
- Gradio needs `server_name="0.0.0.0"` for remote access.
