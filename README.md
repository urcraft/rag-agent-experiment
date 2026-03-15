# OSSYM 2026 — Automated Hardware-Aware RAG Configuration

Experiments for a research paper on automated local RAG system configuration using an agent-based two-phase approach. See [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) for the full paper context.

## How it works

A coding agent (Pi, powered by GPT-5.4) autonomously executes a two-phase algorithm:

1. **Phase 1 — Infrastructure selection:** Profile hardware, compute memory budgets, query live leaderboards (LMArena, Artificial Analysis, MTEB), and select the best feasible model + backend configuration.
2. **Phase 2 — RAG optimization:** Build llama.cpp, download models, index a test corpus (SciFact/BEIR), sweep chunk sizes, compare retrieval strategies, evaluate with RAGAS, and produce a final experiment report.

The algorithm is encoded in agent-readable skill files under `.pi/skills/`:

| Skill | Purpose |
|-------|---------|
| [`hardware-profiler`](.pi/skills/hardware-profiler/SKILL.md) | CPU/RAM/GPU/disk detection → JSON |
| [`memory-budget`](.pi/skills/memory-budget/SKILL.md) | Memory budget per quantization level |
| [`model-selector`](.pi/skills/model-selector/SKILL.md) | Leaderboard-driven model + backend selection |
| [`rag-benchmarker`](.pi/skills/rag-benchmarker/SKILL.md) | llama.cpp build, RAG pipeline, 4 benchmark sweeps, RAGAS evaluation, experiment report |

Trigger prompts: [`.pi/prompts/run-phase1.md`](.pi/prompts/run-phase1.md), [`.pi/prompts/run-phase2.md`](.pi/prompts/run-phase2.md)

## Hardware

DGX Spark (ARM64): 20-core Cortex-X925/A725, 121 GB unified memory, NVIDIA GB10 (CUDA 13.0, compute capability 12.1). Full profile in [`results/phase1/hardware_profile.json`](results/phase1/hardware_profile.json).

## Key results

- **Phase 1** selected `gpt-oss-120b` (117B MoE, 5.1B active) as the best feasible model. It fit the memory budget on paper but failed at runtime during llama-server startup. Phase 2 fell back to `Qwen2.5-32B Q4`.
- **Best retrieval config:** 512-token chunks, vector retrieval, no reranker — recall@5 = 0.774, MRR@5 = 0.690
- **RAGAS scores** are proxy metrics (embedding-based), not true RAGAS — see [`results/issues.md`](results/issues.md)

Full results: [`results/experiment_report.md`](results/experiment_report.md) | [`results/summary_table.json`](results/summary_table.json)

## Repository structure

```
├── AGENTS.md                         # Agent instructions (auto-loaded by Pi)
├── PROJECT_CONTEXT.md                # Paper context, venue, related work
├── config/decisions.json             # Pre-made experiment decisions
├── .pi/
│   ├── prompts/                      # One-command phase triggers
│   └── skills/                       # Phase 1 + Phase 2 algorithm as skill files
├── scripts/
│   ├── phase2_benchmark.py           # Main benchmark script
│   └── finalize_phase2.py            # Post-processing and report generation
├── results/
│   ├── summary_table.json            # Combined metrics (machine-readable)
│   ├── experiment_report.md          # Narrative report with tables and paper text
│   ├── issues.md                     # Problems encountered and workarounds
│   ├── phase1/                       # Hardware profile, memory budget, model selection
│   └── phase2/                       # Benchmark JSONs, CSVs, logs
├── EXPERIMENT_HANDOFF.md             # Detailed handoff for follow-on agents
├── claude_summary.md                 # Claude Code's analysis of Pi's results
└── answers.md                        # Pi's answers to post-experiment review questions
```

## How to reproduce

1. Install Pi: `npm install -g @mariozechner/pi-coding-agent`
2. `cd` into this directory
3. `pi --provider openai --model gpt-5.4`
4. Authenticate with `/login`, then run `/run-phase1` followed by `/run-phase2`

The skill files are agent-agnostic — any coding agent with bash access can follow them. See [`SETUP_LOG.md`](SETUP_LOG.md) for detailed setup notes.

## Known issues

1. `gpt-oss-120b` MXFP4 GGUF failed during tensor loading on GB10 unified memory — see [`results/issues.md`](results/issues.md)
2. RAGAS evaluation hit the 4096-token context limit; reported scores are embedding-based proxies, not true RAGAS
3. Fallback model (32B Q4) underutilizes the hardware — a 72B model would have been a better choice

## License

Experiment code and configuration files. Paper content in `PROJECT_CONTEXT.md` is not licensed for redistribution.
