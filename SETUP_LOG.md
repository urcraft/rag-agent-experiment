# OSSYM 2026 — Setup Log

**Date:** 2026-03-14
**Setup agent:** Claude Code (Opus 4.6)
**Target agent:** Pi coding agent (GPT-5.4 via OpenAI Codex subscription)

---

## Pi Installation

- **Package:** `@mariozechner/pi-coding-agent`
- **Version:** 0.58.1
- **Install method:** `npm install -g @mariozechner/pi-coding-agent`
- **Binary location:** `~/.npm-global/bin/pi`
- **Status:** Installed successfully on ARM64/aarch64. No issues.
- **npm warnings:** Deprecated `node-domexception` and `glob@10.5.0` (non-blocking).

## Authentication

- **Method:** OpenAI Codex subscription via `/login` OAuth flow inside Pi
- **Status:** NOT YET AUTHENTICATED — user must run `/login` interactively inside Pi, then `/model gpt-5.4`
- **No OPENAI_API_KEY** is set in the environment. The Codex subscription OAuth flow is the intended auth method.

## System Packages

### Already installed:
- build-essential, cmake, git, curl, wget, jq, python3-venv, python3-pip

### Needs sudo install (from spark-1 admin account):
```bash
sudo apt-get install -y ninja-build python3-dev libgl1-mesa-dev libgles2-mesa-dev libegl1-mesa-dev
```
These are needed for Phase 2 (building llama.cpp with CUDA, Python C extensions).
Phase 1 can run without them.

## Project Directory

All files created at `~/workspace/ossym-experiments/`:

```
AGENTS.md                              ✓  Project instructions (auto-loaded by Pi)
PROJECT_CONTEXT.md                     ✓  Paper context (copied from ~/workspace/)
README.md                              ✓  Skeleton for Pi to fill
.pi/skills/hardware-profiler/SKILL.md  ✓  Phase 1(a): Hardware detection
.pi/skills/memory-budget/SKILL.md      ✓  Phase 1(b): Memory budget calculation
.pi/skills/model-selector/SKILL.md     ✓  Phase 1(c-e): Model + backend selection
.pi/skills/rag-benchmarker/SKILL.md    ✓  Phase 2: Full RAG benchmark suite
.pi/prompts/run-phase1.md             ✓  One-command Phase 1 trigger
.pi/prompts/run-phase2.md             ✓  One-command Phase 2 trigger
config/decisions.json                  ✓  Pre-made decisions (valid JSON, 6 keys)
scripts/                               ✓  Empty (Pi creates scripts here)
results/phase1/                        ✓  Empty (Pi saves Phase 1 results here)
results/phase2/                        ✓  Empty (Pi saves Phase 2 results here)
```

## Verification Results

- `pi --version` → 0.58.1 ✓
- `pi --help` → CLI options displayed ✓
- `find ~/workspace/ossym-experiments -type f` → 10 files, all present ✓
- `python3 -c "import json; json.load(open('config/decisions.json'))"` → valid ✓
- YAML frontmatter on all 4 SKILL.md files → correct ✓

## Updates

- **2026-03-14:** Added Step 8 (experiment report) to `rag-benchmarker/SKILL.md` — Pi will
  write `results/experiment_report.md` after all benchmarks, with hardware summary, model
  rationale, formatted tables, key findings, and draft paper text. Cross-referenced in
  `AGENTS.md` Rules section.

## Issues Encountered

None.

## How to Launch Pi for the Experiment

```bash
# 1. SSH in as ai-agent (or use existing session)
ssh ai-agent@localhost

# 2. Navigate to project directory
cd ~/workspace/ossym-experiments

# 3. Launch Pi with OpenAI provider
pi --provider openai --model gpt-5.4

# 4. Inside Pi, authenticate (first time only):
#    /login        (select OpenAI, complete OAuth flow)
#    /model gpt-5.4

# 5. Run Phase 1:
#    /run-phase1

# 6. Review the output. If model selection looks good, run Phase 2:
#    /run-phase2
```

## Fallback

If Pi cannot authenticate or hits issues, Claude Code can execute the same
algorithm by following the SKILL.md files step by step. The instruction files
are the primary deliverable — they encode the paper's algorithm in a portable,
agent-readable format that any coding agent can execute.
