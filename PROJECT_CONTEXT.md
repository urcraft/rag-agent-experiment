# OSSYM 2026 Extended Abstract — Project Context

## Paper

**Title:** Toward Automated Hardware-Aware Configuration of Local Retrieval-Augmented Generation Systems

**Submission type:** Extended abstract (1–2 pages, JACoW A4 format)

**Status:** v4 draft complete (6 pages, unconstrained length). Needs trimming to 2 pages for submission. [TODO] placeholders remain for experimental results. System diagram (Figure 1) is complete and embedded.

## Venue

**Conference:** OSSYM 2026 — 8th International Open Search Symposium "Boosting Digital Sovereignty"

**Dates:** 7–9 October 2026 (Berlin, Germany + online)

**Hosted on:** CERN Indico — https://indico.cern.ch/event/1622016/

**Submission deadline:** 15 March 2026, 23:59 CET (extended)

**Submission format:** PDF (A4) using JACoW document templates (Word/LaTeX/ODF available on the conference page)

**Proceedings:** Open-access online publication with DOIs and ISBN, autumn 2026. Selected papers invited for journal extended versions.

**Conference focus:** Open web search, open search architecture, digital sovereignty, European search infrastructure. Programme committee is heavily IR (information retrieval) and web search oriented. The paper must be framed around sovereign/local search and retrieval — not generic LLM tooling.

## Core Idea

An agent-based system that automatically configures the most optimal local RAG (Retrieval-Augmented Generation) setup for a given machine, based on hardware capabilities and user requirements.

The configuration space includes: inference backend (llama.cpp, vLLM, Ollama), language model selection and quantization, embedding model, retrieval strategy (vector, BM25, hybrid), reranking, and chunk size.

## Key Contribution / Novelty Claim

**AutoRAG** (Kim et al., arXiv:2410.20878) already optimizes *which* RAG pipeline modules to use, but it assumes a working inference stack is already in place. Our contribution is the **infrastructure layer**: hardware detection → backend selection → model/quantization fitting → embedding model selection, all before RAG pipeline optimization begins.

This is the gap: nobody automates the step from "I have a laptop" to "I have a working, well-configured local RAG system."

## Two-Phase Architecture

1. **Phase 1 — Leaderboard-Driven Infrastructure Selection:** Detect hardware (CPU, RAM, GPU/VRAM, disk), compute memory budgets, then select the best feasible models by querying independently maintained, community-trusted leaderboards — rather than using hardcoded heuristic rules that go stale as new models are released.

2. **Phase 2 — Empirical RAG Optimization:** Run benchmarks to optimize data-dependent dimensions: embedding model comparison (retrieval recall@k), chunk size sweep, hybrid vs. vector-only retrieval, reranker impact. Evaluate with RAGAS metrics (faithfulness, answer relevancy, context precision).

### Phase 1 Design Decision: Leaderboards Over Hardcoded Heuristics

Static heuristics ("if 16 GB RAM, use Qwen2.5-7B Q4") fail because the open-model landscape changes every few weeks. Leaderboard-driven selection is more systematic and future-proof.

**Leaderboard selection criteria** — we select ranking sources that satisfy:
1. **Independent verification:** Scores produced/verified by third parties, not self-reported by model creators.
2. **Resistance to gaming:** Via crowdsourced human preference, contamination-resistant question generation, or reproducible evaluation pipelines.
3. **Community adoption:** Wide use by researchers and developers (citation counts, use as reference in model releases, active maintenance).

**Leaderboard sources used:**

| Selection Decision | Source | Methodology | Reference |
|---|---|---|---|
| Generation model quality | LMArena / Chatbot Arena | Anonymous pairwise battles, Bradley-Terry ratings from 6M+ crowdsourced votes; style-controlled rankings isolate capability from verbosity bias | Chiang et al., arXiv:2403.04132, 2024 |
| Generation model quality | Artificial Analysis | Independent evaluation across 10 benchmarks (GPQA Diamond, Humanity's Last Exam, Terminal-Bench, etc.); composite Intelligence Index v4.0; also provides throughput/latency data | https://artificialanalysis.ai/methodology |
| Generation model quality (validation) | LiveBench | Monthly-refreshed contamination-resistant questions from recent arXiv papers, news, and competitions; objective ground-truth scoring without LLM judge | White et al., ICLR 2025 Spotlight |
| Embedding model quality | MTEB | 1000+ language coverage; 8 task categories; we use *retrieval subtask* rankings specifically, not the overall average; reproducible evaluation pipeline; community-reviewed submissions; no longer accepts self-reported scores | Muennighoff et al., arXiv:2210.07316, 2023 |

**Note on HuggingFace Open LLM Leaderboard:** Officially retired as of early 2026. The team noted benchmarks were becoming obsolete and could encourage hill-climbing irrelevant directions. Historically important (evaluated 13K+ models over 2 years) but no longer a live data source.

**Note on BFCL (Berkeley Function Calling Leaderboard):** Relevant if we want the agent itself to use the best model for tool calling (now at V4 with agentic evaluation). Secondary concern for the paper — more relevant for implementation.

### Phase 1 Algorithm (5 steps)

1. **(a) Hardware profiling:** Probe CPU, RAM, GPU/VRAM, disk via system tools (lscpu, nvidia-smi, free -h).
2. **(b) Memory budget estimation:** Compute max model size per quantization level. Formula: `required_memory ≈ (parameters × bits_per_weight) / 8 + overhead` (overhead = 10–20% for KV-cache, activations, framework).
3. **(c) Generation model selection:** Query LMArena + Artificial Analysis rankings → filter to open-weight models → intersect with feasible set → select top-ranked. Rank aggregation (average rank) for tie-breaking. LiveBench as secondary validation.
4. **(d) Embedding model selection:** Compute residual memory after generation model → query MTEB *retrieval-task* rankings (not overall average) → filter by budget → select top.
5. **(e) Backend + quantization:** Deterministic rules: no GPU → llama.cpp; GPU with sufficient VRAM → vLLM/Ollama; mixed offloading for borderline cases. Select highest quantization (Q8 > Q5 > Q4) that fits.

Cached fallback ranking bundled with the agent for offline/air-gapped use.

## Implementation Approach

The execution layer is a **coding agent with shell access** (e.g., Pi, Claude Code, or similar). The agent writes and runs scripts to probe hardware, install backends, download models, execute benchmarks, and iterate. This was chosen over a rigid pipeline because:

- The configuration space is too heterogeneous for a fixed script (different OSes, GPU drivers, package managers, etc.)
- A coding agent can adapt, debug, and retry
- The extension/skill system of agents like Pi allows persistent state across sessions

The paper is deliberately **tool-agnostic** — it describes "a coding agent with bash access" rather than tying to a specific product. The specific agent used is an implementation detail, not the contribution.

### About Pi (the coding agent considered for the prototype)

- GitHub: https://github.com/badlogic/pi-mono/
- Blog post by Armin Ronacher: https://lucumr.pocoo.org/2026/1/31/pi/
- Minimal core: 4 tools (Read, Write, Edit, Bash)
- Extension system with persistent session state
- Session branching (explore alternative configs without losing context)
- MIT licensed
- Written by Mario Zechner; used as the engine behind OpenClaw

## Known Limitations to Acknowledge

- **Non-determinism:** LLM-driven agent may produce different configurations on repeated runs. Mitigate by framing the agent as a search procedure: Phase 1 is largely deterministic (given same leaderboard data), Phase 2 produces reproducible benchmark numbers.
- **Bootstrapping problem:** You need an LLM (cloud API or local) to run the agent that sets up your local LLM. Frame as a one-time setup cost. A lightweight local model can also bootstrap.
- **Leaderboard availability and freshness:** System depends on external leaderboards being accessible. Cached fallback provides a reasonable default if network is unavailable. Cached ranking gradually goes stale but is better than no recommendation.
- **Scope:** This is a concept/position paper with preliminary results, not a full system evaluation. Demonstrated on 2 hardware configs; broader validation needed (ARM, Apple Silicon, Intel Arc, multi-GPU).

## Key Related Work

| Reference | Relevance |
|-----------|-----------|
| AutoRAG (Kim et al., 2024) | AutoML for RAG pipeline modules — operates *above* our infrastructure layer. Complementary; could be integrated. |
| LMArena / Chatbot Arena (Chiang et al., 2024) | Crowdsourced pairwise human-preference evaluation with Bradley-Terry ratings. Primary signal for generation model selection in Phase 1. 6M+ votes, style-controlled rankings. |
| LiveBench (White et al., ICLR 2025) | Contamination-resistant LLM benchmark. Monthly-refreshed questions from recent sources, objective ground-truth scoring without LLM judge. Secondary validation signal for Phase 1. |
| Artificial Analysis (2025) | Independent evaluation across 10 benchmarks. Composite Intelligence Index + throughput/latency data. Primary signal alongside LMArena for Phase 1. |
| MTEB (Muennighoff et al., 2023) | Embedding model benchmark, 1000+ languages, 8 task categories. We use retrieval subtask rankings specifically. Primary signal for embedding model selection. |
| RAGAS (Es et al., 2023) | Evaluation framework used in Phase 2 (faithfulness, answer relevancy, context precision). |
| llama.cpp (Gerganov) | Primary CPU inference backend |
| vLLM (Kwon et al., SOSP 2023) | GPU inference with PagedAttention |
| Once-for-All (Cai et al., ICLR 2020) | Precedent for hardware-aware model optimization — principle we extend to RAG configuration |
| OpenWebSearch.EU | Connects to OSSYM's mission — sovereignty framing |

## Remaining TODOs

- [ ] Fill in author name(s) and affiliation(s)
- [ ] Run experiments on at least 1–2 hardware setups
- [ ] Fill in Table 2 results (retrieval recall@5, individual RAGAS scores, latency, RAM usage)
- [ ] Describe the test corpus used
- [ ] Add GitHub URL for the prototype
- [ ] Trim v4 (6 pages) down to 2 pages for submission
- [ ] Convert final draft to JACoW template format (LaTeX or Word template from conference page)
- [ ] Generate compliant PDF and submit via Indico

## Version History

- **v1:** Original draft with hardcoded heuristic Phase 1 ("apply deterministic rules").
- **v2:** Rewrote Phase 1 as leaderboard-driven. Added LMArena, LiveBench, Artificial Analysis, MTEB as named sources with selection criteria. Updated Table 1, Introduction, Related Work, References (8 → 11). No diagram.
- **v3:** Added Figure 1 (system architecture diagram). Trimmed text to fit 2 pages with figure (9pt body, compressed spacing).
- **v4:** Full unconstrained version (6 pages). Expanded Introduction (4 paragraphs), Phase 1 (selection criteria as separate paragraphs, new Table 3 with leaderboard methodology details, 5 algorithm steps with full detail), Phase 2 (separate paragraphs for chunk size, retrieval strategy, reranker), Table 2 (8 rows with individual RAGAS metrics), Related Work (4 subsections), new Discussion and Limitations section, expanded Conclusion and Future Work (5 items).
