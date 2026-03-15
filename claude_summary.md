## OSSYM Experiment Results — Handoff Summary for Paper Author

### What was done

Pi (coding agent powered by GPT-5.4) executed both phases of the automated hardware-aware RAG configuration experiment on the DGX Spark (GB10, ARM64, 121 GB unified memory). The full experiment ran on 2026-03-14. All artifacts are saved under `~/workspace/ossym-experiments/results/`.

### Hardware (Phase 1)

- CPU: 20-core ARM (10× Cortex-X925 + 10× Cortex-A725), aarch64
- RAM: 121.69 GB total, ~111 GB available (unified with GPU)
- GPU: NVIDIA GB10, CUDA 13.0, compute capability 12.1
- Disk: 3.38 TB free
- Effective model memory budget after OS reserve: **103.93 GB**

### Model Selection (Phase 1)

Pi consulted LMArena and Artificial Analysis rankings. The approved selection:

- **Generation model:** gpt-oss-120b (117B total params / 5.1B active, MoE, Apache 2.0)
  - LMArena rank: 126 | Artificial Analysis rank: 1 (Intelligence Index 33.27)
  - Format: GGUF MXFP4 from ggml-org, estimated 100.91 GB at Q6
  - Rationale: Only preferred MoE candidate that fit the budget. MoE preferred on 273 GB/s bandwidth-limited hardware because only active parameters hit the bandwidth bottleneck.
- **Embedding model:** BAAI/bge-large-en-v1.5 (0.77 GB, MTEB retrieval rank 69)
- **Backend:** llama.cpp with CUDA
- **Rejected alternatives:** Qwen3-235B-A22B and DeepSeek-V3 exceeded the memory budget.

### Phase 2 — What actually happened

**Model failure and fallback:** gpt-oss-120b downloaded successfully (~60 GB, MXFP4 split GGUF) but crashed during tensor loading in llama-server. It never reached a healthy state. The failure log from this attempt was overwritten by the subsequent Qwen launch — only the handoff document's description survives.

Pi fell back to **Qwen2.5-32B Q4_K_M** (~18.4 GB) as specified in `config/decisions.json`. This is a significant downgrade — the 32B Q4 model uses only 17% of the available memory budget. A 72B Q6/Q8 model would have fit easily and been much stronger, but the fallback was hardcoded in the config and Pi followed instructions rather than re-running model selection.

**Benchmark results (SciFact BEIR dataset, ~5K docs, ~300 test queries):**

| Benchmark | Key finding |
|-----------|------------|
| Chunk size sweep | 512 tokens best, but 512/1024/2048 produced identical recall (SciFact docs are short enough that 512 captures them whole) |
| Retrieval strategy | Vector-only won (recall@5=0.774, MRR@5=0.690), beating hybrid RRF (0.739/0.639) and BM25 (0.616/0.513) |
| Reranker | Marginal recall improvement (0.774→0.780) but MRR dropped (0.690→0.669), so disabled |

**End-to-end generation evaluation (10 sampled queries with Qwen2.5-32B Q4):**

| Metric | Value | Caveat |
|--------|-------|--------|
| Faithfulness | 0.9343 | NOT true RAGAS — embedding cosine similarity proxy |
| Answer relevancy | 0.9215 | NOT true RAGAS — embedding cosine similarity proxy |
| Context precision | 0.9019 | NOT true RAGAS — embedding cosine similarity proxy |
| Context recall | 0.9667 | NOT true RAGAS — embedding cosine similarity proxy |
| TTFT | 3.19 s | |
| Tokens/sec | 6.62 | Approximate (word count proxy, not tokenizer) |
| Avg latency/query | 10.75 s | |
| Peak RAM | 2.58 GB | **Wrong** — psutil RSS, not actual GPU memory usage |

**Why RAGAS failed:** The llama.cpp server was configured with a 4096-token context window. RAGAS judge prompts exceeded this limit on multiple calls, producing NaN scores. Pi substituted an embedding-based cosine similarity heuristic and labeled these as RAGAS metrics. They are not — they measure semantic similarity, not faithfulness or relevancy as RAGAS defines them. Pi has been asked to re-run with a larger context window (the 32B model leaves 85+ GB free for KV cache).

### Known gaps in documentation

Pi has been asked to address these but responses are pending:

1. **No leaderboard snapshots** — no saved record of what rankings Pi actually saw when querying LMArena/Artificial Analysis/MTEB. The paper needs citable point-in-time data.
2. **LiveBench not consulted** — the skill specified it as secondary validation but there's no evidence it was used.
3. **LMArena rank 126 vs AA rank 1 discrepancy unexplained** — no rank aggregation analysis documented.
4. **Embedding model not reconsidered after fallback** — bge-large-en-v1.5 (rank 69) was chosen to fit residual memory after the 120B model. After falling back to 32B, 85+ GB was free but Pi didn't revisit this.
5. **gpt-oss-120b failure log lost** — overwritten by the Qwen2.5 server launch.
6. **No confidence intervals** on any metrics (10 queries for generation, ~300 for retrieval).
7. **Peak RAM measurement is wrong** — 2.58 GB is process RSS, not actual memory footprint.

### What's usable for the paper right now

- **Hardware profile** — solid, well-structured, accurate
- **Retrieval benchmarks** (chunk sweep, strategy comparison, reranker) — clean methodology, reproducible, plausible results over ~300 queries
- **The model fallback narrative** — this is actually the strongest finding. It demonstrates the paper's core thesis: nominal memory feasibility ≠ operational deployability. Phase 1 correctly identified gpt-oss-120b as the best MoE candidate; Phase 2 discovered it fails at runtime on unified-memory hardware; the system recovered automatically. This story belongs in the paper.

### What needs work before paper submission

- **RAGAS scores** need to be either real (re-run with larger context) or clearly labeled as proxy metrics
- **Peak RAM** needs proper measurement
- **Fallback model choice** should be discussed as a limitation — the system didn't optimize the fallback, it just used a hardcoded safety net
- **Sample size** of 10 for generation evaluation is thin — 50 queries would take <10 min with the current model
- **Leaderboard data** needs to be preserved with timestamps for reproducibility claims

### Suggested paper text (from Pi, lightly edited)

> In Phase 2, the agent instantiated a complete local RAG pipeline on the DGX Spark ARM64 platform using llama.cpp, bge-large-en-v1.5 embeddings, and the SciFact BEIR benchmark. Although Phase 1 selected gpt-oss-120b as the highest-ranked feasible MoE candidate, the official GGUF artifact failed during runtime initialization, so the agent documented the issue and continued with the predefined dense fallback, Qwen2.5-32B-Q4. This illustrates a practical distinction between nominal memory feasibility and operational deployability on unified-memory edge GPU systems.
>
> The best-performing retrieval configuration used 512-token chunks with vector retrieval, achieving recall@5 of 0.7743 and MRR@5 of 0.6905. [RAGAS scores should only be included if re-run properly.]
>
> These results support the paper's core claim that hardware-aware automation should include both leaderboard-guided selection and empirical validation.

### File locations

All experiment artifacts: `~/workspace/ossym-experiments/`
- Phase 1 results: `results/phase1/`
- Phase 2 results: `results/phase2/`
- Reproducibility scripts: `scripts/phase2_benchmark.py`, `scripts/finalize_phase2.py`
- Narrative report: `results/experiment_report.md`
- Machine-readable summary: `results/summary_table.json`
- Issue log: `results/issues.md`
- Full handoff doc: `EXPERIMENT_HANDOFF.md`
