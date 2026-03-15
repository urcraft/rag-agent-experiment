---
name: rag-benchmarker
description: >
  Set up and benchmark a RAG pipeline: build llama.cpp, download models, index a
  test corpus, sweep chunk sizes, compare retrieval strategies, and evaluate with
  RAGAS. Requires approved model selection from model-selector skill. Use after
  Phase 1 is approved.
---

# RAG Pipeline Setup & Benchmarking — Phase 2

Build the infrastructure, set up the RAG pipeline, and run benchmark sweeps.

## Prerequisites

- `results/phase1/selected_config.json` (approved by human)
- `results/phase1/hardware_profile.json`
- `config/decisions.json`

## Step 1: Build llama.cpp

```bash
cd ~/workspace/ossym-experiments
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

Detect GPU and set build flags:
- Read compute capability from the hardware profile.
- Build with CUDA if GPU detected:
  ```bash
  cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=<detected_compute_cap> \
    -DGGML_NATIVE=ON
  cmake --build build --config Release -j$(nproc)
  ```
- If no GPU: `cmake -B build -DGGML_NATIVE=ON && cmake --build build -j$(nproc)`

Verify: `./build/bin/llama-cli --version`

## Step 2: Download models

Download the selected generation model (GGUF) and embedding model from the URLs
in `selected_config.json`. Use `wget` or `huggingface-cli download`.

For large models (> 30 GB), monitor download progress. If download speed is
very slow (< 5 MB/s) and ETA exceeds 60 minutes, check `config/decisions.json`
for a fallback model.

## Step 3: Start llama.cpp server

```bash
./build/bin/llama-server \
  --model <path-to-gguf> \
  --ctx-size 4096 \
  --n-gpu-layers 999 \
  --no-mmap \
  --flash-attn \
  --host 0.0.0.0 \
  --port 8080 &
```

Note: `--no-mmap` and `--flash-attn` improve performance on DGX Spark-class
hardware. These flags are safe to use everywhere (mmap is just slower, flash-attn
falls back gracefully if unsupported).

Verify: `curl http://localhost:8080/health`

## Step 4: Set up Python environment

```bash
cd ~/workspace/ossym-experiments
python3 -m venv venv
source venv/bin/activate
pip install langchain chromadb rank_bm25 sentence-transformers ragas \
  openai faiss-cpu datasets beir
```

If any package fails to install on this architecture, document the error in
`results/issues.md` and try alternatives (e.g., `faiss-cpu` may need conda on
ARM64 — try `pip install faiss-cpu` first, fall back to chromadb's built-in
similarity search).

## Step 5: Download test corpus

From `config/decisions.json`, load the specified BEIR dataset:

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader

dataset = "scifact"  # or from decisions.json
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
```

## Step 6: Phase 2 benchmarks

Run in this order. Save all results to `results/phase2/` as JSON and CSV.

### 6a: Chunk size sweep

Test chunk sizes [256, 512, 1024, 2048] tokens with 10% overlap.
For each, index the corpus and measure retrieval recall@5 and recall@10 using
the ground-truth relevance judgments from BEIR.

Save: `results/phase2/chunk_size_sweep.json`

### 6b: Retrieval strategy comparison

Using the best chunk size from 6a, compare:
- Vector-only (cosine similarity, top-5)
- BM25-only (top-5)
- Hybrid (reciprocal rank fusion of vector + BM25, top-5)

Measure retrieval recall@5 and MRR for each.

Save: `results/phase2/retrieval_comparison.json`

### 6c: Reranker impact

With the best retrieval strategy from 6b, test with and without a cross-encoder
reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2` or similar).

Save: `results/phase2/reranker_comparison.json`

### 6d: End-to-end RAGAS evaluation

Using the best config from 6a–c, generate answers using the llama.cpp server
(OpenAI-compatible API at localhost:8080) and evaluate with RAGAS:
- Faithfulness
- Answer relevancy
- Context precision
- Context recall

Also measure: TTFT, tokens/second, peak RAM usage, total latency per query.

For the RAGAS LLM judge, use the same local model via the llama.cpp server.
If scores look nonsensical (everything > 0.95 or < 0.1), document the limitation.

Save: `results/phase2/ragas_evaluation.json`

## Step 7: Summary table

Compile all results into `results/summary_table.json`:

```json
{
  "hardware_profile": { "...from phase1..." },
  "selected_config": {
    "backend": "...",
    "generation_model": "...",
    "quantization": "...",
    "embedding_model": "...",
    "retrieval_strategy": "...",
    "chunk_size": "...",
    "reranker": "..."
  },
  "metrics": {
    "retrieval_recall_at_5": "...",
    "faithfulness": "...",
    "answer_relevancy": "...",
    "context_precision": "...",
    "context_recall": "...",
    "ttft_seconds": "...",
    "tokens_per_second": "...",
    "peak_ram_gb": "...",
    "latency_per_query_seconds": "..."
  }
}
```

## Step 8: Experiment report

After all benchmarks are complete, write a narrative experiment report to
`results/experiment_report.md`. This document synthesizes findings for direct
use in the OSSYM paper. It must contain the following sections:

### Hardware summary
One paragraph summarizing the machine specs from `results/phase1/hardware_profile.json`
(CPU, GPU, RAM, disk, OS, architecture).

### Model selection rationale
Which generation model was chosen and why — cite leaderboard ranks, MoE preference
from `config/decisions.json`, memory fit from the budget calculation. List alternatives
that were considered and why they were rejected.

### Benchmark results
Present tables (Markdown) for each Phase 2 benchmark, formatted for easy copy-paste
into the paper:
- Chunk size sweep (chunk size × recall@5 × recall@10)
- Retrieval strategy comparison (strategy × recall@5 × MRR)
- Reranker impact (with/without × recall@5 × MRR)
- RAGAS scores (faithfulness, answer relevancy, context precision, context recall)

### Performance metrics
Table of: TTFT (seconds), tokens/second, peak RAM (GB), average latency per query
(seconds). Note measurement methodology (e.g., averaged over N queries).

### Issues encountered
Bullet list of what failed, what was skipped, and any workarounds applied. Reference
`results/issues.md` if it exists.

### Key findings
3–5 bullet points summarizing the most important takeaways from the experiments.
Focus on what would be surprising or useful to the reader.

### Suggested paper text
Draft 2–3 paragraphs suitable for the results section of the extended abstract.
Reference specific numbers from the tables above. Write in third-person academic
style.

---

Save: `results/experiment_report.md`

## Time management

Priority if time is short:
1. Hardware profile + model selection (Phase 1) — minimum viable
2. llama.cpp build + model download + one retrieval comparison — publishable result
3. Full benchmark sweep + RAGAS — complete result

If any step takes > 45 min of debugging, document in `results/issues.md` and skip.
