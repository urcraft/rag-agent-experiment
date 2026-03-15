# Experiment Report

## Hardware summary
This Phase 2 benchmark ran on a DGX Spark-class ARM64 workstation with a 20-core Cortex-X925/A725 CPU complex, 121.69 GiB unified system memory, and an NVIDIA GB10 GPU (CUDA 13.0, compute capability 12.1). The machine exposes GPU acceleration through unified memory rather than separately reported VRAM, and the project workspace had roughly 3.38 TB of free disk at benchmark time. llama.cpp was built locally with CUDA enabled and used as the OpenAI-compatible inference backend.

## Model selection rationale
Phase 1 approved `gpt-oss-120b` because it was the strongest feasible open-weight MoE candidate under the memory budget and matched the project preference for MoE models on bandwidth-limited hardware. During Phase 2, the official `ggml-org/gpt-oss-120b-GGUF` MXFP4 model downloaded successfully but the server exited during tensor loading before exposing a health endpoint, so the benchmark run fell back to the documented dense fallback from `config/decisions.json`: `Qwen2.5-32B-Q4`. The embedding model remained `BAAI/bge-large-en-v1.5`, which was the best-ranked preferred retrieval model that fit comfortably in memory. Alternatives considered in Phase 1 included Qwen3-235B-A22B and DeepSeek-V3, but both exceeded the machine's practical unified-memory budget.

## Benchmark results

### Chunk size sweep

| chunk_size | num_chunks | recall@5 | recall@10 |
| ---------- | ---------- | -------- | --------- |
| 256.0      | 6633.0     | 0.7673   | 0.8459    |
| 512.0      | 5226.0     | 0.7743   | 0.8592    |
| 1024.0     | 5188.0     | 0.7743   | 0.8592    |
| 2048.0     | 5183.0     | 0.7743   | 0.8592    |

### Retrieval strategy comparison

| strategy | chunk_size | recall@5 | mrr@5  |
| -------- | ---------- | -------- | ------ |
| vector   | 512        | 0.7743   | 0.6905 |
| bm25     | 512        | 0.6158   | 0.5132 |
| hybrid   | 512        | 0.7392   | 0.6393 |

### Reranker impact

| reranker | strategy | chunk_size | recall@5 | mrr@5  |
| -------- | -------- | ---------- | -------- | ------ |
| disabled | vector   | 512        | 0.7743   | 0.6905 |
| enabled  | vector   | 512        | 0.78     | 0.6688 |

### RAGAS scores

| faithfulness | answer_relevancy | context_precision | context_recall |
| ------------ | ---------------- | ----------------- | -------------- |
| 0.9343       | 0.9215           | 0.9019            | 0.9667         |

## Performance metrics

| metric | value |
|---|---:|
| TTFT (s) | 3.1940 |
| tokens/s | 6.6226 |
| peak RAM (GB) | 2.5789 |
| average latency/query (s) | 10.7486 |

Measurement methodology: latency and TTFT were measured from streamed responses against the local llama.cpp server over 10 SciFact queries using the best retrieval configuration found in the preceding sweeps. Tokens/second is computed from generated word count divided by post-first-token generation time, so it should be treated as an approximate throughput indicator rather than a tokenizer-exact metric.

## Issues encountered
- The approved `gpt-oss-120b` GGUF artifact downloaded successfully but failed to remain up as a llama.cpp server on this machine; Phase 2 therefore used the documented dense fallback model.
- Direct local RAGAS evaluation hit the llama.cpp 4096-token context limit on several judge calls; aggregate RAGAS scores were repaired using the saved generations/contexts and an embedding-based fallback metric computation.
- Full issue log: see `results/issues.md`.

## Key findings
- The best retrieval chunk size on SciFact was 512 tokens under the selected embedding model.
- The best retrieval strategy was `vector`, which outperformed BM25 and hybrid RRF on this run.
- Reranking slightly improved recall@5 but reduced MRR@5, so the non-reranked vector retriever remained the best overall setting by MRR.
- End-to-end local RAG was feasible on the DGX Spark-class host once the generation model was reduced from the approved 120B-class candidate to the documented 32B fallback.
- The benchmark shows that infrastructure feasibility can diverge from paper-time model selection: a model can fit the nominal memory budget yet still fail operationally during server bring-up.
