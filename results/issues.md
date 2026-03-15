# Issues Encountered

## 2026-03-14T20:01:40+01:00
- Attempted to start llama.cpp server with approved model `gpt-oss-120b` (official GGUF MXFP4 split from `ggml-org/gpt-oss-120b-GGUF`).
- Model files downloaded successfully (~60G), but `llama-server` exited during tensor loading after allocating ~59.9 GiB on CUDA0 and never exposed `/health`.
- Workaround: proceed with the documented dense fallback from `config/decisions.json` (`Qwen2.5-32B-Q4`) for Phase 2 benchmarking so the benchmark sequence can complete.

## 2026-03-14T23:36:10+0100
- Direct local RAGAS evaluation produced NaN metrics because multiple judge requests exceeded the 4096-token server context window. Replaced aggregate RAGAS scores with embedding-based heuristic fallback metrics computed from the saved generations and retrieved contexts.

