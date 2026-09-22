# Model benchmark

Generated at `2026-09-22T23:44:34.489872+00:00` on **Apple M4 Pro**.

- macOS: `26.7`
- Rust: `rustc 1.98.0 (88d9e12ae 2026-08-18)`
- metal-infer commit: `9b90abbec6cdfc3488d5c3f3bbaa6625222718c0`
- configuration: `warmup=1, iterations=5, prompt_tokens=512, generation_tokens=128`

![Model benchmark results](results.svg)

| Backend | Prefill tokens/s | Prefill wall/GPU ms | Decode tokens/s | Decode wall/GPU ms | Memory |
|---|---:|---:|---:|---:|---:|
| metal-infer | 4301.452 | 119.030/118.621 | 161.198 | 794.055/793.600 | 1.290 GB allocated, +0.000 MB measured |
| mlx-lm | 4500.326 | — | 167.812 | — | 1.960 GB peak |

Memory values are not directly comparable: metal-infer reports Metal allocated memory while MLX reports peak memory.

Raw measurements: [results.json](results.json).

The benchmark methodology and reproduction commands are documented in the [benchmark guide](../../../README.md).
