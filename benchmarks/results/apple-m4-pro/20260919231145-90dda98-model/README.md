# Model benchmark

Generated at `2026-09-19T23:11:45.911509+00:00` on **Apple M4 Pro**.

- macOS: `26.6.2`
- Rust: `rustc 1.98.0 (88d9e12ae 2026-08-18)`
- metal-infer commit: `90dda983b10d5b4cf92a3b78544ca8a49a443d00`
- configuration: `warmup=1, iterations=5, prompt_tokens=512, generation_tokens=128`

![Model benchmark results](results.svg)

| Backend | Prefill tokens/s | Prefill wall/GPU ms | Decode tokens/s | Decode wall/GPU ms | Memory |
|---|---:|---:|---:|---:|---:|
| metal-infer | 1248.521 | 410.085/409.097 | 82.109 | 1558.900/1511.820 | 1.266 GB allocated, +0.000 MB measured |
| mlx-lm | 4442.045 | — | 161.471 | — | 1.960 GB peak |

Memory values are not directly comparable: metal-infer reports Metal allocated memory while MLX reports peak memory.

Raw measurements: [results.json](results.json).

The benchmark methodology and reproduction commands are documented in the [benchmark guide](../../../README.md).
