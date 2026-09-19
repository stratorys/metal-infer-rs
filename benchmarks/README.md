# Benchmarks

The benchmark suite compares `metal-infer` with native MLX and, for complete
model runs, llama.cpp. It uses release builds, explicit GPU synchronization,
warm-up runs, and identical FP16 matrix shapes.

## Run the kernel comparison

```sh
uv run benchmarks/compare.py kernels
```

The cases, warm-up count, and measured iterations are defined in `suite.json`.
The command writes normalized raw data to `results/latest.json` and refreshes
the table below. Commands and their complete output are displayed as they run;
each step ends with its elapsed time and primary metrics.

For automation, suppress child output and progress messages with:

```sh
uv run benchmarks/compare.py --quiet kernels
```

## Run the model comparison

```sh
uv run benchmarks/compare.py model \
  --metal-model ./models/Qwen3-0.6B \
  --mlx-model ./models/Qwen3-0.6B
```

Add llama.cpp when an FP16 GGUF converted from the same checkpoint is
available:

```sh
uv run benchmarks/compare.py model \
  --metal-model ./models/Qwen3-0.6B \
  --mlx-model ./models/Qwen3-0.6B \
  --gguf ./models/Qwen3-0.6B-f16.gguf \
  --llama-bench /path/to/llama-bench
```

Model results report prefill and decode separately. Tokenization and sampling
are excluded. Comparisons are meaningful only when the checkpoint, precision,
prompt length, generation length, and power conditions match. llama.cpp is not
included in the kernel table because `llama-bench` measures model execution,
not an isolated generic matrix multiplication.

## Results

Lower latency is better. Higher TFLOP/s and relative throughput are better.
`1.00×` is the `metal-infer` result for the same shape.

<!-- BENCH_RESULTS_START -->

Last generated: `2026-09-19T14:41:30.410545+00:00`.

| Shape (M×N×K) | Backend | Mean ms | TFLOP/s | Relative |
|---|---|---:|---:|---:|
| 64×128×128 | metal-infer | 0.205 | 0.0102 | 1.00× |
| 64×128×128 | mlx | 0.242 | 0.0087 | 0.85× |
| 128×1024×1024 | metal-infer | 0.915 | 0.2933 | 1.00× |
| 128×1024×1024 | mlx | 0.321 | 0.8375 | 2.86× |
| 512×1024×1024 | metal-infer | 1.044 | 1.0285 | 1.00× |
| 512×1024×1024 | mlx | 0.727 | 1.4775 | 1.44× |
| 1024×1024×1024 | metal-infer | 1.652 | 1.2996 | 1.00× |
| 1024×1024×1024 | mlx | 0.782 | 2.7454 | 2.11× |

<!-- BENCH_RESULTS_END -->

These numbers are measurements, not correctness tests. Run the GPU test suite
before collecting them:

```sh
cargo test -p metal-infer-core --test gpu -- --ignored
```
