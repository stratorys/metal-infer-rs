# Benchmarks

The benchmark suite compares `metal-infer` with native MLX and, for complete
model runs, llama.cpp. It uses release builds, explicit GPU synchronization,
warm-up runs, and identical FP16 matrix shapes.

## Run the kernel comparison

```sh
uv run benchmarks/compare.py kernels
```

The cases, warm-up count, and measured iterations are defined in `suite.json`.
They include `m=1` decode projections for Qwen3 hidden, MLP, and vocabulary
dimensions, followed by larger prefill-oriented matrix multiplications.
The command creates a run directory under `results/<machine>/` containing raw
JSON, an SVG graph, and a standalone README. Commands and their complete output
are displayed as they run; each step ends with its elapsed time and primary
metrics.

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

Choose an explicit destination when needed:

```sh
uv run benchmarks/compare.py \
  --result-dir benchmarks/results/apple-m4-pro/manual-run \
  kernels
```

Model results report prefill and decode separately. Tokenization and sampling
are excluded. Comparisons are meaningful only when the checkpoint, precision,
prompt length, generation length, and power conditions match. llama.cpp is not
included in the kernel table because `llama-bench` measures model execution,
not an isolated generic matrix multiplication.

## Results

See the [results index](results/README.md). Each entry contains its exact
values and graph. Lower latency is better; higher TFLOP/s and tokens/s are
better.

These numbers are measurements, not correctness tests. Run the GPU test suite
before collecting them:

```sh
cargo test -p metal-infer-core --test gpu -- --ignored
```
