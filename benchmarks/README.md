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

For dispatch tuning, build the release binary and run repeated, alternating
`reference-msl`, `native-msl`, and `auto` measurements. This reports median
GPU and wall latency across rounds and can save every raw sample:

```sh
cargo build --release --bin metal-infer-bench
uv run benchmarks/tune.py --rounds 3 --iterations 100 --warmup 10 \
  --output /tmp/metal-infer-tuning.json
```

Pass `--model-config /path/to/Qwen3-0.6B` (repeatable for other Qwen3 model
directories) to derive GEMM and GEMV shapes from each `config.json`.
`--prompt-lengths 32 128 512` selects prefill M values.

For decode GEMV bandwidth, rotate distinct FP16 weight buffers in one command
buffer. Choose a copy count whose `working_set` in the benchmark name exceeds
the cache size; for a 1024×1024 matrix, 129 copies occupy 258 MiB:

```sh
target/release/metal-infer-bench kernel --m 1 --n 1024 --k 1024 \
  --rotate 129 --warmup 3 --iterations 10
```

The rotated report divides wall and GPU batch durations by the copy count and
reports effective GPU weight bandwidth in decimal GB/s. `--rows 0|1|2|4|8`
selects a single-matrix GEMV row count on M4 Pro; `--split-k 2|4|8` selects a
two-pass split-K variant. These options require `--matmul-backend auto`. The
ordinary kernel benchmark retains its original single-matrix TFLOP/s metric.
`--rows 1 --half8` selects the 16-byte load variant of the one-row kernel.

The same rotation is available for fused decode projections. For Qwen3-0.6B,
QKV uses widths 2048, 1024, 1024 and gate/up uses 3072, 3072. The commands below
use more than 256 MiB of distinct weights and report both fused and unfused GPU
times per set of projections:

```sh
target/release/metal-infer-bench fusion --kind qkv --k 1024 \
  --query-heads 16 --kv-heads 8 --head-dim 128 \
  --rotate 33 --rows 2 --warmup 5 --iterations 30 --format json
target/release/metal-infer-bench fusion --kind gate-up --k 1024 \
  --intermediate 3072 --rotate 22 --rows 2 \
  --warmup 5 --iterations 30 --format json
```

`--rows 0|2|4|8` selects the fused projection variant. Each iteration dispatches
all copies in one command buffer. The `throughput` field divides the total
projection weight bytes by the fused GPU time; `comparison` gives both paths.

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

Use `--prompt` and `--generate` to override the model suite's default lengths.
Repeat with `--matmul-backend reference-msl` and `--matmul-backend auto` to
measure the full-model effect of the dispatch selection.

The model benchmark starts with every optional fusion disabled. Enable one or
more families explicitly with `--fuse-qkv`, `--fuse-gate-up`,
`--fuse-add-rms-norm`, and `--fuse-qk-rope-cache`. These flags are available on
both `metal-infer-bench model` and `benchmarks/compare.py model`; the inference
CLI is unchanged.

On Apple M4 Pro, Qwen3 selects the measured decode GEMV row counts by default
for the Qwen3-0.6B projection shapes. The model benchmark reports
`decode_gemv_config: "tuned"` in JSON. To compare the previous and current
selection with the same binary and workload, run the model benchmark twice,
changing only `--gemv-config`:

```sh
for config in baseline tuned; do
  target/release/metal-infer-bench model --model ./models/Qwen3-0.6B \
    --prompt 512 --generate 128 --warmup 1 --iterations 5 \
    --fuse-qkv --fuse-gate-up --fuse-add-rms-norm --fuse-qk-rope-cache \
    --gemv-config "$config" --format json
done
```

`benchmarks/compare.py model` uses the tuned default on M4 Pro. Its MLX-LM
result is indicative: the two existing model benchmarks use different token
sequences, despite matching the checkpoint and prompt/decode lengths.

To test whether reusing the normalized gate/up input within each threadgroup
improves the fused decode kernel, compare the model benchmark with and without
`--shared-gate-up-input`. Keep all other arguments the same, including
`--fuse-gate-up --fuse-add-rms-norm`. The existing kernel remains the default.
The JSON field `shared_gate_up_input` records which path ran. For example:

```sh
target/release/metal-infer-bench model --model ./models/Qwen3-0.6B \
  --prompt 512 --generate 128 --warmup 1 --iterations 5 \
  --fuse-qkv --fuse-gate-up --fuse-add-rms-norm --fuse-qk-rope-cache \
  --shared-gate-up-input --format json
```

The same flag is available on `benchmarks/compare.py model` when comparing the
experimental path with MLX-LM. The variant requires Qwen3-0.6B dimensions and
auto matmul on Apple M4 Pro. Use unprofiled runs for throughput comparisons.

For small differences, `model_ab.py` repeats the complete model benchmark in
alternating A/B and B/A order. It builds the release binary once, runs both
variants with the same four fusions and workload, and reports the median of
paired decode differences. Save the commands and raw benchmark JSON with
`--output`:

```sh
uv run python benchmarks/model_ab.py --model ./models/Qwen3-0.6B \
  --candidate-args=--shared-gate-up-input --rounds 4 \
  --output /tmp/qwen3-model-ab.json
```

`--baseline-args` defaults to an empty string. Both variant arguments are
split as shell words without invoking a shell. For example, compare two GEMV
configurations with `--baseline-args='--gemv-config baseline'` and
`--candidate-args='--gemv-config tuned'`. Pass `--skip-build` when the release
binary has already been built.

Choose the matrix implementation with `--matmul-backend auto`,
`--matmul-backend reference-msl`, `--matmul-backend native-msl`, or
`--matmul-backend mps`. The comparison pins
`mlx-lm==0.31.3` and `mlx==0.32.2` so successive reports use a stable baseline.

To diagnose the GPU cost by kernel, run the model benchmark with
`--profile-kernels` after building the release binary:

```sh
target/release/metal-infer-bench model --model ./models/Qwen3-0.6B \
  --prompt 512 --generate 128 --iterations 1 --warmup 1 \
  --fuse-qkv --fuse-gate-up --matmul-backend auto \
  --profile-kernels --format json
```

The optional `kernel_profile` field groups dispatch counts and GPU timestamp
durations by kernel, separately for prefill and decode. Warm-up runs are excluded.
The profiler creates a separate compute pass for each dispatch, so its latency and tokens/s must not be used
for performance comparisons. Compare throughput with a separate run that omits
`--profile-kernels`. `unattributed_ms` is the phase GPU time minus the sum of
timed Metal compute dispatches; it includes work outside those dispatches and
measurement gaps.

## Run the fusion microbenchmarks

Compare each fused kernel with its individual operations using Qwen3-0.6B
dimensions:

```sh
target/release/metal-infer-bench fusion --kind qkv
target/release/metal-infer-bench fusion --kind gate-up
target/release/metal-infer-bench fusion --kind add-rms-norm
target/release/metal-infer-bench fusion --kind qk-rope-cache
```

On M4 Pro, `--matmul-backend auto` selects the tuned fused decode GEMV kernels
when `k` is divisible by 256. Use `--matmul-backend native-msl` to select them
explicitly.

Projection benchmarks default to the vectorized `K=1024` path. Pass `--k 1023`
to QKV or gate/up to measure the scalar fallback. Reports contain separate GPU
and wall-clock distributions plus the fused/unfused speedup.

Measure decode attention independently at the target cache length with:

```sh
target/release/metal-infer-bench attention --tokens 1 --length 512
```

For prefill, run `attention --tokens 512 --length 512`; `--kind compare` also
reports flash-prefill for multiple query tokens. The model selects flash-prefill
from 32 query tokens onward, flash-decode for single-token attention from 256
active KV tokens onward when there are two query heads per KV head, and split-KV
for shorter decode contexts. QKV and QK+RoPE+cache are the default model
fusions; benchmark flags still select an explicit fusion set so unfused
baselines remain reproducible.

## Results

See the [results index](results/README.md). Each entry contains its exact
values and graph. Lower latency is better; higher TFLOP/s and tokens/s are
better.

These numbers are measurements, not correctness tests. Run the GPU test suite
before collecting them:

```sh
cargo test -p metal-infer-core --test gpu -- --ignored
```
