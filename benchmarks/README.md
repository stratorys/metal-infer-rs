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

For decode GEMV bandwidth, rotate distinct FP16 weight buffers in one command
buffer. Choose a copy count whose `working_set` in the benchmark name exceeds
the cache size; for a 1024×1024 matrix, 129 copies occupy 258 MiB:

```sh
target/release/metal-infer-bench kernel --m 1 --n 1024 --k 1024 \
  --rotate 129 --warmup 3 --iterations 10
```

The rotated report divides wall and GPU batch durations by the copy count and
reports effective GPU weight bandwidth in decimal GB/s. The ordinary kernel
benchmark retains its original single-matrix TFLOP/s metric.

The same rotation is available for fused decode projections. For Qwen3-0.6B,
QKV uses widths 2048, 1024, 1024 and gate/up uses 3072, 3072. The commands below
use more than 256 MiB of distinct weights and report both fused and unfused GPU
times per set of projections:

```sh
target/release/metal-infer-bench fusion --kind qkv --k 1024 \
  --query-heads 16 --kv-heads 8 --head-dim 128 \
  --rotate 33 --warmup 5 --iterations 30 --format json
target/release/metal-infer-bench fusion --kind gate-up --k 1024 \
  --intermediate 3072 --rotate 22 \
  --warmup 5 --iterations 30 --format json
```

Each iteration dispatches all copies in one command buffer. The `throughput`
field divides the total projection weight bytes by the fused GPU time;
`comparison` gives both paths.

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

The model benchmark runs the same plan as `metal-infer generate`: every fusion
and kernel variant is chosen by the autotune at load time. The JSON report
contains the plan that ran in its `plan` field. To test a variant, override one
decision with `--with KEY=VALUE` (repeatable). `--with list` prints every key
with its current value:

```sh
target/release/metal-infer-bench model --model ./models/Qwen3-0.6B --with list
target/release/metal-infer-bench model --model ./models/Qwen3-0.6B \
  --with fusion.qkv=off --with gemv.config=baseline --format json
```

The same option is available on `metal-infer generate`, `metal-infer serve`,
and `benchmarks/compare.py model`. The MLX-LM result of `compare.py` is
indicative: the two model benchmarks use different token sequences, despite
matching the checkpoint and prompt/decode lengths. The comparison pins
`mlx-lm==0.31.3` and `mlx==0.32.2` so successive reports use a stable baseline.

For small differences, `model_ab.py` repeats the complete model benchmark in
alternating A/B and B/A order. It builds the release binary once, runs both
variants with the same workload, and reports the median of paired decode
differences. Save the commands and raw benchmark JSON with `--output`:

```sh
uv run python benchmarks/model_ab.py --model ./models/Qwen3-0.6B \
  --candidate-args='--with fusion.qkv=off' --rounds 4 \
  --output /tmp/qwen3-model-ab.json
```

`--baseline-args` defaults to an empty string. Both variant arguments are
split as shell words without invoking a shell. Pass `--skip-build` when the
release binary has already been built.

To diagnose the GPU cost by kernel, run the model benchmark with
`--profile-kernels` after building the release binary:

```sh
target/release/metal-infer-bench model --model ./models/Qwen3-0.6B \
  --prompt 512 --generate 128 --iterations 1 --warmup 1 \
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

On M4 Pro, the tuned fused decode GEMV kernels run when `k` is divisible by 256.

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
for shorter decode contexts. Override the attention choice of a complete model
with `--with attention=...`.

## Results

See the [results index](results/README.md). Each entry contains its exact
values and graph. Lower latency is better; higher TFLOP/s and tokens/s are
better.

These numbers are measurements, not correctness tests. Run the GPU test suite
before collecting them:

```sh
cargo test -p metal-infer-kernels --test gpu -- --ignored
```
