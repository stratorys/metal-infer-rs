# Benchmarks

One script, one protocol, the same metrics for every engine:

```sh
uv run benchmarks/bench.py --model Qwen/Qwen3-0.6B
```

It builds the release binaries, then runs each engine alone and writes
`results/<chip>/<date>.json` (every raw sample) and `<date>.md` (the tables).

## Protocol

- One engine at a time. Its server or benchmark process is started alone,
  measured, stopped, and followed by a cooldown (30 s by default). The script
  refuses to start while another engine process runs or the port is busy.
- Rounds alternate the engine order (A, B then B, A) so no engine always runs
  on a cooler or warmer machine. Default: 2 rounds.
- Same model, same precision, greedy decoding, fixed output length
  (`ignore_eos`); outputs that stop early are flagged in the table.
- Recorded with each run: chip, macOS, memory, power source, thermal state,
  metal-infer commit (and whether the tree is dirty), MLX versions.
- Run on AC power, with other applications closed.

## Metrics

Offline, with each engine's own tool (`metal-infer-bench`, `mlx_lm.benchmark`):

| Metric | Definition |
|---|---|
| ppN | prompt tokens per second for an N-token prefill (llama-bench format) |
| tgN | generated tokens per second for N tokens after the prefill |
| MBU | (weight bytes + KV bytes read per token) × tg / peak memory bandwidth |
| MFU | 2 × parameters × pp / peak FP16 FLOPS (needs `--peak-tflops`) |

Server, one OpenAI streaming client for every engine, at concurrency 1, 2, 4, 8:

| Metric | Definition |
|---|---|
| TTFT | request sent → first token received |
| TPOT | (E2EL − TTFT) / (output tokens − 1) |
| ITL | time between two consecutive streamed tokens |
| E2EL | request sent → last token received |
| Output tok/s | output tokens of all requests / wall time of the level |

Every latency is reported as p50, p90, p99, mean and standard deviation.

`--energy` adds joules per output token: `powermetrics` combined CPU + GPU +
ANE power during the measurement, minus the idle power measured first. It needs
sudo; run `sudo -v` just before.

## Options

| Option | Use |
|---|---|
| `--engines metal-infer mlx` | engines to compare |
| `--candidate KEY=VALUE` | add a metal-infer variant with a plan override, for A/B |
| `--mode offline\|server\|all` | which measurements to run |
| `--prompt 512 --generate 128` | workload |
| `--rounds`, `--iterations`, `--requests`, `--concurrency`, `--cooldown` | repetitions |
| `--peak-bandwidth`, `--peak-tflops` | chip peaks when the built-in table does not know the chip |

A/B example (same machine state, alternated):

```sh
uv run benchmarks/bench.py --model Qwen/Qwen3-0.6B --engines metal-infer \
  --candidate fusion.qkv=off --mode offline
```

## Kernel profile

```sh
target/release/metal-infer-bench --model Qwen/Qwen3-0.6B --profile
```

prints the GPU time of each kernel per iteration. Profiling uses separate
compute passes, so its timings are diagnostic only.
