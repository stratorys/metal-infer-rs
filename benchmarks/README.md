# Benchmarks

One script, one protocol, the same client for every engine:

```sh
uv run --project benchmarks benchmarks/bench.py --model Qwen/Qwen3-0.6B --suite showcase --gguf ~/.cache/metal-infer/gguf/Qwen3-0.6B-F16.gguf
uv run --project benchmarks benchmarks/bench.py --model Qwen/Qwen3-0.6B
uv run --project benchmarks benchmarks/bench.py --model Qwen/Qwen3-0.6B --workload sharegpt --mode server
```

The script builds the release binaries, then runs each engine alone. It creates
`results/<chip>/<date>-<commit>/` as soon as it starts, with:

- `run.json`: every parameter and the full list of planned measurements;
- `progress.jsonl`: one line per finished measurement, appended as the run goes;
- `results.json`, `results.svg` and `README.md`: the aggregates, the chart, and
  the tables with every parameter and the exact commands. They are rewritten
  after every measurement, with an "in progress" banner until the run ends.

Each finished measurement prints one line, for example
`[12/60] round 2 · mlx · ctx 8192 · pp 3,950 tok/s · tg 98.4 tok/s · TTFT 2.07 s · left ~14 min`.

If a run stops (Ctrl-C, crash, error), `--resume results/<chip>/<run>` continues
it and skips the measurements already saved. A finished run rebuilds the
[results index](results/README.md) and `results/<chip>/latest-<model>.svg`.
Unfinished runs are never listed.

`--render results.json` writes the report of an existing results file, older
formats included, without measuring.

The code lives in `bench.py` (command line and run loop) and `benchlib/`
(one module per concern: engines, client, workloads, calibration, statistics,
progress, reports, chart).

## Latest results

![Latest Apple M4 Pro results for Qwen3-0.6B](results/apple-m4-pro/latest-qwen-qwen3-0-6b.svg)

Reproduce:

```sh
uv run --project benchmarks benchmarks/bench.py --model Qwen/Qwen3-0.6B --suite showcase --gguf ~/.cache/metal-infer/gguf/Qwen3-0.6B-F16.gguf
```

All runs: [results index](results/README.md).

## Showcase suite

`--suite showcase` is a three-engine long-context preset:

```
--workload context --prompt 512 2048 8192 32000 --generate 128
--engines metal-infer mlx llama.cpp --rounds 5 --requests 3 --mode server
```

Any option given explicitly overrides the suite, for example
`--suite showcase --prompt 512 8192 --rounds 1` for a quick check.

Every engine runs its own OpenAI-compatible server and is measured by the same
client, one request at a time. Prompts are calibrated to the same token count
per server, but their exact text may differ. For each engine and context length,
the report gives the median of the per-round medians of observed TTFT and TPOT
with a 95 % bootstrap interval (2,000 draws, fixed seed). The verdict against the first engine of `--engines` is
"faster" or "slower" only when the two intervals do not overlap, "tie (noise)"
otherwise.

## Workloads

Offline (`--mode offline`, diagnostic) runs each engine's own benchmark tool at
every `--prompt` length: `metal-infer-bench`, `mlx_lm.benchmark` and
`llama-bench` (prefill of N tokens, then decode after N tokens of context).
The tools do not time exactly the same things, so the published numbers come
from the server workloads.

Server (`--mode server`) starts each engine's OpenAI-compatible server and sends
streamed chat completions:

| Workload | Requests | Load |
|---|---|---|
| `context` | ShareGPT text cut to exactly each `--prompt` length, `--generate` output tokens | `--requests` requests one at a time, per context length |
| `synthetic` | fixed English text cut to exactly the first `--prompt` length, `--generate` output tokens | `--requests` per level at `--concurrency 1 2 4 8` |
| `sharegpt` | ShareGPT V3 conversations: first human turn as prompt, first assistant turn length as output length | `--num-prompts` requests per rate, Poisson arrivals at `--request-rate 1 2 4 inf` |

The ShareGPT file is downloaded once to `~/.cache/metal-infer/datasets/` and
never committed. The `context` text is every ShareGPT turn in file order, joined
by blank lines and cut to its first 400,000 characters.

The `sharegpt` sampling is the one of `vllm bench serve --dataset-name
sharegpt`: conversations with at least two turns, shuffled with `--seed`,
prompt 4–1024 tokens, output at least 4, prompt + output at most 2048. With the
same `--num-prompts` and `--seed`, `vllm bench serve` replays the same requests
against any OpenAI-compatible server. Each run README prints the exact command:

```sh
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --base-url http://127.0.0.1:8931 --model Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B \
  --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 100 --seed 0 --request-rate 2 --ignore-eos
```

### Exact prompt lengths

For the `context` and `synthetic` workloads, the prompt is calibrated against
each server: probe requests with one output token read `usage.prompt_tokens`,
and a search on the text length finds the prefix that gives exactly N prompt
tokens with that engine's own tokenizer and chat template. Any measured request
that reports another count stops the run. The report shows the calibrated and
observed counts for each engine.

### Same work for every engine

Every request is greedy, uses `ignore_eos` and sets `max_tokens` to the output
length. A unique, fixed-length prefix (`Request 000001.`) defeats prompt caches.
Engine-specific settings:

- `mlx_lm.server` has no `ignore_eos`: the model's end-of-sequence ids (from
  `generation_config.json`) get `logit_bias` −100. It runs with
  `--prompt-cache-size 1` so that long prompts do not fill memory.
- `llama-server` runs with one slot (`-np 1`), full GPU offload (`-ngl 99`),
  and requests set `cache_prompt: false`.
- metal-infer and llama.cpp get a context of the longest prompt + output + 64.

Current metal-infer limits, visible in the server results: requests are
served one at a time (no batching), and the KV cache is rebuilt for every
request (no prefix cache).

## llama.cpp and the F16 GGUF

```sh
brew install llama.cpp
llama-server --version
```

There is no official F16 GGUF of Qwen3 (`Qwen/Qwen3-0.6B-GGUF` has Q8_0 only).
Convert the same Hugging Face weights that metal-infer loads, with the converter
of the installed llama.cpp build (Homebrew does not ship it):

```sh
build=$(llama-server --version 2>&1 | sed -nE 's/^version: .*build ([0-9]+).*/\1/p; s/^version: ([0-9]+) \(.*/\1/p')
git clone --depth 1 --branch b$build https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
mkdir -p ~/.cache/metal-infer/gguf
uv run --no-project --with torch --with transformers --with sentencepiece --with /tmp/llama.cpp/gguf-py \
  python /tmp/llama.cpp/convert_hf_to_gguf.py \
  ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<revision> \
  --outtype f16 --outfile ~/.cache/metal-infer/gguf/Qwen3-0.6B-F16.gguf
```

Pass the file with `--gguf`; its path and sha256 are recorded in each run.
Lighter alternative: download `Qwen3-0.6B-BF16.gguf` from
`unsloth/Qwen3-0.6B-GGUF` at a fixed revision, then
`llama-quantize Qwen3-0.6B-BF16.gguf Qwen3-0.6B-F16.gguf F16`.

## Protocol

- One engine at a time. Its server or benchmark process is started alone,
  measured, stopped, and followed by a cooldown (30 s by default). The script
  refuses to start while another engine process runs or the port is busy.
  Server output goes to `server-<engine>.log` in the run directory.
- Rounds alternate the engine order (A, B, C then C, B, A) so no engine always
  runs on a cooler or warmer machine.
- Same model family, with the actual weight format recorded for each engine;
  greedy decoding and fixed output length;
  outputs that stop early are flagged and failed requests are counted.
- Recorded with each run: chip, macOS, memory, power source, thermal state,
  metal-infer commit (and whether the tree is dirty), MLX and llama.cpp
  versions, GGUF sha256.
- Run on AC power, with other applications closed.

## Metrics

The server client uses observed streamed output and server-reported token usage.
Its values must not be treated as bit-for-bit identical to a separate run of
`vllm bench serve`; offline measurements use each engine's own benchmark tool.

Offline:

| Metric | Definition |
|---|---|
| ppN | prompt tokens per second for an N-token prefill (llama-bench format) |
| tgN | generated tokens per second for N tokens after the prefill |

Server:

| Metric | Definition |
|---|---|
| TTFT | request sent → first output-bearing chunk received (content or reasoning) |
| TPOT | (E2EL − TTFT) / (output tokens − 1) |
| ITL | time between two consecutive output-bearing chunks |
| E2EL | request sent → last output-bearing chunk received |
| req/s, output tok/s | completed requests or output tokens / wall time of the level |
| Goodput | completed requests per second with TTFT ≤ `--slo-ttft-ms` (1000) and TPOT ≤ `--slo-tpot-ms` (50) |

Every latency is reported as p50, p90, p99, mean and standard deviation, and
the main metrics also as a median with a 95 % bootstrap interval over rounds.

`--energy` adds joules per output token: `powermetrics` combined CPU + GPU +
ANE power during the measurement, minus the idle power measured first. It needs
sudo; run `sudo -v` just before.

## Options

| Option | Use |
|---|---|
| `--suite showcase` | three-engine long-context preset; explicit options win |
| `--engines metal-infer mlx llama.cpp` | engines to compare; the first one is the verdict reference |
| `--gguf PATH_OR_REPO` | local BF16/F16 file or cached `unsloth/Qwen3-0.6B-GGUF` for llama.cpp |
| `--candidate KEY=VALUE` | add a metal-infer variant with a plan override, for A/B |
| `--mode offline\|server\|all` | which measurements to run |
| `--workload context\|synthetic\|sharegpt` | server requests |
| `--prompt 512 2048 --generate 128` | prompt or context lengths and generated tokens |
| `--num-prompts`, `--request-rate`, `--seed` | ShareGPT sample and arrivals |
| `--slo-ttft-ms`, `--slo-tpot-ms` | goodput limits |
| `--rounds`, `--iterations`, `--requests`, `--concurrency`, `--cooldown` | repetitions |
| `--peak-bandwidth`, `--peak-tflops` | legacy inputs; MBU/MFU are no longer reported |
| `--resume RUN_DIRECTORY` | continue an interrupted run |
| `--render results.json` | rewrite the report of an existing run, without measuring |

A/B example (same machine state, alternated):

```sh
uv run --project benchmarks benchmarks/bench.py --model Qwen/Qwen3-0.6B --engines metal-infer \
  --candidate fusion.qkv=off --mode offline
```

## Kernel profile

```sh
target/release/metal-infer-bench --model Qwen/Qwen3-0.6B --profile
```

prints the GPU time of each kernel per iteration. Profiling uses separate
compute passes, so its timings are diagnostic only.
