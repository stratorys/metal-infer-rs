#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Benchmark metal-infer against other engines, one engine at a time."""

import argparse
import concurrent.futures
import datetime
import html
import http.client
import json
import os
import pathlib
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
METAL_BENCH = ROOT / "target" / "release" / "metal-infer-bench"
METAL_SERVER = ROOT / "target" / "release" / "metal-infer"
MLX_LM_VERSION = "0.31.3"
MLX_VERSION = "0.32.2"
MLX = ["uvx", "--from", f"mlx-lm=={MLX_LM_VERSION}", "--with", f"mlx=={MLX_VERSION}"]
ENGINE_PROCESSES = ("metal-infer", "mlx_lm", "llama-server", "llama-bench", "vllm")
PEAK_BANDWIDTH_GBS = {"Apple M4 Pro": 273.0, "Apple M4": 120.0}
PERCENTILES = (50, 90, 99)
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
FONT = "-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif"
PROMPT_TEXT = (
    "You are reviewing the design of a small inference engine for Apple Silicon. "
    "The engine loads FP16 weights, runs a prefill pass over the prompt, then "
    "generates tokens one at a time while keeping a key and value cache in GPU "
    "memory. Explain in detail how each part works, which parts are limited by "
    "memory bandwidth and which by compute, and how you would measure them. "
)


class BenchmarkError(RuntimeError):
    pass


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Hugging Face id or local directory")
    parser.add_argument("--render", type=pathlib.Path, metavar="RESULTS_JSON",
                        help="write the report of an existing results file without measuring")
    parser.add_argument("--mlx-model", help="model id for MLX (default: --model)")
    parser.add_argument("--engines", nargs="+", default=["metal-infer", "mlx"],
                        choices=["metal-infer", "mlx"])
    parser.add_argument("--candidate", action="append", default=[], metavar="KEY=VALUE",
                        help="add a metal-infer variant with this plan override (repeatable)")
    parser.add_argument("--mode", choices=["offline", "server", "all"], default="all")
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--generate", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--requests", type=int, default=8, help="requests per concurrency level")
    parser.add_argument("--cooldown", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8931)
    parser.add_argument("--peak-bandwidth", type=float, help="GB/s, overrides the chip table")
    parser.add_argument("--peak-tflops", type=float, help="FP16 TFLOPS, enables MFU")
    parser.add_argument("--energy", action="store_true",
                        help="measure joules per token with sudo powermetrics (run `sudo -v` first)")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def run(command: list[str], **kwargs: Any) -> str:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False,
                               **kwargs)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BenchmarkError(f"{' '.join(command)} failed: {detail}")
    return completed.stdout


def optional(command: list[str]) -> str | None:
    try:
        return run(command).strip()
    except (BenchmarkError, OSError):
        return None


def model_directory(model: str) -> pathlib.Path:
    path = pathlib.Path(model).expanduser()
    if path.is_dir():
        return path
    cache = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache/huggingface"))
    repository = cache / "hub" / f"models--{model.replace('/', '--')}"
    reference = repository / "refs" / "main"
    if reference.is_file():
        snapshot = repository / "snapshots" / reference.read_text().strip()
        if snapshot.is_dir():
            return snapshot
    raise BenchmarkError(f"model {model} is not a directory nor in the Hugging Face cache")


def model_facts(directory: pathlib.Path) -> dict[str, Any]:
    config = json.loads((directory / "config.json").read_text())
    hidden = config["hidden_size"]
    head_dim = config.get("head_dim", hidden // config["num_attention_heads"])
    query = config["num_attention_heads"] * head_dim
    kv = config["num_key_value_heads"] * head_dim
    layer = (hidden * (query + 2 * kv) + query * hidden
             + 3 * hidden * config["intermediate_size"])
    parameters = config["num_hidden_layers"] * layer + config["vocab_size"] * hidden
    return {
        "parameters_read_per_token": parameters,
        "weight_bytes_read_per_token": parameters * 2,
        "kv_bytes_per_token": 2 * config["num_hidden_layers"] * kv * 2,
    }


def system() -> dict[str, Any]:
    chip = optional(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.machine()
    return {
        "chip": chip,
        "macos": platform.mac_ver()[0],
        "memory_bytes": int(optional(["sysctl", "-n", "hw.memsize"]) or 0),
        "power_source": (optional(["pmset", "-g", "batt"]) or "").split("\n")[0],
        "thermal": optional(["pmset", "-g", "therm"]),
        "metal_infer_commit": optional(["git", "rev-parse", "HEAD"]),
        "metal_infer_dirty": bool(optional(["git", "status", "--porcelain"])),
        "mlx_lm": MLX_LM_VERSION,
        "mlx": MLX_VERSION,
    }


def require_idle(port: int) -> None:
    listing = optional(["ps", "-axo", "pid=,command="]) or ""
    own = os.getpid()
    busy = [
        line.strip() for line in listing.splitlines()
        if any(name in line for name in ENGINE_PROCESSES)
        and "bench.py" not in line and str(own) != line.split()[0]
    ]
    if busy:
        raise BenchmarkError("another engine is running:\n" + "\n".join(busy))
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        connection.request("GET", "/v1/models")
        raise BenchmarkError(f"port {port} is already in use")
    except OSError:
        pass


def cooldown(seconds: float) -> None:
    if seconds > 0:
        print(f"  cooldown {seconds:.0f} s", flush=True)
        time.sleep(seconds)


def summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    result = {
        "mean": statistics.fmean(ordered),
        "std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }
    for percentile in PERCENTILES:
        position = (len(ordered) - 1) * percentile / 100
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        result[f"p{percentile}"] = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return result


class Energy:
    """Samples package power with powermetrics while a measurement runs."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.idle_watts: float | None = None
        if enabled:
            self.idle_watts = self._average_watts(lambda: time.sleep(5))[0]

    def measure(self, action: Any) -> tuple[Any, float | None]:
        if not self.enabled:
            return action(), None
        watts, result, seconds = self._average_watts(action)
        net = max(watts - (self.idle_watts or 0.0), 0.0)
        return result, net * seconds

    def _average_watts(self, action: Any) -> Any:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as output:
            path = output.name
        process = subprocess.Popen(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power",
             "-i", "200", "-o", path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        started = time.perf_counter()
        result = action()
        seconds = time.perf_counter() - started
        subprocess.run(["sudo", "-n", "kill", "-INT", str(process.pid)], check=False)
        process.wait(timeout=10)
        text = pathlib.Path(path).read_text(errors="ignore")
        samples = [float(value) for value in re.findall(r"Combined Power \(CPU \+ GPU \+ ANE\): (\d+) mW", text)]
        if not samples:
            raise BenchmarkError("powermetrics produced no power samples; run `sudo -v` first")
        return statistics.fmean(samples) / 1000, result, seconds


def metal_bench_command(args: argparse.Namespace, overrides: list[str]) -> list[str]:
    command = [str(METAL_BENCH), "--model", args.model, "--prompt", str(args.prompt),
               "--generate", str(args.generate), "--iterations", str(args.iterations)]
    for override in overrides:
        command += ["--with", override]
    return command


def mlx_bench_command(args: argparse.Namespace) -> list[str]:
    return MLX + ["mlx_lm.benchmark", "--model", args.mlx_model, "-p", str(args.prompt),
                  "-g", str(args.generate), "-n", str(args.iterations)]


def offline_metal(args: argparse.Namespace, overrides: list[str]) -> list[dict[str, float]]:
    report = json.loads(run(metal_bench_command(args, overrides)))
    return [
        {
            "pp_tps": args.prompt / (sample["prefill_ms"] / 1000),
            "tg_tps": args.generate / (sample["decode_ms"] / 1000),
            "peak_memory_gb": report["allocated_bytes"] / 1e9,
            "plan": report["plan"],
        }
        for sample in report["samples"]
    ]


def offline_mlx(args: argparse.Namespace) -> list[dict[str, float]]:
    output = run(mlx_bench_command(args))
    trials = re.findall(r"Trial \d+:\s*prompt_tps=([0-9.]+), generation_tps=([0-9.]+), "
                        r"peak_memory=([0-9.]+)", output)
    if not trials:
        trials = re.findall(r"Averages: prompt_tps=([0-9.]+), generation_tps=([0-9.]+), "
                            r"peak_memory=([0-9.]+)", output)
    if not trials:
        raise BenchmarkError("mlx_lm.benchmark output has no trial results:\n" + output)
    return [{"pp_tps": float(pp), "tg_tps": float(tg), "peak_memory_gb": float(memory)}
            for pp, tg, memory in trials]


def server_command(engine: str, args: argparse.Namespace, overrides: list[str]) -> list[str]:
    if engine == "mlx":
        return MLX + ["mlx_lm.server", "--model", args.mlx_model, "--host", "127.0.0.1",
                      "--port", str(args.port)]
    command = [str(METAL_SERVER), "serve", "--model", args.model,
               "--bind", f"127.0.0.1:{args.port}"]
    for override in overrides:
        command += ["--with", override]
    return command


def wait_ready(port: int, process: subprocess.Popen, timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BenchmarkError("server exited during startup")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/v1/models")
            if connection.getresponse().status == 200:
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise BenchmarkError("server did not become ready")


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def request(port: int, prompt: str, generate: int) -> dict[str, Any]:
    unique = f"Request {time.time_ns()}. {prompt}"
    body = json.dumps({
        "messages": [{"role": "user", "content": unique}],
        "max_tokens": generate,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    started = time.perf_counter()
    connection.request("POST", "/v1/chat/completions", body,
                       {"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        raise BenchmarkError(f"server answered {response.status}: {response.read()[:500]!r}")
    arrivals: list[float] = []
    usage: dict[str, Any] = {}
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                arrivals.append(time.perf_counter())
    finished = time.perf_counter()
    if not arrivals:
        raise BenchmarkError("stream contained no tokens")
    output_tokens = int(usage.get("completion_tokens") or len(arrivals))
    ttft = arrivals[0] - started
    e2el = finished - started
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": output_tokens,
        "ttft_ms": ttft * 1000,
        "e2el_ms": e2el * 1000,
        "tpot_ms": (e2el - ttft) / max(output_tokens - 1, 1) * 1000,
        "itl_ms": [(later - earlier) * 1000 for earlier, later in zip(arrivals, arrivals[1:])],
    }


def server_level(port: int, prompt: str, args: argparse.Namespace, concurrency: int,
                 energy: Energy) -> dict[str, Any]:
    def batch() -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
            results = list(pool.map(lambda _: request(port, prompt, args.generate),
                                    range(args.requests)))
        return results, time.perf_counter() - started

    (results, seconds), joules = energy.measure(batch)
    output_tokens = sum(result["output_tokens"] for result in results)
    short = [result["output_tokens"] for result in results if result["output_tokens"] != args.generate]
    return {
        "concurrency": concurrency,
        "requests": results,
        "prompt_tokens": sorted({result["prompt_tokens"] for result in results}, key=str),
        "short_outputs": short,
        "output_tps": output_tokens / seconds,
        "joules_per_token": None if joules is None else joules / output_tokens,
    }


def offline_run(engine: str, overrides: list[str], args: argparse.Namespace,
                energy: Energy) -> dict[str, Any]:
    action = (lambda: offline_mlx(args)) if engine == "mlx" else (lambda: offline_metal(args, overrides))
    samples, joules = energy.measure(action)
    generated = args.generate * args.iterations
    return {"samples": samples,
            "joules_per_token": None if joules is None else joules / generated}


def server_run(engine: str, overrides: list[str], args: argparse.Namespace,
               energy: Energy) -> list[dict[str, Any]]:
    prompt = PROMPT_TEXT * max(1, args.prompt // 95)
    process = subprocess.Popen(server_command(engine, args, overrides), cwd=ROOT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True)
    try:
        wait_ready(args.port, process)
        request(args.port, prompt, args.generate)
        return [server_level(args.port, prompt, args, level, energy) for level in args.concurrency]
    finally:
        stop(process)


def aggregate(entry: dict[str, Any], facts: dict[str, Any], args: argparse.Namespace,
              bandwidth: float | None) -> dict[str, Any]:
    result: dict[str, Any] = {"engine": entry["label"]}
    offline = [sample for run in entry["offline"] for sample in run["samples"]]
    if offline:
        pp = summary([sample["pp_tps"] for sample in offline])
        tg = summary([sample["tg_tps"] for sample in offline])
        bytes_per_token = (facts["weight_bytes_read_per_token"]
                           + facts["kv_bytes_per_token"] * (args.prompt + args.generate / 2))
        joules = [run["joules_per_token"] for run in entry["offline"] if run["joules_per_token"]]
        result["offline"] = {
            f"pp{args.prompt}_tps": pp,
            f"tg{args.generate}_tps": tg,
            "mbu": bytes_per_token * tg["p50"] / (bandwidth * 1e9) if bandwidth else None,
            "mfu": (2 * facts["parameters_read_per_token"] * pp["p50"] / (args.peak_tflops * 1e12)
                    if args.peak_tflops else None),
            "peak_memory_gb": max(sample["peak_memory_gb"] for sample in offline),
            "joules_per_token": statistics.fmean(joules) if joules else None,
            "plan": next((sample["plan"] for sample in offline if "plan" in sample), None),
        }
    levels: dict[int, list[dict[str, Any]]] = {}
    for run in entry["server"]:
        for level in run:
            levels.setdefault(level["concurrency"], []).append(level)
    result["server"] = []
    for concurrency, runs in sorted(levels.items()):
        requests = [request for run in runs for request in run["requests"]]
        joules = [run["joules_per_token"] for run in runs if run["joules_per_token"]]
        result["server"].append({
            "concurrency": concurrency,
            "ttft_ms": summary([request["ttft_ms"] for request in requests]),
            "tpot_ms": summary([request["tpot_ms"] for request in requests]),
            "itl_ms": summary([value for request in requests for value in request["itl_ms"]]),
            "e2el_ms": summary([request["e2el_ms"] for request in requests]),
            "output_tps": summary([run["output_tps"] for run in runs]),
            "joules_per_token": statistics.fmean(joules) if joules else None,
            "prompt_tokens": sorted({value for run in runs for value in run["prompt_tokens"]}, key=str),
            "short_outputs": [value for run in runs for value in run["short_outputs"]],
        })
    return result


def number(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f} %"


def markdown(document: dict[str, Any], image: bool = False) -> str:
    args = document["arguments"]
    system = document["system"]
    power = (system.get("power_source") or "").removeprefix("Now drawing from ").strip("'")
    lines = [
        f"# {system['chip']} — {args['model']}",
        "",
        f"{document['generated_at']} · commit `{system['metal_infer_commit']}`"
        f"{' (dirty)' if system['metal_infer_dirty'] else ''} · "
        f"prompt {args['prompt']}, generate {args['generate']}, {args['rounds']} rounds",
        "",
        f"macOS {system['macos']} · {system['memory_bytes'] / 2**30:.0f} GB · {power or 'unknown power'} · "
        f"mlx-lm {system['mlx_lm']}, mlx {system['mlx']}",
        "",
    ]
    if image:
        lines += ["![Benchmark results](results.svg)", ""]
    offline = [result for result in document["results"] if "offline" in result]
    if offline:
        lines += [
            "## Offline",
            "",
            f"| engine | pp{args['prompt']} tok/s p50 (p90/p99) | tg{args['generate']} tok/s p50 (p90/p99) "
            "| mean ± std tg | MBU | MFU | peak GB | J/token |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for result in offline:
            values = result["offline"]
            pp = values[f"pp{args['prompt']}_tps"]
            tg = values[f"tg{args['generate']}_tps"]
            lines.append(
                f"| {result['engine']} | {number(pp['p50'])} ({number(pp['p90'])}/{number(pp['p99'])}) "
                f"| {number(tg['p50'])} ({number(tg['p90'])}/{number(tg['p99'])}) "
                f"| {number(tg['mean'])} ± {number(tg['std'])} "
                f"| {percent(values['mbu'])} | {percent(values['mfu'])} "
                f"| {number(values['peak_memory_gb'], 2)} | {number(values['joules_per_token'], 3)} |")
        lines.append("")
    server = [result for result in document["results"] if result.get("server")]
    if server:
        lines += ["## Server", "",
                  "p50 / p90 / p99 (mean ± std), milliseconds.", ""]
        concurrencies = sorted({level["concurrency"] for result in server for level in result["server"]})
        for concurrency in concurrencies:
            lines += [
                f"### Concurrency {concurrency}",
                "",
                "| engine | TTFT | TPOT | ITL | E2EL | output tok/s | J/token |",
                "|---|---|---|---|---|---|---|",
            ]
            for result in server:
                level = next((item for item in result["server"] if item["concurrency"] == concurrency), None)
                if level is None:
                    continue
                cells = []
                for key in ("ttft_ms", "tpot_ms", "itl_ms", "e2el_ms"):
                    stats = level[key]
                    cells.append(f"{number(stats['p50'])} / {number(stats['p90'])} / {number(stats['p99'])} "
                                 f"({number(stats['mean'])} ± {number(stats['std'])})")
                lines.append(f"| {result['engine']} | {' | '.join(cells)} "
                             f"| {number(level['output_tps']['p50'])} | {number(level['joules_per_token'], 3)} |")
                if level["short_outputs"]:
                    lines.append(f"| ⚠ {result['engine']} stopped early: {level['short_outputs']} tokens | | | | | | |")
            lines.append("")
    return "\n".join(lines) + "\n"


def setup(document: dict[str, Any]) -> str:
    arguments = document["arguments"]
    args = argparse.Namespace(**arguments)
    system = document["system"]
    facts = document["model"]
    modes = ["offline", "server"] if args.mode == "all" else [args.mode]
    rows = [
        ("model", f"`{args.model}`"),
        ("MLX model", f"`{args.mlx_model}`"),
        ("engines", ", ".join(args.engines)),
        ("candidates", ", ".join(f"`{candidate}`" for candidate in args.candidate) or "none"),
        ("modes", ", ".join(modes)),
        ("prompt / generated tokens", f"{args.prompt} / {args.generate}"),
        ("iterations per offline run", str(args.iterations)),
        ("rounds (engine order reversed every other round)", str(args.rounds)),
        ("cooldown after each run", f"{args.cooldown:.0f} s"),
        ("peak bandwidth for MBU", f"{number(document['peak_bandwidth_gbs'])} GB/s"),
        ("peak FP16 TFLOPS for MFU", number(args.peak_tflops)),
        ("energy", "powermetrics, idle power subtracted" if args.energy else "not measured"),
        ("parameters read per token", f"{facts['parameters_read_per_token']:,}"),
        ("weight bytes read per token", f"{facts['weight_bytes_read_per_token'] / 1e9:.3f} GB"),
        ("KV bytes per context token", f"{facts['kv_bytes_per_token']:,}"),
        ("chip / memory", f"{system['chip']} / {system['memory_bytes'] / 2**30:.0f} GB"),
        ("macOS", system["macos"]),
        ("power source", system.get("power_source") or "unknown"),
        ("thermal state", "; ".join((system.get("thermal") or "unknown").split("\n"))),
        ("metal-infer commit", f"`{system['metal_infer_commit']}`"
                               f"{' (dirty working tree)' if system['metal_infer_dirty'] else ''}"),
        ("MLX versions", f"mlx-lm {system['mlx_lm']}, mlx {system['mlx']}"),
    ]
    if "server" in modes:
        rows += [
            ("server concurrency levels", ", ".join(str(level) for level in args.concurrency)),
            ("requests per level", str(args.requests)),
            ("server prompt", f"fixed English text repeated {max(1, args.prompt // 95)} times, "
                              "unique prefix per request, greedy, `ignore_eos`"),
        ]
    lines = ["## Setup", "", "| parameter | value |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    entries = [(engine, []) for engine in args.engines if engine == "metal-infer"]
    entries += [(f"metal-infer {candidate}", [candidate]) for candidate in args.candidate]
    commands = []
    if "offline" in modes:
        commands += [" ".join(metal_bench_command(args, overrides)) for _, overrides in entries]
        if "mlx" in args.engines:
            commands.append(" ".join(mlx_bench_command(args)))
    if "server" in modes:
        commands += [" ".join(server_command("metal-infer", args, overrides)) for _, overrides in entries]
        if "mlx" in args.engines:
            commands.append(" ".join(server_command("mlx", args, [])))
    lines += [
        "",
        "Offline: pp = prompt tokens / prefill time and tg = generated tokens / decode time, "
        "one sample per iteration; MBU = (weight bytes + KV bytes × (prompt + generated / 2)) "
        "× tg p50 / peak bandwidth.",
        "",
        "### Commands",
        "",
        "```sh",
        *commands,
        "```",
    ]
    for result in document["results"]:
        plan = (result.get("offline") or {}).get("plan")
        if plan:
            lines += ["", f"### Plan — {result['engine']}", "", "| key | value |", "|---|---|"]
            lines += [f"| `{key}` | `{value}` |" for key, value in plan.items()]
    return "\n".join(lines) + "\n"


def run_readme(document: dict[str, Any]) -> str:
    return (markdown(document, image=True) + setup(document)
            + "\nRaw measurements: [results.json](results.json). Protocol and metrics: "
            "[benchmark guide](../../../README.md).\n")


def svg_text(x: float, y: float, value: str, anchor: str = "middle", size: int = 13,
             weight: str = "normal", fill: str = "#24292f") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{FONT}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(value)}</text>')


def chart_panels(document: dict[str, Any]) -> list[tuple[str, list[tuple[str, dict[str, float]]]]]:
    args = document["arguments"]
    panels = []
    offline = [result for result in document["results"] if "offline" in result]
    if offline:
        for key, name in ((f"pp{args['prompt']}_tps", "Prefill"), (f"tg{args['generate']}_tps", "Decode")):
            panels.append((f"{name} {key.removesuffix('_tps')} (tok/s)",
                           [(result["engine"], result["offline"][key]) for result in offline]))
    server = [result for result in document["results"] if result.get("server")]
    if server:
        concurrency = min(level["concurrency"] for result in server for level in result["server"])
        levels = [(result["engine"], next((level for level in result["server"]
                                           if level["concurrency"] == concurrency), None))
                  for result in server]
        for key, name in (("ttft_ms", "TTFT"), ("tpot_ms", "TPOT")):
            panels.append((f"{name} at concurrency {concurrency} (ms)",
                           [(engine, level[key]) for engine, level in levels if level]))
    return panels


def svg(document: dict[str, Any]) -> str:
    panels = chart_panels(document)
    if not panels:
        raise BenchmarkError("the results contain nothing to plot")
    args = document["arguments"]
    width, height, top, chart_height, gap = 1000, 440, 110, 250, 70
    baseline = top + chart_height
    panel_width = (width - 90 - gap * (len(panels) - 1)) / len(panels)
    body = [
        svg_text(width / 2, 36, f"{args['model']} on {document['system']['chip']}", size=22, weight="600"),
        svg_text(width / 2, 60, f"prompt {args['prompt']}, generate {args['generate']} · "
                 "bar = p50, whisker = mean ± std", size=12, fill="#57606a"),
    ]
    for panel_index, (title, entries) in enumerate(panels):
        left = 70 + panel_index * (panel_width + gap)
        maximum = max(max(stats["p50"], stats["mean"] + stats["std"]) for _, stats in entries) * 1.12 or 1.0
        digits = 0 if maximum >= 100 else 1
        body.append(svg_text(left + panel_width / 2, top - 24, title, size=15, weight="600"))
        for tick in range(6):
            y = baseline - chart_height * tick / 5
            body.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + panel_width:.1f}" '
                        f'y2="{y:.1f}" stroke="#d8dee4"/>')
            body.append(svg_text(left - 6, y + 4, number(maximum * tick / 5, digits), anchor="end",
                                 size=11, fill="#57606a"))
        slot = panel_width / len(entries)
        bar_width = min(80.0, slot * 0.6)
        for index, (engine, stats) in enumerate(entries):
            center = left + slot * (index + 0.5)
            y = baseline - chart_height * stats["p50"] / maximum
            low = baseline - chart_height * max(stats["mean"] - stats["std"], 0.0) / maximum
            high = baseline - chart_height * (stats["mean"] + stats["std"]) / maximum
            body.append(f'<rect x="{center - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                        f'height="{baseline - y:.1f}" fill="{COLORS[index % len(COLORS)]}" rx="2"/>')
            body.append(f'<line x1="{center:.1f}" y1="{low:.1f}" x2="{center:.1f}" y2="{high:.1f}" '
                        'stroke="#24292f" stroke-width="1.5"/>')
            body.append(svg_text(center, min(y, high) - 8, number(stats["p50"]), size=12, weight="600"))
            body.append(svg_text(center, baseline + 20, engine, size=11))
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(args['model'])} benchmark</title>",
        "<desc>Throughput and latency per engine, p50 with mean ± standard deviation.</desc>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        *body,
        "</svg>",
        "",
    ])


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def run_directory(document: dict[str, Any]) -> pathlib.Path:
    system = document["system"]
    generated = datetime.datetime.fromisoformat(document["generated_at"])
    commit = (system.get("metal_infer_commit") or "uncommitted")[:7]
    dirty = "-dirty" if system.get("metal_infer_dirty") else ""
    return RESULTS / slug(system["chip"]) / f"{generated:%Y%m%d-%H%M%S}-{commit}{dirty}"


def rebuild_index() -> None:
    runs = []
    for path in RESULTS.glob("*/*/results.json"):
        try:
            runs.append((path.parent, json.loads(path.read_text())))
        except (OSError, json.JSONDecodeError):
            continue
    runs.sort(key=lambda run: run[1]["generated_at"], reverse=True)
    lines = [
        "# Benchmark results",
        "",
        "Written by `benchmarks/bench.py`, newest first.",
        "",
        "| generated | chip | model | workload | decode tok/s p50 | report |",
        "|---|---|---|---|---|---|",
    ]
    latest: dict[pathlib.Path, pathlib.Path] = {}
    for directory, document in runs:
        args = document["arguments"]
        key = f"tg{args['generate']}_tps"
        decode = " · ".join(f"{result['engine']} {number(result['offline'][key]['p50'])}"
                            for result in document["results"] if "offline" in result)
        lines.append(f"| `{document['generated_at'][:19]}` | {document['system']['chip']} | {args['model']} "
                     f"| pp{args['prompt']} / tg{args['generate']} | {decode or '—'} "
                     f"| [open]({directory.relative_to(RESULTS).as_posix()}/README.md) |")
        latest.setdefault(directory.parent, directory)
    (RESULTS / "README.md").write_text("\n".join(lines) + "\n")
    for chip_directory, directory in latest.items():
        shutil.copyfile(directory / "results.svg", chip_directory / "latest.svg")


def write_report(document: dict[str, Any]) -> pathlib.Path:
    directory = run_directory(document)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(document, indent=2) + "\n")
    (directory / "results.svg").write_text(svg(document))
    (directory / "README.md").write_text(run_readme(document))
    rebuild_index()
    return directory


def main() -> None:
    args = arguments()
    if args.render:
        directory = write_report(json.loads(args.render.read_text()))
        print(f"report: {directory / 'README.md'}")
        return
    if not args.model:
        raise BenchmarkError("--model is required unless --render is given")
    args.mlx_model = args.mlx_model or args.model
    facts = model_facts(model_directory(args.model))
    if not args.skip_build:
        print("building release binaries", flush=True)
        run(["cargo", "build", "--release", "--bin", "metal-infer", "--bin", "metal-infer-bench"])
    machine = system()
    bandwidth = args.peak_bandwidth or PEAK_BANDWIDTH_GBS.get(machine["chip"])
    entries = [{"label": engine, "engine": engine, "overrides": [], "offline": [], "server": []}
               for engine in args.engines]
    entries += [{"label": f"metal-infer {candidate}", "engine": "metal-infer",
                 "overrides": [candidate], "offline": [], "server": []}
                for candidate in args.candidate]
    energy = Energy(args.energy)
    modes = ["offline", "server"] if args.mode == "all" else [args.mode]
    for round_index in range(args.rounds):
        order = entries if round_index % 2 == 0 else list(reversed(entries))
        for entry in order:
            for mode in modes:
                require_idle(args.port)
                print(f"round {round_index + 1}/{args.rounds} · {entry['label']} · {mode}", flush=True)
                if mode == "offline":
                    entry["offline"].append(offline_run(entry["engine"], entry["overrides"], args, energy))
                else:
                    entry["server"].append(server_run(entry["engine"], entry["overrides"], args, energy))
                cooldown(args.cooldown)
    generated_at = datetime.datetime.now(datetime.timezone.utc)
    document = {
        "schema_version": 3,
        "generated_at": generated_at.isoformat(),
        "system": machine,
        "arguments": {key: value for key, value in vars(args).items()},
        "model": facts,
        "peak_bandwidth_gbs": bandwidth,
        "results": [aggregate(entry, facts, args, bandwidth) for entry in entries],
        "raw": entries,
    }
    directory = write_report(document)
    print(markdown(document))
    print(f"report: {directory / 'README.md'}")


if __name__ == "__main__":
    try:
        main()
    except BenchmarkError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
