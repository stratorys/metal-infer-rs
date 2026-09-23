#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Benchmark metal-infer against other engines, one engine at a time."""

import argparse
import concurrent.futures
import datetime
import http.client
import json
import os
import pathlib
import platform
import re
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
    parser.add_argument("--model", required=True, help="Hugging Face id or local directory")
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


def offline_metal(args: argparse.Namespace, overrides: list[str]) -> list[dict[str, float]]:
    command = [str(METAL_BENCH), "--model", args.model, "--prompt", str(args.prompt),
               "--generate", str(args.generate), "--iterations", str(args.iterations)]
    for override in overrides:
        command += ["--with", override]
    report = json.loads(run(command))
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
    output = run(MLX + ["mlx_lm.benchmark", "--model", args.mlx_model, "-p", str(args.prompt),
                        "-g", str(args.generate), "-n", str(args.iterations)])
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


def markdown(document: dict[str, Any]) -> str:
    args = document["arguments"]
    lines = [
        f"# {document['system']['chip']} — {args['model']}",
        "",
        f"{document['generated_at']} · commit `{document['system']['metal_infer_commit']}`"
        f"{' (dirty)' if document['system']['metal_infer_dirty'] else ''} · "
        f"prompt {args['prompt']}, generate {args['generate']}, {args['rounds']} rounds",
        "",
    ]
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


def main() -> None:
    args = arguments()
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
    directory = RESULTS / re.sub(r"[^a-z0-9]+", "-", machine["chip"].lower()).strip("-")
    directory.mkdir(parents=True, exist_ok=True)
    stem = generated_at.strftime("%Y%m%d-%H%M%S")
    (directory / f"{stem}.json").write_text(json.dumps(document, indent=2) + "\n")
    table = markdown(document)
    (directory / f"{stem}.md").write_text(table)
    print(table)
    print(f"results: {directory / stem}.json")


if __name__ == "__main__":
    try:
        main()
    except BenchmarkError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
