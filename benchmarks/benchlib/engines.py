"""How each engine is started, queried and benchmarked offline."""

import argparse
import contextlib
import http.client
import json
import os
import pathlib
import re
import signal
import subprocess
import time
from typing import Any, Iterator

from benchlib.client import Target, stream
from benchlib.common import METAL_BENCH, METAL_SERVER, MLX, ROOT, BenchmarkError, eos_ids, run
from benchlib.plan import Engine
from benchlib.workloads import PROMPT_TEXT, SHAREGPT_FILTERS

CONTEXT_MARGIN = 64


def context_size(args: argparse.Namespace) -> int:
    """Server context: the longest request plus its output, with a small margin."""
    longest = max(args.prompt)
    if args.workload == "sharegpt":
        longest = max(longest, SHAREGPT_FILTERS["max_total_tokens"])
    return longest + args.generate + CONTEXT_MARGIN


def server_command(engine: Engine, args: argparse.Namespace) -> list[str]:
    port = str(args.port)
    if engine.kind == "mlx":
        return MLX + [
            "mlx_lm.server",
            "--model",
            args.mlx_model,
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--prompt-cache-size",
            "1",
        ]
    if engine.kind == "llama.cpp":
        return [
            "llama-server",
            "-m",
            str(args.gguf),
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "-c",
            str(context_size(args)),
            "-ngl",
            "99",
            "-np",
            "1",
        ]
    command = [
        str(METAL_SERVER),
        "serve",
        "--model",
        args.model,
        "--bind",
        f"127.0.0.1:{port}",
        "--context",
        str(context_size(args)),
    ]
    for override in engine.overrides:
        command += ["--with", override]
    return command


def extra_body(engine: Engine, directory: pathlib.Path) -> dict[str, Any]:
    """Request fields one engine needs so that every engine does the same work.

    mlx_lm.server has no ignore_eos, so its end-of-sequence tokens are banned with logit_bias.
    llama-server would otherwise keep the previous prompt in its slot and reuse its prefix.
    """
    if engine.kind == "mlx":
        return {"logit_bias": {str(token): -100 for token in eos_ids(directory)}}
    if engine.kind == "llama.cpp":
        return {"cache_prompt": False}
    return {}


def wait_ready(
    port: int, process: subprocess.Popen, log: pathlib.Path, timeout: float = 600
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = "\n".join(log.read_text(errors="ignore").splitlines()[-20:])
            raise BenchmarkError(f"server exited during startup, see {log}:\n{tail}")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/v1/models")
            if connection.getresponse().status == 200:
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise BenchmarkError(f"server did not become ready, see {log}")


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


@contextlib.contextmanager
def server_session(
    engine: Engine, args: argparse.Namespace, directory: pathlib.Path, log: pathlib.Path
) -> Iterator[Target]:
    """A started, warmed-up server; it is always stopped on exit, Ctrl-C included."""
    with log.open("a") as output:
        process = subprocess.Popen(
            server_command(engine, args),
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_ready(args.port, process, log)
            target = Target(
                port=args.port,
                extra=extra_body(engine, directory),
                prefix_requests=args.workload != "sharegpt",
            )
            stream(target, PROMPT_TEXT, 16, "", usage_only=True)
            yield target
        finally:
            stop(process)


def metal_bench_command(
    args: argparse.Namespace, overrides: list[str], prompt: int, test: str
) -> list[str]:
    command = [
        str(METAL_BENCH),
        "--model",
        args.model,
        "--prompt",
        str(prompt),
        "--generate",
        str(args.generate),
        "--iterations",
        str(args.iterations),
        "--test",
        test,
    ]
    if test == "tg":
        command += ["--depth", str(prompt)]
    for override in overrides:
        command += ["--with", override]
    return command


def mlx_bench_command(args: argparse.Namespace, prompt: int) -> list[str]:
    return MLX + [
        "mlx_lm.benchmark",
        "--model",
        args.mlx_model,
        "-p",
        str(prompt),
        "-g",
        str(args.generate),
        "-n",
        str(args.iterations),
    ]


def llama_bench_commands(args: argparse.Namespace, prompt: int) -> list[list[str]]:
    """Prefill of `prompt` tokens, then decode after `prompt` tokens of context, like the others."""
    common = [
        "llama-bench",
        "-m",
        str(args.gguf),
        "-ngl",
        "99",
        "-r",
        str(args.iterations),
        "-o",
        "json",
    ]
    return [
        common + ["-p", str(prompt), "-n", "0"],
        common + ["-p", "0", "-n", str(args.generate), "-d", str(prompt)],
    ]


def offline_commands(engine: Engine, args: argparse.Namespace, prompt: int) -> list[list[str]]:
    if engine.kind == "mlx":
        return [mlx_bench_command(args, prompt)]
    if engine.kind == "llama.cpp":
        return llama_bench_commands(args, prompt)
    return [metal_bench_command(args, engine.overrides, prompt, test) for test in ("pp", "tg")]


def offline_metal(
    args: argparse.Namespace, overrides: list[str], prompt: int
) -> list[dict[str, Any]]:
    samples = []
    for test in ("pp", "tg"):
        report = json.loads(run(metal_bench_command(args, overrides, prompt, test)))
        for sample in report["samples"]:
            samples.append(
                {
                    "pp_tps": prompt / (sample["prefill_ms"] / 1000) if test == "pp" else None,
                    "tg_tps": args.generate / (sample["decode_ms"] / 1000)
                    if test == "tg"
                    else None,
                    "peak_memory_gb": report["allocated_bytes"] / 1e9,
                    "plan": report["plan"],
                }
            )
    return samples


def offline_mlx(args: argparse.Namespace, prompt: int) -> list[dict[str, Any]]:
    output = run(mlx_bench_command(args, prompt))
    numbers = r"prompt_tps=([0-9.]+), generation_tps=([0-9.]+), peak_memory=([0-9.]+)"
    trials = list(re.finditer(r"Trial \d+:\s*" + numbers, output))
    if not trials:
        trials = list(re.finditer(r"Averages: " + numbers, output))
    if not trials:
        raise BenchmarkError("mlx_lm.benchmark output has no trial results:\n" + output)
    return [
        {
            "pp_tps": float(trial.group(1)),
            "tg_tps": float(trial.group(2)),
            "peak_memory_gb": float(trial.group(3)),
        }
        for trial in trials
    ]


def offline_llama(args: argparse.Namespace, prompt: int) -> list[dict[str, Any]]:
    commands = llama_bench_commands(args, prompt)
    prefill = json.loads(run(commands[0]))
    decode = json.loads(run(commands[1]))
    pp = next(
        (
            entry["samples_ts"]
            for entry in prefill
            if entry.get("n_prompt") == prompt and entry.get("n_gen") == 0
        ),
        None,
    )
    tg = next(
        (
            entry["samples_ts"]
            for entry in decode
            if entry.get("n_gen") == args.generate and entry.get("n_prompt") == 0
        ),
        None,
    )
    if not pp or not tg:
        raise BenchmarkError("llama-bench output has no prefill or decode samples")
    return [{"pp_tps": float(value), "tg_tps": None, "peak_memory_gb": None} for value in pp] + [
        {"pp_tps": None, "tg_tps": float(value), "peak_memory_gb": None} for value in tg
    ]


def offline_samples(engine: Engine, args: argparse.Namespace, prompt: int) -> list[dict[str, Any]]:
    if engine.kind == "mlx":
        return offline_mlx(args, prompt)
    if engine.kind == "llama.cpp":
        return offline_llama(args, prompt)
    return offline_metal(args, engine.overrides, prompt)
