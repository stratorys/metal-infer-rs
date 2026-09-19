#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Run reproducible metal-infer comparisons and store normalized JSON."""

import argparse
import datetime
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "benchmarks" / "suite.json"
README_PATH = ROOT / "benchmarks" / "README.md"
DEFAULT_RESULTS = ROOT / "benchmarks" / "results" / "latest.json"
RESULTS_START = "<!-- BENCH_RESULTS_START -->"
RESULTS_END = "<!-- BENCH_RESULTS_END -->"


class BenchmarkError(RuntimeError):
    """A benchmark command or result could not be processed."""


@dataclass
class CommandResult:
    stdout: str
    elapsed_seconds: float


class Runner:
    def __init__(self, total_steps: int, quiet: bool) -> None:
        self.total_steps = total_steps
        self.quiet = quiet
        self.step = 0

    def run(self, command: list[str], label: str) -> CommandResult:
        self.step += 1
        if not self.quiet:
            print(f"\n[{self.step}/{self.total_steps}] {label}", flush=True)
            print(f"$ {' '.join(command)}", flush=True)
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stdout_thread = threading.Thread(
            target=self._forward,
            args=(process.stdout, stdout_chunks, sys.stdout),
        )
        stderr_thread = threading.Thread(
            target=self._forward,
            args=(process.stderr, stderr_chunks, sys.stderr),
        )
        stdout_thread.start()
        stderr_thread.start()
        return_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        elapsed = time.perf_counter() - started
        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
        if return_code != 0:
            detail = stderr.strip() or stdout.strip()
            raise BenchmarkError(
                f"command failed after {elapsed:.2f}s ({' '.join(command)}): {detail}"
            )
        return CommandResult(stdout=stdout, elapsed_seconds=elapsed)

    def run_json(
        self,
        command: list[str],
        label: str,
    ) -> tuple[dict[str, Any] | list[dict[str, Any]], float]:
        result = self.run(command, label)
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as parse_error:
            raise BenchmarkError(
                f"command did not emit valid JSON ({' '.join(command)})"
            ) from parse_error
        if not isinstance(value, (dict, list)):
            raise BenchmarkError("benchmark JSON must be an object or an array")
        return value, result.elapsed_seconds

    def completed(self, elapsed: float, summary: str) -> None:
        if not self.quiet:
            print(f"completed in {elapsed:.2f}s: {summary}", flush=True)

    def _forward(
        self,
        pipe: Any,
        chunks: list[str],
        destination: Any,
    ) -> None:
        if pipe is None:
            return
        for line in iter(pipe.readline, ""):
            chunks.append(line)
            if not self.quiet:
                print(line, end="", file=destination, flush=True)
        pipe.close()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_RESULTS)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("kernels")
    model = subparsers.add_parser("model")
    model.add_argument("--metal-model", type=pathlib.Path, required=True)
    model.add_argument("--mlx-model", required=True)
    model.add_argument("--gguf", type=pathlib.Path)
    model.add_argument("--llama-bench", type=pathlib.Path)
    return parser.parse_args()


def capture(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BenchmarkError(f"command failed ({' '.join(command)}): {detail}")
    return completed.stdout


def load_suite() -> dict[str, Any]:
    value = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkError("suite.json must contain an object")
    return value


def build(runner: Runner) -> None:
    result = runner.run(
        ["cargo", "build", "--release", "--bin", "metal-infer-bench"],
        "build metal-infer-bench (release)",
    )
    runner.completed(result.elapsed_seconds, "release binary ready")


def system_metadata() -> dict[str, Any]:
    chip = platform.machine()
    try:
        chip = capture(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    except BenchmarkError:
        pass
    commit = None
    try:
        commit = capture(["git", "rev-parse", "HEAD"]).strip()
    except BenchmarkError:
        pass
    return {
        "chip": chip,
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "metal_infer_commit": commit,
        "rustc": capture(["rustc", "--version"]).strip(),
    }


def kernel_results(config: dict[str, Any], runner: Runner) -> list[dict[str, Any]]:
    iterations = int(config["iterations"])
    warmup = int(config["warmup"])
    results = []
    for shape in config["shapes"]:
        m = int(shape["m"])
        n = int(shape["n"])
        k = int(shape["k"])
        common = [
            "--m",
            str(m),
            "--n",
            str(n),
            "--k",
            str(k),
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
        ]
        metal, elapsed = runner.run_json(
            [
                str(ROOT / "target" / "release" / "metal-infer-bench"),
                "kernel",
                *common,
                "--format",
                "json",
            ],
            f"metal-infer matmul {m}×{n}×{k}",
        )
        if not isinstance(metal, dict):
            raise BenchmarkError("metal-infer kernel output must be an object")
        metal["backend"] = "metal-infer"
        metal["dtype"] = "f16"
        metal["dimensions"] = {"m": m, "n": n, "k": k}
        runner.completed(
            elapsed,
            f"{float(metal['mean_ms']):.3f} ms, "
            f"{float(metal['throughput']):.4f} TFLOP/s",
        )
        mlx, elapsed = runner.run_json(
            ["uv", "run", "benchmarks/mlx_reference.py", *common],
            f"MLX matmul {m}×{n}×{k}",
        )
        if not isinstance(mlx, dict):
            raise BenchmarkError("MLX kernel output must be an object")
        runner.completed(
            elapsed,
            f"{float(mlx['mean_ms']):.3f} ms, "
            f"{float(mlx['throughput']):.4f} TFLOP/s",
        )
        results.extend([metal, mlx])
    return results


def parse_mlx_model(output: str, prompt: int, generate: int) -> dict[str, Any]:
    matches = re.findall(
        r"Averages: prompt_tps=([0-9.]+), generation_tps=([0-9.]+), "
        r"peak_memory=([0-9.]+)",
        output,
    )
    if not matches:
        raise BenchmarkError("could not find MLX-LM averages in benchmark output")
    prompt_tps, generation_tps, peak_memory = matches[-1]
    return {
        "backend": "mlx-lm",
        "benchmark": "qwen3_model",
        "prefill": {
            "tokens": prompt,
            "tokens_per_second": float(prompt_tps),
        },
        "decode": {
            "tokens": generate,
            "tokens_per_second": float(generation_tps),
        },
        "peak_memory_gb": float(peak_memory),
    }


def parse_llama_model(
    value: dict[str, Any] | list[dict[str, Any]],
    prompt: int,
    generate: int,
) -> dict[str, Any]:
    rows = value if isinstance(value, list) else [value]
    prefill = next(
        (
            row
            for row in rows
            if int(row.get("n_prompt", 0)) == prompt
            and int(row.get("n_gen", 0)) == 0
        ),
        None,
    )
    decode = next(
        (
            row
            for row in rows
            if int(row.get("n_prompt", 0)) == 0
            and int(row.get("n_gen", 0)) == generate
        ),
        None,
    )
    if prefill is None or decode is None:
        raise BenchmarkError("llama-bench JSON lacks separate pp and tg results")
    return {
        "backend": "llama.cpp",
        "benchmark": "qwen3_model",
        "prefill": {
            "tokens": prompt,
            "tokens_per_second": float(prefill["avg_ts"]),
        },
        "decode": {
            "tokens": generate,
            "tokens_per_second": float(decode["avg_ts"]),
        },
        "raw": rows,
    }


def model_results(
    config: dict[str, Any],
    args: argparse.Namespace,
    runner: Runner,
) -> list[dict[str, Any]]:
    prompt = int(config["prompt_tokens"])
    generate = int(config["generation_tokens"])
    iterations = int(config["iterations"])
    warmup = int(config["warmup"])
    metal, elapsed = runner.run_json(
        [
            str(ROOT / "target" / "release" / "metal-infer-bench"),
            "model",
            "--model",
            str(args.metal_model),
            "--prompt",
            str(prompt),
            "--generate",
            str(generate),
            "--iterations",
            str(iterations),
            "--warmup",
            str(warmup),
            "--format",
            "json",
        ],
        f"metal-infer model prefill={prompt}, decode={generate}",
    )
    if not isinstance(metal, dict):
        raise BenchmarkError("metal-infer model output must be an object")
    metal["backend"] = "metal-infer"
    runner.completed(
        elapsed,
        f"prefill {float(metal['prefill']['tokens_per_second']):.3f} tokens/s, "
        f"decode {float(metal['decode']['tokens_per_second']):.3f} tokens/s",
    )
    mlx_result = runner.run(
        [
            "uvx",
            "--from",
            "mlx-lm",
            "mlx_lm.benchmark",
            "--model",
            args.mlx_model,
            "-p",
            str(prompt),
            "-g",
            str(generate),
            "-n",
            str(iterations),
        ],
        f"MLX-LM model prefill={prompt}, decode={generate}",
    )
    mlx = parse_mlx_model(mlx_result.stdout, prompt, generate)
    runner.completed(
        mlx_result.elapsed_seconds,
        f"prefill {float(mlx['prefill']['tokens_per_second']):.3f} tokens/s, "
        f"decode {float(mlx['decode']['tokens_per_second']):.3f} tokens/s",
    )
    results = [metal, mlx]
    if args.gguf is not None or args.llama_bench is not None:
        if args.gguf is None or args.llama_bench is None:
            raise BenchmarkError("--gguf and --llama-bench must be provided together")
        llama, elapsed = runner.run_json(
            [
                str(args.llama_bench),
                "-m",
                str(args.gguf),
                "-p",
                str(prompt),
                "-n",
                str(generate),
                "-r",
                str(iterations),
                "-ngl",
                "99",
                "-o",
                "json",
            ],
            f"llama.cpp model prefill={prompt}, decode={generate}",
        )
        parsed_llama = parse_llama_model(llama, prompt, generate)
        runner.completed(
            elapsed,
            f"prefill {float(parsed_llama['prefill']['tokens_per_second']):.3f} "
            f"tokens/s, decode "
            f"{float(parsed_llama['decode']['tokens_per_second']):.3f} tokens/s",
        )
        results.append(parsed_llama)
    return results


def render_kernel_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Shape (M×N×K) | Backend | Mean ms | TFLOP/s | Relative |",
        "|---|---|---:|---:|---:|",
    ]
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for result in results:
        dimensions = result.get("dimensions")
        if not isinstance(dimensions, dict):
            continue
        shape = (int(dimensions["m"]), int(dimensions["n"]), int(dimensions["k"]))
        grouped.setdefault(shape, []).append(result)
    for shape, rows in grouped.items():
        metal = next((row for row in rows if row["backend"] == "metal-infer"), None)
        baseline = float(metal["throughput"]) if metal is not None else 0.0
        for row in rows:
            throughput = float(row["throughput"])
            relative = throughput / baseline if baseline > 0.0 else 0.0
            lines.append(
                f"| {shape[0]}×{shape[1]}×{shape[2]} | {row['backend']} | "
                f"{float(row['mean_ms']):.3f} | {throughput:.4f} | {relative:.2f}× |"
            )
    return "\n".join(lines)


def update_readme(results: list[dict[str, Any]], generated_at: str) -> None:
    if not README_PATH.exists():
        return
    current = README_PATH.read_text(encoding="utf-8")
    if RESULTS_START not in current or RESULTS_END not in current:
        raise BenchmarkError("benchmark README result markers are missing")
    table = render_kernel_table(results)
    replacement = (
        f"{RESULTS_START}\n\nLast generated: `{generated_at}`.\n\n"
        f"{table}\n\n{RESULTS_END}"
    )
    prefix, remainder = current.split(RESULTS_START, maxsplit=1)
    _, suffix = remainder.split(RESULTS_END, maxsplit=1)
    README_PATH.write_text(prefix + replacement + suffix, encoding="utf-8")


def write_results(
    output: pathlib.Path,
    command: str,
    suite: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    document = {
        "schema_version": 1,
        "generated_at": generated_at,
        "system": system_metadata(),
        "suite": suite[command[:-1] if command == "kernels" else command],
        "results": results,
    }
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    if command == "kernels":
        update_readme(results, generated_at)
    print(f"wrote {output}")


def main() -> None:
    args = arguments()
    started = time.perf_counter()
    try:
        suite = load_suite()
        benchmark_steps = (
            len(suite["kernel"]["shapes"]) * 2 if args.command == "kernels" else 2
        )
        if args.command == "model" and args.gguf is not None:
            benchmark_steps += 1
        runner = Runner(
            total_steps=benchmark_steps + int(not args.skip_build),
            quiet=args.quiet,
        )
        if not args.skip_build:
            build(runner)
        if args.command == "kernels":
            results = kernel_results(suite["kernel"], runner)
        else:
            results = model_results(suite["model"], args, runner)
        write_results(args.output, args.command, suite, results)
        print(
            f"completed {len(results)} benchmark results in "
            f"{time.perf_counter() - started:.2f}s",
            flush=True,
        )
    except (BenchmarkError, OSError, KeyError, ValueError) as benchmark_error:
        print(f"benchmark failed: {benchmark_error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
