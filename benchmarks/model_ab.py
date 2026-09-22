#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Compare two model decode variants with alternating A/B and B/A rounds."""

import argparse
import json
import math
import pathlib
import shlex
import statistics
import subprocess
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
BINARY = ROOT / "target" / "release" / "metal-infer-bench"
FUSIONS = (
    "--fuse-qkv",
    "--fuse-gate-up",
    "--fuse-add-rms-norm",
    "--fuse-qk-rope-cache",
)
COMMON_OPTIONS = frozenset((
    "--model", "--prompt", "--generate", "--warmup", "--iterations",
    "--format", "--profile-kernels", *FUSIONS,
))


class BenchmarkError(RuntimeError):
    """A benchmark command or result could not be processed."""


def variant_args(parser: argparse.ArgumentParser, value: str, name: str) -> list[str]:
    try:
        arguments = shlex.split(value)
    except ValueError as error:
        parser.error(f"{name}: {error}")
    for argument in arguments:
        if argument.split("=", 1)[0] in COMMON_OPTIONS:
            parser.error(f"{name} cannot override the shared workload: {argument}")
    return arguments


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-args", default="", help="extra arguments for variant A")
    parser.add_argument("--candidate-args", required=True, help="extra arguments for variant B")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--generate", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, help="save commands and raw results as JSON")
    args = parser.parse_args()
    if min(args.rounds, args.prompt, args.generate, args.iterations) < 1 or args.warmup < 0:
        parser.error("rounds, prompt, generate and iterations must be positive; warmup cannot be negative")
    if not (args.model / "config.json").is_file():
        parser.error(f"model directory has no config.json: {args.model}")
    args.model = args.model.resolve()
    args.baseline_args = variant_args(parser, args.baseline_args, "--baseline-args")
    args.candidate_args = variant_args(parser, args.candidate_args, "--candidate-args")
    if not args.candidate_args or args.baseline_args == args.candidate_args:
        parser.error("the candidate must have arguments different from the baseline")
    return args


def round_order(round_index: int) -> tuple[str, str]:
    return ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    except OSError as error:
        raise BenchmarkError(f"could not run {shlex.join(command)}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BenchmarkError(f"{shlex.join(command)} failed: {detail}")
    return completed.stdout


def decode_tps(report: Any, prompt: int, generate: int, iterations: int) -> float:
    if not isinstance(report, dict) or report.get("benchmark") != "qwen3_model":
        raise BenchmarkError("model benchmark did not return a qwen3_model JSON object")
    if report.get("iterations") != iterations:
        raise BenchmarkError("model benchmark returned a different iteration count")
    if report.get("kernel_profile") is not None:
        raise BenchmarkError("profiled model runs cannot be used for throughput comparisons")
    prefill = report.get("prefill")
    decode = report.get("decode")
    if not isinstance(prefill, dict) or not isinstance(decode, dict):
        raise BenchmarkError("model benchmark has no prefill or decode result")
    if prefill.get("tokens") != prompt or decode.get("tokens") != generate:
        raise BenchmarkError("model benchmark returned different prompt or decode lengths")
    fusions = report.get("fusions", {})
    if not isinstance(fusions, dict) or not all(
        fusions.get(flag) is True
        for flag in ("qkv", "gate_up", "add_rms_norm", "qk_rope_cache")
    ):
        raise BenchmarkError("model benchmark did not enable all four fusions")
    try:
        tps = float(decode["tokens_per_second"])
    except (KeyError, TypeError, ValueError) as error:
        raise BenchmarkError("model benchmark has no valid decode tokens_per_second") from error
    if not math.isfinite(tps) or tps <= 0:
        raise BenchmarkError("model benchmark returned a non-positive or non-finite decode rate")
    return tps


def model_command(args: argparse.Namespace, extra: list[str]) -> list[str]:
    return [
        str(BINARY), "model", "--model", str(args.model),
        "--prompt", str(args.prompt), "--generate", str(args.generate),
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
        *FUSIONS, *extra, "--format", "json",
    ]


def summarize(runs: list[dict[str, Any]], rounds: int) -> dict[str, float | int]:
    baseline = [run["decode_tps"] for run in runs if run["variant"] == "baseline"]
    candidate = [run["decode_tps"] for run in runs if run["variant"] == "candidate"]
    if len(baseline) != rounds or len(candidate) != rounds:
        raise BenchmarkError("each round must contain one result per variant")
    paired_deltas = [b - a for a, b in zip(baseline, candidate)]
    paired_percent = [100.0 * delta / a for a, delta in zip(baseline, paired_deltas)]
    return {
        "baseline_median_tps": statistics.median(baseline),
        "candidate_median_tps": statistics.median(candidate),
        "paired_median_delta_tps": statistics.median(paired_deltas),
        "paired_median_delta_percent": statistics.median(paired_percent),
        "candidate_wins": sum(delta > 0 for delta in paired_deltas),
        "rounds": rounds,
    }


def main() -> None:
    args = arguments()
    if not args.skip_build:
        print("Building release benchmark binary", flush=True)
        run_command(["cargo", "build", "--release", "--bin", "metal-infer-bench"])
    variants = {"baseline": args.baseline_args, "candidate": args.candidate_args}
    runs: list[dict[str, Any]] = []
    for round_index in range(args.rounds):
        order = round_order(round_index)
        print(f"Round {round_index + 1}/{args.rounds}: {' -> '.join(order)}", flush=True)
        for variant in order:
            command = model_command(args, variants[variant])
            print(f"  $ {shlex.join(command)}", flush=True)
            try:
                report = json.loads(run_command(command))
            except json.JSONDecodeError as error:
                raise BenchmarkError(f"{variant} returned invalid JSON") from error
            tps = decode_tps(report, args.prompt, args.generate, args.iterations)
            runs.append({
                "round": round_index + 1,
                "variant": variant,
                "command": command,
                "decode_tps": tps,
                "report": report,
            })
            print(f"  {variant}: {tps:.3f} tok/s", flush=True)
    summary = summarize(runs, args.rounds)
    print(
        f"Median: baseline {summary['baseline_median_tps']:.3f}, "
        f"candidate {summary['candidate_median_tps']:.3f} tok/s",
        flush=True,
    )
    print(
        f"Paired median: {summary['paired_median_delta_tps']:+.3f} tok/s "
        f"({summary['paired_median_delta_percent']:+.2f}%); "
        f"candidate wins {summary['candidate_wins']}/{args.rounds}",
        flush=True,
    )
    if args.output is not None:
        document = {
            "schema_version": 1,
            "model": str(args.model),
            "workload": {
                "rounds": args.rounds,
                "prompt": args.prompt,
                "generate": args.generate,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "fusions": list(FUSIONS),
            },
            "variants": variants,
            "runs": runs,
            "summary": summary,
        }
        try:
            args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            raise BenchmarkError(f"could not write {args.output}: {error}") from error
        print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BenchmarkError as error:
        raise SystemExit(str(error)) from error
