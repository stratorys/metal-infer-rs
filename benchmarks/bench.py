"""Benchmark metal-infer against other engines, one engine at a time."""

import argparse
import math
import pathlib
import sys
import time
from typing import Any

from benchlib.client import Job, Target, server_level
from benchlib.common import (
    PEAK_BANDWIDTH_GBS,
    SCHEMA_VERSION,
    BenchmarkError,
    Energy,
    cooldown,
    file_sha256,
    model_directory,
    model_facts,
    require_idle,
    run,
    slug,
    system,
)
from benchlib.engines import offline_samples, server_session
from benchlib.gguf import resolve_gguf
from benchlib.plan import SUITES, Measurement, apply_suite, engines, measurements
from benchlib.progress import RunState, now, progress_line
from benchlib.report import build_document, markdown, render, write_report
from benchlib.workloads import Workload, calibrate, workload


def request_rate(value: str) -> str:
    if value == "inf":
        return value
    try:
        if math.isfinite(float(value)) and float(value) > 0:
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        "a request rate is a positive number of requests per second or inf"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--model", help="Hugging Face id or local directory")
    result.add_argument(
        "--render",
        type=pathlib.Path,
        metavar="RESULTS_JSON",
        help="write the report of an existing results file without measuring",
    )
    result.add_argument(
        "--resume",
        type=pathlib.Path,
        metavar="RUN_DIRECTORY",
        help="continue an interrupted run, skipping the measurements already done",
    )
    result.add_argument(
        "--suite",
        choices=sorted(SUITES),
        help="preset: showcase = long-context serving against MLX and llama.cpp; "
        "options given explicitly win",
    )
    result.add_argument("--mlx-model", help="model id for MLX (default: --model)")
    result.add_argument(
        "--gguf", help="local BF16/F16 file or unsloth/Qwen3-0.6B-GGUF (cached BF16)"
    )
    result.add_argument(
        "--engines",
        nargs="+",
        default=["metal-infer", "mlx"],
        choices=["metal-infer", "mlx", "llama.cpp"],
    )
    result.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="add a metal-infer variant with this plan override (repeatable)",
    )
    result.add_argument("--mode", choices=["offline", "server", "all"], default="all")
    result.add_argument(
        "--workload",
        choices=["synthetic", "sharegpt", "context"],
        default="synthetic",
        help="server requests: fixed text at fixed concurrency, ShareGPT conversations "
        "with Poisson arrivals, or one long prompt at a time per context length",
    )
    result.add_argument(
        "--prompt",
        type=int,
        nargs="+",
        default=[512],
        help="offline prompt lengths and context lengths; the synthetic workload uses the first",
    )
    result.add_argument("--generate", type=int, default=128)
    result.add_argument("--iterations", type=int, default=5)
    result.add_argument("--rounds", type=int, default=2)
    result.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="synthetic workload concurrency levels",
    )
    result.add_argument(
        "--requests",
        type=int,
        default=8,
        help="synthetic and context workloads: requests per measurement",
    )
    result.add_argument("--num-prompts", type=int, default=100, help="ShareGPT requests per rate")
    result.add_argument(
        "--request-rate",
        type=request_rate,
        nargs="+",
        default=["1", "2", "4", "inf"],
        help="ShareGPT Poisson arrival rates in requests per second, inf sends all at once",
    )
    result.add_argument("--seed", type=int, default=0, help="ShareGPT sampling and arrival seed")
    result.add_argument("--slo-ttft-ms", type=float, default=1000.0, help="goodput TTFT limit")
    result.add_argument("--slo-tpot-ms", type=float, default=50.0, help="goodput TPOT limit")
    result.add_argument(
        "--cooldown", type=float, default=30.0, help="seconds after each engine and mode"
    )
    result.add_argument("--port", type=int, default=8931)
    result.add_argument("--peak-bandwidth", type=float, help="GB/s, overrides the chip table")
    result.add_argument("--peak-tflops", type=float, help="FP16 TFLOPS, enables MFU")
    result.add_argument(
        "--energy",
        action="store_true",
        help="measure joules per token with sudo powermetrics (run `sudo -v` first)",
    )
    result.add_argument("--skip-build", action="store_true")
    return result


def explicit_options(options: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    found = set()
    for action in options._actions:
        for option in action.option_strings:
            if any(word == option or word.startswith(option + "=") for word in argv):
                found.add(action.dest)
    return found


def validate(args: argparse.Namespace) -> None:
    if not args.model:
        raise BenchmarkError("--model is required unless --render or --resume is given")
    args.mlx_model = args.mlx_model or args.model
    if (
        args.rounds < 1
        or args.iterations < 1
        or args.requests < 1
        or args.num_prompts < 1
        or args.generate < 1
        or any(value < 1 for value in args.prompt)
        or any(value < 1 for value in args.concurrency)
    ):
        raise BenchmarkError(
            "rounds, iterations, requests, prompts, concurrency and generation must be positive"
        )
    if "llama.cpp" in args.engines and not args.gguf:
        raise BenchmarkError("llama.cpp needs --gguf, a local GGUF or supported Hugging Face repo")
    if args.gguf:
        gguf, args.gguf_source = resolve_gguf(args.gguf)
        args.gguf = str(gguf)


def header(args: argparse.Namespace, directory: pathlib.Path, load: Workload) -> dict[str, Any]:
    machine = system(args.engines)
    files = {}
    if args.gguf:
        print("hashing the GGUF file", flush=True)
        files["gguf"] = {
            "path": args.gguf,
            "sha256": file_sha256(pathlib.Path(args.gguf)),
            **getattr(args, "gguf_source", {}),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "started_at": now(),
        "system": machine,
        "arguments": {
            key: vars(args)[key] for key in vars(args) if key not in ["render", "resume"]
        },
        "model": model_facts(directory),
        "workload": load.description,
        "peak_bandwidth_gbs": args.peak_bandwidth or PEAK_BANDWIDTH_GBS.get(machine["chip"]),
        "files": files,
        "engines": [engine.as_dict() for engine in engines(args)],
        "measurements": [
            {
                "key": item.key,
                "round": item.round,
                "engine": item.engine.label,
                "mode": item.mode,
                "load": item.load.as_dict(),
            }
            for item in measurements(args)
        ],
    }


def measure_server(
    measurement: Measurement,
    target: Target,
    args: argparse.Namespace,
    load: Workload,
    energy: Energy,
    characters: dict[int, int],
) -> dict[str, Any]:
    if measurement.load.type == "rate":
        jobs = load.jobs
        return server_level(
            target,
            energy,
            jobs,
            len(jobs),
            load.description["schedules"][measurement.load.value],
            args.slo_ttft_ms,
            args.slo_tpot_ms,
        )
    tokens = int(measurement.load.value) if measurement.load.type == "context" else args.prompt[0]
    if tokens not in characters:
        print(f"  calibrating the prompt to {tokens} tokens", flush=True)
        characters[tokens] = calibrate(target, load.text, tokens)
    jobs = [Job(prompt=load.text[: characters[tokens]], generate=args.generate)] * args.requests
    workers = 1 if measurement.load.type == "context" else int(measurement.load.value)
    data = server_level(target, energy, jobs, workers, None, args.slo_ttft_ms, args.slo_tpot_ms)
    wrong = [
        item["prompt_tokens"]
        for item in data["requests"]
        if "error" not in item and item["prompt_tokens"] != tokens
    ]
    if wrong:
        raise BenchmarkError(
            f"{measurement.engine.label}: requests calibrated to {tokens} prompt tokens "
            f"reported {wrong}"
        )
    return {**data, "calibration": {"tokens": tokens, "characters": characters[tokens]}}


def publish(state: RunState) -> None:
    write_report(build_document(state.header, state.records), state.directory, index=False)


def record(
    state: RunState,
    measurement: Measurement,
    data: dict[str, Any],
    started: float,
    remaining: list[Measurement],
    total: int,
) -> None:
    state.append(
        {
            "key": measurement.key,
            "round": measurement.round,
            "engine": measurement.engine.label,
            "mode": measurement.mode,
            "load": measurement.load.as_dict(),
            "finished_at": now(),
            "seconds": time.perf_counter() - started,
            "data": data,
        }
    )
    publish(state)
    print(
        progress_line(
            len(state.records), total, measurement, data, state.remaining_seconds(remaining)
        ),
        flush=True,
    )


def run_group(
    group: list[Measurement],
    later: list[Measurement],
    args: argparse.Namespace,
    state: RunState,
    load: Workload,
    directory: pathlib.Path,
    energy: Energy,
    total: int,
) -> None:
    """Measurements of one round, engine and mode: one server session or one offline batch."""
    first = group[0]
    require_idle(args.port)
    print(
        f"round {first.round + 1}/{args.rounds} · {first.engine.label} · {first.mode}", flush=True
    )
    if first.mode == "offline":
        for position in range(len(group)):
            measurement = group[position]
            started = time.perf_counter()
            measured = energy.measure(
                lambda: offline_samples(measurement.engine, args, int(measurement.load.value))
            )
            generated = args.generate * args.iterations
            data = {
                "samples": measured.result,
                "joules_per_token": None
                if measured.joules is None
                else measured.joules / generated,
            }
            record(state, measurement, data, started, group[position + 1 :] + later, total)
        return
    log = state.directory / f"server-{slug(first.engine.label)}.log"
    with server_session(first.engine, args, directory, log) as target:
        characters: dict[int, int] = {}
        for position in range(len(group)):
            measurement = group[position]
            started = time.perf_counter()
            data = measure_server(measurement, target, args, load, energy, characters)
            record(state, measurement, data, started, group[position + 1 :] + later, total)


def execute(
    args: argparse.Namespace, state: RunState, load: Workload, directory: pathlib.Path
) -> None:
    planned = measurements(args)
    if [item.key for item in planned] != [item["key"] for item in state.header["measurements"]]:
        raise BenchmarkError(
            "the measurement list no longer matches run.json; this run cannot be resumed"
        )
    done = state.done_keys()
    pending = [item for item in planned if item.key not in done]
    print(
        f"run directory: {state.directory}\n{len(done)}/{len(planned)} measurements already done",
        flush=True,
    )
    publish(state)
    energy = Energy(args.energy)
    resume = f"uv run --project benchmarks benchmarks/bench.py --resume {state.directory}"
    try:
        index = 0
        while index < len(pending):
            group = [pending[index]]
            while (
                index + len(group) < len(pending)
                and pending[index + len(group)].group == group[0].group
            ):
                group.append(pending[index + len(group)])
            index += len(group)
            run_group(group, pending[index:], args, state, load, directory, energy, len(planned))
            if index < len(pending):
                cooldown(args.cooldown)
    except KeyboardInterrupt:
        print(
            f"\ninterrupted; the finished measurements are saved. Resume with:\n  {resume}",
            file=sys.stderr,
        )
        sys.exit(130)
    except BenchmarkError as error:
        raise BenchmarkError(
            f"{error}\nthe finished measurements are saved. Resume with:\n  {resume}"
        )
    state.finish()
    document = build_document(state.header, state.records)
    write_report(document, state.directory, index=True)
    print(markdown(document))
    print(f"report: {state.directory / 'README.md'}")


def main() -> None:
    options = parser()
    args = options.parse_args()
    explicit = explicit_options(options, sys.argv[1:])
    if args.render:
        print(f"report: {render(args.render) / 'README.md'}")
        return
    state = None
    if args.resume:
        state = RunState.resume(args.resume)
        stored = argparse.Namespace(**state.header["arguments"])
        stored.skip_build = args.skip_build
        if "cooldown" in explicit:
            stored.cooldown = args.cooldown
        args = stored
    else:
        apply_suite(args, explicit)
        validate(args)
    directory = model_directory(args.model)
    load = workload(args, directory)
    if not args.skip_build and any(engine.kind == "metal-infer" for engine in engines(args)):
        print("building release binaries", flush=True)
        run(["cargo", "build", "--release", "--bin", "metal-infer", "--bin", "metal-infer-bench"])
    if state is None:
        state = RunState.create(header(args, directory, load))
    execute(args, state, load, directory)


if __name__ == "__main__":
    try:
        main()
    except BenchmarkError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
