#!/usr/bin/env python3
"""Compare Metal matmul backends with alternating, repeated measurements."""

import argparse
import json
import pathlib
import statistics
import subprocess
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
BINARY = ROOT / "target" / "release" / "metal-infer-bench"
SUITE = ROOT / "benchmarks" / "suite.json"


def qwen_shapes(config_paths: list[pathlib.Path], prompt_lengths: list[int]) -> list[tuple[int, int, int]]:
    shapes: set[tuple[int, int, int]] = set()
    for path in config_paths:
        config_path = path / "config.json" if path.is_dir() else path
        config = json.loads(config_path.read_text(encoding="utf-8"))
        hidden = int(config["hidden_size"])
        intermediate = int(config["intermediate_size"])
        query = int(config["num_attention_heads"]) * int(config["head_dim"])
        kv = int(config["num_key_value_heads"]) * int(config["head_dim"])
        vocabulary = int(config["vocab_size"])
        projections = ((query, hidden), (kv, hidden), (hidden, query),
                       (intermediate, hidden), (hidden, intermediate))
        shapes.update((m, n, k) for m in prompt_lengths for n, k in projections)
        shapes.update((1, n, k) for n, k in projections)
        shapes.add((1, vocabulary, hidden))
    return sorted(shapes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--shapes", nargs="*", help="M,N,K triples; defaults to suite.json")
    parser.add_argument("--model-config", action="append", type=pathlib.Path,
                        help="Qwen3 config.json or model directory; may be repeated")
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--output", type=pathlib.Path, help="write all raw backend results to JSON")
    args = parser.parse_args()
    if args.rounds < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("rounds and iterations must be positive; warmup cannot be negative")
    if not args.prompt_lengths or min(args.prompt_lengths) < 1:
        parser.error("prompt lengths must be positive")
    if args.shapes:
        try:
            shapes = [tuple(map(int, shape.split(","))) for shape in args.shapes]
        except ValueError:
            parser.error("shapes must be M,N,K triples")
        if any(len(shape) != 3 or min(shape) < 1 for shape in shapes):
            parser.error("shapes must be positive M,N,K triples")
    elif args.model_config:
        shapes = qwen_shapes(args.model_config, args.prompt_lengths)
    else:
        suite = json.loads(SUITE.read_text(encoding="utf-8"))
        shapes = [(s["m"], s["n"], s["k"]) for s in suite["kernel"]["shapes"]]

    backends = ("reference-msl", "native-msl", "auto")
    raw: list[dict[str, Any]] = []
    for m, n, k in shapes:
        samples: dict[str, list[dict]] = {backend: [] for backend in backends}
        for round_index in range(args.rounds):
            order = backends if round_index % 2 == 0 else tuple(reversed(backends))
            for backend in order:
                command = [
                    str(BINARY), "kernel", "--m", str(m), "--n", str(n),
                    "--k", str(k), "--warmup", str(args.warmup),
                    "--iterations", str(args.iterations), "--matmul-backend",
                    backend, "--format", "json",
                ]
                output = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
                samples[backend].append(json.loads(output.stdout))
        raw.append({"dimensions": {"m": m, "n": n, "k": k}, "samples": samples})
        print(f"{m}x{n}x{k}")
        for backend in backends:
            gpu = statistics.median(row["gpu_median_ms"] for row in samples[backend])
            wall = statistics.median(row["median_ms"] for row in samples[backend])
            print(f"  {backend:14} GPU {gpu:8.4f} ms  wall {wall:8.4f} ms")
    if args.output:
        args.output.write_text(json.dumps({"rounds": args.rounds, "iterations": args.iterations,
                                          "warmup": args.warmup, "results": raw}, indent=2) + "\n",
                               encoding="utf-8")


if __name__ == "__main__":
    main()
