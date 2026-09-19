#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx>=0.30"]
# ///
"""Manual MLX baseline for metal-infer's row-major FP16 matmul benchmark."""

import argparse
import json
import statistics
import time

import mlx.core as mx


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if min(args.m, args.n, args.k, args.iterations) <= 0 or args.warmup < 0:
        raise ValueError("dimensions and iterations must be positive")

    left = mx.full((args.m, args.k), 0.01, dtype=mx.float16)
    weight = mx.full((args.n, args.k), 0.02, dtype=mx.float16)

    for _ in range(args.warmup):
        mx.eval(left @ weight.T)
    mx.synchronize()

    samples = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        mx.eval(left @ weight.T)
        mx.synchronize()
        samples.append(time.perf_counter() - started)

    samples.sort()
    mean = statistics.fmean(samples)
    p95 = samples[round((len(samples) - 1) * 0.95)]
    operations = 2 * args.m * args.n * args.k
    print(
        json.dumps(
            {
                "benchmark": f"mlx_matmul_f16[{args.m},{args.n},{args.k}]",
                "iterations": args.iterations,
                "mean_ms": mean * 1000,
                "median_ms": statistics.median(samples) * 1000,
                "p95_ms": p95 * 1000,
                "throughput": operations / mean / 1e12,
                "throughput_unit": "TFLOP/s",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
