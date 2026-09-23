"""Distributions, bootstrap confidence intervals and verdicts."""

import random
import statistics
from typing import Any

PERCENTILES = [50, 90, 99]
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0


def percentile(ordered: list[float], value: float) -> float:
    position = (len(ordered) - 1) * value / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    result = {
        "mean": statistics.fmean(ordered),
        "std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "max": ordered[-1],
    }
    for value in PERCENTILES:
        result[f"p{value}"] = percentile(ordered, value)
    return result


def confidence(round_medians: list[float]) -> dict[str, Any] | None:
    """Median of the per-round medians with a 95 % bootstrap interval."""
    if not round_medians:
        return None
    generator = random.Random(BOOTSTRAP_SEED)
    draws = sorted(
        statistics.median(generator.choices(round_medians, k=len(round_medians)))
        for _ in range(BOOTSTRAP_DRAWS)
    )
    return {
        "median": statistics.median(round_medians),
        "low": percentile(draws, 2.5),
        "high": percentile(draws, 97.5),
        "rounds": len(round_medians),
    }


def verdict(
    candidate: dict[str, Any] | None, reference: dict[str, Any] | None, higher_is_better: bool
) -> str | None:
    """faster or slower only when the two intervals do not overlap."""
    if not candidate or not reference:
        return None
    if candidate["low"] > reference["high"]:
        return "faster" if higher_is_better else "slower"
    if candidate["high"] < reference["low"]:
        return "slower" if higher_is_better else "faster"
    return "tie (noise)"
