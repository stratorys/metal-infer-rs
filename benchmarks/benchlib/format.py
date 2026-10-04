"""Number and load formatting shared by the Markdown and SVG reports."""

from typing import Any


def number(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def adaptive(value: float) -> str:
    return number(value, 0 if value >= 100 else 1 if value >= 1 else 2)


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f} %"


def p50(stats: dict[str, float] | None) -> float | None:
    return stats["p50"] if stats else None


def stats_cell(stats: dict[str, float] | None) -> str:
    if not stats:
        return "—"
    return (
        f"{number(stats['p50'])} / {number(stats['p90'])} / {number(stats['p99'])} "
        f"({number(stats['mean'])} ± {number(stats['std'])})"
    )


def interval_cell(interval: dict[str, Any] | None) -> str:
    if not interval:
        return "—"
    return (
        f"{adaptive(interval['median'])} [{adaptive(interval['low'])}–{adaptive(interval['high'])}]"
    )


def load_label(load: dict[str, Any]) -> str:
    if load["type"] == "concurrency":
        return f"Concurrency {load['value']}"
    if load["type"] == "context":
        return f"Context {load['value']}"
    return (
        "All requests at once"
        if load["value"] == "inf"
        else f"{load['value']} requests/s (Poisson)"
    )


def load_tick(load: dict[str, Any]) -> str:
    if load["type"] == "concurrency":
        return f"c={load['value']}"
    if load["type"] == "context":
        value = int(load["value"])
        return f"{value // 1024}k" if value >= 1024 and value % 1024 == 0 else str(value)
    return "∞" if load["value"] == "inf" else f"{load['value']}/s"


def server_loads(document: dict[str, Any], context: bool) -> list[dict[str, Any]]:
    """Server loads in first-seen order, either the context loads or all the others."""
    loads: list[dict[str, Any]] = []
    for result in document["results"]:
        for level in result.get("server") or []:
            if (level["load"]["type"] == "context") == context and level["load"] not in loads:
                loads.append(level["load"])
    return loads


def level_for(result: dict[str, Any], load: dict[str, Any]) -> dict[str, Any] | None:
    return next((level for level in result.get("server") or [] if level["load"] == load), None)
