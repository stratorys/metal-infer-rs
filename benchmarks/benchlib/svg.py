"""The results chart: offline bars, serving curves per load, and curves against context length."""

import html
import math
from typing import Any

from benchlib.format import adaptive, level_for, load_tick, number, p50, server_loads

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
FONT = "-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif"
WIDTH = 1000
HEADER = 110
ROW_HEIGHT = 320
CHART_HEIGHT = 210
GAP = 60


def svg_text(
    x: float,
    y: float,
    value: str,
    anchor: str = "middle",
    size: int = 13,
    weight: str = "normal",
    fill: str = "#24292f",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{FONT}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(value)}</text>'
    )


def color(index: int) -> str:
    return COLORS[index % len(COLORS)]


def categorical(count: int) -> list[float]:
    return [(index + 0.5) / count for index in range(count)]


def logarithmic(values: list[int]) -> list[float]:
    """Positions on a log2 axis, with a margin on both sides."""
    low = math.log2(min(values))
    high = math.log2(max(values))
    if high == low:
        return [0.5 for _ in values]
    return [0.06 + 0.88 * (math.log2(value) - low) / (high - low) for value in values]


def offline_panels(document: dict[str, Any]) -> list[dict[str, Any]]:
    args = document["arguments"]
    results = document["results"]
    indices = [index for index in range(len(results)) if results[index].get("offline")]
    if not indices:
        return []
    panels = []
    for metric in [
        {"key": "pp_tps", "title": "Offline prefill (tok/s)"},
        {"key": "tg_tps", "title": f"Offline decode tg{args['generate']} (tok/s)"},
    ]:
        groups = []
        for prompt in args["prompt"]:
            entries = []
            for index in indices:
                values = next(
                    (
                        item[metric["key"]]
                        for item in results[index]["offline"]
                        if item["prompt"] == prompt
                    ),
                    None,
                )
                entries.append({"color": index, "values": values})
            groups.append({"label": f"prompt {prompt}", "entries": entries})
        panels.append({"kind": "bars", "title": metric["title"], "groups": groups})
    return panels


def context_panels(document: dict[str, Any]) -> list[dict[str, Any]]:
    loads = server_loads(document, context=True)
    if not loads:
        return []
    results = document["results"]
    positions = logarithmic([int(load["value"]) for load in loads])
    panels = []
    for metric in [
        {"key": "ttft_ms", "title": "TTFT (ms) by context length"},
        {"key": "tpot_ms", "title": "TPOT (ms) by context length"},
    ]:
        series = []
        for index in range(len(results)):
            intervals = [
                ((level_for(results[index], load) or {}).get("ci") or {}).get(metric["key"])
                for load in loads
            ]
            if not any(intervals):
                continue
            series.append(
                {
                    "color": index,
                    "values": [interval["median"] if interval else None for interval in intervals],
                    "low": [interval["low"] if interval else None for interval in intervals],
                    "high": [interval["high"] if interval else None for interval in intervals],
                }
            )
        if series:
            panels.append(
                {
                    "kind": "lines",
                    "title": metric["title"],
                    "labels": [load_tick(load) for load in loads],
                    "positions": positions,
                    "series": series,
                }
            )
    return panels


def load_panels(document: dict[str, Any]) -> list[dict[str, Any]]:
    loads = server_loads(document, context=False)
    if not loads:
        return []
    results = document["results"]
    panels = []
    for metric in [
        {"key": "ttft_ms", "title": "TTFT p50 (ms)"},
        {"key": "tpot_ms", "title": "TPOT p50 (ms)"},
        {"key": "itl_ms", "title": "ITL p50 (ms)"},
        {"key": "e2el_ms", "title": "E2EL p50 (ms)"},
        {"key": "request_tps", "title": "Requests/s"},
        {"key": "output_tps", "title": "Output tok/s"},
        {"key": "goodput_rps", "title": "Goodput (req/s)"},
    ]:
        if len(loads) == 1:
            entries = []
            for index, result in enumerate(results):
                level = level_for(result, loads[0])
                value = p50(level.get(metric["key"])) if level else None
                entries.append({"color": index, "values": {"p50": value} if value is not None else None})
            if any(entry["values"] for entry in entries):
                panels.append(
                    {
                        "kind": "bars",
                        "title": metric["title"],
                        "groups": [{"label": load_tick(loads[0]), "entries": entries}],
                        "whiskers": False,
                    }
                )
            continue
        series = []
        for index in range(len(results)):
            if not results[index].get("server"):
                continue
            values = [
                p50((level_for(results[index], load) or {}).get(metric["key"])) for load in loads
            ]
            if any(value is not None for value in values):
                series.append({"color": index, "values": values, "low": None, "high": None})
        if series:
            panels.append(
                {
                    "kind": "lines",
                    "title": metric["title"],
                    "labels": [load_tick(load) for load in loads],
                    "positions": categorical(len(loads)),
                    "series": series,
                }
            )
    return panels


def chart_rows(document: dict[str, Any]) -> list[list[dict[str, Any]]]:
    rows = [context_panels(document), offline_panels(document)]
    server = load_panels(document)
    rows.extend(server[index : index + 3] for index in range(0, len(server), 3))
    return [panels for panels in rows if panels]


def axis(
    body: list[str], left: float, baseline: float, width: float, height: float, maximum: float
) -> None:
    digits = 0 if maximum >= 100 else 1 if maximum >= 1 else 2
    for tick in range(6):
        y = baseline - height * tick / 5
        body.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="#d8dee4"/>'
        )
        body.append(
            svg_text(
                left - 6,
                y + 4,
                number(maximum * tick / 5, digits),
                anchor="end",
                size=11,
                fill="#57606a",
            )
        )


def bars(
    body: list[str],
    panel: dict[str, Any],
    left: float,
    baseline: float,
    width: float,
    height: float,
) -> None:
    stats = [
        entry["values"]
        for group in panel["groups"]
        for entry in group["entries"]
        if entry["values"]
    ]
    whiskers = panel.get("whiskers", True)
    maximum = max(
        max(values["p50"], values["mean"] + values["std"] if whiskers else values["p50"])
        for values in stats
    ) * 1.15 or 1.0
    axis(body, left, baseline, width, height, maximum)
    group_width = width / len(panel["groups"])
    for group_index in range(len(panel["groups"])):
        group = panel["groups"][group_index]
        entries = group["entries"]
        center = left + group_width * (group_index + 0.5)
        bar_width = min(46.0, group_width * 0.8 / len(entries))
        for position in range(len(entries)):
            values = entries[position]["values"]
            if not values:
                continue
            x = center + (position - (len(entries) - 1) / 2) * bar_width
            y = baseline - height * values["p50"] / maximum
            body.append(
                f'<rect x="{x - bar_width * 0.42:.1f}" y="{y:.1f}" width="{bar_width * 0.84:.1f}" '
                f'height="{baseline - y:.1f}" fill="{color(entries[position]["color"])}" rx="2"/>'
            )
            label_y = y
            if whiskers:
                low = baseline - height * max(values["mean"] - values["std"], 0.0) / maximum
                high = baseline - height * (values["mean"] + values["std"]) / maximum
                body.append(
                    f'<line x1="{x:.1f}" y1="{low:.1f}" x2="{x:.1f}" y2="{high:.1f}" '
                    'stroke="#24292f" stroke-width="1.5"/>'
                )
                label_y = min(y, high)
            body.append(svg_text(x, label_y - 6, chart_number(values["p50"]), size=10, weight="600"))
        body.append(svg_text(center, baseline + 18, group["label"], size=11))


def lines_panel(
    body: list[str],
    panel: dict[str, Any],
    left: float,
    baseline: float,
    width: float,
    height: float,
) -> None:
    values = [
        value
        for series in panel["series"]
        for key in ["values", "high"]
        for value in series[key] or []
        if value is not None
    ]
    maximum = max(values) * 1.15 or 1.0
    axis(body, left, baseline, width, height, maximum)
    xs = [left + width * position for position in panel["positions"]]
    for index in range(len(panel["labels"])):
        body.append(svg_text(xs[index], baseline + 18, panel["labels"][index], size=11))
    for series in panel["series"]:
        stroke = color(series["color"])
        present = [index for index in range(len(xs)) if series["values"][index] is not None]
        if series["low"] and series["high"]:
            banded = [
                index
                for index in present
                if series["low"][index] is not None and series["high"][index] is not None
            ]
            upper = [
                f"{xs[index]:.1f},{baseline - height * series['high'][index] / maximum:.1f}"
                for index in banded
            ]
            lower = [
                f"{xs[index]:.1f},{baseline - height * series['low'][index] / maximum:.1f}"
                for index in reversed(banded)
            ]
            if len(banded) > 1:
                body.append(
                    f'<polygon points="{" ".join(upper + lower)}" fill="{stroke}" fill-opacity="0.15"/>'
                )
            for index in banded:
                top = baseline - height * series["high"][index] / maximum
                bottom = baseline - height * series["low"][index] / maximum
                body.append(
                    f'<line x1="{xs[index]:.1f}" y1="{top:.1f}" x2="{xs[index]:.1f}" y2="{bottom:.1f}" '
                    f'stroke="{stroke}" stroke-width="1.5"/>'
                )
        points = [
            f"{xs[index]:.1f},{baseline - height * series['values'][index] / maximum:.1f}"
            for index in present
        ]
        body.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{stroke}" stroke-width="2"/>'
        )
        for index in present:
            value = series["values"][index]
            y = baseline - height * value / maximum
            body.append(f'<circle cx="{xs[index]:.1f}" cy="{y:.1f}" r="3.5" fill="{stroke}"/>')
            body.append(svg_text(xs[index], y - 8, chart_number(value), size=10, fill=stroke))


def chart_number(value: float) -> str:
    return number(value, 1 if value >= 1 else 2)


def subtitle(document: dict[str, Any]) -> str:
    args = document["arguments"]
    parts = [f"workload {document['workload']['name']}"]
    parts.append(f"prompt {', '.join(map(str, args['prompt']))} · output {args['generate']}")
    parts.append(f"{args['rounds']} rounds · {args['iterations']} offline iterations")
    if server_loads(document, context=False) or server_loads(document, context=True):
        parts.append(f"{args['requests']} server requests per load")
    if server_loads(document, context=True):
        parts.append("lines = median of per-round medians, band = 95 % bootstrap interval")
    if document["results"] and any(result.get("offline") for result in document["results"]):
        parts.append("bars = p50, whisker = mean ± std")
    if not document.get("finished_at"):
        status = document.get("status") or {}
        parts.append(f"in progress {status.get('done')}/{status.get('total')}")
    return " · ".join(parts)


def svg(document: dict[str, Any]) -> str:
    rows = chart_rows(document)
    args = document["arguments"]
    engines = [result["engine"] for result in document["results"]]
    height = HEADER + ROW_HEIGHT * max(len(rows), 1) + 36
    body = [
        svg_text(
            WIDTH / 2, 34, f"{args['model']} on {document['system']['chip']}", size=22, weight="600"
        ),
        svg_text(WIDTH / 2, 56, subtitle(document), size=12, fill="#57606a"),
    ]
    system = document["system"]
    power = "battery" if "Battery" in system.get("power_source", "") else system.get("power_source", "unknown power")
    dirty = "dirty tree" if system.get("metal_infer_dirty") else "clean tree"
    body.append(svg_text(WIDTH / 2, 100, f"{power} · {dirty} · commit {system.get('metal_infer_commit', 'unknown')[:8]}", size=11, fill="#57606a"))
    legend_width = 190
    legend_left = WIDTH / 2 - legend_width * len(engines) / 2
    for index in range(len(engines)):
        x = legend_left + legend_width * index
        body.append(
            f'<rect x="{x:.1f}" y="72" width="12" height="12" fill="{color(index)}" rx="2"/>'
        )
        body.append(svg_text(x + 18, 82, engines[index], anchor="start", size=12))
    if not rows:
        body.append(
            svg_text(
                WIDTH / 2, HEADER + ROW_HEIGHT / 2, "no measurement yet", size=14, fill="#57606a"
            )
        )
    for row_index in range(len(rows)):
        panels = rows[row_index]
        top = HEADER + row_index * ROW_HEIGHT + 40
        baseline = top + CHART_HEIGHT
        panel_width = (WIDTH - 90 - GAP * (len(panels) - 1)) / len(panels)
        for panel_index in range(len(panels)):
            panel = panels[panel_index]
            left = 70 + panel_index * (panel_width + GAP)
            body.append(
                svg_text(left + panel_width / 2, top - 16, panel["title"], size=14, weight="600")
            )
            if panel["kind"] == "bars":
                bars(body, panel, left, baseline, panel_width, CHART_HEIGHT)
            else:
                lines_panel(body, panel, left, baseline, panel_width, CHART_HEIGHT)
    failures = sum(level.get("failures", 0) for result in document["results"] for level in result.get("server") or [])
    short_outputs = sum(len(level.get("short_outputs") or []) for result in document["results"] for level in result.get("server") or [])
    body.append(svg_text(WIDTH / 2, height - 18, f"Server: {failures} failed requests · {short_outputs} short outputs · offline and server throughput are different measurements", size=11, fill="#57606a"))
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
            f'viewBox="0 0 {WIDTH} {height}" role="img">',
            f"<title>{html.escape(args['model'])} benchmark</title>",
            "<desc>Throughput and latency by engine: against context length, per offline prompt length, "
            "and per server load.</desc>",
            f'<rect width="{WIDTH}" height="{height}" fill="#ffffff"/>',
            *body,
            "</svg>",
            "",
        ]
    )
