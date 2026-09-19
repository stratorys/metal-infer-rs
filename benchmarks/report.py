#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Generate deterministic SVG and Markdown benchmark reports."""

import argparse
import html
import json
import pathlib
import re
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "benchmarks" / "results"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


class ReportError(RuntimeError):
    """A benchmark report could not be generated."""


def benchmark_kind(document: dict[str, Any]) -> str:
    kind = document.get("kind")
    if kind in {"kernels", "model"}:
        return str(kind)
    suite = document.get("suite", {})
    if isinstance(suite, dict) and "shapes" in suite:
        return "kernels"
    if isinstance(suite, dict) and "prompt_tokens" in suite:
        return "model"
    raise ReportError("cannot determine benchmark kind")


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "unknown"


def default_run_directory(
    document: dict[str, Any],
    results_root: pathlib.Path = RESULTS_ROOT,
) -> pathlib.Path:
    system = document.get("system", {})
    chip = slug(str(system.get("chip", "unknown-device")))
    generated = str(document.get("generated_at", "unknown-date"))
    timestamp = re.sub(r"[^0-9]", "", generated)[:14] or "unknown-date"
    commit = str(system.get("metal_infer_commit") or "uncommitted")[:7]
    dirty = "-dirty" if system.get("metal_infer_dirty", False) else ""
    kind = benchmark_kind(document)
    return results_root / chip / f"{timestamp}-{commit}{dirty}-{kind}"


def svg_text(
    x: float,
    y: float,
    value: str,
    anchor: str = "middle",
    size: int = 13,
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="#24292f">'
        f"{html.escape(value)}</text>"
    )


def svg_document(title: str, description: str, body: list[str], height: int) -> str:
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1000" '
            f'height="{height}" viewBox="0 0 1000 {height}" role="img">',
            f"<title>{html.escape(title)}</title>",
            f"<desc>{html.escape(description)}</desc>",
            '<rect width="1000" height="100%" fill="#ffffff"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def kernel_svg(results: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    backends: list[str] = []
    for result in results:
        dimensions = result.get("dimensions")
        if not isinstance(dimensions, dict):
            continue
        shape = (
            int(dimensions["m"]),
            int(dimensions["n"]),
            int(dimensions["k"]),
        )
        grouped.setdefault(shape, []).append(result)
        backend = str(result["backend"])
        if backend not in backends:
            backends.append(backend)
    if not grouped or not backends:
        raise ReportError("kernel report has no plottable results")
    maximum = max(float(result["throughput"]) for result in results)
    width = 820
    left = 110
    top = 75
    chart_height = 310
    baseline = top + chart_height
    group_width = width / len(grouped)
    bar_width = min(55.0, group_width / (len(backends) + 1))
    body = [svg_text(500, 35, "FP16 matrix multiplication", size=22, weight="600")]
    for tick in range(6):
        value = maximum * tick / 5
        y = baseline - chart_height * tick / 5
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="930" y2="{y:.1f}" stroke="#d8dee4"/>')
        body.append(svg_text(left - 12, y + 5, f"{value:.2f}", anchor="end", size=12))
    for group_index, (shape, rows) in enumerate(grouped.items()):
        center = left + group_width * (group_index + 0.5)
        by_backend = {str(row["backend"]): row for row in rows}
        for backend_index, backend in enumerate(backends):
            row = by_backend.get(backend)
            if row is None:
                continue
            value = float(row["throughput"])
            height = chart_height * value / maximum if maximum > 0 else 0
            x = center + (backend_index - (len(backends) - 1) / 2) * bar_width - bar_width * 0.4
            y = baseline - height
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.8:.1f}" '
                f'height="{height:.1f}" fill="{COLORS[backend_index % len(COLORS)]}" rx="2"/>'
            )
            body.append(svg_text(x + bar_width * 0.4, y - 7, f"{value:.3f}", size=11))
        body.append(svg_text(center, baseline + 24, f"{shape[0]}×{shape[1]}×{shape[2]}", size=12))
    body.append(svg_text(28, 230, "TFLOP/s", size=13, weight="600"))
    legend_x = 250
    for index, backend in enumerate(backends):
        x = legend_x + index * 230
        body.append(f'<rect x="{x}" y="425" width="16" height="16" fill="{COLORS[index % len(COLORS)]}" rx="2"/>')
        body.append(svg_text(x + 24, 438, backend, anchor="start", size=13))
    return svg_document(
        "Kernel benchmark results",
        "Grouped FP16 matrix multiplication throughput by shape and backend.",
        body,
        470,
    )


def model_svg(results: list[dict[str, Any]]) -> str:
    if not results:
        raise ReportError("model report has no results")
    body = [svg_text(500, 35, "Qwen3 model throughput", size=22, weight="600")]
    phases = [("prefill", "Prefill"), ("decode", "Decode")]
    for panel_index, (phase_key, phase_name) in enumerate(phases):
        panel_left = 65 + panel_index * 490
        panel_width = 400
        top = 90
        chart_height = 260
        baseline = top + chart_height
        values = [float(result[phase_key]["tokens_per_second"]) for result in results]
        maximum = max(values)
        body.append(svg_text(panel_left + panel_width / 2, 70, phase_name, size=17, weight="600"))
        for tick in range(6):
            value = maximum * tick / 5
            y = baseline - chart_height * tick / 5
            body.append(
                f'<line x1="{panel_left}" y1="{y:.1f}" x2="{panel_left + panel_width}" '
                f'y2="{y:.1f}" stroke="#d8dee4"/>'
            )
            body.append(svg_text(panel_left - 8, y + 5, f"{value:.0f}", anchor="end", size=11))
        slot = panel_width / len(results)
        for index, result in enumerate(results):
            value = float(result[phase_key]["tokens_per_second"])
            bar_height = chart_height * value / maximum if maximum > 0 else 0
            bar_width = min(90.0, slot * 0.6)
            x = panel_left + slot * (index + 0.5) - bar_width / 2
            y = baseline - bar_height
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="{COLORS[index % len(COLORS)]}" rx="2"/>'
            )
            body.append(svg_text(x + bar_width / 2, y - 7, f"{value:.1f}", size=12))
            body.append(svg_text(x + bar_width / 2, baseline + 22, str(result["backend"]), size=12))
    body.append(svg_text(22, 225, "tokens/s", size=13, weight="600"))
    return svg_document(
        "Model benchmark results",
        "Separate Qwen3 prefill and decode throughput by backend.",
        body,
        410,
    )


def kernel_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Shape (M×N×K) | Backend | Mean ms | GPU mean ms | TFLOP/s | Relative |",
        "|---|---|---:|---:|---:|---:|",
    ]
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for result in results:
        dimensions = result.get("dimensions")
        if isinstance(dimensions, dict):
            shape = (int(dimensions["m"]), int(dimensions["n"]), int(dimensions["k"]))
            grouped.setdefault(shape, []).append(result)
    for shape, rows in grouped.items():
        metal = next((row for row in rows if row["backend"] == "metal-infer"), None)
        baseline = float(metal["throughput"]) if metal is not None else 0.0
        for row in rows:
            throughput = float(row["throughput"])
            relative = throughput / baseline if baseline > 0 else 0.0
            gpu_mean = row.get("gpu_mean_ms")
            gpu_mean_text = f"{float(gpu_mean):.3f}" if gpu_mean is not None else "—"
            lines.append(
                f"| {shape[0]}×{shape[1]}×{shape[2]} | {row['backend']} | "
                f"{float(row['mean_ms']):.3f} | {gpu_mean_text} | "
                f"{throughput:.4f} | {relative:.2f}× |"
            )
    return "\n".join(lines)


def model_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Backend | Prefill tokens/s | Prefill wall/GPU ms | Decode tokens/s | Decode wall/GPU ms | Memory |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        memory = "not comparable"
        if "peak_memory_gb" in result:
            memory = f"{float(result['peak_memory_gb']):.3f} GB peak"
        elif "allocated_bytes" in result:
            memory = f"{int(result['allocated_bytes']) / 1_000_000_000:.3f} GB allocated"
            growth = result.get("allocation_growth_bytes")
            if growth is not None:
                memory += f", +{int(growth) / 1_000_000:.3f} MB measured"
        prefill = result["prefill"]
        decode = result["decode"]
        prefill_timing = "—"
        decode_timing = "—"
        if "wall_mean_ms" in prefill and "gpu_mean_ms" in prefill:
            prefill_timing = (
                f"{float(prefill['wall_mean_ms']):.3f}/"
                f"{float(prefill['gpu_mean_ms']):.3f}"
            )
        if "wall_mean_ms" in decode and "gpu_mean_ms" in decode:
            decode_timing = (
                f"{float(decode['wall_mean_ms']):.3f}/"
                f"{float(decode['gpu_mean_ms']):.3f}"
            )
        lines.append(
            f"| {result['backend']} | "
            f"{float(prefill['tokens_per_second']):.3f} | {prefill_timing} | "
            f"{float(decode['tokens_per_second']):.3f} | {decode_timing} | {memory} |"
        )
    return "\n".join(lines)


def run_readme(document: dict[str, Any]) -> str:
    kind = benchmark_kind(document)
    system = document.get("system", {})
    suite = document.get("suite", {})
    results = document.get("results", [])
    title = "Kernel benchmark" if kind == "kernels" else "Model benchmark"
    table = kernel_table(results) if kind == "kernels" else model_table(results)
    configuration = ", ".join(f"{key}={value}" for key, value in suite.items() if key != "shapes")
    commit = str(system.get("metal_infer_commit") or "uncommitted")
    dirty = " (dirty working tree)" if system.get("metal_infer_dirty", False) else ""
    memory_note = ""
    if kind == "model":
        memory_note = (
            "\nMemory values are not directly comparable: metal-infer reports Metal "
            "allocated memory while MLX reports peak memory.\n"
        )
    return (
        f"# {title}\n\n"
        f"Generated at `{document.get('generated_at', 'unknown')}` on "
        f"**{system.get('chip', 'unknown device')}**.\n\n"
        f"- macOS: `{system.get('macos', 'unknown')}`\n"
        f"- Rust: `{system.get('rustc', 'unknown')}`\n"
        f"- metal-infer commit: `{commit}`{dirty}\n"
        f"- configuration: `{configuration}`\n\n"
        f"![{title} results](results.svg)\n\n"
        f"{table}\n"
        f"{memory_note}\n"
        "Raw measurements: [results.json](results.json).\n\n"
        "The benchmark methodology and reproduction commands are documented in "
        "the [benchmark guide](../../../README.md).\n"
    )


def generate_run_report(document: dict[str, Any], run_directory: pathlib.Path) -> None:
    results = document.get("results")
    if not isinstance(results, list):
        raise ReportError("benchmark results must be an array")
    kind = benchmark_kind(document)
    svg = kernel_svg(results) if kind == "kernels" else model_svg(results)
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "results.svg").write_text(svg, encoding="utf-8")
    (run_directory / "README.md").write_text(run_readme(document), encoding="utf-8")


def rebuild_index(results_root: pathlib.Path = RESULTS_ROOT) -> None:
    rows = []
    for path in sorted(results_root.glob("*/*/results.json"), reverse=True):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            system = document.get("system", {})
            rows.append(
                (
                    str(document.get("generated_at", "unknown")),
                    str(system.get("chip", "unknown")),
                    benchmark_kind(document),
                    path.parent.relative_to(results_root).as_posix(),
                )
            )
        except (json.JSONDecodeError, OSError, ReportError):
            continue
    lines = [
        "# Benchmark results",
        "",
        "Generated benchmark runs, grouped by machine and execution.",
        "",
        "| Generated | Device | Type | Report |",
        "|---|---|---|---|",
    ]
    for generated, chip, kind, relative in rows:
        lines.append(f"| `{generated}` | {chip} | {kind} | [open]({relative}/README.md) |")
    lines.append("")
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    try:
        results_path = args.results if args.results.is_absolute() else ROOT / args.results
        document = json.loads(results_path.read_text(encoding="utf-8"))
        generate_run_report(document, results_path.parent)
        rebuild_index()
        print(f"generated report in {results_path.parent}")
    except (OSError, json.JSONDecodeError, ReportError, ValueError) as report_error:
        print(f"report failed: {report_error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
