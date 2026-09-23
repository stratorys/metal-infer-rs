"""Aggregation of progress.jsonl into results, the run README, and the results index."""

import argparse
import json
import pathlib
import shutil
import statistics
from typing import Any

from benchlib.common import RESULTS, ROOT, SCHEMA_VERSION, slug
from benchlib.engines import offline_commands, server_command
from benchlib.format import (
    interval_cell,
    level_for,
    load_label,
    number,
    p50,
    percent,
    server_loads,
    stats_cell,
)
from benchlib.plan import engines, modes
from benchlib.progress import run_directory
from benchlib.stats import confidence, summary, verdict
from benchlib.svg import svg

HIGHER_IS_BETTER = {
    "pp_tps": True,
    "tg_tps": True,
    "output_tps": True,
    "ttft_ms": False,
    "tpot_ms": False,
    "e2el_ms": False,
}
CONTEXT_METRICS = ["ttft_ms", "tpot_ms", "e2el_ms"]
LOAD_METRICS = ["ttft_ms", "tpot_ms"]
VERDICT_NAMES = {
    "pp_tps": "prefill",
    "tg_tps": "decode",
    "ttft_ms": "TTFT",
    "tpot_ms": "TPOT",
    "output_tps": "output",
}


def median_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def offline_summary(
    prompt: int,
    records: list[dict[str, Any]],
    facts: dict[str, Any],
    args: argparse.Namespace,
    bandwidth: float | None,
) -> dict[str, Any]:
    samples = [sample for record in records for sample in record["data"]["samples"]]
    pp = summary([sample["pp_tps"] for sample in samples if sample.get("pp_tps") is not None])
    tg = summary([sample["tg_tps"] for sample in samples if sample.get("tg_tps") is not None])
    bytes_per_token = facts["weight_bytes_read_per_token"] + facts["kv_bytes_per_token"] * (
        prompt + args.generate / 2
    )
    joules = [
        record["data"]["joules_per_token"]
        for record in records
        if record["data"]["joules_per_token"]
    ]
    memory = [
        sample["peak_memory_gb"] for sample in samples if sample.get("peak_memory_gb") is not None
    ]
    ci = {}
    for metric in ["pp_tps", "tg_tps"]:
        ci[metric] = confidence(
            [
                statistics.median(values)
                for record in records
                if (
                    values := [
                        sample[metric]
                        for sample in record["data"]["samples"]
                        if sample.get(metric) is not None
                    ]
                )
            ]
        )
    return {
        "prompt": prompt,
        "pp_tps": pp,
        "tg_tps": tg,
        "ci": ci,
        "verdict": {},
        "mbu": bytes_per_token * tg["p50"] / (bandwidth * 1e9) if bandwidth else None,
        "mfu": (
            2 * facts["parameters_read_per_token"] * pp["p50"] / (args.peak_tflops * 1e12)
            if args.peak_tflops
            else None
        ),
        "peak_memory_gb": max(memory) if memory else None,
        "joules_per_token": statistics.fmean(joules) if joules else None,
        "plan": next((sample["plan"] for sample in samples if "plan" in sample), None),
    }


def server_summary(load: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    runs = [record["data"] for record in records]
    requests = [item for run in runs for item in run["requests"] if "error" not in item]
    multi_token_requests = [item for item in requests if item["output_tokens"] > 1]
    joules = [run["joules_per_token"] for run in runs if run["joules_per_token"]]
    ci = {}
    for metric in CONTEXT_METRICS if load["type"] == "context" else LOAD_METRICS:
        medians = []
        for run in runs:
            value = median_or_none(
                [
                    item.get(metric)
                    for item in run["requests"]
                    if "error" not in item and (metric != "tpot_ms" or item["output_tokens"] > 1)
                ]
            )
            if value is not None:
                medians.append(value)
        ci[metric] = confidence(medians)
    if load["type"] != "context":
        ci["output_tps"] = confidence([run["output_tps"] for run in runs])
    return {
        "load": load,
        "ttft_ms": summary([item["ttft_ms"] for item in requests]),
        "tpot_ms": summary([item["tpot_ms"] for item in multi_token_requests]),
        "itl_ms": summary([value for item in requests for value in item["itl_ms"]]),
        "e2el_ms": summary([item["e2el_ms"] for item in requests]),
        "request_tps": summary([run["request_tps"] for run in runs]),
        "input_tps": summary([run["input_tps"] for run in runs]),
        "output_tps": summary([run["output_tps"] for run in runs]),
        "goodput_rps": summary([run["goodput_rps"] for run in runs]),
        "failures": sum(len(run["failures"]) for run in runs),
        "short_outputs": [value for run in runs for value in run["short_outputs"]],
        "joules_per_token": statistics.fmean(joules) if joules else None,
        "ci": ci,
        "verdict": {},
        "prompt_tokens": sorted(
            {item["prompt_tokens"] for item in requests if item["prompt_tokens"] is not None}
        ),
        "calibrated_tokens": next(
            (run["calibration"]["tokens"] for run in runs if run.get("calibration")), None
        ),
    }


def verdicts(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, str | None]:
    return {
        metric: verdict(candidate[metric], reference.get(metric), HIGHER_IS_BETTER[metric])
        for metric in candidate
        if metric in reference
    }


def add_verdicts(results: list[dict[str, Any]]) -> None:
    """Each engine against the first one of --engines."""
    if len(results) < 2:
        return
    reference = results[0]
    for result in results[1:]:
        for level in result["server"]:
            base = level_for(reference, level["load"])
            if base:
                level["verdict"] = verdicts(level["ci"], base["ci"])
        for values in result["offline"]:
            base = next(
                (item for item in reference["offline"] if item["prompt"] == values["prompt"]), None
            )
            if base:
                values["verdict"] = verdicts(values["ci"], base["ci"])


def planned_loads(header: dict[str, Any], mode: str) -> list[dict[str, str]]:
    loads: list[dict[str, str]] = []
    for measurement in header["measurements"]:
        if measurement["mode"] == mode and measurement["load"] not in loads:
            loads.append(measurement["load"])
    return loads


def aggregate(header: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    args = argparse.Namespace(**header["arguments"])
    results = []
    for engine in engines(args):
        mine = [record for record in records if record["engine"] == engine.label]
        result: dict[str, Any] = {
            "engine": engine.label,
            "kind": engine.kind,
            "offline": [],
            "server": [],
        }
        for load in planned_loads(header, "offline"):
            chosen = [
                record for record in mine if record["mode"] == "offline" and record["load"] == load
            ]
            if chosen:
                result["offline"].append(
                    offline_summary(
                        int(load["value"]),
                        chosen,
                        header["model"],
                        args,
                        header["peak_bandwidth_gbs"],
                    )
                )
        for load in planned_loads(header, "server"):
            chosen = [
                record for record in mine if record["mode"] == "server" and record["load"] == load
            ]
            if chosen:
                result["server"].append(server_summary(load, chosen))
        results.append(result)
    add_verdicts(results)
    return results


def build_document(header: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **header,
        "generated_at": header["started_at"],
        "status": {"done": len(records), "total": len(header["measurements"])},
        "results": aggregate(header, records),
    }


def upgrade_v3(document: dict[str, Any]) -> None:
    args = document["arguments"]
    prompt = args["prompt"]
    generate = args["generate"]
    args["prompt"] = [prompt]
    defaults = {
        "workload": "synthetic",
        "num_prompts": None,
        "request_rate": [],
        "seed": None,
        "slo_ttft_ms": None,
        "slo_tpot_ms": None,
    }
    for key in defaults:
        args.setdefault(key, defaults[key])
    for result in document["results"]:
        values = result.get("offline")
        if values:
            kept = {
                key: values.get(key)
                for key in ["mbu", "mfu", "peak_memory_gb", "joules_per_token", "plan"]
            }
            result["offline"] = [
                {
                    "prompt": prompt,
                    "pp_tps": values[f"pp{prompt}_tps"],
                    "tg_tps": values[f"tg{generate}_tps"],
                    **kept,
                }
            ]
        result["server"] = [
            {
                **level,
                "load": {"type": "concurrency", "value": level["concurrency"]},
                "request_tps": None,
                "input_tps": None,
                "goodput_rps": None,
                "failures": 0,
            }
            for level in result.get("server", [])
        ]
    document["workload"] = {"name": "synthetic"}


def upgrade_v4(document: dict[str, Any]) -> None:
    args = document["arguments"]
    defaults = {"suite": None, "gguf": None, "resume": None}
    for key in defaults:
        args.setdefault(key, defaults[key])
    document["system"].setdefault("llama_cpp", None)
    document["started_at"] = document["generated_at"]
    document["finished_at"] = document["generated_at"]
    for result in document["results"]:
        result.setdefault(
            "kind",
            "metal-infer" if result["engine"].startswith("metal-infer") else result["engine"],
        )
        result["offline"] = result.get("offline") or []
        result["server"] = result.get("server") or []
        for values in result["offline"]:
            values.setdefault("ci", {})
            values.setdefault("verdict", {})
        for level in result["server"]:
            level["load"] = {"type": level["load"]["type"], "value": str(level["load"]["value"])}
            level.setdefault("ci", {})
            level.setdefault("verdict", {})
            level.setdefault("prompt_tokens", [])
            level.setdefault("calibrated_tokens", None)


def upgrade(document: dict[str, Any]) -> dict[str, Any]:
    version = document.get("schema_version", 3)
    if version < 4:
        upgrade_v3(document)
    if version < 5:
        upgrade_v4(document)
    document["schema_version"] = SCHEMA_VERSION
    return document


def verdict_text(level: dict[str, Any], metrics: list[str]) -> str:
    found = level.get("verdict") or {}
    return " · ".join(
        f"{VERDICT_NAMES[metric]} {found[metric]}" for metric in metrics if found.get(metric)
    )


def summary_section(document: dict[str, Any]) -> list[str]:
    loads = server_loads(document, context=True)
    if not loads:
        return []
    args = document["arguments"]
    results = document["results"]
    reference = results[0]["engine"]
    lines = [
        "## Summary",
        "",
        f"Chat completions through each engine's OpenAI-compatible server, one request at a time, prompts "
        f"calibrated to exactly N prompt tokens, {args['generate']} generated tokens. Each cell is the median "
        f"of the per-round medians [95 % bootstrap interval] over {args['rounds']} rounds of {args['requests']} "
        f"requests. Verdicts compare against {reference}: faster or slower only when the intervals do not overlap.",
        "",
        "| context | engine | TTFT ms | TPOT ms | E2EL ms | verdict |",
        "|---|---|---|---|---|---|",
    ]
    warnings = []
    for load in loads:
        for result in results:
            level = level_for(result, load)
            if level is None:
                continue
            ci = level.get("ci") or {}
            text = (
                "reference"
                if result is results[0]
                else verdict_text(level, ["ttft_ms", "tpot_ms"]) or "—"
            )
            lines.append(
                f"| {load['value']} | {result['engine']} | {interval_cell(ci.get('ttft_ms'))} "
                f"| {interval_cell(ci.get('tpot_ms'))} | {interval_cell(ci.get('e2el_ms'))} | {text} |"
            )
            if level["failures"] or level["short_outputs"]:
                warnings.append(
                    f"- ⚠ {result['engine']}, context {load['value']}: {level['failures']} failed, "
                    f"{len(level['short_outputs'])} outputs shorter than requested"
                )
    return lines + [""] + (warnings + [""] if warnings else [])


def offline_section(document: dict[str, Any]) -> list[str]:
    args = document["arguments"]
    offline = [result for result in document["results"] if result.get("offline")]
    if not offline:
        return []
    lines = [
        "## Offline (diagnostic)",
        "",
        f"Each engine's own benchmark tool, {args['generate']} generated tokens after the prompt, "
        f"{args['iterations']} iterations per run.",
        "",
        f"| engine | prompt | pp tok/s p50 (p90/p99) | tg{args['generate']} tok/s p50 (p90/p99) "
        "| mean ± std tg | MBU | MFU | peak GB | J/token | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in offline:
        for values in result["offline"]:
            pp = values["pp_tps"]
            tg = values["tg_tps"]
            lines.append(
                f"| {result['engine']} | {values['prompt']} "
                f"| {number(pp['p50'])} ({number(pp['p90'])}/{number(pp['p99'])}) "
                f"| {number(tg['p50'])} ({number(tg['p90'])}/{number(tg['p99'])}) "
                f"| {number(tg['mean'])} ± {number(tg['std'])} "
                f"| {percent(values['mbu'])} | {percent(values['mfu'])} "
                f"| {number(values['peak_memory_gb'], 2)} | {number(values['joules_per_token'], 3)} "
                f"| {verdict_text(values, ['pp_tps', 'tg_tps']) or '—'} |"
            )
    return lines + [""]


def server_section(document: dict[str, Any]) -> list[str]:
    args = document["arguments"]
    server = [result for result in document["results"] if result.get("server")]
    if not server:
        return []
    lines = []
    loads = server_loads(document, context=False)
    if loads:
        slo = (
            f"Goodput: completed requests per second with TTFT ≤ {number(args['slo_ttft_ms'], 0)} ms "
            f"and TPOT ≤ {number(args['slo_tpot_ms'], 0)} ms."
            if args.get("slo_ttft_ms")
            else ""
        )
        lines += [
            "## Server",
            "",
            f"Workload: {workload_line(document)}.",
            "",
            f"Latencies in milliseconds: p50 / p90 / p99 (mean ± std). {slo}",
            "",
        ]
    for load in loads:
        lines += [
            f"### {load_label(load)}",
            "",
            "| engine | TTFT | TPOT | ITL | E2EL | req/s | output tok/s | goodput req/s | failed | J/token | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for result in server:
            level = level_for(result, load)
            if level is None:
                continue
            cells = [stats_cell(level[key]) for key in ["ttft_ms", "tpot_ms", "itl_ms", "e2el_ms"]]
            lines.append(
                f"| {result['engine']} | {' | '.join(cells)} | {number(p50(level['request_tps']), 2)} "
                f"| {number(p50(level['output_tps']))} | {number(p50(level['goodput_rps']), 2)} "
                f"| {level['failures']} | {number(level['joules_per_token'], 3)} "
                f"| {verdict_text(level, ['ttft_ms', 'tpot_ms', 'output_tps']) or '—'} |"
            )
            if level["short_outputs"]:
                lines.append(
                    f"| ⚠ {result['engine']}: {len(level['short_outputs'])} outputs shorter "
                    "than requested | | | | | | | | | | |"
                )
        lines.append("")
    contexts = server_loads(document, context=True)
    if contexts:
        lines += [
            "## Long context: latency detail",
            "",
            "Milliseconds, p50 / p90 / p99 (mean ± std) over every request of every round.",
            "",
            "| context | engine | prompt tokens | TTFT | TPOT | E2EL | failed |",
            "|---|---|---|---|---|---|---|",
        ]
        for load in contexts:
            for result in server:
                level = level_for(result, load)
                if level is None:
                    continue
                observed = ", ".join(str(value) for value in level["prompt_tokens"]) or "—"
                lines.append(
                    f"| {load['value']} | {result['engine']} | {observed} | {stats_cell(level['ttft_ms'])} "
                    f"| {stats_cell(level['tpot_ms'])} | {stats_cell(level['e2el_ms'])} "
                    f"| {level['failures']} |"
                )
        lines.append("")
    return lines


def workload_line(document: dict[str, Any]) -> str:
    args = document["arguments"]
    workload = document["workload"]
    if workload["name"] == "sharegpt":
        return (
            f"ShareGPT, {workload['num_prompts']} conversations per rate (seed {workload['seed']}), "
            f"input tokens p50 {number(p50(workload['input_tokens']), 0)}, "
            f"output tokens p50 {number(p50(workload['output_tokens']), 0)}"
        )
    calibrated = any(
        level.get("calibrated_tokens")
        for result in document["results"]
        for level in result.get("server") or []
    )
    length = f"exactly {args['prompt'][0]}" if calibrated else f"about {args['prompt'][0]}"
    return (
        f"synthetic: fixed English text of {length} prompt tokens, "
        f"{args['generate']} generated tokens, {args['requests']} requests per level"
    )


def markdown(document: dict[str, Any], image: bool = False) -> str:
    args = document["arguments"]
    system = document["system"]
    power = (system.get("power_source") or "").removeprefix("Now drawing from ").strip("'")
    versions = f"mlx-lm {system['mlx_lm']}, mlx {system['mlx']}"
    if system.get("llama_cpp"):
        versions += f", llama.cpp {system['llama_cpp']}"
    lines = [
        f"# {system['chip']} — {args['model']}",
        "",
        f"{document['started_at']} · commit `{system['metal_infer_commit']}`"
        f"{' (dirty)' if system['metal_infer_dirty'] else ''} · "
        f"workload {document['workload']['name']} · {args['rounds']} rounds",
        "",
        f"macOS {system['macos']} · {system['memory_bytes'] / 2**30:.0f} GB · {power or 'unknown power'} · {versions}",
        "",
    ]
    if not document.get("finished_at"):
        status = document.get("status") or {}
        lines += [
            f"> **Run in progress: {status.get('done')}/{status.get('total')} measurements.** "
            "The figures cover the finished measurements only.",
            "",
        ]
    if image:
        lines += ["![Benchmark results](results.svg)", ""]
    lines += summary_section(document) + offline_section(document) + server_section(document)
    return "\n".join(lines) + "\n"


def distribution(stats: dict[str, float] | None) -> str:
    if not stats:
        return "—"
    return (
        f"p50 {number(stats['p50'], 0)}, p90 {number(stats['p90'], 0)}, "
        f"max {number(stats.get('max'), 0)}, mean {number(stats['mean'], 0)}"
    )


def row(name: str, value: str) -> dict[str, str]:
    return {"name": name, "value": value}


def display_path(value: str) -> str:
    path = pathlib.Path(value)
    if path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    home = pathlib.Path.home()
    if path.is_relative_to(home):
        return str(pathlib.Path("~") / path.relative_to(home))
    return value


def display_command(command: list[str]) -> str:
    return " ".join(display_path(part) for part in command)


def workload_rows(document: dict[str, Any], args: argparse.Namespace) -> list[dict[str, str]]:
    workload = document["workload"]
    if workload["name"] == "sharegpt":
        filters = workload["filters"]
        return [
            row(
                "server workload",
                f"ShareGPT V3, first human turn as prompt, first assistant turn length "
                f"as output length ([source]({workload['source']}))",
            ),
            row("dataset sha256", f"`{workload['sha256']}`"),
            row(
                "sampling",
                f"{workload['num_prompts']} conversations, shuffled with seed {workload['seed']}",
            ),
            row(
                "filters",
                f"prompt {filters['min_prompt_tokens']}–{filters['max_prompt_tokens']} tokens, "
                f"output ≥ {filters['min_output_tokens']}, prompt + output ≤ {filters['max_total_tokens']}",
            ),
            row("input tokens", distribution(workload["input_tokens"])),
            row("output tokens", distribution(workload["output_tokens"])),
            row(
                "request rates",
                ", ".join(
                    f"{rate} req/s" if rate != "inf" else "all at once"
                    for rate in args.request_rate
                ),
            ),
            row("arrivals", f"Poisson process, seed {args.seed}, one client per request"),
        ]
    calibrated: list[str] = []
    for result in document["results"]:
        for level in result.get("server") or []:
            if level.get("calibrated_tokens"):
                observed = ", ".join(str(value) for value in level["prompt_tokens"])
                calibrated.append(f"{result['engine']} {level['calibrated_tokens']} → {observed}")
    lengths = (
        "calibrated per engine to exactly N prompt tokens, chat template included, by probing "
        "`usage.prompt_tokens`; a request with any other count stops the run"
    )
    if workload["name"] == "context":
        rows = [
            row("server workload", f"{workload['text']} ([source]({workload['source']}))"),
            row("dataset sha256", f"`{workload['sha256']}`"),
            row("contexts", ", ".join(str(prompt) for prompt in args.prompt)),
            row("requests per context and round", f"{args.requests}, one at a time"),
            row("generated tokens", str(args.generate)),
            row("prompt length", lengths),
        ]
    elif calibrated:
        rows = [
            row("server workload", "fixed English text cut to the prompt length"),
            row("prompt length", lengths),
            row("server concurrency levels", ", ".join(str(level) for level in args.concurrency)),
            row("requests per level", str(args.requests)),
            row("server generated tokens", str(args.generate)),
        ]
    else:
        rows = [
            row(
                "server workload",
                f"fixed English text repeated {max(1, args.prompt[0] // 95)} times",
            ),
            row("server concurrency levels", ", ".join(str(level) for level in args.concurrency)),
            row("requests per level", str(args.requests)),
            row("server generated tokens", str(args.generate)),
        ]
    if calibrated:
        rows.append(row("prompt tokens, calibrated → observed", "; ".join(calibrated)))
    return rows


def setup(document: dict[str, Any]) -> str:
    args = argparse.Namespace(**document["arguments"])
    system = document["system"]
    facts = document["model"]
    run_modes = modes(args)
    run_engines = engines(args)
    rows = [
        row("model", f"`{args.model}`"),
        row("MLX model", f"`{args.mlx_model}`"),
        row("engines", ", ".join(engine.label for engine in run_engines)),
        row("suite", args.suite or "none"),
        row("modes", ", ".join(run_modes)),
        row("rounds (engine order reversed every other round)", str(args.rounds)),
        row("cooldown after each engine and mode", f"{args.cooldown:.0f} s"),
    ]
    if "offline" in run_modes:
        rows += [
            row("offline prompt lengths", ", ".join(str(prompt) for prompt in args.prompt)),
            row("offline generated tokens", str(args.generate)),
            row("offline iterations per run", str(args.iterations)),
        ]
    if "server" in run_modes:
        rows += workload_rows(document, args)
        rows += [
            row(
                "requests",
                "chat completion, streamed, greedy, `ignore_eos`, `max_tokens` = output length, "
                "unique fixed-length prefix per request",
            ),
            row(
                "goodput SLO",
                f"TTFT ≤ {number(args.slo_ttft_ms, 0)} ms, TPOT ≤ {number(args.slo_tpot_ms, 0)} ms",
            ),
        ]
        kinds = [engine.kind for engine in run_engines]
        if "mlx" in kinds:
            rows.append(
                row(
                    "MLX requests",
                    "`mlx_lm.server` has no `ignore_eos`: its end-of-sequence ids get "
                    "`logit_bias` −100; prompt cache limited to one entry",
                )
            )
        if "llama.cpp" in kinds:
            rows.append(row("llama.cpp requests", "`cache_prompt: false`, one slot (`-np 1`)"))
    gguf = (document.get("files") or {}).get("gguf")
    if gguf:
        rows.append(
            row("llama.cpp model", f"`{display_path(gguf['path'])}`, sha256 `{gguf['sha256']}`")
        )
    rows += [
        row("peak bandwidth for MBU", f"{number(document['peak_bandwidth_gbs'])} GB/s"),
        row("peak FP16 TFLOPS for MFU", number(args.peak_tflops)),
        row("energy", "powermetrics, idle power subtracted" if args.energy else "not measured"),
        row("parameters read per token", f"{facts['parameters_read_per_token']:,}"),
        row("weight bytes read per token", f"{facts['weight_bytes_read_per_token'] / 1e9:.3f} GB"),
        row("KV bytes per context token", f"{facts['kv_bytes_per_token']:,}"),
        row("chip / memory", f"{system['chip']} / {system['memory_bytes'] / 2**30:.0f} GB"),
        row("macOS", system["macos"]),
        row("power source", system.get("power_source") or "unknown"),
        row("thermal state", "; ".join((system.get("thermal") or "unknown").split("\n"))),
        row(
            "metal-infer commit",
            f"`{system['metal_infer_commit']}`"
            f"{' (dirty working tree)' if system['metal_infer_dirty'] else ''}",
        ),
        row("MLX versions", f"mlx-lm {system['mlx_lm']}, mlx {system['mlx']}"),
        row("llama.cpp version", system.get("llama_cpp") or "not used"),
    ]
    lines = ["## Setup", "", "| parameter | value |", "|---|---|"]
    lines += [f"| {item['name']} | {item['value']} |" for item in rows]
    commands = []
    for engine in run_engines:
        if "offline" in run_modes:
            for prompt in args.prompt:
                commands += [
                    display_command(command) for command in offline_commands(engine, args, prompt)
                ]
        if "server" in run_modes:
            commands.append(display_command(server_command(engine, args)))
    lines += [
        "",
        "Offline: pp = prompt tokens / prefill time and tg = generated tokens / decode time, "
        "one sample per iteration; MBU = (weight bytes + KV bytes × (prompt + generated / 2)) "
        "× tg p50 / peak bandwidth. MBU/MFU are derived indicators, not hardware-counter measurements.",
        "",
        "Server: TTFT = request sent → first streamed output chunk; TPOT = (E2EL − TTFT) / "
        "(output tokens − 1); ITL = time between output-bearing chunks; E2EL = request sent → last "
        "output-bearing chunk. Role, finish and usage events do not count as output. Throughputs "
        "are divided by the observed wall time of the level.",
        "",
        "### Commands",
        "",
        "```sh",
        *commands,
        "```",
    ]
    workload = document["workload"]
    if "server" in run_modes and workload["name"] == "sharegpt":
        lines += [
            "",
            "### Reproduce with a standard tool",
            "",
            "The sampling matches `vllm bench serve --dataset-name sharegpt` (same filters, same seed, same "
            "shuffle), so the same requests can be replayed against any OpenAI-compatible server started "
            "with the commands above:",
            "",
            "```sh",
            *[
                f"vllm bench serve --backend openai-chat --endpoint /v1/chat/completions "
                f"--base-url http://127.0.0.1:{args.port} --model {args.model} --tokenizer {args.model} "
                f"--dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json "
                f"--num-prompts {workload['num_prompts']} --seed {workload['seed']} --request-rate {rate} "
                f"--ignore-eos --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 "
                f"--goodput ttft:{args.slo_ttft_ms:g} tpot:{args.slo_tpot_ms:g}"
                for rate in args.request_rate
            ],
            "```",
        ]
    for result in document["results"]:
        plan = next(
            (values["plan"] for values in result.get("offline") or [] if values.get("plan")), None
        )
        if plan:
            lines += ["", f"### Plan — {result['engine']}", "", "| key | value |", "|---|---|"]
            lines += [f"| `{key}` | `{plan[key]}` |" for key in plan]
    return "\n".join(lines) + "\n"


def run_readme(document: dict[str, Any]) -> str:
    raw = (
        "[progress.jsonl](progress.jsonl), one line per measurement, with the run parameters in "
        "[run.json](run.json)"
        if document.get("measurements")
        else "[results.json](results.json)"
    )
    return (
        markdown(document, image=True)
        + setup(document)
        + f"\nRaw measurements: {raw}. Protocol and metrics: [benchmark guide](../../../README.md).\n"
    )


def write_report(document: dict[str, Any], directory: pathlib.Path, index: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(document, indent=2) + "\n")
    (directory / "results.svg").write_text(svg(document))
    (directory / "README.md").write_text(run_readme(document))
    if index:
        rebuild_index()


def render(path: pathlib.Path) -> pathlib.Path:
    document = upgrade(json.loads(path.read_text()))
    directory = run_directory(document["started_at"], document["system"])
    write_report(document, directory, index=True)
    return directory


def headline(document: dict[str, Any]) -> str:
    contexts = server_loads(document, context=True)
    if contexts:
        load = contexts[-1]
        parts = []
        for result in document["results"]:
            level = level_for(result, load)
            interval = ((level or {}).get("ci") or {}).get("ttft_ms")
            if interval:
                parts.append(f"{result['engine']} {number(interval['median'])} ms")
        return f"TTFT at {load['value']}: " + " · ".join(parts) if parts else "—"
    parts = [
        f"{result['engine']} {number(p50(result['offline'][0]['tg_tps']))} (pp{result['offline'][0]['prompt']})"
        for result in document["results"]
        if result.get("offline")
    ]
    return "offline decode tok/s p50: " + " · ".join(parts) if parts else "—"


def rebuild_index() -> None:
    """Only finished runs; the newest one of each chip and model also becomes latest-<model>.svg."""
    runs = []
    for path in RESULTS.glob("*/*/results.json"):
        try:
            document = upgrade(json.loads(path.read_text()))
        except OSError:
            continue
        except json.JSONDecodeError:
            continue
        except KeyError:
            continue
        if document.get("finished_at"):
            runs.append({"directory": path.parent, "document": document})
    runs.sort(key=lambda run: run["document"]["started_at"], reverse=True)
    lines = [
        "# Benchmark results",
        "",
        "Written by `benchmarks/bench.py`, newest first. Runs still in progress are not listed.",
        "",
        "| started | chip | model | workload | headline | report |",
        "|---|---|---|---|---|---|",
    ]
    latest: dict[str, dict[str, Any]] = {}
    for run in runs:
        directory = run["directory"]
        document = run["document"]
        args = document["arguments"]
        workload = document["workload"]
        if workload["name"] == "sharegpt":
            name = f"sharegpt × {workload['num_prompts']}"
        elif workload["name"] == "context":
            name = "context " + ", ".join(str(prompt) for prompt in args["prompt"])
        else:
            name = f"synthetic tg{args['generate']}"
        lines.append(
            f"| `{document['started_at'][:19]}` | {document['system']['chip']} | {args['model']} "
            f"| {name} | {headline(document)} "
            f"| [open]({directory.relative_to(RESULTS).as_posix()}/README.md) |"
        )
        latest.setdefault(
            f"{directory.parent}/{slug(args['model'])}",
            {"directory": directory, "model": args["model"]},
        )
    (RESULTS / "README.md").write_text("\n".join(lines) + "\n")
    for key in latest:
        directory = latest[key]["directory"]
        shutil.copyfile(
            directory / "results.svg", directory.parent / f"latest-{slug(latest[key]['model'])}.svg"
        )
