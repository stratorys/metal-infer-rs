"""A run on disk: run.json, progress.jsonl appended after every measurement, and resuming."""

import datetime
import json
import os
import pathlib
import statistics
from typing import Any

from benchlib.common import RESULTS, BenchmarkError, slug
from benchlib.plan import Measurement


def run_directory(started_at: str, system: dict[str, Any]) -> pathlib.Path:
    started = datetime.datetime.fromisoformat(started_at)
    commit = (system.get("metal_infer_commit") or "uncommitted")[:7]
    dirty = "-dirty" if system.get("metal_infer_dirty") else ""
    return RESULTS / slug(system["chip"]) / f"{started:%Y%m%d-%H%M%S}-{commit}{dirty}"


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class RunState:
    def __init__(
        self, directory: pathlib.Path, header: dict[str, Any], records: list[dict[str, Any]]
    ) -> None:
        self.directory = directory
        self.header = header
        self.records = records
        self.seconds: dict[str, list[float]] = {}

    @staticmethod
    def create(header: dict[str, Any]) -> "RunState":
        directory = run_directory(header["started_at"], header["system"])
        directory.mkdir(parents=True, exist_ok=False)
        state = RunState(directory, header, [])
        state.write_header()
        (directory / "progress.jsonl").touch()
        return state

    @staticmethod
    def resume(directory: pathlib.Path) -> "RunState":
        header_path = directory / "run.json"
        if not header_path.is_file():
            raise BenchmarkError(f"{directory} has no run.json to resume")
        header = json.loads(header_path.read_text())
        if header.get("finished_at"):
            raise BenchmarkError(f"the run in {directory} is already finished")
        records = []
        lines = (directory / "progress.jsonl").read_text().splitlines()
        for index in range(len(lines)):
            try:
                records.append(json.loads(lines[index]))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise BenchmarkError(f"progress.jsonl line {index + 1} is corrupt")
        return RunState(directory, header, records)

    def write_header(self) -> None:
        (self.directory / "run.json").write_text(json.dumps(self.header, indent=2) + "\n")

    def done_keys(self) -> set[str]:
        return {record["key"] for record in self.records}

    def append(self, record: dict[str, Any]) -> None:
        with (self.directory / "progress.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
            file.flush()
            os.fsync(file.fileno())
        self.records.append(record)
        self.seconds.setdefault(record["mode"], []).append(record["seconds"])

    def finish(self) -> None:
        self.header["finished_at"] = now()
        self.write_header()

    def remaining_seconds(self, pending: list[Measurement]) -> float | None:
        """Mean duration of this session's measurements of the same mode, times what is left."""
        total = 0.0
        for measurement in pending:
            durations = self.seconds.get(measurement.mode)
            if not durations:
                return None
            total += statistics.fmean(durations)
        return total


def short_load(measurement: Measurement) -> str:
    load = measurement.load
    if load.type == "context":
        return f"ctx {load.value}"
    if load.type == "prompt":
        return f"prompt {load.value}"
    if load.type == "concurrency":
        return f"c={load.value}"
    return "all at once" if load.value == "inf" else f"{load.value} req/s"


def median_of(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def figures(measurement: Measurement, data: dict[str, Any]) -> list[str]:
    if measurement.mode == "offline":
        samples = data["samples"]
        return [
            f"pp {median_of([sample['pp_tps'] for sample in samples]):,.0f} tok/s",
            f"tg {median_of([sample['tg_tps'] for sample in samples]):,.1f} tok/s",
        ]
    completed = [item for item in data["requests"] if "error" not in item]
    result = []
    ttft = median_of([item["ttft_ms"] for item in completed])
    if measurement.load.type == "context":
        tpot = median_of([item["tpot_ms"] for item in completed if item["output_tokens"] > 1])
        if tpot is not None:
            result.append(f"TPOT {tpot:,.1f} ms")
    else:
        result.append(f"output {data['output_tps']:,.1f} tok/s")
        tpot = median_of([item["tpot_ms"] for item in completed])
        if tpot is not None:
            result.append(f"TPOT {tpot:,.1f} ms")
    if ttft is not None:
        result.append(f"TTFT {ttft / 1000:,.2f} s")
    if data.get("failures"):
        result.append(f"{len(data['failures'])} failed")
    return result


def progress_line(
    done: int, total: int, measurement: Measurement, data: dict[str, Any], remaining: float | None
) -> str:
    parts = [
        f"[{done}/{total}] round {measurement.round + 1}",
        measurement.engine.label,
        short_load(measurement),
    ]
    parts += figures(measurement, data)
    parts.append("left ?" if remaining is None else f"left ~{max(remaining / 60, 0):.0f} min")
    return " · ".join(parts)
