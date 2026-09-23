"""Paths, versions, process helpers and machine facts shared by every module."""

import hashlib
import http.client
import json
import os
import pathlib
import platform
import re
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"
DATASETS = pathlib.Path.home() / ".cache" / "metal-infer" / "datasets"
METAL_BENCH = ROOT / "target" / "release" / "metal-infer-bench"
METAL_SERVER = ROOT / "target" / "release" / "metal-infer"
MLX_LM_VERSION = "0.31.3"
MLX_VERSION = "0.32.2"
MLX = ["uvx", "--from", f"mlx-lm=={MLX_LM_VERSION}", "--with", f"mlx=={MLX_VERSION}"]
ENGINE_PROCESSES = ["metal-infer", "mlx_lm", "llama-server", "llama-bench", "vllm"]
PEAK_BANDWIDTH_GBS = {"Apple M4 Pro": 273.0, "Apple M4": 120.0}
SCHEMA_VERSION = 5


class BenchmarkError(RuntimeError):
    pass


def run(command: list[str], **kwargs: Any) -> str:
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False, **kwargs
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BenchmarkError(f"{' '.join(command)} failed: {detail}")
    return completed.stdout


def optional(command: list[str]) -> str | None:
    try:
        return run(command).strip()
    except BenchmarkError:
        return None
    except OSError:
        return None


def version_output(command: list[str]) -> str | None:
    """First line of a version command, which some tools print on stderr."""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return None
    text = (completed.stdout + "\n" + completed.stderr).strip()
    lines = [line for line in text.splitlines() if "version" in line.lower()]
    return lines[0].strip() if lines else None


def model_directory(model: str) -> pathlib.Path:
    path = pathlib.Path(model).expanduser()
    if path.is_dir():
        return path
    cache = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache/huggingface"))
    repository = cache / "hub" / f"models--{model.replace('/', '--')}"
    reference = repository / "refs" / "main"
    if reference.is_file():
        snapshot = repository / "snapshots" / reference.read_text().strip()
        if snapshot.is_dir():
            return snapshot
    raise BenchmarkError(f"model {model} is not a directory nor in the Hugging Face cache")


def model_facts(directory: pathlib.Path) -> dict[str, Any]:
    config = json.loads((directory / "config.json").read_text())
    hidden = config["hidden_size"]
    head_dim = config.get("head_dim", hidden // config["num_attention_heads"])
    query = config["num_attention_heads"] * head_dim
    kv = config["num_key_value_heads"] * head_dim
    layer = hidden * (query + 2 * kv) + query * hidden + 3 * hidden * config["intermediate_size"]
    parameters = config["num_hidden_layers"] * layer + config["vocab_size"] * hidden
    return {
        "parameters_read_per_token": parameters,
        "weight_bytes_read_per_token": parameters * 2,
        "kv_bytes_per_token": 2 * config["num_hidden_layers"] * kv * 2,
    }


def eos_ids(directory: pathlib.Path) -> list[int]:
    """End-of-sequence ids from generation_config.json, falling back to config.json."""
    for name in ["generation_config.json", "config.json"]:
        path = directory / name
        if not path.is_file():
            continue
        value = json.loads(path.read_text()).get("eos_token_id")
        if isinstance(value, int):
            return [value]
        if isinstance(value, list) and value:
            return [int(item) for item in value]
    raise BenchmarkError(f"no eos_token_id in {directory}")


def system(engines: list[str]) -> dict[str, Any]:
    chip = optional(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.machine()
    return {
        "chip": chip,
        "macos": optional(["sw_vers", "-productVersion"]) or "unknown",
        "memory_bytes": int(optional(["sysctl", "-n", "hw.memsize"]) or 0),
        "power_source": (optional(["pmset", "-g", "batt"]) or "").split("\n")[0],
        "thermal": optional(["pmset", "-g", "therm"]),
        "metal_infer_commit": optional(["git", "rev-parse", "HEAD"]),
        "metal_infer_dirty": bool(optional(["git", "status", "--porcelain"])),
        "mlx_lm": MLX_LM_VERSION,
        "mlx": MLX_VERSION,
        "llama_cpp": version_output(["llama-server", "--version"])
        if "llama.cpp" in engines
        else None,
    }


def require_idle(port: int) -> None:
    listing = optional(["ps", "-axo", "pid=,command="]) or ""
    own = str(os.getpid())
    busy = []
    for line in listing.splitlines():
        words = line.split()
        if not words or words[0] == own or "bench.py" in line:
            continue
        if any(name in line for name in ENGINE_PROCESSES):
            busy.append(line.strip())
    if busy:
        raise BenchmarkError("another engine is running:\n" + "\n".join(busy))
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        connection.request("GET", "/v1/models")
    except OSError:
        return
    raise BenchmarkError(f"port {port} is already in use")


def cooldown(seconds: float) -> None:
    if seconds > 0:
        print(f"  cooldown {seconds:.0f} s", flush=True)
        time.sleep(seconds)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Measured:
    result: Any
    joules: float | None


@dataclass(frozen=True)
class PowerSample:
    watts: float
    result: Any
    seconds: float


class Energy:
    """Samples package power with powermetrics while a measurement runs."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.idle_watts = 0.0
        if enabled:
            self.idle_watts = self._sample(lambda: time.sleep(5)).watts

    def measure(self, action: Callable[[], Any]) -> Measured:
        if not self.enabled:
            return Measured(result=action(), joules=None)
        sample = self._sample(action)
        return Measured(
            result=sample.result, joules=max(sample.watts - self.idle_watts, 0.0) * sample.seconds
        )

    def _sample(self, action: Callable[[], Any]) -> PowerSample:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as output:
            path = output.name
        process = subprocess.Popen(
            [
                "sudo",
                "-n",
                "powermetrics",
                "--samplers",
                "cpu_power,gpu_power",
                "-i",
                "200",
                "-o",
                path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)
        started = time.perf_counter()
        result = action()
        seconds = time.perf_counter() - started
        subprocess.run(["sudo", "-n", "kill", "-INT", str(process.pid)], check=False)
        process.wait(timeout=10)
        text = pathlib.Path(path).read_text(errors="ignore")
        samples = [
            float(value)
            for value in re.findall(r"Combined Power \(CPU \+ GPU \+ ANE\): (\d+) mW", text)
        ]
        if not samples:
            raise BenchmarkError("powermetrics produced no power samples; run `sudo -v` first")
        return PowerSample(watts=statistics.fmean(samples) / 1000, result=result, seconds=seconds)
