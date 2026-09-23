"""The one OpenAI-compatible client every engine is measured with."""

import concurrent.futures
import http.client
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from benchlib.common import BenchmarkError, Energy

CALIBRATION_PREFIX = "Request 000000."


class Prefixes:
    """Unique, fixed-length request prefixes, so no engine reuses a cached prompt."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.count = 0

    def next(self) -> str:
        with self.lock:
            self.count += 1
            return f"Request {self.count:06d}."


@dataclass(frozen=True)
class Target:
    port: int
    extra: dict[str, Any] = field(default_factory=dict)
    prefixes: Prefixes = field(default_factory=Prefixes)
    prefix_requests: bool = True


@dataclass(frozen=True)
class Job:
    prompt: str
    generate: int


def stream(
    target: Target, prompt: str, generate: int, prefix: str, *, usage_only: bool = False
) -> dict[str, Any]:
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": f"{prefix} {prompt}" if prefix else prompt}],
            "max_tokens": generate,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            **target.extra,
        }
    )
    connection = http.client.HTTPConnection("127.0.0.1", target.port, timeout=1800)
    started = time.perf_counter()
    try:
        connection.request(
            "POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        if response.status != 200:
            raise BenchmarkError(f"server answered {response.status}: {response.read()[:500]!r}")
        usage: dict[str, Any] = {}
        first = None
        last_output = None
        intervals: list[float] = []
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if payload == "[DONE]":
                break
            timestamp = time.perf_counter()
            chunk = json.loads(payload)
            has_output = any(
                choice.get("delta", {}).get(field)
                for choice in chunk.get("choices", [])
                for field in ["content", "reasoning", "reasoning_content"]
            )
            if has_output:
                if first is None:
                    first = timestamp
                else:
                    intervals.append(timestamp - last_output)
                last_output = timestamp
            usage = chunk.get("usage") or usage
        if usage.get("completion_tokens") is None or usage.get("prompt_tokens") is None:
            raise BenchmarkError("stream did not report prompt and completion token usage")
        prompt_tokens = int(usage["prompt_tokens"])
        output_tokens = int(usage["completion_tokens"])
        if usage_only:
            return {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens}
        if first is None:
            raise BenchmarkError("stream contained no output-bearing completion chunks")
        ttft = first - started
        e2el = last_output - started
        tpot = (e2el - ttft) / (output_tokens - 1) if output_tokens > 1 else 0.0
    finally:
        connection.close()
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": ttft * 1000,
        "e2el_ms": e2el * 1000,
        "tpot_ms": tpot * 1000,
        "itl_ms": [value * 1000 for value in intervals],
    }


def request(target: Target, prompt: str, generate: int) -> dict[str, Any]:
    """A measured request: failures are recorded instead of stopping the run."""
    try:
        return stream(
            target, prompt, generate, target.prefixes.next() if target.prefix_requests else ""
        )
    except OSError as error:
        return {"error": str(error) or type(error).__name__}
    except http.client.HTTPException as error:
        return {"error": str(error) or type(error).__name__}
    except json.JSONDecodeError as error:
        return {"error": str(error)}
    except UnicodeDecodeError as error:
        return {"error": str(error)}
    except BenchmarkError as error:
        return {"error": str(error)}


def arrivals(rate: str, count: int, seed: int) -> list[float]:
    if rate == "inf":
        return [0.0] * count
    import numpy as np

    delays = np.random.RandomState(seed).gamma(shape=1.0, scale=1.0 / float(rate), size=count)
    moments = np.cumsum(delays)
    if count:
        moments *= (count / float(rate)) / moments[-1]
    return moments.tolist()


def server_level(
    target: Target,
    energy: Energy,
    jobs: list[Job],
    workers: int,
    schedule: list[float] | None,
    slo_ttft_ms: float,
    slo_tpot_ms: float,
) -> dict[str, Any]:
    def batch() -> dict[str, Any]:
        started = time.perf_counter()

        def send(index: int) -> dict[str, Any]:
            if schedule is not None:
                delay = started + schedule[index] - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            job = jobs[index]
            sent = time.perf_counter() - started
            return {
                **request(target, job.prompt, job.generate),
                "target_tokens": job.generate,
                "scheduled_s": schedule[index] if schedule is not None else None,
                "sent_s": sent,
            }

        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(send, range(len(jobs))))
        return {"requests": results, "seconds": time.perf_counter() - started}

    measured = energy.measure(batch)
    results = measured.result["requests"]
    seconds = measured.result["seconds"]
    completed = [result for result in results if "error" not in result]
    output_tokens = sum(result["output_tokens"] for result in completed)
    good = [
        result
        for result in completed
        if result["ttft_ms"] <= slo_ttft_ms and result["tpot_ms"] <= slo_tpot_ms
    ]
    return {
        "requests": results,
        "seconds": seconds,
        "request_tps": len(completed) / seconds,
        "input_tps": sum(result["prompt_tokens"] or 0 for result in completed) / seconds,
        "output_tps": output_tokens / seconds,
        "goodput_rps": len(good) / seconds,
        "failures": [result["error"] for result in results if "error" in result],
        "short_outputs": [
            result["output_tokens"]
            for result in completed
            if result["output_tokens"] != result["target_tokens"]
        ],
        "joules_per_token": None
        if measured.joules is None or not output_tokens
        else measured.joules / output_tokens,
    }
