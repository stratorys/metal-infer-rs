"""Request texts: synthetic, ShareGPT conversations, long-context text, and exact calibration."""

import argparse
import json
import pathlib
import random
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from benchlib.client import CALIBRATION_PREFIX, Job, Target, arrivals, stream
from benchlib.common import DATASETS, BenchmarkError, file_sha256, slug
from benchlib.stats import summary

SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/"
    "resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
SHAREGPT_FILTERS = {
    "min_prompt_tokens": 4,
    "min_output_tokens": 4,
    "max_prompt_tokens": 1024,
    "max_total_tokens": 2048,
}
CONTEXT_CHARS = 400_000
CALIBRATION_PROBES = 40
PROMPT_TEXT = (
    "You are reviewing the design of a small inference engine for Apple Silicon. "
    "The engine loads FP16 weights, runs a prefill pass over the prompt, then "
    "generates tokens one at a time while keeping a key and value cache in GPU "
    "memory. Explain in detail how each part works, which parts are limited by "
    "memory bandwidth and which by compute, and how you would measure them. "
)


@dataclass(frozen=True)
class Workload:
    """What the server measurements send: a text cut to N tokens, or fixed ShareGPT jobs."""

    description: dict[str, Any]
    text: str = ""
    jobs: list[Job] = field(default_factory=list)


def sharegpt_file() -> pathlib.Path:
    path = DATASETS / "ShareGPT_V3_unfiltered_cleaned_split.json"
    if not path.is_file():
        DATASETS.mkdir(parents=True, exist_ok=True)
        print(f"downloading ShareGPT (about 670 MB) to {path}", flush=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(SHAREGPT_URL, partial)
        partial.rename(path)
    return path


def sample_sharegpt(args: argparse.Namespace, directory: pathlib.Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    source = sharegpt_file()
    print("sampling ShareGPT conversations", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, use_fast=True)
    conversations = [
        conversation
        for conversation in json.loads(source.read_text())
        if len(conversation.get("conversations") or []) >= 2
    ]
    random.Random(args.seed).shuffle(conversations)
    limits = SHAREGPT_FILTERS
    samples = []
    for conversation in conversations:
        turns = conversation["conversations"]
        prompt = turns[0]["value"]
        prompt_tokens = len(tokenizer(prompt).input_ids)
        output_tokens = len(tokenizer(turns[1]["value"]).input_ids)
        if (
            prompt_tokens < limits["min_prompt_tokens"]
            or output_tokens < limits["min_output_tokens"]
            or prompt_tokens > limits["max_prompt_tokens"]
            or prompt_tokens + output_tokens > limits["max_total_tokens"]
        ):
            continue
        samples.append(
            {
                "id": conversation.get("id"),
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
            }
        )
        if len(samples) == args.num_prompts:
            break
    if len(samples) < args.num_prompts:
        raise BenchmarkError(f"ShareGPT has only {len(samples)} conversations within the filters")
    return {"sha256": file_sha256(source), "samples": samples}


def sharegpt_workload(args: argparse.Namespace, directory: pathlib.Path) -> Workload:
    source = sharegpt_file()
    source_hash = file_sha256(source)
    tokenizer_files = [
        directory / name
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    ]
    tokenizer_hash = ":".join(file_sha256(path) for path in tokenizer_files if path.is_file())
    cache = DATASETS / f"sharegpt-{slug(args.model)}-{args.num_prompts}-seed{args.seed}.json"
    if cache.is_file():
        sampled = json.loads(cache.read_text())
    if (
        not cache.is_file()
        or sampled.get("sha256") != source_hash
        or sampled.get("tokenizer_hash") != tokenizer_hash
    ):
        sampled = sample_sharegpt(args, directory)
        sampled["tokenizer_hash"] = tokenizer_hash
        cache.write_text(json.dumps(sampled))
    samples = sampled["samples"]
    description = {
        "name": "sharegpt",
        "source": SHAREGPT_URL,
        "sha256": sampled["sha256"],
        "tokenizer_hash": tokenizer_hash,
        "seed": args.seed,
        "num_prompts": args.num_prompts,
        "filters": SHAREGPT_FILTERS,
        "input_tokens": summary([sample["prompt_tokens"] for sample in samples]),
        "output_tokens": summary([sample["output_tokens"] for sample in samples]),
        "conversation_ids": [sample["id"] for sample in samples],
        "requests": samples,
        "schedules": {rate: arrivals(rate, len(samples), args.seed) for rate in args.request_rate},
    }
    jobs = [Job(prompt=sample["prompt"], generate=sample["output_tokens"]) for sample in samples]
    return Workload(description=description, jobs=jobs)


def context_workload() -> Workload:
    """Every ShareGPT turn in file order, joined, cut to the first CONTEXT_CHARS characters."""
    cache = DATASETS / "context-text.json"
    if cache.is_file():
        cached = json.loads(cache.read_text())
    else:
        source = sharegpt_file()
        print("building the long-context text from ShareGPT", flush=True)
        parts = []
        length = 0
        for conversation in json.loads(source.read_text()):
            for turn in conversation.get("conversations") or []:
                parts.append(turn["value"])
                length += len(turn["value"]) + 2
            if length >= CONTEXT_CHARS:
                break
        cached = {"sha256": file_sha256(source), "text": "\n\n".join(parts)[:CONTEXT_CHARS]}
        cache.write_text(json.dumps(cached))
    description = {
        "name": "context",
        "source": SHAREGPT_URL,
        "sha256": cached["sha256"],
        "text": f"ShareGPT turns in file order joined by blank lines, first {CONTEXT_CHARS:,} characters",
    }
    return Workload(description=description, text=cached["text"])


def synthetic_workload(args: argparse.Namespace) -> Workload:
    repeats = args.prompt[0] // 20 + 10
    return Workload(description={"name": "synthetic"}, text=PROMPT_TEXT * repeats)


def workload(args: argparse.Namespace, directory: pathlib.Path) -> Workload:
    if args.mode == "offline":
        return Workload(description={"name": args.workload})
    if args.workload == "sharegpt":
        return sharegpt_workload(args, directory)
    if args.workload == "context":
        return context_workload()
    return synthetic_workload(args)


def calibrate(target: Target, text: str, tokens: int) -> int:
    """Number of leading characters of text that makes a request of exactly `tokens` prompt tokens.

    The engine's own tokenizer and chat template decide, through usage.prompt_tokens. The
    prefix has the same token length as the measured requests' prefixes.
    """
    counts: dict[int, int] = {}

    def count(chars: int) -> int:
        if chars not in counts:
            result = stream(target, text[:chars], 1, CALIBRATION_PREFIX, usage_only=True)
            if result["prompt_tokens"] is None:
                raise BenchmarkError("the server does not report usage.prompt_tokens in its stream")
            counts[chars] = result["prompt_tokens"]
        return counts[chars]

    overhead = count(0)
    if overhead >= tokens:
        raise BenchmarkError(
            f"the chat template alone takes {overhead} tokens, not fewer than {tokens}"
        )
    low = 0
    low_count = overhead
    high = -1
    high_count = -1
    guess = min(len(text), tokens * 4)
    for probe in range(CALIBRATION_PROBES):
        value = count(guess)
        if value == tokens:
            return guess
        if value < tokens:
            low = guess
            low_count = value
        else:
            high = guess
            high_count = value
        if high < 0:
            if low == len(text):
                raise BenchmarkError(f"the text gives only {low_count} tokens, fewer than {tokens}")
            per_token = low / max(low_count - overhead, 1)
            guess = min(len(text), low + max(1, int((tokens - low_count) * per_token * 1.05) + 1))
            continue
        if high - low <= 1:
            raise BenchmarkError(
                f"no prefix gives exactly {tokens} prompt tokens: {low} characters give {low_count}, "
                f"{high} give {high_count}"
            )
        if probe % 2 == 0:
            guess = low + (tokens - low_count) * (high - low) // (high_count - low_count)
            guess = min(max(guess, low + 1), high - 1)
        else:
            guess = (low + high) // 2
    raise BenchmarkError(
        f"calibration to {tokens} tokens did not converge in {CALIBRATION_PROBES} probes"
    )
