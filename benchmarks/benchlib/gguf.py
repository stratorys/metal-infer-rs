"""Resolve a GGUF file supplied as a local path or a supported Hugging Face repo."""

import pathlib
from dataclasses import dataclass

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

from benchlib.common import BenchmarkError

QWEN_GGUF_REPO = "unsloth/Qwen3-0.6B-GGUF"
QWEN_GGUF_FILE = "Qwen3-0.6B-BF16.gguf"
QWEN_GGUF_REVISION = "50968a4468ef4233ed78cd7c3de230dd1d61a56b"


@dataclass(frozen=True)
class ResolvedGguf:
    path: pathlib.Path
    source: dict[str, str]


def resolve_gguf(value: str) -> ResolvedGguf:
    """Use only the local HF cache; a benchmark never downloads weights implicitly."""
    local = pathlib.Path(value).expanduser()
    if local.is_file():
        return ResolvedGguf(path=local.resolve(), source={})
    if value != QWEN_GGUF_REPO:
        raise BenchmarkError(
            f"--gguf {value} is not a file or the supported Hugging Face repo {QWEN_GGUF_REPO}"
        )
    try:
        cached = pathlib.Path(
            hf_hub_download(
                repo_id=QWEN_GGUF_REPO,
                filename=QWEN_GGUF_FILE,
                revision=QWEN_GGUF_REVISION,
                local_files_only=True,
            )
        )
    except (LocalEntryNotFoundError, FileNotFoundError, OSError) as error:
        raise BenchmarkError(
            f"{QWEN_GGUF_FILE} at revision {QWEN_GGUF_REVISION} is not in the "
            "Hugging Face cache; download it before benchmarking"
        ) from error
    if not cached.is_file():
        raise BenchmarkError(f"cached GGUF {cached} is not a file")
    return ResolvedGguf(
        path=cached.resolve(),
        source={
            "repo": QWEN_GGUF_REPO,
            "revision": QWEN_GGUF_REVISION,
            "filename": QWEN_GGUF_FILE,
        },
    )
