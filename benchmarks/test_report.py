#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Tests for deterministic benchmark report generation."""

import json
import pathlib
import tempfile
import unittest
import xml.etree.ElementTree as element_tree

import report


class ReportTests(unittest.TestCase):
    def test_kernel_report_is_valid_svg(self) -> None:
        document = kernel_document()
        with tempfile.TemporaryDirectory() as directory:
            destination = pathlib.Path(directory)
            report.generate_run_report(document, destination)
            element_tree.parse(destination / "results.svg")
            readme = (destination / "README.md").read_text(encoding="utf-8")
            self.assertIn("metal-infer", readme)
            self.assertNotIn("/Users/", readme)

    def test_model_report_uses_independent_panels(self) -> None:
        document = model_document()
        with tempfile.TemporaryDirectory() as directory:
            destination = pathlib.Path(directory)
            report.generate_run_report(document, destination)
            svg = (destination / "results.svg").read_text(encoding="utf-8")
            self.assertIn("Prefill", svg)
            self.assertIn("Decode", svg)
            element_tree.fromstring(svg)

    def test_index_links_runs(self) -> None:
        document = kernel_document()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            run = root / "apple-m4-pro" / "run-one"
            run.mkdir(parents=True)
            (run / "results.json").write_text(json.dumps(document), encoding="utf-8")
            report.rebuild_index(root)
            index = (root / "README.md").read_text(encoding="utf-8")
            self.assertIn("apple-m4-pro/run-one/README.md", index)


def kernel_document() -> dict:
    return {
        "kind": "kernels",
        "generated_at": "2026-09-19T14:41:30+00:00",
        "system": {
            "chip": "Apple M4 Pro",
            "macos": "26.6.2",
            "rustc": "rustc test",
            "metal_infer_commit": "abcdef0",
            "metal_infer_dirty": False,
        },
        "suite": {"warmup": 5, "iterations": 20, "shapes": []},
        "results": [
            {
                "backend": "metal-infer",
                "dimensions": {"m": 64, "n": 128, "k": 128},
                "mean_ms": 0.2,
                "throughput": 0.01,
            },
            {
                "backend": "mlx<&>",
                "dimensions": {"m": 64, "n": 128, "k": 128},
                "mean_ms": 0.1,
                "throughput": 0.02,
            },
        ],
    }


def model_document() -> dict:
    document = kernel_document()
    document["kind"] = "model"
    document["suite"] = {
        "warmup": 1,
        "iterations": 5,
        "prompt_tokens": 512,
        "generation_tokens": 128,
    }
    document["results"] = [
        {
            "backend": "metal-infer",
            "allocated_bytes": 1_000_000,
            "prefill": {"tokens_per_second": 600.0},
            "decode": {"tokens_per_second": 8.0},
        },
        {
            "backend": "mlx-lm",
            "peak_memory_gb": 2.0,
            "prefill": {"tokens_per_second": 4_500.0},
            "decode": {"tokens_per_second": 170.0},
        },
    ]
    return document


if __name__ == "__main__":
    unittest.main()
