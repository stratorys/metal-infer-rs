#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Tests for the repeated model A/B benchmark runner."""

import argparse
import contextlib
import io
import json
import pathlib
import subprocess
import unittest
from unittest import mock

import model_ab


def report(tps: float) -> dict:
    return {
        "benchmark": "qwen3_model",
        "iterations": 5,
        "prefill": {"tokens": 512},
        "decode": {"tokens": 128, "tokens_per_second": tps},
        "fusions": {
            "qkv": True,
            "gate_up": True,
            "add_rms_norm": True,
            "qk_rope_cache": True,
        },
    }


class ModelAbTests(unittest.TestCase):
    def test_main_alternates_order_and_summarizes_pairs(self) -> None:
        args = argparse.Namespace(
            model=pathlib.Path("/tmp/Qwen3-0.6B"),
            baseline_args=[],
            candidate_args=["--shared-gate-up-input"],
            rounds=2,
            prompt=512,
            generate=128,
            warmup=1,
            iterations=5,
            skip_build=True,
            output=None,
        )
        calls: list[bool] = []

        def fake_run(command: list[str]) -> str:
            candidate = "--shared-gate-up-input" in command
            calls.append(candidate)
            return json.dumps(report([150.0, 152.0, 151.0, 149.0][len(calls) - 1]))

        output = io.StringIO()
        with mock.patch.object(model_ab, "arguments", return_value=args), mock.patch.object(
            model_ab, "run_command", side_effect=fake_run
        ), contextlib.redirect_stdout(output):
            model_ab.main()

        self.assertEqual(calls, [False, True, True, False])
        self.assertIn("candidate wins 2/2", output.getvalue())
        self.assertIn("+2.000 tok/s", output.getvalue())

    def test_rejects_profiled_or_invalid_model_output(self) -> None:
        profiled = report(151.0)
        profiled["kernel_profile"] = {"decode": {}}
        with self.assertRaisesRegex(model_ab.BenchmarkError, "profiled"):
            model_ab.decode_tps(profiled, 512, 128, 5)
        invalid = report(151.0)
        invalid["decode"] = None
        with self.assertRaisesRegex(model_ab.BenchmarkError, "prefill or decode"):
            model_ab.decode_tps(invalid, 512, 128, 5)

    def test_failed_command_stops_the_run(self) -> None:
        failed = subprocess.CompletedProcess(args=["bench"], returncode=1, stdout="", stderr="GPU error")
        with mock.patch.object(model_ab.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(model_ab.BenchmarkError, "GPU error"):
                model_ab.run_command(["bench"])


if __name__ == "__main__":
    unittest.main()
