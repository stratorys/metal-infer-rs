"""The deterministic list of measurements a run is made of."""

import argparse
from dataclasses import dataclass, field
from typing import Any

SUITES: dict[str, dict[str, Any]] = {
    "showcase": {
        "workload": "context",
        "prompt": [512, 2048, 8192, 32000],
        "generate": 128,
        "engines": ["metal-infer", "mlx", "llama.cpp"],
        "rounds": 5,
        "requests": 3,
        "mode": "server",
    },
}


@dataclass(frozen=True)
class Engine:
    label: str
    kind: str
    overrides: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "kind": self.kind, "overrides": list(self.overrides)}


@dataclass(frozen=True)
class Load:
    type: str
    value: str

    @property
    def key(self) -> str:
        return f"{self.type}={self.value}"

    def as_dict(self) -> dict[str, str]:
        return {"type": self.type, "value": self.value}


@dataclass(frozen=True)
class Measurement:
    round: int
    engine: Engine
    mode: str
    load: Load

    @property
    def key(self) -> str:
        return f"r{self.round + 1}/{self.engine.label}/{self.mode}/{self.load.key}"

    @property
    def group(self) -> str:
        """Measurements sharing a group run in one server session or one offline batch."""
        return f"r{self.round + 1}/{self.engine.label}/{self.mode}"


def apply_suite(args: argparse.Namespace, explicit: set[str]) -> None:
    """Fill the suite's values for every option the command line did not set."""
    if not args.suite:
        return
    values = SUITES[args.suite]
    for name in values:
        if name not in explicit:
            setattr(args, name, values[name])


def engines(args: argparse.Namespace) -> list[Engine]:
    result = [Engine(label=kind, kind=kind) for kind in args.engines]
    result += [
        Engine(label=f"metal-infer {candidate}", kind="metal-infer", overrides=[candidate])
        for candidate in args.candidate
    ]
    return result


def modes(args: argparse.Namespace) -> list[str]:
    return ["offline", "server"] if args.mode == "all" else [args.mode]


def loads(args: argparse.Namespace, mode: str) -> list[Load]:
    if mode == "offline":
        return [Load(type="prompt", value=str(prompt)) for prompt in args.prompt]
    if args.workload == "context":
        return [Load(type="context", value=str(prompt)) for prompt in args.prompt]
    if args.workload == "sharegpt":
        return [Load(type="rate", value=rate) for rate in args.request_rate]
    return [Load(type="concurrency", value=str(level)) for level in args.concurrency]


def measurements(args: argparse.Namespace) -> list[Measurement]:
    """Rounds, then engines in alternating order, then modes, then loads."""
    all_engines = engines(args)
    result = []
    for round_index in range(args.rounds):
        order = all_engines if round_index % 2 == 0 else list(reversed(all_engines))
        for engine in order:
            for mode in modes(args):
                for load in loads(args, mode):
                    result.append(
                        Measurement(round=round_index, engine=engine, mode=mode, load=load)
                    )
    return result
