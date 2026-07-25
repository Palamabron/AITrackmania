"""Portable demonstration-transition files shared by recorders and the learner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from tmrl.core.data import Transition
from tmrl.distributed.codec import WireCodec
from tmrl.distributed.protocol import transition_from_wire, transition_to_wire

DEMO_FORMAT = "tmrl-demo-v1"
DEMO_SUFFIX = ".tmdemo"
_DEMO_CODEC_BYTES = 1 << 30


def save_demonstration(
    path: Path, transitions: Sequence[Transition], metadata: Mapping[str, object]
) -> Path:
    codec = WireCodec(_DEMO_CODEC_BYTES)
    payload = {
        "format": DEMO_FORMAT,
        "metadata": dict(metadata),
        "transitions": [transition_to_wire(item) for item in transitions],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(codec.encode(payload))
    temporary.replace(path)
    return path


def load_demonstration_transitions(path: Path) -> list[Transition]:
    codec = WireCodec(_DEMO_CODEC_BYTES)
    payload = codec.decode(path.read_bytes())
    if not isinstance(payload, Mapping) or payload.get("format") != DEMO_FORMAT:
        raise ValueError(f"{path} is not a {DEMO_FORMAT} demonstration file")
    return [
        replace(transition_from_wire(item), info={"is_demo": True})
        for item in payload["transitions"]
    ]


def resolve_demo_files(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob(f"*{DEMO_SUFFIX}")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"demonstration path not found: {path}")
    if not files:
        raise FileNotFoundError("no .tmdemo files found in the given demonstration paths")
    return files
