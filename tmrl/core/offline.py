"""Offline expert trajectories loaded into replay before online collection."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from tmrl.core.contracts import ReplayStore
from tmrl.core.data import Trajectory, Transition

TRAJECTORY_FORMAT = "tmrl-offline-trajectory-v1"
_TRAJECTORY_SUFFIXES = {".pkl"}


def save_trajectory(path: str | Path, trajectory: Trajectory) -> Path:
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".h5":
        raise ValueError("HDF5 trajectories are not supported; save as .pkl")
    if suffix != ".pkl":
        target = target.with_suffix(".pkl")
    if not trajectory.transitions:
        raise ValueError("trajectory must contain at least one transition")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": TRAJECTORY_FORMAT,
        "episode_id": trajectory.episode_id,
        "transitions": list(trajectory.transitions),
        "metadata": dict(trajectory.metadata),
    }
    with target.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def load_trajectory(path: str | Path) -> Trajectory:
    source = Path(path)
    if source.suffix.lower() == ".h5":
        raise ValueError("HDF5 trajectories are not supported; load a .pkl file")
    with source.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict) or payload.get("format") != TRAJECTORY_FORMAT:
        raise ValueError("unsupported offline trajectory format")
    transitions = payload["transitions"]
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("trajectory must contain at least one transition")
    if not all(isinstance(item, Transition) for item in transitions):
        raise ValueError("trajectory transitions must be Transition objects")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("trajectory metadata must be a mapping")
    return Trajectory(
        episode_id=str(payload["episode_id"]),
        transitions=transitions,
        metadata=dict(metadata),
    )


def demonstration_files(path: str | Path) -> tuple[Path, ...]:
    source = Path(path)
    if source.is_dir():
        matches = tuple(sorted(item.resolve() for item in source.glob("*.pkl") if item.is_file()))
        if not matches:
            raise FileNotFoundError(f"offline trajectory directory has no .pkl files: {source}")
        return matches
    if source.suffix.lower() not in _TRAJECTORY_SUFFIXES:
        raise ValueError(f"offline trajectory file must be a .pkl archive: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"offline trajectory path does not exist: {source}")
    return (source.resolve(),)


def protect_demo(transition: Transition) -> Transition:
    info = dict(transition.info)
    info["is_demo"] = True
    info["source"] = "demo"
    return replace(transition, info=info)


class OfflineBufferLoader:
    """Seeds a replay store with expert transitions marked as protected demos."""

    def __init__(self, store: ReplayStore) -> None:
        self.store = store

    def load_demonstrations(self, path: Path) -> int:
        imported = 0
        for file in demonstration_files(path):
            for transition in load_trajectory(file).transitions:
                self.store.append(protect_demo(transition))
                imported += 1
        return imported
