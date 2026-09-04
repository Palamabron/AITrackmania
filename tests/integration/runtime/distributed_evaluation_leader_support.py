from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Any

from tests.integration.runtime.distributed_evaluation_support import _evaluation_coordinator
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_checkpoint import EvaluatedPolicyCheckpoint
from trackmaniarl.distributed.coordinator_leaders import (
    EvaluationCandidate,
    record_evaluation_leaders,
)
from trackmaniarl.distributed.coordinator_support import (
    _AsyncCheckpointWriter,
    _CheckpointWrite,
)


def _candidate(version: int, finish_rate: float, times: tuple[float, float]) -> EvaluationCandidate:
    trials = 5
    median_s, best_s = times
    return EvaluationCandidate(
        finish_rate,
        round(finish_rate * trials),
        trials,
        best_s,
        median_s,
        median_s,
        version,
    )


class _FailingLeaderProbe:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.attempts = 0

    def save(self, evaluated: EvaluatedPolicyCheckpoint) -> Path:
        self.attempts += 1
        assert evaluated.on_failed is not None
        evaluated.on_failed(OSError("disk full"))
        return self.directory / "fastest-eval.pt"


class _DelayedFirstCheckpointFailure:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.attempts = 0

    def save(self, state: Mapping[str, Any], path: Path) -> None:
        del state
        self.attempts += 1
        if self.attempts != 1:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")
            return
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise TimeoutError("checkpoint failure was not released")
        raise OSError("delayed disk failure")


class _ObservedCheckpointWriter(_AsyncCheckpointWriter):
    def __init__(self, codec: Any) -> None:
        super().__init__(codec)
        self._observe_wait = False
        self.wait_entered = Event()

    def observe_next_wait(self) -> None:
        self._observe_wait = True
        self.wait_entered.clear()

    def wait(self) -> None:
        if self._observe_wait:
            self._observe_wait = False
            self.wait_entered.set()
        super().wait()


class _AsyncLeaderCheckpointProbe:
    def __init__(self, writer: _AsyncCheckpointWriter, directory: Path) -> None:
        self.writer = writer
        self.directory = directory

    def save(self, evaluated: EvaluatedPolicyCheckpoint) -> Path:
        path = self.directory / f"{evaluated.kind.value}-{evaluated.version}.pt"

        def saved() -> None:
            if evaluated.on_saved is not None:
                evaluated.on_saved(path)

        self.writer.submit(_CheckpointWrite({}, path, saved, evaluated.on_failed))
        return path


class _DeferredLeaderCheckpointProbe:
    def __init__(self, directory: Path) -> None:
        self.path = directory / "fastest-eval.pt"
        self.evaluated: EvaluatedPolicyCheckpoint | None = None

    def save(self, evaluated: EvaluatedPolicyCheckpoint) -> Path:
        self.evaluated = evaluated
        return self.path

    def complete(self) -> None:
        assert self.evaluated is not None
        assert self.evaluated.on_saved is not None
        self.evaluated.on_saved(self.path)


class _StepRecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, int | None]] = []

    def log(self, event: str, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        del payload
        self.records.append((event, step))


@dataclass(slots=True)
class _DelayedFailureScenario:
    coordinator: Coordinator
    codec: _DelayedFirstCheckpointFailure
    writer: _ObservedCheckpointWriter
    events: list[tuple[str, dict[str, Any]]]
    errors: list[BaseException]
    first: EvaluationCandidate
    second: EvaluationCandidate

    def begin(self) -> tuple[Thread, tuple[float, float, float] | None]:
        record_evaluation_leaders(self.coordinator, self.first)
        assert self.codec.started.wait(timeout=5.0)
        first_rank = self.coordinator._fastest_evaluation
        self.writer.observe_next_wait()
        thread = Thread(target=self.record_second)
        thread.start()
        assert self.writer.wait_entered.wait(timeout=5.0)
        return thread, first_rank

    def record_second(self) -> None:
        try:
            record_evaluation_leaders(self.coordinator, self.second)
        except BaseException as exc:
            self.errors.append(exc)

    def resolve_failure(
        self, thread: Thread, first_rank: tuple[float, float, float] | None
    ) -> None:
        assert self.coordinator._fastest_evaluation == first_rank
        self.codec.release.set()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert len(self.errors) == 1
        assert isinstance(self.errors[0], OSError)
        assert self.coordinator._fastest_evaluation is None

    def retry(self) -> None:
        record_evaluation_leaders(self.coordinator, self.second)
        self.writer.wait()
        assert self.coordinator._fastest_evaluation is not None
        fastest = [payload for event, payload in self.events if event == "eval/fastest_checkpoint"]
        assert [payload["policy_version"] for payload in fastest] == [42]

    def close(self) -> None:
        self.codec.release.set()
        self.writer.close()


def _delayed_failure_scenario(tmp_path: Path) -> _DelayedFailureScenario:
    events: list[tuple[str, dict[str, Any]]] = []
    coordinator = _evaluation_coordinator(tmp_path, events, [])
    coordinator.run.spec.evaluation.min_finish_rate = 1.0
    coordinator._evaluation_policy_states[42] = {"weight": 2.0}
    codec = _DelayedFirstCheckpointFailure()
    writer = _ObservedCheckpointWriter(codec)
    coordinator._checkpoint_writer = writer
    coordinator._checkpoint = _AsyncLeaderCheckpointProbe(writer, tmp_path).save
    first = _candidate(41, 0.8, (42.0, 41.0))
    second = _candidate(42, 0.8, (40.0, 39.0))
    return _DelayedFailureScenario(coordinator, codec, writer, events, [], first, second)
