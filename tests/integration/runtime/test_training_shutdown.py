import logging
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from trackmaniarl.commands.training import (
    _LocalProcesses,
    _stop_local_processes_with_notice,
)
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_runtime import _run_offline_pretraining, close_runtime
from trackmaniarl.distributed.coordinator_support import _AsyncCheckpointWriter


class _ShutdownProcess:
    exitcode = 0
    pid = 1

    def __init__(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, *, timeout: float) -> None:
        del timeout
        self.alive = False

    def terminate(self) -> None:
        self.alive = False


class _InterruptingShutdownProcess(_ShutdownProcess):
    def __init__(self) -> None:
        super().__init__()
        self.pending_interrupt = True

    def join(self, *, timeout: float) -> None:
        if self.pending_interrupt:
            self.pending_interrupt = False
            raise KeyboardInterrupt
        super().join(timeout=timeout)


class _InterruptedWait:
    def __init__(self) -> None:
        self.calls = 0

    def result(self) -> None:
        self.calls += 1
        if self.calls == 1:
            raise KeyboardInterrupt

    def done(self) -> bool:
        return self.calls > 1


class _CloseProbe:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.closed = False

    def close(self) -> None:
        self.closed = True
        if self.failure is not None:
            raise self.failure


def test_graceful_shutdown_ignores_repeated_interrupt_until_checkpoint_is_safe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    processes = _LocalProcesses(
        learner=_ShutdownProcess(),
        actor=_InterruptingShutdownProcess(),
        shutdown=SimpleNamespace(set=lambda: None),
        endpoint="127.0.0.1:8787",
    )

    _stop_local_processes_with_notice(processes)

    output = capsys.readouterr().out
    assert "Ctrl+C was ignored to protect the checkpoint" in output
    assert "Safe shutdown complete" in output


def test_checkpoint_writer_retains_an_in_progress_write_after_interrupted_wait() -> None:
    writer = _AsyncCheckpointWriter(SimpleNamespace())
    pending = _InterruptedWait()
    writer.pending = pending
    try:
        with pytest.raises(KeyboardInterrupt):
            writer.wait()
        assert writer.pending is pending
        writer.wait()
        assert writer.pending is None
    finally:
        writer.close()


def test_runtime_closes_the_journal_after_a_checkpoint_writer_failure() -> None:
    writer = _CloseProbe(OSError("checkpoint failed"))
    journal = _CloseProbe()
    coordinator = cast(
        Coordinator,
        SimpleNamespace(
            _server=None,
            _rpc_executor=None,
            _checkpoint_writer=writer,
            journal=journal,
        ),
    )

    with pytest.raises(OSError, match="checkpoint failed"):
        close_runtime(coordinator)

    assert writer.closed
    assert journal.closed


def _offline_spec() -> SimpleNamespace:
    training = SimpleNamespace(offline_pretrain_updates=1, save_final_checkpoint=True)
    return SimpleNamespace(training=training)


class _OfflineInterruptHarness:
    def __init__(self, tmp_path: Path, captured: dict[str, int]) -> None:
        self.tmp_path = tmp_path
        self.captured = captured

    def interrupt(self) -> None:
        raise KeyboardInterrupt

    def save_checkpoint(self) -> Path:
        self.captured["checkpoints"] += 1
        return self.tmp_path / "interrupted.pt"

    def wait(self) -> None:
        self.captured["waits"] += 1


def _interrupted_coordinator(tmp_path: Path, captured: dict[str, int]) -> Coordinator:
    harness = _OfflineInterruptHarness(tmp_path, captured)
    coordinator = SimpleNamespace(
        run=SimpleNamespace(spec=_offline_spec()),
        demo_paths=(tmp_path / "demo.npz",),
        journal=SimpleNamespace(has_history=lambda: False),
        _prepare_training=lambda: None,
        _import_demonstrations=lambda: None,
        _offline_pretrain=harness.interrupt,
        _checkpoint=harness.save_checkpoint,
        _checkpoint_writer=SimpleNamespace(wait=harness.wait),
    )
    return cast(Coordinator, coordinator)


def test_interrupted_offline_pretraining_writes_a_final_checkpoint(tmp_path: Path) -> None:
    captured = {"checkpoints": 0, "waits": 0}
    coordinator = _interrupted_coordinator(tmp_path, captured)

    with pytest.raises(KeyboardInterrupt):
        _run_offline_pretraining(coordinator)

    assert captured == {"checkpoints": 1, "waits": 1}


def test_interrupted_offline_pretraining_reports_disabled_final_checkpoint(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    captured = {"checkpoints": 0, "waits": 0}
    coordinator = _interrupted_coordinator(tmp_path, captured)
    coordinator.run.spec.training.save_final_checkpoint = False

    with caplog.at_level(logging.WARNING), pytest.raises(KeyboardInterrupt):
        _run_offline_pretraining(coordinator)

    assert captured == {"checkpoints": 0, "waits": 1}
    assert "Final checkpoint is disabled" in caplog.text
    assert "Saving a final checkpoint" not in caplog.text
