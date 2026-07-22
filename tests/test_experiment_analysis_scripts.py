from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FETCH = _load_script("fetch_analysis")
REPORT = _load_script("iteration_report")


class FakeRun:
    id = "z67iytmc"
    name = "professional-run"
    state = "running"
    url = "https://wandb.ai/dsc-pjatk-warsaw/my-trackmania-agent/runs/z67iytmc"
    created_at = "2026-07-21T12:00:00Z"
    updated_at = "2026-07-21T12:05:00Z"
    config: Mapping[str, Any] = {"training": {"batch_size": 512}}
    summary: Mapping[str, Any] = {"learner/loss/value": 1.0}

    def scan_history(self, *, page_size: int) -> Iterable[Mapping[str, Any]]:
        assert page_size == 1_000
        return (
            {
                "_step": index,
                "learner/loss/value": float(index),
                "episode/return": float(index * 2),
                "replay/fill_fraction": index / 12,
                "episode/finished": index == 12,
                "episode/termination": "finished" if index == 12 else "no_progress",
                "label": "not numeric",
            }
            for index in range(1, 13)
        )


def test_parse_run_path_requires_entity_project_and_run_id() -> None:
    assert FETCH.parse_run_path("/entity/project/run/") == "entity/project/run"
    with pytest.raises(ValueError, match="ENTITY/PROJECT/RUN_ID"):
        FETCH.parse_run_path("entity/project")


def test_create_api_rejects_an_incomplete_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wandb", object())

    with pytest.raises(RuntimeError, match="incomplete"):
        FETCH._create_api(30)


def test_explicit_run_does_not_read_the_experiment_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called() -> list[dict[str, Any]]:
        raise AssertionError("explicit W&B runs must not query the registry")

    monkeypatch.setattr(FETCH, "_read_registry", fail_if_called)
    args = argparse.Namespace(
        run=["entity/project/run"],
        exp_id="",
        entity="unused",
        project="unused",
    )

    assert FETCH._requests_from_args(args) == [FETCH.RunRequest("entity/project/run", "run")]


def test_analyze_run_normalizes_tmrl_history() -> None:
    analysis = FETCH.analyze_run(FakeRun(), "entity/project/z67iytmc", "z67iytmc")

    assert analysis["schema_version"] == "2.0"
    assert analysis["history"]["rows_scanned"] == 12
    assert analysis["history"]["metric_groups"] == {
        "episode": ["episode/finished", "episode/return"],
        "learner": ["learner/loss/value"],
        "replay": ["replay/fill_fraction"],
    }
    loss = analysis["metrics"]["learner/loss/value"]
    assert loss["last"] == 12.0
    assert loss["recent_delta"] == 6.0
    assert loss["recent_relative_delta"] == pytest.approx(4.0)
    assert analysis["metrics"]["episode/finished"]["last"] == 1.0
    assert analysis["categories"]["episode/termination"] == {"finished": 1, "no_progress": 11}


def test_analyze_run_reports_empty_history() -> None:
    class EmptyRun(FakeRun):
        def scan_history(self, *, page_size: int) -> Iterable[Mapping[str, Any]]:
            return ()

    analysis = FETCH.analyze_run(EmptyRun(), "entity/project/empty", "empty")

    assert analysis["history"]["rows_scanned"] == 0
    assert analysis["metrics"] == {}
    assert analysis["warnings"] == [
        "History is empty; only run metadata and W&B summary are available."
    ]


def test_report_renders_highlights_and_stability_warning() -> None:
    analysis = FETCH.analyze_run(FakeRun(), "entity/project/z67iytmc", "z67iytmc")
    report = REPORT.build_json_report({"z67iytmc": analysis})
    markdown = REPORT.build_markdown_report(report)

    experiment = report["experiments"][0]
    assert experiment["run"]["state"] == "running"
    episode_rows = experiment["sections"]["episode_health"]
    assert next(row for row in episode_rows if row["name"] == "episode/return")["last"] == 24.0
    assert any("learner/loss/value rose" in alert for alert in experiment["alerts"])
    assert "## z67iytmc" in markdown
    assert "`episode/return`" in markdown
