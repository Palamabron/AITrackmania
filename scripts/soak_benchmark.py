from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

if TYPE_CHECKING or __package__:
    from scripts.soak_evidence import checkpoint_file
    from scripts.soak_types import (
        Check,
        Checkpoint,
        VerificationInputError,
        add_check,
        integer,
        mapping,
        number,
        sha256,
        string,
    )
else:
    from soak_evidence import checkpoint_file
    from soak_types import (
        Check,
        Checkpoint,
        VerificationInputError,
        add_check,
        integer,
        mapping,
        number,
        sha256,
        string,
    )


@dataclass(frozen=True, slots=True)
class AcceptanceThresholds:
    trials_per_map: int
    minimum_finish_rate: float
    target_median_s: float


@dataclass(frozen=True, slots=True)
class TrialAcceptance:
    passed: bool
    finished: int
    median_s: float | None


@dataclass(frozen=True, slots=True)
class BenchmarkContext:
    evaluation: dict[str, object] | None
    manifest: dict[str, object]
    assets: list[dict[str, object]]
    final_checkpoint: Checkpoint | None
    run_dir: Path
    checks: list[Check]


@dataclass(frozen=True, slots=True)
class BenchmarkAnalysis:
    trials: list[dict[str, object]]
    acceptance: TrialAcceptance
    error_indices: list[int]
    structure_valid: bool
    assets_valid: bool
    checkpoint_bound: bool


@dataclass(frozen=True, slots=True)
class BenchmarkAnalysisInput:
    context: BenchmarkContext
    evaluation: dict[str, object]
    suite: dict[str, object]
    trials: list[dict[str, object]]
    acceptance: TrialAcceptance


@dataclass(frozen=True, slots=True)
class BenchmarkReportData:
    analysis: BenchmarkAnalysis
    final_checkpoint: Checkpoint | None
    run_dir: Path


def _manifest_evaluation(manifest: dict[str, object]) -> dict[str, object]:
    config = mapping(manifest.get("config"), "manifest.config")
    return mapping(config.get("evaluation"), "manifest.config.evaluation")


def _benchmark_trials(evaluation: dict[str, object]) -> list[dict[str, object]]:
    value = evaluation.get("trials")
    if not isinstance(value, list) or not value:
        raise VerificationInputError("evaluation.json.trials must be a non-empty list")
    return [mapping(item, f"evaluation.json.trials[{index}]") for index, item in enumerate(value)]


def _trial_acceptance(
    trials: list[dict[str, object]],
    suite: dict[str, object],
    assets: list[dict[str, object]],
) -> TrialAcceptance:
    thresholds = _acceptance_thresholds(suite)
    expected = _expected_trials(assets, thresholds.trials_per_map)
    observed = _observed_trials(trials)
    finished_times = _finished_times(trials)
    finish_rate = len(finished_times) / sum(expected.values()) if expected else 0.0
    median_s = median(finished_times) if finished_times else None
    passed = (
        observed == expected
        and finish_rate >= thresholds.minimum_finish_rate
        and median_s is not None
        and median_s < thresholds.target_median_s
    )
    return TrialAcceptance(passed, len(finished_times), median_s)


def _finished_times(trials: list[dict[str, object]]) -> list[float]:
    return [
        number(trial.get("finish_time_s"), "finished trial finish_time_s")
        for trial in trials
        if trial.get("finished") is True
    ]


def _acceptance_thresholds(suite: dict[str, object]) -> AcceptanceThresholds:
    return AcceptanceThresholds(
        integer(suite.get("trials_per_map"), "evaluation trials_per_map"),
        number(suite.get("min_finish_rate"), "evaluation min_finish_rate"),
        number(suite.get("target_median_s"), "evaluation target_median_s"),
    )


def _expected_trials(assets: list[dict[str, object]], trials_per_map: int) -> Counter[str]:
    return Counter(
        {string(asset.get("map_id"), "evaluation asset map_id"): trials_per_map for asset in assets}
    )


def _observed_trials(trials: list[dict[str, object]]) -> Counter[str]:
    return Counter(string(trial.get("map_id"), "benchmark trial map_id") for trial in trials)


def _benchmark_structure(
    evaluation: dict[str, object],
    suite: dict[str, object],
    assets: list[dict[str, object]],
) -> bool:
    artifact_suite = mapping(evaluation.get("suite"), "evaluation.json.suite")
    protocols = {
        string(asset.get("plugin_protocol_version"), "evaluation asset protocol")
        for asset in assets
    }
    protocol = evaluation.get("plugin_protocol_version")
    return (
        evaluation.get("schema_version") == "1"
        and artifact_suite.get("name") == suite.get("name")
        and artifact_suite.get("version") == suite.get("version")
        and isinstance(protocol, str)
        and protocol in protocols
        and isinstance(evaluation.get("metrics"), dict)
    )


def benchmark_evidence(context: BenchmarkContext) -> dict[str, object]:
    evaluation = context.evaluation
    if evaluation is None:
        _record_missing_benchmark(context.checks)
        return {"present": False}
    suite = _manifest_evaluation(context.manifest)
    trials = _benchmark_trials(evaluation)
    acceptance = _trial_acceptance(trials, suite, context.assets)
    analysis_input = BenchmarkAnalysisInput(context, evaluation, suite, trials, acceptance)
    analysis = _benchmark_analysis(analysis_input)
    _record_benchmark_checks(context, evaluation, analysis)
    data = BenchmarkReportData(analysis, context.final_checkpoint, context.run_dir)
    return _benchmark_report(evaluation, data)


def _benchmark_analysis(data: BenchmarkAnalysisInput) -> BenchmarkAnalysis:
    expected_uids = {
        string(asset.get("map_uid"), "evaluation asset map_uid") for asset in data.context.assets
    }
    observed_uids = {
        string(trial.get("map_uid"), "benchmark trial map_uid") for trial in data.trials
    }
    return BenchmarkAnalysis(
        data.trials,
        data.acceptance,
        _error_trial_indices(data.trials),
        _benchmark_structure(data.evaluation, data.suite, data.context.assets),
        observed_uids == expected_uids,
        _checkpoint_is_bound(
            data.evaluation,
            data.context.final_checkpoint,
            data.context.run_dir,
        ),
    )


def _record_benchmark_checks(
    context: BenchmarkContext,
    evaluation: dict[str, object],
    analysis: BenchmarkAnalysis,
) -> None:
    for check in (
        _final_benchmark_check(analysis),
        _trial_health_check(analysis),
        _benchmark_binding_check(evaluation, analysis),
    ):
        add_check(context.checks, check)


def _final_benchmark_check(analysis: BenchmarkAnalysis) -> Check:
    passed = analysis.structure_valid and analysis.assets_valid and analysis.acceptance.passed
    return Check("final_benchmark_artifact", passed, _benchmark_detail(analysis))


def _trial_health_check(analysis: BenchmarkAnalysis) -> Check:
    return Check(
        "benchmark_trial_health",
        not analysis.error_indices,
        f"error_trial_indices={analysis.error_indices}",
    )


def _benchmark_binding_check(evaluation: dict[str, object], analysis: BenchmarkAnalysis) -> Check:
    name = _reported_checkpoint_name(evaluation)
    return Check("benchmark_checkpoint_binding", analysis.checkpoint_bound, f"checkpoint={name}")


def _benchmark_detail(analysis: BenchmarkAnalysis) -> str:
    acceptance = analysis.acceptance
    return (
        f"trials={len(analysis.trials)}, finished={acceptance.finished}, "
        f"median_s={acceptance.median_s}"
    )


def _record_missing_benchmark(checks: list[Check]) -> None:
    for name in (
        "final_benchmark_artifact",
        "benchmark_trial_health",
        "benchmark_checkpoint_binding",
    ):
        add_check(checks, Check(name, False, "evaluation.json is missing"))


def _error_trial_indices(trials: list[dict[str, object]]) -> list[int]:
    return [
        index
        for index, trial in enumerate(trials)
        if "telemetry_error" not in trial
        or trial.get("telemetry_error") is not None
        or "controller_error" not in trial
        or trial.get("controller_error") is not None
    ]


def _reported_checkpoint(evaluation: dict[str, object]) -> str:
    value = evaluation.get("checkpoint")
    return value if isinstance(value, str) else ""


def _reported_checkpoint_name(evaluation: dict[str, object]) -> str:
    reported = _reported_checkpoint(evaluation)
    return Path(reported).name if reported else "-"


def _checkpoint_is_bound(
    evaluation: dict[str, object], final_checkpoint: Checkpoint | None, run_dir: Path
) -> bool:
    reported = _reported_checkpoint(evaluation)
    return (
        final_checkpoint is not None
        and bool(reported)
        and checkpoint_file(run_dir, reported) == final_checkpoint.file
        and final_checkpoint.file.is_file()
    )


def _checkpoint_fields(
    final_checkpoint: Checkpoint | None, run_dir: Path
) -> tuple[str | None, str | None]:
    if final_checkpoint is None:
        return None, None
    path = final_checkpoint.file.relative_to(run_dir).as_posix()
    digest = sha256(final_checkpoint.file) if final_checkpoint.file.is_file() else None
    return path, digest


def _benchmark_report(
    evaluation: dict[str, object], data: BenchmarkReportData
) -> dict[str, object]:
    checkpoint_path, checkpoint_digest = _checkpoint_fields(data.final_checkpoint, data.run_dir)
    evaluation_path = data.run_dir / "evaluation.json"
    acceptance = data.analysis.acceptance
    return {
        "present": True,
        "path": "evaluation.json",
        "sha256": sha256(evaluation_path),
        "size_bytes": evaluation_path.stat().st_size,
        "schema_version": evaluation.get("schema_version"),
        "trial_count": len(data.analysis.trials),
        "finished_trials": acceptance.finished,
        "median_finish_time_s": acceptance.median_s,
        "error_trial_indices": data.analysis.error_indices,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_digest,
    }
