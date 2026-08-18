"""Structured experiment proposal strategies with a Gemini default and safe fallback."""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class StudySpec(BaseModel):
    """Budgeted HPO description; parameter values are validated before a trial starts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    max_trials: PositiveInt
    search_space: dict[str, list[Any]]
    evaluation_suite: str
    strategy: str = "gemini"
    max_invalid_proposals: int = 2


@dataclass(frozen=True, slots=True)
class Proposal:
    """One candidate config patch and a human-readable hypothesis."""

    patch: Mapping[str, Any]
    rationale: str
    source: str


class ProposalStrategy(Protocol):
    def propose(self, study: StudySpec, history: list[Mapping[str, Any]]) -> Proposal: ...


class GridStrategy:
    """Deterministic fallback for offline or provider-failure operation."""

    def propose(self, study: StudySpec, history: list[Mapping[str, Any]]) -> Proposal:
        keys = sorted(study.search_space)
        values = tuple(study.search_space[key] for key in keys)
        candidate_count = 1
        for choices in values:
            candidate_count *= len(choices)
        if candidate_count == 0:
            return Proposal({}, "No search dimensions; run the baseline.", "grid")
        index = min(len(history), candidate_count - 1)
        candidate = next(itertools.islice(itertools.product(*values), index, index + 1))
        return Proposal(
            dict(zip(keys, candidate, strict=True)),
            "Deterministic grid trial.",
            "grid",
        )


class GeminiStrategy:
    """Optional Gemini adapter. It emits JSON only and never launches a process itself."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    def propose(self, study: StudySpec, history: list[Mapping[str, Any]]) -> Proposal:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Install trackmaniarl[orchestrator] to use the Gemini strategy"
            ) from exc
        client = genai.Client(api_key=self.api_key)
        prompt = {
            "instruction": (
                "Return only JSON with keys patch and rationale. Patch may only use "
                "search_space keys and values."
            ),
            "study": study.model_dump(mode="json"),
            "history": history,
        }
        response = client.models.generate_content(model=self.model, contents=json.dumps(prompt))
        try:
            result = json.loads(response.text)
            patch = result["patch"]
            rationale = result["rationale"]
        except (AttributeError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Gemini returned an invalid proposal payload") from exc
        if not isinstance(patch, dict) or not isinstance(rationale, str):
            raise RuntimeError("Gemini proposal must contain an object patch and string rationale")
        _validate_patch(study, patch)
        return Proposal(patch=patch, rationale=rationale, source="gemini")


class OptunaStrategy:
    """Categorical Optuna proposal strategy for the declared YAML search space."""

    def __init__(self, study_name: str) -> None:
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("Install trackmaniarl[orchestrator] to use OptunaStrategy") from exc
        self._optuna: Any = optuna
        self._study = optuna.create_study(study_name=study_name, direction="maximize")
        self._trials: dict[int, Any] = {}

    def propose(self, study: StudySpec, history: list[Mapping[str, Any]]) -> Proposal:
        del history
        trial = self._study.ask()
        patch = {
            key: trial.suggest_categorical(key, choices)
            for key, choices in sorted(study.search_space.items())
        }
        self._trials[trial.number] = trial
        return Proposal(patch, "Optuna categorical proposal.", f"optuna:{trial.number}")

    def complete(self, proposal: Proposal, score: float) -> None:
        """Report a completed trial score after the evaluation suite has finished."""

        _, _, number = proposal.source.partition(":")
        trial = self._trials.pop(int(number))
        self._study.tell(trial, score)


class FallbackStrategy:
    """Use the configured LLM first, then switch deterministically after failures."""

    def __init__(
        self, primary: ProposalStrategy, fallback: ProposalStrategy, max_failures: int = 2
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.max_failures = max_failures
        self.failures = 0

    def propose(self, study: StudySpec, history: list[Mapping[str, Any]]) -> Proposal:
        if self.failures >= self.max_failures:
            return self.fallback.propose(study, history)
        try:
            return self.primary.propose(study, history)
        except (RuntimeError, ValueError):
            self.failures += 1
            return self.fallback.propose(study, history)


def _validate_patch(study: StudySpec, patch: Mapping[str, Any]) -> None:
    unknown = set(patch) - set(study.search_space)
    if unknown:
        raise ValueError(f"Proposal changes keys outside search_space: {sorted(unknown)}")
    invalid = {key: value for key, value in patch.items() if value not in study.search_space[key]}
    if invalid:
        raise ValueError(f"Proposal values are outside search_space: {invalid}")


class StudyLedger:
    """Append-only record of inputs, decisions and outcomes for reproducible autonomy."""

    def __init__(self, directory: str | Path) -> None:
        self.path = Path(directory) / "study.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, payload: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "kind": kind,
                        "payload": dict(payload),
                    },
                    default=str,
                )
                + "\n"
            )


class StudyRunner:
    """Record every proposal and outcome around an externally supplied trial executor."""

    def __init__(self, strategy: ProposalStrategy, ledger: StudyLedger) -> None:
        self.strategy = strategy
        self.ledger = ledger

    def run(self, study: StudySpec, execute: Callable[[Mapping[str, Any]], float]) -> list[float]:
        history: list[Mapping[str, Any]] = []
        scores: list[float] = []
        for trial_index in range(study.max_trials):
            proposal = self.strategy.propose(study, history)
            self.ledger.append(
                "proposal",
                {
                    "trial": trial_index,
                    "patch": proposal.patch,
                    "rationale": proposal.rationale,
                    "source": proposal.source,
                },
            )
            try:
                score = float(execute(proposal.patch))
            except Exception as exc:
                outcome: dict[str, Any] = {
                    "trial": trial_index,
                    "source": proposal.source,
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
                self.ledger.append("outcome", outcome)
                history.append(outcome)
                continue
            outcome = {
                "trial": trial_index,
                "source": proposal.source,
                "status": "completed",
                "score": score,
            }
            self.ledger.append("outcome", outcome)
            history.append(outcome)
            scores.append(score)
            complete = getattr(self.strategy, "complete", None)
            if callable(complete):
                complete(proposal, score)
        return scores
