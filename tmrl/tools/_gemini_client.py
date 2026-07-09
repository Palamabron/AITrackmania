"""Gemini AI client and make_decision dispatcher for the orchestrator."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
from typing import Any

from tmrl.tools._experiment_io import EXPERIMENTS_DIR, REPO_ROOT
from tmrl.tools._experiment_io import load_dotenv as _load_dotenv
from tmrl.tools._heuristic_decisions import _decide, _propose
from tmrl.tools._orchestrator_utils import _log
from tmrl.tools._snapshot_analyze import _extract_json_from_output

VALID_OVERRIDE_SECTIONS = {
    "algorithm",
    "training",
    "model",
    "environment",
    "player_runs",
    "run",
    "wandb",
    "distributed",
}

# Map common param names to their correct section
PARAM_TO_SECTION: dict[str, str] = {
    "iqn_lr": "algorithm",
    "gamma": "algorithm",
    "iqn_grad_clip": "algorithm",
    "iqn_epsilon_decay_steps": "algorithm",
    "iqn_epsilon_start": "algorithm",
    "iqn_epsilon_end": "algorithm",
    "n_steps": "algorithm",
    "iqn_soft_target_tau": "algorithm",
    "backup_clip_range": "algorithm",
    "iqn_huber_kappa": "algorithm",
    "iqn_dueling": "algorithm",
    "iqn_double_dqn": "algorithm",
    "iqn_sort_quantiles": "algorithm",
    "reward_normalize_scale": "algorithm",
    "batch_size": "training",
    "training_steps_per_round": "training",
    "rounds_per_epoch": "training",
    "environment_steps_before_training": "training",
    "max_training_steps_per_environment_step": "training",
    "update_model_interval": "training",
    "update_buffer_interval": "training",
    "residual_mlp_hidden_dim": "model",
    "residual_mlp_num_blocks": "model",
    "gnn_layers": "model",
    "gnn_hidden": "model",
    "binary_brake": "model",
    "end_of_track_reward": "environment",
    "crash_penalty": "environment",
    "speed_reward_weight": "environment",
    "constant_penalty": "environment",
    "demo_max_batch_fraction": "player_runs",
    "buffers_maxlen": "run",
    "rw_max_samples_per_episode": "run",
}


def _fix_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Fix malformed overrides from Gemini (wrong section keys, dot notation)."""
    fixed: dict[str, Any] = {}

    for section, params in overrides.items():
        if not isinstance(params, dict):
            continue

        if section in VALID_OVERRIDE_SECTIONS:
            # Valid section -- but check for dot-notation keys inside
            clean_params: dict[str, Any] = {}
            for key, val in params.items():
                if "." in key:
                    # e.g. "algorithm.iqn_grad_clip" -> extract param name
                    real_key = key.split(".")[-1]
                    real_section = PARAM_TO_SECTION.get(real_key, section)
                    fixed.setdefault(real_section, {})[real_key] = val
                else:
                    clean_params[key] = val
            if clean_params:
                fixed.setdefault(section, {}).update(clean_params)
        else:
            # Invalid section (e.g. "optimization", "rl_algorithm")
            # Try to remap each param to its correct section
            for key, val in params.items():
                real_key = key.split(".")[-1] if "." in key else key
                real_section = PARAM_TO_SECTION.get(real_key)  # type: ignore[assignment]
                if real_section:
                    fixed.setdefault(real_section, {})[real_key] = val
                else:
                    # Unknown param -- put in algorithm as best guess
                    fixed.setdefault("algorithm", {})[real_key] = val

    return fixed


def _call_gemini(prompt: str, *, retries: int = 3) -> str | None:
    """Call Gemini API and return the text response.

    Retries transient failures with exponential backoff.
    """
    _load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        _log("WARNING: GEMINI_API_KEY not set, falling back to heuristic")
        return None

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        _log(f"Gemini SDK not installed: {exc}")
        return None

    client = genai.Client(api_key=api_key)
    last_err: str = ""
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    top_p=0.9,
                ),
            )
            return response.text
        except Exception as exc:
            last_err = str(exc)
            if attempt < retries:
                delay = 10 * (2 ** (attempt - 1))
                _log(
                    f"Gemini attempt {attempt}/{retries} failed ({exc!r}), retrying in {delay}s..."
                )
                time.sleep(delay)

    _log(f"Gemini API failed after {retries} attempts: {last_err}")
    return None


def _build_decide_prompt(context: dict[str, Any]) -> str:
    snapshot = context.get("snapshot", {})
    target = context.get("target_finish_time_s", 36.0)
    elapsed = context.get("elapsed_hours", 0)
    max_hours = context.get("current_max_hours", 4)
    exp_entry = context.get("exp_entry", {})
    recent = snapshot.get("recent_metrics", {})

    return f"""You are an ML experiment analyst for a TrackMania reinforcement learning agent.

TARGET: Finish the track in {target}s or less.
Primary metric: eval/finish_time_test_s > 0 (lower is better).
Note: eval logs 0.0 when the agent did NOT finish that eval episode;
use min of positive values only.

CURRENT EXPERIMENT: {exp_entry.get("exp_id", "unknown")}
HYPOTHESIS: {exp_entry.get("hypothesis", "N/A")}
ELAPSED: {elapsed:.1f}h / {max_hours}h max
OVERRIDES: {json.dumps(exp_entry.get("config_overrides", {}), indent=2)}

RECENT METRICS (last ~100 trainer steps):
{json.dumps(recent, indent=2, default=str)}

SNAPSHOT SUMMARY (trust these before claiming "never finished"):
- best_finish_time_s: {snapshot.get("best_finish_time_s", "None (no positive finish yet)")}
- last_finish_time_s: {snapshot.get("last_finish_time_s", "N/A")}
- worker_best_finish_time_s: {snapshot.get("worker_best_finish_time_s", "N/A")}
- worker_finish_count: {snapshot.get("worker_finish_count", 0)} (episodes with run/finish_time > 0)
- trainer_state: {snapshot.get("trainer_state", "unknown")}
- worker_state: {snapshot.get("worker_state", "unknown")}

IMPORTANT - IQN loss scale for this project:
- loss/iqn_loss in the 30-90 range is COMMON while Q-values are stable (max_q roughly 15-50).
- Do NOT stop solely because recent loss last/p95 exceeds 50.
- Only treat loss as diverged if last > 100, or Q-values explode (|max_q| > 200),
  or loss trends sharply upward with collapsing returns.

NOTE ON GRADIENTS: Pre-clip gradient norms are structurally 50-150x the clip limit for this
architecture. This is NORMAL and must NOT be used as a stop criterion. Only stop on gradients
if grad_norm itself is NaN.

SIGNS TO STOP EARLY:
- Q-values exploding (max_q > 200 or min_q < -50)
- Loss last > 100 or NaN
- No positive best_finish_time_s AND worker_finish_count == 0 after 2+ hours
- Buffer empty (connection issue)

SIGNS TO CONTINUE:
- worker_finish_count >= 5 — STRONG signal to continue, even if other metrics look bad
- best_finish_time_s > 0 (even if >> target) — learning to finish
- worker_finish_count increasing
- Returns increasing (even slowly)
- Q-values in reasonable range (roughly 0-50 for max_q in this setup)
- Epsilon still decaying (still exploring)

Respond with ONLY a JSON object (no markdown, no explanation):
{{"action": "continue" or "stop", "reason": "brief explanation"}}"""


def _build_propose_prompt(context: dict[str, Any]) -> str:
    registry = context.get("registry", [])
    target = context.get("target_finish_time_s", 36.0)

    # Load search space
    search_space_text = ""
    sp_path = EXPERIMENTS_DIR / "search_space.yaml"
    if sp_path.exists():
        search_space_text = sp_path.read_text(encoding="utf-8")[:3000]

    # Load decisions log
    decisions_text = ""
    dec_path = EXPERIMENTS_DIR / "decisions.md"
    if dec_path.exists():
        decisions_text = dec_path.read_text(encoding="utf-8")[-3000:]

    # Load validation report (produced by scripts/validate_decisions.py)
    validation_text = ""
    val_path = EXPERIMENTS_DIR / "validation_report.json"
    if val_path.exists():
        try:
            val = json.loads(val_path.read_text(encoding="utf-8"))
            parts = [
                f"Errors: {val.get('error_count', 0)}, Warnings: {val.get('warning_count', 0)}"
            ]
            for lb in val.get("leaderboard", [])[:5]:
                parts.append(f"  #{lb['rank']} {lb['exp_id']}: {lb['best_time_s']:.1f}s")
            for f in val.get("findings", []):
                if f.get("severity") == "WARNING" and f.get("category") in (
                    "gradient_obsession",
                    "premature_stops",
                    "leaderboard_mismatch",
                ):
                    parts.append(f"  [{f['category']}] {f['claim']}")
            validation_text = "VALIDATION REPORT:\n" + "\n".join(parts)
        except Exception:
            pass

    # Build rich experiment summary with analysis data
    reg_summary = []
    for e in registry:
        parts = [
            f"- {e['exp_id']}: status={e.get('status')}",
            f"  overrides={json.dumps(e.get('config_overrides', {}))}",
        ]
        sm = e.get("summary_metrics") or {}
        ft = sm.get("best_finish_time_s")
        if ft and ft > 0:
            parts.append(f"  best_finish={ft:.2f}s")

        ap = EXPERIMENTS_DIR / "analysis" / f"{e['exp_id']}.json"
        if ap.exists():
            try:
                a = json.loads(ap.read_text(encoding="utf-8"))
                lo = a.get("metrics", {}).get("loss/iqn_loss", {})
                if lo.get("median"):
                    parts.append(f"  loss_median={lo['median']:.1f}")
                ret = a.get("metrics", {}).get("metrics/return_train", {})
                if ret.get("last"):
                    parts.append(f"  return_last={ret['last']:.0f}")
                w = a.get("worker", {})
                if w.get("finish_rate"):
                    parts.append(f"  worker_finish_rate={w['finish_rate']:.1%}")
                if w.get("finish_count"):
                    parts.append(f"  worker_finishes={w['finish_count']}")
                trends = a.get("training_trends", {})
                if trends:
                    dirs = {
                        k: v.get("direction", "?") if isinstance(v, dict) else v
                        for k, v in trends.items()
                    }
                    parts.append(f"  trends={dirs}")
            except Exception:
                pass

        if e.get("stop_reason"):
            parts.append(f"  stop_reason={e['stop_reason']}")

        reg_summary.append("\n".join(parts))

    # Parameter effects summary
    param_effects_text = ""
    completed = [e for e in registry if e.get("status") in ("completed", "stopped_early")]
    if completed:
        effects: dict[str, list[str]] = {}
        for e in completed:
            ap = EXPERIMENTS_DIR / "analysis" / f"{e['exp_id']}.json"
            ana: dict[str, Any] = {}
            if ap.exists():
                with contextlib.suppress(Exception):
                    ana = json.loads(ap.read_text(encoding="utf-8"))
            ft = ana.get("best_finish_time_s")
            ft_s = f"{ft:.1f}s" if ft and ft > 0 else "DNF"
            lo = ana.get("metrics", {}).get("loss/iqn_loss", {}).get("median")
            lo_s = f"loss={lo:.1f}" if lo else ""
            for dk, val in _flatten_dict_orch(e.get("config_overrides", {})):
                effects.setdefault(dk, []).append(f"{val}({e['exp_id']}:{ft_s} {lo_s})")
        lines = []
        for pk, trials in effects.items():
            lines.append(f"  {pk}: {', '.join(trials)}")
        if lines:
            param_effects_text = "PARAMETER EFFECTS (param: value(exp:result) ...):\n" + "\n".join(
                lines
            )

    return f"""You are an ML experiment designer for a TrackMania RL agent (IQN algorithm).

TARGET: Finish the track in {target} seconds.

PAST EXPERIMENTS (with metrics):
{chr(10).join(reg_summary)}

{param_effects_text}

DECISIONS LOG (recent):
{decisions_text[-2000:]}

{validation_text}

SEARCH SPACE (available parameters to tune):
{search_space_text[:2000]}

CRITICAL FINDINGS FROM PAST EXPERIMENTS:
- Gradient clipping saturation is STRUCTURAL. Do NOT propose experiments that try to "fix"
  gradient clipping (changing iqn_grad_clip, adam_eps, weight_decay for gradient purposes,
  or iqn_soft_target_tau for gradient purposes). These have been tried 11 times and all failed.
- The best config is: batch_size=512, iqn_lr=3e-5, iqn_grad_clip=1.0. Always use
  stable-learning-with-strict-clip (61.65s) as parent. Adding n_steps=7 (long-horizon-planning-v2)
  improved finish rate (25%) but not best time (78.18s).
- UNTRIED directions that should be explored: iqn_epsilon_decay_steps (currently 2M, try 800k),
  end_of_track_reward (currently 8, try 16+), speed_reward_weight, constant_penalty,
  reward_normalize_scale, Munchausen RL, model capacity changes, longer training time.
- Do NOT propose experiments similar to: adam-eps-for-stability, softer-target-network,
  stable-clip-regularized-tau, weight-decay variants, or any "fix gradient clipping" idea.

STRATEGY:
1. Look at which experiments had the best finish times and return values.
2. Check the parameter effects -- which params improved performance?
3. Consider combining parameters that individually helped.
4. Check which search space params haven't been tried yet.
5. Avoid configs similar to experiments that failed or were stopped early for bad metrics.
6. If the best experiment's training trends show "improving", consider extending its approach.
7. ALWAYS use "stable-learning-with-strict-clip" (best: 61.65s) as parent
   unless you have a specific reason not to.

CRITICAL - OVERRIDE FORMAT RULES:
The "overrides" dict must use EXACTLY these top-level section keys:
- "algorithm": iqn_lr, gamma, iqn_grad_clip, iqn_epsilon_decay_steps,
  n_steps, iqn_soft_target_tau, backup_clip_range, etc.
- "training": batch_size, training_steps_per_round,
  rounds_per_epoch, environment_steps_before_training, etc.
- "model" for: residual_mlp_hidden_dim, residual_mlp_num_blocks, gnn_layers, gnn_hidden, etc.
- "environment" for: end_of_track_reward, reward (nested: crash_penalty, speed_reward_weight, etc.)
- "player_runs" for: demo_sampling_weight, demo_max_batch_fraction, etc.
- "run" for: buffers_maxlen, rw_max_samples_per_episode, etc.

WRONG: {{"optimization": {{"algorithm.iqn_grad_clip": 5.0}}}}
CORRECT: {{"algorithm": {{"iqn_grad_clip": 5.0}}}}

WRONG: {{"rl_algorithm": {{"algorithm.gamma": 0.99}}}}
CORRECT: {{"algorithm": {{"gamma": 0.99}}}}

Respond with ONLY a JSON object (no markdown, no code fences):
{{"exp_id": "kebab-case-name", "parent": "gtn-baseline",
"hypothesis": "why this should help",
"overrides": {{"section_name": {{"param_name": value}}}}}}"""


def _flatten_dict_orch(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_dict_orch(v, path))
        else:
            items.append((path, v))
    return items


def _is_gradient_stop(reason: str) -> bool:
    """True if the stop reason is about gradient norms/clipping (a known false alarm)."""
    lower = reason.lower()
    gradient_keywords = [
        "gradient norm",
        "grad_norm",
        "gradient clip",
        "grad clip",
        "pre-clip",
        "pre_clip",
        "clipping",
        "saturating",
        "truncated",
        "truncation",
    ]
    return any(kw in lower for kw in gradient_keywords)


def _make_decision(mode: str, context: dict[str, Any]) -> dict[str, Any]:
    """Make a decision using Gemini AI, with heuristic fallback.

    Hard overrides (applied AFTER Gemini):
    1. Worker finishing tracks (count >= 5) => always continue.
    2. Gradient-based stop reasons => override to continue (structural, not a bug).
    """
    if mode == "decide":
        heuristic = _decide(context)

        prompt = _build_decide_prompt(context)
        response = _call_gemini(prompt)

        gemini_decision: dict[str, Any] | None = None
        if response:
            try:
                text = response.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text
                    text = text.rsplit("```", 1)[0]
                try:
                    parsed: dict[str, Any] = json.loads(text.strip())
                except json.JSONDecodeError:
                    parsed = _extract_json_from_output(text) or {}
                if "action" in parsed:
                    _log(f"Gemini decision: {parsed}")
                    gemini_decision = parsed
                else:
                    _log(f"Gemini response missing 'action' key: {text[:200]}")
            except Exception as exc:
                _log(f"Error parsing Gemini decision: {exc}, response: {response[:200]}")

        decision = gemini_decision or heuristic

        if decision.get("action") == "stop":
            reason = decision.get("reason", "")

            # Override 1: worker is finishing tracks => keep running
            if heuristic.get("action") == "continue" and "Worker is finishing" in heuristic.get(
                "reason", ""
            ):
                _log(
                    f"OVERRIDE: Gemini said stop ({reason!r}) but worker is "
                    f"finishing tracks. Forcing continue."
                )
                return heuristic

            # Override 2: gradient-based stop reason => structural, not a problem
            if _is_gradient_stop(reason):
                _log(
                    f"OVERRIDE: Ignoring gradient-based stop ({reason!r}). "
                    f"Pre-clip >> clip is structural for this architecture."
                )
                return {
                    "action": "continue",
                    "reason": (
                        "Gradient stop overridden (structural). "
                        f"Heuristic: {heuristic.get('reason', 'N/A')}"
                    ),
                }

        return decision

    elif mode == "propose":
        prompt = _build_propose_prompt(context)
        response = _call_gemini(prompt)

        if response:
            try:
                text = response.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text
                    text = text.rsplit("```", 1)[0]

                # Try to extract JSON even if wrapped in extra text
                parsed_json = _extract_json_from_output(text) if "{" in text else None
                proposal: dict[str, Any] = parsed_json or json.loads(text.strip())

                if "exp_id" in proposal and "overrides" in proposal:
                    exp_id = str(proposal["exp_id"])
                    parent = str(proposal.get("parent", "gtn-baseline"))
                    hypothesis = str(proposal.get("hypothesis", "AI-proposed experiment"))
                    raw_overrides = proposal["overrides"]
                    if not isinstance(raw_overrides, dict):
                        _log(f"Gemini overrides not a dict: {type(raw_overrides)}")
                        raise ValueError("overrides must be a dict")
                    overrides = _fix_overrides(raw_overrides)
                    if overrides != raw_overrides:
                        _log(f"Fixed overrides: {raw_overrides} -> {overrides}")

                    try:
                        result = subprocess.run(
                            [
                                "uv",
                                "run",
                                "python",
                                "-m",
                                "tmrl.tools.experiment_manager",
                                "register",
                                "--exp-id",
                                exp_id,
                                "--parent",
                                parent,
                                "--hypothesis",
                                hypothesis,
                                "--overrides",
                                json.dumps(overrides),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=120,
                            cwd=str(REPO_ROOT),
                        )
                        if result.returncode == 0:
                            _log(f"Gemini proposed & registered: {exp_id}")
                            return {"action": "proposed", "exp_id": exp_id}
                        _log(f"Registration failed: {result.stderr[:200]}")
                    except subprocess.TimeoutExpired:
                        _log("Registration subprocess timed out")
                    except Exception as exc:
                        _log(f"Registration error: {exc}")
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                _log(f"Gemini propose parse error: {exc}, response: {response[:300]}")
            except Exception as exc:
                _log(f"Unexpected error processing Gemini proposal: {exc}")

        _log("Falling back to heuristic proposal...")
        return _propose(context)

    return {"action": "continue", "reason": "Unknown mode"}
