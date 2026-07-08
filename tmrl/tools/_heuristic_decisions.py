"""Heuristic decide/propose functions for the orchestrator."""

from __future__ import annotations

import contextlib
import json
import subprocess
from typing import Any

from tmrl.tools._experiment_io import EXPERIMENTS_DIR, REPO_ROOT
from tmrl.tools._orchestrator_utils import _log


def _decide(context: dict[str, Any]) -> dict[str, Any]:
    """Decide whether to continue or stop the current experiment.

    Analyzes training dynamics: loss stability, Q-value health,
    reward trends, and convergence signals.
    """
    snapshot = context.get("snapshot", {})
    recent = snapshot.get("recent_metrics", {})
    target_time = context.get("target_finish_time_s", 36.0)
    elapsed_h = context.get("elapsed_hours", 0)
    best_ft = snapshot.get("best_finish_time_s")

    # --- Target check ---
    if best_ft is not None and best_ft > 0 and best_ft <= target_time:
        return {"action": "stop", "reason": f"Target reached: {best_ft:.2f}s"}

    # --- Worker finishing = learning is happening, keep running ---
    worker_finish_count = snapshot.get("worker_finish_count", 0)
    worker_best = snapshot.get("worker_best_finish_time_s")
    if worker_finish_count >= 5 and worker_best and worker_best > 0:
        return {
            "action": "continue",
            "reason": f"Worker is finishing tracks ({worker_finish_count} finishes, "
            f"best {worker_best:.1f}s). Learning is progressing.",
        }

    # --- Catastrophic failure checks ---
    loss = recent.get("loss/iqn_loss", {})
    loss_last = loss.get("last")
    loss_p95 = loss.get("p95")
    loss_median = loss.get("median")

    if loss_last and loss_last > 100:
        return {"action": "stop", "reason": f"Loss diverged catastrophically: {loss_last:.4g}"}

    q_max = recent.get("q/max_q", {})
    q_max_last = q_max.get("last")
    if q_max_last and abs(q_max_last) > 500:
        return {"action": "stop", "reason": f"Q-values exploded: {q_max_last:.2f}"}

    q_min = recent.get("q/min_q", {})
    q_min_last = q_min.get("last")
    if q_min_last is not None and q_min_last < -50:
        return {"action": "stop", "reason": f"Q-values collapsed (min_q={q_min_last:.2f})"}

    # NOTE: Gradient saturation (pre-clip >> clip) is structural for this
    # architecture and must NOT be used as a stop criterion.  All 18 past
    # experiments showed pre-clip/clip ratios >20x; stopping on this wasted
    # 11 experiments.  Only NaN gradients warrant a stop.
    grad = recent.get("debug/grad_norm", {})
    grad_last = grad.get("last")
    if grad_last is not None and (grad_last != grad_last):  # NaN check
        return {"action": "stop", "reason": "Gradient norm is NaN."}

    # --- Loss spike detection ---
    if loss_p95 and loss_median and loss_median > 0:
        spike_ratio = loss_p95 / loss_median
        if spike_ratio > 5 and elapsed_h >= 2:
            return {
                "action": "stop",
                "reason": f"Loss highly unstable (p95/median={spike_ratio:.1f}). "
                f"Consider lower lr or tighter grad_clip.",
            }

    # --- Stagnation after extended time ---
    ret_train = recent.get("metrics/return_train", {})
    if elapsed_h >= 3 and ret_train:
        ret_last = ret_train.get("last", 0)
        ret_p95 = ret_train.get("p95", 0)
        if ret_last > 0 and ret_p95 > 0 and ret_last < ret_p95 * 0.3:
            pass  # Return dropped significantly but might be exploration; don't stop

    # --- Memory buffer health ---
    buffer_len = recent.get("buffer/memory_len", {}).get("last")
    if buffer_len is not None and buffer_len < 100 and elapsed_h > 0.5:
        return {
            "action": "stop",
            "reason": f"Buffer nearly empty ({buffer_len}) after {elapsed_h:.1f}h. "
            f"Worker/server connection issue.",
        }

    # --- All clear ---
    reasons = []
    if loss_last and loss_median:
        reasons.append(f"loss={loss_last:.2f}(med={loss_median:.2f})")
    if q_max_last:
        reasons.append(f"Q_max={q_max_last:.1f}")
    if best_ft and best_ft > 0:
        reasons.append(f"best_finish={best_ft:.1f}s")
    elif best_ft is None or best_ft == 0:
        reasons.append("no finish yet")
    if ret_train.get("last"):
        reasons.append(f"return={ret_train['last']:.1f}")

    summary = ", ".join(reasons) if reasons else "metrics unavailable"
    return {"action": "continue", "reason": f"Training healthy: {summary}"}


def _propose(context: dict[str, Any]) -> dict[str, Any]:
    """Propose the next experiment based on completed results.

    Reads the search space and past experiments to suggest what to try next.
    """
    registry = context.get("registry", [])
    completed = [e for e in registry if e.get("status") in ("completed", "stopped_early")]

    # Check what we've already tried
    tried_params: set[str] = set()
    for e in registry:
        overrides = e.get("config_overrides", {})
        for section in overrides.values():
            if isinstance(section, dict):
                tried_params.update(section.keys())
            else:
                tried_params.add(str(section))

    # Load analyses for completed experiments
    analyses: list[dict[str, Any]] = []
    for e in completed:
        ap = EXPERIMENTS_DIR / "analysis" / f"{e['exp_id']}.json"
        if ap.exists():
            with contextlib.suppress(Exception):
                analyses.append(json.loads(ap.read_text(encoding="utf-8")))

    best_ft = float("inf")
    best_parent = "gtn-baseline"
    best_return = -float("inf")
    for a in analyses:
        ft = a.get("best_finish_time_s")
        if ft and ft > 0 and ft < best_ft:
            best_ft = ft
            best_parent = a.get("exp_id", "gtn-baseline")
        ret = a.get("metrics", {}).get("metrics/return_train", {}).get("last", 0)
        if ret > best_return:
            best_return = ret

    # Proposal logic based on what hasn't been tried
    proposals = []

    if "iqn_lr" not in tried_params:
        proposals.append(
            {
                "exp_id": "higher-lr-5e5",
                "hypothesis": "Increase learning rate to 5e-5 for faster convergence.",
                "overrides": {"algorithm": {"iqn_lr": 5e-5}},
            }
        )

    if "batch_size" not in tried_params:
        proposals.append(
            {
                "exp_id": "batch-512",
                "hypothesis": "Double batch size to 512 for lower gradient variance.",
                "overrides": {"training": {"batch_size": 512}},
            }
        )

    if "iqn_epsilon_decay_steps" not in tried_params:
        proposals.append(
            {
                "exp_id": "fast-exploit-800k",
                "hypothesis": "Reduce epsilon decay to 800k steps for faster exploitation.",
                "overrides": {"algorithm": {"iqn_epsilon_decay_steps": 800000}},
            }
        )

    if "gamma" not in tried_params:
        proposals.append(
            {
                "exp_id": "shorter-horizon-gamma99",
                "hypothesis": (
                    "Lower gamma to 0.99 for more stable Q-values and faster credit assignment."
                ),
                "overrides": {"algorithm": {"gamma": 0.99}},
            }
        )

    if "end_of_track_reward" not in tried_params:
        proposals.append(
            {
                "exp_id": "big-finish-bonus-16",
                "hypothesis": (
                    "Double finish reward to 16.0 to strongly incentivize track completion."
                ),
                "overrides": {"environment": {"end_of_track_reward": 16.0}},
            }
        )

    if "n_steps" not in tried_params:
        proposals.append(
            {
                "exp_id": "nsteps-5-longer-returns",
                "hypothesis": (
                    "Increase n_steps to 5 for better multi-step returns (longer TD targets)."
                ),
                "overrides": {"algorithm": {"n_steps": 5}},
            }
        )

    # If best experiment had high loss, propose lower lr
    for a in analyses:
        loss_p95 = a.get("metrics", {}).get("loss/iqn_loss", {}).get("p95", 0)
        if loss_p95 > 30:
            proposals.append(
                {
                    "exp_id": f"lower-lr-from-{a.get('exp_id', 'unknown')}",
                    "hypothesis": (
                        f"Loss was high (p95={loss_p95:.1f}) in {a.get('exp_id')}. Try lr=2e-5."
                    ),
                    "overrides": {"algorithm": {"iqn_lr": 2e-5}},
                    "parent": a.get("exp_id", "gtn-baseline"),
                }
            )
            break

    if not proposals:
        return {
            "action": "no_proposal",
            "reason": "All standard variations have been tried. Manual review needed.",
        }

    # Pick the first untried proposal
    existing_ids = {e["exp_id"] for e in registry}
    for prop in proposals:
        if prop["exp_id"] not in existing_ids:
            parent = prop.get("parent", best_parent)
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
                        str(prop["exp_id"]),
                        "--parent",
                        str(parent),
                        "--hypothesis",
                        str(prop["hypothesis"]),
                        "--overrides",
                        json.dumps(prop["overrides"]),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(REPO_ROOT),
                )
                if result.returncode == 0:
                    _log(f"Proposed & registered: {prop['exp_id']}")
                    return {"action": "proposed", "exp_id": prop["exp_id"]}
                _log(f"Registration failed: {result.stderr[:200]}")
            except Exception as exc:
                _log(f"Registration error: {exc}")

    return {
        "action": "no_proposal",
        "reason": "Could not register new experiment.",
    }
