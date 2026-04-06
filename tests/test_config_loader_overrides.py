"""Integration tests for loader precedence, Hydra overrides, and scheduler keys."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.tmrl_test_min_assets import write_min_tmrl_test_pickles


def _run_probe(repo_root: Path, env: dict[str, str]) -> dict[str, object]:
    probe = """
import json
from tmrl.config import MAIN_CONFIG
from tmrl.config.constants import SCHEDULER_CONFIG

out = {
    "algorithm_name": MAIN_CONFIG.algorithm.name,
    "model_use_rnn": MAIN_CONFIG.model.use_rnn,
    "lr_actor": MAIN_CONFIG.algorithm.lr_actor,
    "batch_size": MAIN_CONFIG.training.batch_size,
    "run_name": MAIN_CONFIG.run.name,
    "wandb_key": MAIN_CONFIG.wandb.api_key,
    "distributed_password": MAIN_CONFIG.distributed.password,
    "scheduler_keys": sorted(SCHEDULER_CONFIG.keys()),
}
print(json.dumps(out))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(repo_root),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    # loguru can print before/after payload; parse the last non-empty line as JSON.
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_loader_precedence_and_hydra_override(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    tmrl_data = tmp_path / "TmrlData"
    cfg_dir = tmrl_data / "config"
    cfg_dir.mkdir(parents=True)
    write_min_tmrl_test_pickles(tmrl_data)
    (cfg_dir / "local.yaml").write_text(
        "\n".join(
            [
                "run:",
                "  name: local_run_name",
                "algorithm:",
                "  lr_actor: 0.00011",
                "training:",
                "  batch_size: 111",
            ]
        )
        + "\n"
    )

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path),
            "LOGURU_LEVEL": "ERROR",
            "TMRL_HYDRA_OVERRIDES": "model=mlp_actor_critic,algorithm=sac",
            "TMRL_CONFIG_OVERRIDES": json.dumps(
                {
                    "run": {"name": "json_patch_run"},
                    "training": {"batch_size": 222},
                }
            ),
            "WANDB_API_KEY": "wandb_from_env",
            "TMRL_PASSWORD": "password_from_env",
        }
    )

    out = _run_probe(repo_root, env)

    assert out["algorithm_name"] == "SAC"
    assert out["model_use_rnn"] is False
    assert out["lr_actor"] == 0.00011  # from local.yaml
    assert out["batch_size"] == 222  # JSON override wins over local.yaml
    assert out["run_name"] == "json_patch_run"
    assert out["wandb_key"] == "wandb_from_env"
    assert out["distributed_password"] == "password_from_env"


def test_scheduler_config_uses_lowercase_keys(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    tmrl_data = tmp_path / "TmrlData"
    cfg_dir = tmrl_data / "config"
    cfg_dir.mkdir(parents=True)
    write_min_tmrl_test_pickles(tmrl_data)
    (cfg_dir / "local.yaml").write_text("")

    env = os.environ.copy()
    env.update({"HOME": str(tmp_path), "LOGURU_LEVEL": "ERROR"})

    out = _run_probe(repo_root, env)
    assert "name" in out["scheduler_keys"]
    assert "t_0" in out["scheduler_keys"]
    assert "T_0" not in out["scheduler_keys"]
