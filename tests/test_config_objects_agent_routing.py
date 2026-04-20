"""Smoke tests for ``_build_agent`` routing in :mod:`tmrl.config.config_objects`.

Regression guard for two bugs fixed on this branch:

* The TQC branch used to be dead-code nested inside the SAC block, so
  ``algorithm.name: TQC`` fell through to ``raise ValueError("Unknown algorithm")``.
* ``algorithm.clip_weights_value`` was ignored at runtime; only the TQC path is
  currently wired through ``_build_agent``, so we assert the value is threaded
  into the bound ``AGENT`` partial for TQC.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.tmrl_test_min_assets import write_min_tmrl_test_pickles

_TQC_CLIP_VALUE = 0.75

_ALG_ENV_MATRIX: list[tuple[str, str]] = [
    ("SAC", "LIDAR"),
    ("REDQSAC", "LIDAR"),
    ("IQN", "LIDAR"),
    ("SDSAC", "LIDAR"),
    ("TQC", "MTQC"),
]


def _run_probe(
    repo_root: Path, tmp_home: Path, alg_name: str, rtgym_iface: str
) -> dict[str, object]:
    tmrl_data = tmp_home / "TmrlData"
    (tmrl_data / "config").mkdir(parents=True, exist_ok=True)
    write_min_tmrl_test_pickles(tmrl_data)

    alg_block = ["algorithm:", f"  name: {alg_name}"]
    if alg_name == "TQC":
        alg_block += [
            "  clipping_weights: true",
            f"  clip_weights_value: {_TQC_CLIP_VALUE}",
        ]
    env_block = ["environment:", f"  rtgym_interface: {rtgym_iface}"]
    (tmrl_data / "config" / "local.yaml").write_text("\n".join(alg_block + env_block) + "\n")

    probe = """
import functools, json
import tmrl.config.config_objects as co

agent = co.AGENT
assert isinstance(agent, functools.partial), f"AGENT is not a partial: {type(agent)!r}"
out = {
    "alg_name": co.ALG_NAME,
    "clip_weights_value": agent.keywords.get("clip_weights_value"),
    "weight_clipping_enabled": agent.keywords.get("weight_clipping_enabled"),
}
print(json.dumps(out))
"""

    env = os.environ.copy()
    env.update({"HOME": str(tmp_home), "LOGURU_LEVEL": "ERROR"})

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(repo_root),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


@pytest.mark.parametrize(("alg_name", "rtgym_iface"), _ALG_ENV_MATRIX)
def test_build_agent_routes_every_supported_algorithm(
    alg_name: str, rtgym_iface: str, tmp_path: Path
):
    """Every registered algorithm produces a bound ``AGENT`` partial.

    For TQC we additionally assert that ``algorithm.clip_weights_value`` is threaded
    into the agent partial (regression test for the hard-coded ``1.0`` bound).
    """
    repo_root = Path(__file__).resolve().parents[1]
    out = _run_probe(repo_root, tmp_path, alg_name, rtgym_iface)
    assert out["alg_name"] == alg_name
    if alg_name == "TQC":
        assert out["weight_clipping_enabled"] is True
        assert out["clip_weights_value"] == pytest.approx(_TQC_CLIP_VALUE)
