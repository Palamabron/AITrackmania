"""Minimal smoke tests for Buffer behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_BUFFER_PATH = Path(__file__).resolve().parents[1] / "tmrl" / "networking" / "buffer.py"
_SPEC = importlib.util.spec_from_file_location("tmrl_buffer_module", _BUFFER_PATH)
assert _SPEC
assert _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Buffer = _MODULE.Buffer


def _make_sample(rew: float = 1.0):
    return (None, None, rew, False, False, {"reward_sum": rew})


def test_append_and_overflow_clips_to_maxlen():
    buf = Buffer(maxlen=3)
    for i in range(10):
        buf.append_sample(_make_sample(rew=float(i)))
    assert len(buf) == 3
    assert [s[2] for s in buf.memory] == [7.0, 8.0, 9.0]


def test_clear_empties_buffer():
    buf = Buffer(maxlen=10)
    buf.append_sample(_make_sample())
    buf.clear()
    assert len(buf) == 0


def test_speed_bonus_updates_rewards():
    buf = Buffer(maxlen=10)
    for _ in range(4):
        buf.append_sample(_make_sample(rew=1.0))
    buf.apply_speed_bonus(16.0)
    expected_rew = 1.0 + (16.0 / 16.0)
    assert all(abs(s[2] - expected_rew) < 1e-6 for s in buf.memory)
