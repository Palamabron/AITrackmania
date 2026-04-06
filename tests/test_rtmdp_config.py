"""Tests for RT-MDP configuration defaults."""

import tmrl.config as cfg
from tmrl.config.loader import MAIN_CONFIG


class TestRTGYMConfig:
    def test_reset_act_buf_set_true(self):
        rtgym = MAIN_CONFIG.environment.rtgym
        assert rtgym.reset_act_buf is True, (
            "reset_act_buf must default to True to clear stale pre-reset actions"
        )

    def test_act_buf_len_positive(self):
        abl = MAIN_CONFIG.environment.rtgym.act_buf_len
        assert abl > 0, f"act_buf_len must be > 0 for RT-MDP, got {abl}"

    def test_act_buf_len_is_integer(self):
        abl = MAIN_CONFIG.environment.rtgym.act_buf_len
        assert isinstance(abl, int), f"act_buf_len should be int, got {type(abl)}"


class TestUpdateModelInterval:
    def test_update_interval_exists(self):
        assert cfg.UPDATE_MODEL_INTERVAL > 0

    def test_update_interval_positive(self):
        val = cfg.UPDATE_MODEL_INTERVAL
        assert val > 0, f"UPDATE_MODEL_INTERVAL must be > 0, got {val}"
