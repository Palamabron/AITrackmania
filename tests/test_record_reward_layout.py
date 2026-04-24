from tmrl.tools.record_reward import _is_lap_finished


def test_finish_flag_uses_index_9_for_tqc_20f_layout() -> None:
    data = [0.0] * 20
    data[8] = 1.0  # braking in TQC layout
    data[9] = 0.0  # finish flag
    assert _is_lap_finished(tuple(data)) is False
    data[9] = 1.0
    assert _is_lap_finished(tuple(data)) is True


def test_finish_flag_uses_index_8_for_legacy_19f_layout() -> None:
    data = [0.0] * 19
    data[8] = 1.0
    assert _is_lap_finished(tuple(data)) is True
