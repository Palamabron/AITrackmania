"""Smoke tests for Buffer behavior (via the public tmrl.networking import path)."""

from __future__ import annotations

import threading

from tmrl.networking import Buffer


def _make_sample(rew: float = 1.0):
    return (None, None, rew, False, False, {"reward_sum": rew})


# ---------------------------------------------------------------------------
# Basic append / clear / speed-bonus
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# thread_safe=True: lock-protected append path
# ---------------------------------------------------------------------------


def test_thread_safe_append_and_overflow():
    """Lock-guarded append and clip_to_maxlen work correctly with thread_safe=True."""
    buf = Buffer(maxlen=3, thread_safe=True)
    for i in range(10):
        buf.append_sample(_make_sample(rew=float(i)))
    assert len(buf) == 3
    assert [s[2] for s in buf.memory] == [7.0, 8.0, 9.0]


def test_thread_safe_concurrent_appends_stay_within_maxlen():
    """Concurrent appends from multiple threads never exceed maxlen."""
    buf = Buffer(maxlen=50, thread_safe=True)
    barrier = threading.Barrier(5)

    def _worker():
        barrier.wait()
        for i in range(20):
            buf.append_sample(_make_sample(rew=float(i)))

    threads = [threading.Thread(target=_worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(buf) <= 50


# ---------------------------------------------------------------------------
# __iadd__: buffer-merging and stats copy logic
# ---------------------------------------------------------------------------


def test_iadd_merges_samples_and_copies_stats():
    """__iadd__ appends other.memory into dst and copies all stat fields."""
    dst = Buffer(maxlen=20)
    src = Buffer(maxlen=20)
    src.stat_train_return = 42.0
    src.stat_test_return = 10.0
    src.stat_train_steps = 5
    src.stat_test_steps = 3.0
    src.stat_test_finish_time = 7.5
    src.stat_test_finished_track = True
    src.stat_test_finished_count = 2
    src.stat_test_competition_eliminated = True
    src.stat_test_competition_crashes = 1
    for i in range(3):
        src.append_sample(_make_sample(rew=float(i)))
    for i in range(2):
        dst.append_sample(_make_sample(rew=float(i + 10)))

    dst += src

    assert len(dst) == 5
    assert dst.stat_train_return == 42.0
    assert dst.stat_test_return == 10.0
    assert dst.stat_train_steps == 5
    assert dst.stat_test_steps == 3.0
    assert dst.stat_test_finish_time == 7.5
    assert dst.stat_test_finished_track is True
    assert dst.stat_test_finished_count == 2
    assert dst.stat_test_competition_eliminated is True
    assert dst.stat_test_competition_crashes == 1


def test_iadd_with_thread_safe_dst():
    """__iadd__ into a thread_safe destination acquires the lock correctly."""
    dst = Buffer(maxlen=20, thread_safe=True)
    src = Buffer(maxlen=20)
    for i in range(3):
        src.append_sample(_make_sample(rew=float(i)))
    dst += src
    assert len(dst) == 3


def test_iadd_overflow_clips_to_maxlen():
    """__iadd__ respects maxlen on the destination buffer."""
    dst = Buffer(maxlen=3)
    dst.append_sample(_make_sample(rew=0.0))
    src = Buffer(maxlen=10)
    for i in range(5):
        src.append_sample(_make_sample(rew=float(i + 1)))
    dst += src
    assert len(dst) == 3
