"""Durable idempotency journal for accepted rollout chunks."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from threading import RLock

_RECOVERY_BATCH_ROWS = 256


class RolloutJournal:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = RLock()
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                payload BLOB NOT NULL,
                UNIQUE(session_id, sequence)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS actor_profiles (
                actor_id TEXT PRIMARY KEY,
                profile_index INTEGER NOT NULL
            )
            """
        )
        self._connection.commit()

    def append(self, session_id: str, sequence: int, payload: bytes) -> tuple[int, bool]:
        with self._lock:
            cursor = self._connection.execute(
                "INSERT OR IGNORE INTO chunks(session_id, sequence, payload) VALUES (?, ?, ?)",
                (session_id, sequence, payload),
            )
            self._connection.commit()
            row = self._connection.execute(
                "SELECT id FROM chunks WHERE session_id = ? AND sequence = ?",
                (session_id, sequence),
            ).fetchone()
        if row is None:
            raise RuntimeError("rollout journal failed to persist a chunk")
        return int(row[0]), cursor.rowcount == 1

    def rows_after(self, watermark: int) -> Iterator[tuple[int, bytes]]:
        last = watermark
        while True:
            with self._lock:
                rows = self._connection.execute(
                    "SELECT id, payload FROM chunks WHERE id > ? ORDER BY id LIMIT ?",
                    (last, _RECOVERY_BATCH_ROWS),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                last = int(row[0])
                yield last, bytes(row[1])

    def has_rows(self) -> bool:
        with self._lock:
            return self._connection.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is not None

    def prune(self, watermark: int) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM chunks WHERE id <= ?", (watermark,))
            self._connection.commit()

    def discard(self, session_id: str, sequence: int) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM chunks WHERE session_id = ? AND sequence = ?",
                (session_id, sequence),
            )
            self._connection.commit()

    def actor_profile(self, actor_id: str, profile_count: int) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT profile_index FROM actor_profiles WHERE actor_id = ?", (actor_id,)
            ).fetchone()
            if row is not None:
                return int(row[0])
            count = int(
                self._connection.execute("SELECT COUNT(*) FROM actor_profiles").fetchone()[0]
            )
            profile = count % profile_count
            self._connection.execute(
                "INSERT INTO actor_profiles(actor_id, profile_index) VALUES (?, ?)",
                (actor_id, profile),
            )
            self._connection.commit()
            return profile

    def close(self) -> None:
        self._connection.close()
