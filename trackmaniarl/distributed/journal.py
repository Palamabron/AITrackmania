"""Durable idempotency journal for accepted rollout chunks."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from threading import RLock
from uuid import uuid4

_RECOVERY_BATCH_ROWS = 256


class JournalPayloadConflictError(ValueError):
    """A rollout sequence was reused for different bytes."""


class RolloutJournal:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = RLock()
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
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
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS receipts (
                session_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                row_id INTEGER NOT NULL,
                payload_sha256 BLOB NOT NULL,
                PRIMARY KEY(session_id, sequence)
            )
            """
        )
        self._connection.execute("CREATE INDEX IF NOT EXISTS receipts_row_id ON receipts(row_id)")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'journal_id'"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('journal_id', ?)",
                (uuid4().hex,),
            )
        self._connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES ('pruned_through', '0')"
        )
        self._migrate_receipts()
        self._connection.commit()
        identity = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'journal_id'"
        ).fetchone()
        if identity is None:
            raise RuntimeError("rollout journal failed to persist its identity")
        self.identity = str(identity[0])

    def _migrate_receipts(self) -> None:
        cursor = self._connection.execute(
            """
            SELECT chunks.id, chunks.session_id, chunks.sequence, chunks.payload
            FROM chunks
            LEFT JOIN receipts
              ON receipts.session_id = chunks.session_id
             AND receipts.sequence = chunks.sequence
            WHERE receipts.session_id IS NULL
            ORDER BY chunks.id
            """
        )
        while rows := cursor.fetchmany(_RECOVERY_BATCH_ROWS):
            self._connection.executemany(
                """
                INSERT OR IGNORE INTO receipts(session_id, sequence, row_id, payload_sha256)
                VALUES (?, ?, ?, ?)
                """,
                (
                    (str(session_id), int(sequence), int(row_id), sha256(bytes(payload)).digest())
                    for row_id, session_id, sequence, payload in rows
                ),
            )

    def append(self, session_id: str, sequence: int, payload: bytes) -> tuple[int, bool]:
        digest = sha256(payload).digest()
        with self._lock:
            receipt = self._connection.execute(
                """
                SELECT row_id, payload_sha256 FROM receipts
                WHERE session_id = ? AND sequence = ?
                """,
                (session_id, sequence),
            ).fetchone()
            if receipt is not None:
                if bytes(receipt[1]) != digest:
                    raise JournalPayloadConflictError(
                        f"rollout {session_id!r} sequence {sequence} reused with different payload"
                    )
                return int(receipt[0]), False
            cursor = self._connection.execute(
                "INSERT INTO chunks(session_id, sequence, payload) VALUES (?, ?, ?)",
                (session_id, sequence, payload),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("rollout journal failed to assign a row ID")
            row_id = cursor.lastrowid
            self._connection.execute(
                """
                INSERT INTO receipts(session_id, sequence, row_id, payload_sha256)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, sequence, row_id, digest),
            )
            self._connection.commit()
        return row_id, True

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

    def has_history(self) -> bool:
        with self._lock:
            return (
                self._connection.execute("SELECT 1 FROM receipts LIMIT 1").fetchone() is not None
                or self.pruned_through > 0
            )

    def has_rows_after(self, watermark: int) -> bool:
        with self._lock:
            return (
                self._connection.execute(
                    "SELECT 1 FROM chunks WHERE id > ? LIMIT 1", (watermark,)
                ).fetchone()
                is not None
            )

    @property
    def pruned_through(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = 'pruned_through'"
            ).fetchone()
        if row is None:
            raise RuntimeError("rollout journal has no prune frontier")
        return int(row[0])

    def validate_checkpoint(self, journal_id: object, applied_frontier: int) -> None:
        """Reject checkpoints that cannot resume without losing or duplicating rows."""

        if applied_frontier < 0:
            raise ValueError("journal applied frontier cannot be negative")
        with self._lock:
            if journal_id != self.identity:
                raise ValueError("checkpoint belongs to a different rollout journal")
            if applied_frontier < self.pruned_through:
                raise ValueError(
                    "checkpoint journal frontier predates data already pruned from the WAL"
                )
            maximum = self._connection.execute("SELECT MAX(row_id) FROM receipts").fetchone()
            maximum_row_id = 0 if maximum is None or maximum[0] is None else int(maximum[0])
            if applied_frontier > maximum_row_id:
                raise ValueError("checkpoint journal frontier is ahead of durable WAL history")

    def prune(self, watermark: int) -> None:
        if watermark < 0:
            raise ValueError("journal prune frontier cannot be negative")
        with self._lock:
            if watermark < self.pruned_through:
                return
            maximum = self._connection.execute("SELECT MAX(row_id) FROM receipts").fetchone()
            maximum_row_id = 0 if maximum is None or maximum[0] is None else int(maximum[0])
            if watermark > maximum_row_id:
                raise ValueError("journal prune frontier is ahead of durable WAL history")
            self._connection.execute("DELETE FROM chunks WHERE id <= ?", (watermark,))
            self._connection.execute(
                "UPDATE metadata SET value = ? WHERE key = 'pruned_through'",
                (str(watermark),),
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
