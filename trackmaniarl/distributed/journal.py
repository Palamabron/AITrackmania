"""Durable idempotency journal for accepted rollout chunks."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock
from uuid import uuid4

_RECOVERY_BATCH_ROWS = 256
_JOURNAL_SCHEMA_VERSION = "1"
_SCHEMA_TABLES = frozenset({"chunks", "actor_profiles", "receipts", "metadata"})
_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        payload BLOB NOT NULL,
        UNIQUE(session_id, sequence)
    )
    """,
    """
    CREATE TABLE actor_profiles (
        actor_id TEXT PRIMARY KEY,
        profile_index INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE receipts (
        session_id TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        row_id INTEGER NOT NULL,
        payload_sha256 BLOB NOT NULL,
        PRIMARY KEY(session_id, sequence)
    )
    """,
    """
    CREATE TABLE metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
)
_INSERT_RECEIPT_QUERY = """
    INSERT INTO receipts(session_id, sequence, row_id, payload_sha256)
    VALUES (?, ?, ?, ?)
"""


class JournalPayloadConflictError(ValueError):
    """A rollout sequence was reused for different bytes."""


class JournalSchemaError(ValueError):
    """The journal does not implement the current durable schema."""


@dataclass(frozen=True, slots=True)
class JournalStatistics:
    pending_rows: int
    pending_payload_bytes: int
    receipt_rows: int
    database_bytes: int
    wal_bytes: int


@dataclass(frozen=True, slots=True)
class _JournalChunk:
    session_id: str
    sequence: int
    payload: bytes
    digest: bytes


@dataclass(frozen=True, slots=True)
class _Receipt:
    row_id: int
    digest: bytes


@dataclass(frozen=True, slots=True)
class _JournalCounts:
    pending_rows: int
    pending_payload_bytes: int
    receipt_rows: int


class RolloutJournal:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = RLock()
        try:
            self._initialize_database()
        except BaseException:
            self._connection.close()
            raise
        self.identity = self._read_identity()

    def _initialize_database(self) -> None:
        self._configure_connection()
        if self._database_tables():
            self._validate_schema()
        else:
            self._create_schema()
            self._initialize_metadata()
        self._connection.commit()

    def _configure_connection(self) -> None:
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")

    def _create_schema(self) -> None:
        for statement in _SCHEMA_STATEMENTS:
            self._connection.execute(statement)
        self._connection.execute("CREATE INDEX receipts_row_id ON receipts(row_id)")

    def _initialize_metadata(self) -> None:
        self._connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("schema_version", _JOURNAL_SCHEMA_VERSION),
                ("journal_id", uuid4().hex),
                ("pruned_through", "0"),
            ),
        )

    def _database_tables(self) -> frozenset[str]:
        rows = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return frozenset(str(row[0]) for row in rows)

    def _validate_schema(self) -> None:
        tables = self._database_tables()
        if tables != _SCHEMA_TABLES:
            raise JournalSchemaError(
                f"rollout journal tables differ: expected={sorted(_SCHEMA_TABLES)}, "
                f"actual={sorted(tables)}"
            )
        if self._metadata_value("schema_version") != _JOURNAL_SCHEMA_VERSION:
            raise JournalSchemaError("rollout journal schema version is unsupported")
        self._validate_receipts()

    def _validate_receipts(self) -> None:
        row = self._connection.execute(
            """
            SELECT 1 FROM chunks
            LEFT JOIN receipts
              ON receipts.session_id = chunks.session_id
             AND receipts.sequence = chunks.sequence
            WHERE receipts.session_id IS NULL OR receipts.row_id != chunks.id
            LIMIT 1
            """
        ).fetchone()
        if row is not None:
            raise JournalSchemaError("rollout journal contains chunks without current receipts")

    def _metadata_value(self, key: str) -> str:
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise JournalSchemaError(f"rollout journal metadata is missing {key!r}")
        return str(row[0])

    def _read_identity(self) -> str:
        return self._metadata_value("journal_id")

    def append(self, session_id: str, sequence: int, payload: bytes) -> tuple[int, bool]:
        chunk = _JournalChunk(session_id, sequence, payload, sha256(payload).digest())
        with self._lock:
            receipt = self._find_receipt(chunk)
            if receipt is not None:
                return self._receipt_row_id(chunk, receipt), False
            row_id = self._insert_chunk(chunk)
            self._connection.commit()
        return row_id, True

    def _find_receipt(self, chunk: _JournalChunk) -> _Receipt | None:
        row = self._connection.execute(
            """
            SELECT row_id, payload_sha256 FROM receipts
            WHERE session_id = ? AND sequence = ?
            """,
            (chunk.session_id, chunk.sequence),
        ).fetchone()
        return None if row is None else _Receipt(int(row[0]), bytes(row[1]))

    @staticmethod
    def _receipt_row_id(chunk: _JournalChunk, receipt: _Receipt) -> int:
        if receipt.digest != chunk.digest:
            raise JournalPayloadConflictError(
                f"rollout {chunk.session_id!r} sequence {chunk.sequence} "
                "reused with different payload"
            )
        return receipt.row_id

    def _insert_chunk(self, chunk: _JournalChunk) -> int:
        cursor = self._connection.execute(
            "INSERT INTO chunks(session_id, sequence, payload) VALUES (?, ?, ?)",
            (chunk.session_id, chunk.sequence, chunk.payload),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("rollout journal failed to assign a row ID")
        row_id = int(cursor.lastrowid)
        self._connection.execute(
            _INSERT_RECEIPT_QUERY,
            (chunk.session_id, chunk.sequence, row_id, chunk.digest),
        )
        return row_id

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

    def statistics(self, applied_frontier: int) -> JournalStatistics:
        if applied_frontier < 0:
            raise ValueError("journal applied frontier cannot be negative")
        with self._lock:
            counts = self._statistics_rows(applied_frontier)
        return JournalStatistics(
            pending_rows=counts.pending_rows,
            pending_payload_bytes=counts.pending_payload_bytes,
            receipt_rows=counts.receipt_rows,
            database_bytes=self._file_size(self.path),
            wal_bytes=self._file_size(self.path.with_name(f"{self.path.name}-wal")),
        )

    def _statistics_rows(self, applied_frontier: int) -> _JournalCounts:
        pending = self._connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM chunks WHERE id > ?",
            (applied_frontier,),
        ).fetchone()
        receipt = self._connection.execute("SELECT COUNT(*) FROM receipts").fetchone()
        if pending is None or receipt is None:
            raise RuntimeError("rollout journal statistics query returned no row")
        return _JournalCounts(int(pending[0]), int(pending[1]), int(receipt[0]))

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0

    @property
    def pruned_through(self) -> int:
        with self._lock:
            return int(self._metadata_value("pruned_through"))

    def validate_checkpoint(self, journal_id: object, applied_frontier: int) -> None:
        """Reject checkpoints that cannot resume without losing or duplicating rows."""

        if applied_frontier < 0:
            raise ValueError("journal applied frontier cannot be negative")
        with self._lock:
            self._validate_journal_identity(journal_id)
            self._validate_frontier(applied_frontier)

    def _validate_journal_identity(self, journal_id: object) -> None:
        if journal_id != self.identity:
            raise ValueError("checkpoint belongs to a different rollout journal")

    def _validate_frontier(self, applied_frontier: int) -> None:
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
