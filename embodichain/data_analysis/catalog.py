# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Durable SQLite catalog and immutable analysis snapshots."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from .schema import IDENTITY_FIELDS, validate_record

_ALLOWED_TRANSITIONS = {
    "proposed": {
        "planning_failed",
        "rollout_failed",
        "rejected",
        "pending_write",
        "partial_commit",
        "committed",
    },
    "pending_write": {"partial_commit", "committed"},
    "partial_commit": {"committed"},
    "planning_failed": set(),
    "rollout_failed": set(),
    "rejected": set(),
    "committed": set(),
}

__all__ = ["Catalog"]


class Catalog:
    """SQLite-backed current records, event ledger and immutable snapshots."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS records (
                episode_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                status TEXT NOT NULL,
                record_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS snapshot_records (
                snapshot_id INTEGER NOT NULL REFERENCES snapshots(snapshot_id),
                episode_id TEXT NOT NULL,
                record_json TEXT NOT NULL,
                PRIMARY KEY (snapshot_id, episode_id)
            );
            CREATE INDEX IF NOT EXISTS records_status_idx ON records(status);
            """)
        self._connection.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def upsert(
        self, record: Mapping[str, Any]
    ) -> Literal["inserted", "updated", "unchanged"]:
        """Persist a complete record snapshot and append one event for a real change."""
        normalized = validate_record(record, artifact_root=self.path.parent)
        payload = json.dumps(
            normalized, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT status, record_json FROM records WHERE episode_id = ?",
                (normalized["episode_id"],),
            ).fetchone()
            outcome: Literal["inserted", "updated", "unchanged"] = "inserted"
            if row is not None:
                previous = json.loads(row[1])
                for field in IDENTITY_FIELDS:
                    if previous[field] != normalized[field]:
                        raise ValueError(
                            f"Episode {normalized['episode_id']!r} has conflicting immutable identity field '{field}'."
                        )
                if row[1] == payload:
                    return "unchanged"
                if normalized["status"] == row[0]:
                    raise ValueError(
                        f"Episode {normalized['episode_id']!r} already exists with different content at status {row[0]!r}."
                    )
                if (
                    normalized["status"] != row[0]
                    and normalized["status"] not in _ALLOWED_TRANSITIONS[row[0]]
                ):
                    raise ValueError(
                        f"Illegal status transition {row[0]!r} -> {normalized['status']!r}."
                    )
                outcome = "updated"
            self._connection.execute(
                "INSERT INTO records(episode_id, status, record_json) VALUES (?, ?, ?) "
                "ON CONFLICT(episode_id) DO UPDATE SET status=excluded.status, record_json=excluded.record_json",
                (normalized["episode_id"], normalized["status"], payload),
            )
            self._connection.execute(
                "INSERT INTO events(episode_id, status, record_json) VALUES (?, ?, ?)",
                (normalized["episode_id"], normalized["status"], payload),
            )
        return outcome

    def get(self, episode_id: str) -> dict[str, Any] | None:
        """Return one detached current record, or ``None`` when absent."""
        with self._lock:
            row = self._connection.execute(
                "SELECT record_json FROM records WHERE episode_id = ?", (episode_id,)
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def records(
        self, filters: Mapping[str, Any] | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        """Return current records matching status and categorical or numeric filters."""
        with self._lock:
            if status is None:
                rows = self._connection.execute(
                    "SELECT record_json FROM records ORDER BY episode_id"
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT record_json FROM records WHERE status = ? ORDER BY episode_id",
                    (status,),
                ).fetchall()
        result = [json.loads(row[0]) for row in rows]
        for name, condition in (filters or {}).items():
            result = [
                record for record in result if self._matches(record, name, condition)
            ]
        return result

    @staticmethod
    def _matches(record: Mapping[str, Any], name: str, condition: Any) -> bool:
        if isinstance(condition, Mapping) and (
            "availability" in condition or "values" in condition
        ):
            from .dimensions import match_dimension

            return match_dimension(record, name, condition)
        measurement = record.get("dimensions", {}).get(name)
        value = (
            measurement.get("value")
            if isinstance(measurement, Mapping)
            else record.get(name)
        )
        if isinstance(condition, Mapping):
            if (
                value is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                return False
            minimum = condition.get("min")
            maximum = condition.get("max")
            return (minimum is None or value >= minimum) and (
                maximum is None or value <= maximum
            )
        if isinstance(condition, (list, tuple, set, frozenset)):
            return value in condition
        return value == condition

    def summary(self) -> dict[str, Any]:
        """Return current total/status counts and durable event count."""
        with self._lock:
            statuses = {
                status: count
                for status, count in self._connection.execute(
                    "SELECT status, COUNT(*) FROM records GROUP BY status ORDER BY status"
                ).fetchall()
            }
            events = self._connection.execute("SELECT COUNT(*) FROM events").fetchone()[
                0
            ]
        return {"total": sum(statuses.values()), "statuses": statuses, "events": events}

    def snapshot(self, name: str) -> int:
        """Atomically capture all current full records and return the snapshot ID."""
        if not name:
            raise ValueError("Snapshot name cannot be empty.")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "INSERT INTO snapshots(name) VALUES (?)", (name,)
            )
            snapshot_id = int(cursor.lastrowid)
            self._connection.execute(
                "INSERT INTO snapshot_records(snapshot_id, episode_id, record_json) "
                "SELECT ?, episode_id, record_json FROM records",
                (snapshot_id,),
            )
        return snapshot_id

    def snapshot_records(self, snapshot_id: int) -> list[dict[str, Any]]:
        """Return detached records from an immutable snapshot."""
        with self._lock:
            exists = self._connection.execute(
                "SELECT 1 FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(f"Unknown snapshot_id {snapshot_id}.")
            rows = self._connection.execute(
                "SELECT record_json FROM snapshot_records WHERE snapshot_id = ? ORDER BY episode_id",
                (snapshot_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def import_jsonl(self, path: str | Path) -> dict[str, int]:
        """Idempotently import full record snapshots from a JSON Lines file."""
        counts = {"inserted": 0, "updated": 0, "unchanged": 0}
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    outcome = self.upsert(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid record at {path}:{line_number}: {exc}"
                    ) from exc
                counts[outcome] += 1
        return counts
