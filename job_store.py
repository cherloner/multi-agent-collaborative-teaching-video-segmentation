from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Iterator


ACTIVE_STATUSES = ("queued", "processing")
TERMINAL_STATUSES = ("completed", "failed", "cancelled")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    phase TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    content_hash TEXT,
                    target_duration INTEGER NOT NULL,
                    output_format TEXT NOT NULL DEFAULT 'source',
                    captions_enabled INTEGER NOT NULL DEFAULT 1,
                    result_json TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS jobs_status_created
                    ON jobs(status, created_at);

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    clip_filename TEXT NOT NULL,
                    rating TEXT NOT NULL CHECK (rating IN ('keep', 'reject')),
                    reason TEXT,
                    features_json TEXT,
                    score REAL,
                    profile_version INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(job_id, clip_filename)
                );

                CREATE TABLE IF NOT EXISTS scoring_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0,
                    profile_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS scoring_profiles_active
                    ON scoring_profiles(active, id);
                """
            )
            feedback_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(feedback)").fetchall()
            }
            for column, definition in {
                "features_json": "TEXT",
                "score": "REAL",
                "profile_version": "INTEGER",
            }.items():
                if column not in feedback_columns:
                    connection.execute(
                        f"ALTER TABLE feedback ADD COLUMN {column} {definition}"
                    )
            job_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            for column, definition in {
                "output_format": "TEXT NOT NULL DEFAULT 'source'",
                "captions_enabled": "INTEGER NOT NULL DEFAULT 1",
                "content_hash": "TEXT",
            }.items():
                if column not in job_columns:
                    connection.execute(
                        f"ALTER TABLE jobs ADD COLUMN {column} {definition}"
                    )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS jobs_cache_lookup
                ON jobs(content_hash, status, updated_at)
                """
            )

    def create_job(self, job: dict, capacity: int) -> bool:
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active_count = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'processing')"
            ).fetchone()[0]
            if active_count >= capacity:
                connection.rollback()
                return False
            connection.execute(
                """
                INSERT INTO jobs (
                    id, status, progress, phase, filename, source_path,
                    content_hash, target_duration, output_format,
                    captions_enabled, created_at, updated_at
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job["id"],
                    int(job.get("progress", 4)),
                    job.get("phase", "视频已接收"),
                    job["filename"],
                    job["source_path"],
                    job.get("content_hash"),
                    int(job["target_duration"]),
                    job.get("output_format", "source"),
                    int(bool(job.get("captions_enabled", True))),
                    now,
                    now,
                ),
            )
            connection.commit()
        return True

    def update_job(self, job_id: str, **changes: object) -> None:
        if not changes:
            return
        allowed = {
            "status",
            "progress",
            "phase",
            "error",
            "cancel_requested",
            "result",
        }
        invalid = set(changes) - allowed
        if invalid:
            raise ValueError(f"unsupported job fields: {sorted(invalid)}")

        assignments = []
        values = []
        for key, value in changes.items():
            column = "result_json" if key == "result" else key
            if key == "result":
                value = json.dumps(value, ensure_ascii=False)
            if key == "cancel_requested":
                value = int(bool(value))
            assignments.append(f"{column} = ?")
            values.append(value)
        assignments.append("updated_at = ?")
        values.extend((utc_now(), job_id))
        with self.connection() as connection:
            cursor = connection.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            if cursor.rowcount != 1:
                raise KeyError(job_id)

    def get_job(self, job_id: str, include_internal: bool = False) -> dict | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if not row:
                return None
            feedback = connection.execute(
                """
                SELECT clip_filename, rating, reason, created_at, updated_at
                FROM feedback WHERE job_id = ? ORDER BY clip_filename
                """,
                (job_id,),
            ).fetchall()
        return self._row_to_job(row, feedback, include_internal)

    def list_jobs(self, limit: int = 20) -> list[dict]:
        safe_limit = min(max(int(limit), 1), 100)
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [self._row_to_job(row, [], False) for row in rows]

    def count_active(self) -> int:
        with self.connection() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'processing')"
                ).fetchone()[0]
            )

    def performance_stats(self, limit: int = 50) -> dict:
        safe_limit = min(max(int(limit), 1), 500)
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT result_json FROM jobs
                WHERE status = 'completed' AND result_json IS NOT NULL
                ORDER BY updated_at DESC LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        totals = []
        throughputs = []
        cache_hits = 0
        for row in rows:
            try:
                result = json.loads(row["result_json"])
                performance = result.get("performance", {})
            except (json.JSONDecodeError, TypeError):
                continue
            total_ms = performance.get("total_ms")
            throughput = performance.get(
                "source_seconds_per_processing_second"
            )
            if isinstance(total_ms, (int, float)) and total_ms >= 0:
                totals.append(float(total_ms))
                cache = result.get("cache")
                cache_hits += int(
                    bool(cache.get("hit"))
                    if isinstance(cache, dict)
                    else False
                )
            if isinstance(throughput, (int, float)) and throughput >= 0:
                throughputs.append(float(throughput))

        def percentile(values: list[float], ratio: float) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            index = max(0, math.ceil(len(ordered) * ratio) - 1)
            return round(ordered[index], 2)

        return {
            "sample_count": len(totals),
            "median_total_ms": (
                round(median(totals), 2)
                if totals
                else None
            ),
            "p95_total_ms": percentile(totals, 0.95),
            "median_throughput": (
                round(median(throughputs), 2)
                if throughputs
                else None
            ),
            "cache_hit_count": cache_hits,
            "cache_hit_rate": (
                round(cache_hits / len(totals), 4)
                if totals
                else 0.0
            ),
        }

    def find_reusable_job(
        self,
        *,
        content_hash: str | None,
        target_duration: int,
        output_format: str,
        captions_enabled: bool,
        scoring_profile_version: int,
        processing_signature: str,
        exclude_job_id: str,
        limit: int = 20,
    ) -> dict | None:
        if not content_hash:
            return None
        safe_limit = min(max(int(limit), 1), 100)
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'completed'
                    AND content_hash = ?
                    AND target_duration = ?
                    AND output_format = ?
                    AND captions_enabled = ?
                    AND id != ?
                    AND result_json IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (
                    content_hash,
                    int(target_duration),
                    output_format,
                    int(bool(captions_enabled)),
                    exclude_job_id,
                    safe_limit,
                ),
            ).fetchall()
        for row in rows:
            try:
                result = json.loads(row["result_json"])
                profile = result.get("scoring_profile")
                cache = result.get("cache")
                if not isinstance(profile, dict) or not isinstance(cache, dict):
                    continue
                version = int(profile.get("version", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            cached_signature = cache.get("processing_signature")
            if (
                version == int(scoring_profile_version)
                and cached_signature == processing_signature
            ):
                return self._row_to_job(row, [], False)
        return None

    def artifact_references(self) -> dict:
        upload_paths = set()
        clip_filenames = set()
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT status, source_path, result_json FROM jobs"
            ).fetchall()
        for row in rows:
            if row["status"] in ACTIVE_STATUSES:
                upload_paths.add(str(Path(row["source_path"]).resolve()))
            if not row["result_json"]:
                continue
            try:
                result = json.loads(row["result_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            clip_filenames.update(
                clip["filename"]
                for clip in result.get("clips", [])
                if isinstance(clip, dict) and clip.get("filename")
            )
        return {
            "upload_paths": upload_paths,
            "clip_filenames": clip_filenames,
        }

    def list_expired_terminal_jobs(
        self,
        before: str,
        limit: int = 100,
    ) -> list[dict]:
        safe_limit = min(max(int(limit), 1), 1000)
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('completed', 'failed', 'cancelled')
                    AND updated_at < ?
                ORDER BY updated_at
                LIMIT ?
                """,
                (before, safe_limit),
            ).fetchall()
        return [
            self._row_to_job(row, [], True)
            for row in rows
        ]

    def claim_next_job(self) -> dict | None:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'queued' AND cancel_requested = 0
                ORDER BY created_at
                LIMIT 1
                """
            ).fetchone()
            if not row:
                connection.rollback()
                return None
            now = utc_now()
            connection.execute(
                """
                UPDATE jobs
                SET status = 'processing', progress = 8, phase = '准备分析',
                    updated_at = ?
                WHERE id = ?
                """,
                (now, row["id"]),
            )
            connection.commit()
        return self.get_job(row["id"], include_internal=True)

    def request_cancel(self, job_id: str) -> dict | None:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if not row:
                connection.rollback()
                return None
            if row["status"] in TERMINAL_STATUSES:
                connection.rollback()
                return self.get_job(job_id)
            queued = row["status"] == "queued"
            connection.execute(
                """
                UPDATE jobs
                SET cancel_requested = 1,
                    status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END,
                    progress = CASE WHEN status = 'queued' THEN progress ELSE progress END,
                    phase = CASE
                        WHEN status = 'queued' THEN '任务已取消'
                        ELSE '正在取消'
                    END,
                    updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), job_id),
            )
            connection.commit()
        job = self.get_job(job_id, include_internal=queued)
        return job

    def is_cancel_requested(self, job_id: str) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def recover_incomplete(self) -> dict:
        recovered = failed = 0
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT id, source_path, cancel_requested FROM jobs WHERE status IN ('queued', 'processing')"
            ).fetchall()
            for row in rows:
                source_exists = Path(row["source_path"]).is_file()
                if row["cancel_requested"]:
                    Path(row["source_path"]).unlink(missing_ok=True)
                    connection.execute(
                        """
                        UPDATE jobs SET status = 'cancelled', phase = '任务已取消',
                            updated_at = ? WHERE id = ?
                        """,
                        (utc_now(), row["id"]),
                    )
                    failed += 1
                elif source_exists:
                    connection.execute(
                        """
                        UPDATE jobs SET status = 'queued', progress = 4,
                            phase = '服务重启，任务已恢复', updated_at = ?
                        WHERE id = ?
                        """,
                        (utc_now(), row["id"]),
                    )
                    recovered += 1
                else:
                    connection.execute(
                        """
                        UPDATE jobs SET status = 'failed', phase = '恢复失败',
                            error = '服务重启后未找到原视频', updated_at = ?
                        WHERE id = ?
                        """,
                        (utc_now(), row["id"]),
                    )
                    failed += 1
        return {"recovered": recovered, "failed": failed}

    def save_feedback(
        self,
        job_id: str,
        clip_filename: str,
        rating: str,
        reason: str | None,
    ) -> dict:
        now = utc_now()
        with self.connection() as connection:
            job_row = connection.execute(
                "SELECT result_json FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if not job_row:
                raise KeyError(job_id)
            features_json = None
            score = None
            profile_version = None
            if job_row["result_json"]:
                result = json.loads(job_row["result_json"])
                clip = next(
                    (
                        item
                        for item in result.get("clips", [])
                        if item.get("filename") == clip_filename
                    ),
                    None,
                )
                if clip and isinstance(clip.get("features"), dict):
                    features_json = json.dumps(
                        {
                            "features": clip["features"],
                            "available_features": clip.get("available_features"),
                        },
                        ensure_ascii=False,
                    )
                    score = clip.get("score")
                    profile_version = result.get("scoring_profile", {}).get(
                        "version"
                    )
            connection.execute(
                """
                INSERT INTO feedback (
                    job_id, clip_filename, rating, reason, features_json,
                    score, profile_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, clip_filename) DO UPDATE SET
                    rating = excluded.rating,
                    reason = excluded.reason,
                    features_json = COALESCE(excluded.features_json, feedback.features_json),
                    score = COALESCE(excluded.score, feedback.score),
                    profile_version = COALESCE(
                        excluded.profile_version,
                        feedback.profile_version
                    ),
                    updated_at = excluded.updated_at
                """,
                (
                    job_id,
                    clip_filename,
                    rating,
                    reason,
                    features_json,
                    score,
                    profile_version,
                    now,
                    now,
                ),
            )
        return {
            "clip_filename": clip_filename,
            "rating": rating,
            "reason": reason,
            "updated_at": now,
        }

    def feedback_training_examples(self) -> dict:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT feedback.job_id, feedback.clip_filename, feedback.rating,
                    feedback.features_json, feedback.score,
                    feedback.profile_version, feedback.updated_at,
                    jobs.result_json
                FROM feedback
                JOIN jobs ON jobs.id = feedback.job_id
                ORDER BY feedback.job_id, feedback.clip_filename
                """
            ).fetchall()

        examples = []
        skipped = 0
        for row in rows:
            snapshot = None
            if row["features_json"]:
                try:
                    snapshot = json.loads(row["features_json"])
                except json.JSONDecodeError:
                    snapshot = None
            if not snapshot and row["result_json"]:
                result = json.loads(row["result_json"])
                clip = next(
                    (
                        item
                        for item in result.get("clips", [])
                        if item.get("filename") == row["clip_filename"]
                    ),
                    None,
                )
                if clip and isinstance(clip.get("features"), dict):
                    snapshot = {
                        "features": clip["features"],
                        "available_features": clip.get("available_features"),
                    }
            if not snapshot or not isinstance(snapshot.get("features"), dict):
                skipped += 1
                continue
            examples.append(
                {
                    "job_id": row["job_id"],
                    "clip_filename": row["clip_filename"],
                    "rating": row["rating"],
                    "features": snapshot["features"],
                    "available_features": snapshot.get("available_features"),
                    "score": row["score"],
                    "profile_version": row["profile_version"],
                    "updated_at": row["updated_at"],
                }
            )
        return {"examples": examples, "skipped": skipped, "total": len(rows)}

    def save_scoring_profile(self, profile: dict, activate: bool) -> dict:
        payload = dict(profile)
        payload.pop("version", None)
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if activate:
                connection.execute(
                    "UPDATE scoring_profiles SET active = 0 WHERE active = 1"
                )
            cursor = connection.execute(
                """
                INSERT INTO scoring_profiles (
                    status, active, profile_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    payload.get("status", "unknown"),
                    int(activate),
                    json.dumps(payload, ensure_ascii=False),
                    now,
                ),
            )
            profile_id = int(cursor.lastrowid)
            connection.commit()
        return {
            **payload,
            "version": profile_id,
            "active": bool(activate),
            "created_at": now,
        }

    def active_scoring_profile(self) -> dict | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM scoring_profiles
                WHERE active = 1 ORDER BY id DESC LIMIT 1
                """
            ).fetchone()
        return self._row_to_profile(row)

    def latest_scoring_profile(self) -> dict | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM scoring_profiles ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return self._row_to_profile(row)

    def feedback_stats(self) -> dict:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT rating, COUNT(*) AS count FROM feedback GROUP BY rating"
            ).fetchall()
        counts = {row["rating"]: row["count"] for row in rows}
        keep = int(counts.get("keep", 0))
        reject = int(counts.get("reject", 0))
        total = keep + reject
        return {
            "keep": keep,
            "reject": reject,
            "total": total,
            "keep_rate": round(keep / total, 4) if total else None,
        }

    @staticmethod
    def _row_to_profile(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        profile = json.loads(row["profile_json"])
        return {
            **profile,
            "version": int(row["id"]),
            "active": bool(row["active"]),
            "created_at": row["created_at"],
        }

    def delete_job(self, job_id: str) -> dict | None:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if not row:
                connection.rollback()
                return None
            if row["status"] in ACTIVE_STATUSES:
                connection.rollback()
                raise RuntimeError("active job")
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            connection.commit()
        return self._row_to_job(row, [], True)

    @staticmethod
    def _row_to_job(
        row: sqlite3.Row,
        feedback_rows: list[sqlite3.Row],
        include_internal: bool,
    ) -> dict:
        job = {
            "id": row["id"],
            "status": row["status"],
            "progress": row["progress"],
            "phase": row["phase"],
            "filename": row["filename"],
            "target_duration": row["target_duration"],
            "output_format": row["output_format"],
            "captions_enabled": bool(row["captions_enabled"]),
            "error": row["error"],
            "cancel_requested": bool(row["cancel_requested"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "feedback": [dict(feedback) for feedback in feedback_rows],
        }
        if row["result_json"]:
            job["result"] = json.loads(row["result_json"])
        if include_internal:
            job["source_path"] = row["source_path"]
            job["content_hash"] = row["content_hash"]
        return job
