import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from job_store import JobStore


class JobStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.store = JobStore(self.root / "jobs.db")
        self.store.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def create_job(self, job_id: str = "job-1", capacity: int = 4) -> Path:
        source = self.root / f"{job_id}.mp4"
        source.write_bytes(b"fixture")
        created = self.store.create_job(
            {
                "id": job_id,
                "filename": source.name,
                "source_path": str(source),
                "target_duration": 30,
            },
            capacity,
        )
        self.assertTrue(created)
        return source

    def test_job_survives_store_recreation(self) -> None:
        self.create_job()
        claimed = self.store.claim_next_job()
        self.assertEqual(claimed["status"], "processing")
        self.store.update_job(
            "job-1",
            status="completed",
            progress=100,
            phase="成片已就绪",
            result={"clips": [{"filename": "job-1-01.mp4"}]},
        )

        reopened = JobStore(self.root / "jobs.db")
        reopened.initialize()
        job = reopened.get_job("job-1")
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["result"]["clips"][0]["filename"], "job-1-01.mp4")
        self.assertEqual(job["output_format"], "source")
        self.assertTrue(job["captions_enabled"])

    def test_export_preferences_survive_store_recreation(self) -> None:
        source = self.root / "portrait-job.mp4"
        source.write_bytes(b"fixture")
        self.assertTrue(
            self.store.create_job(
                {
                    "id": "portrait-job",
                    "filename": source.name,
                    "source_path": str(source),
                    "target_duration": 30,
                    "output_format": "portrait",
                    "captions_enabled": False,
                },
                capacity=4,
            )
        )
        reopened = JobStore(self.root / "jobs.db")
        reopened.initialize()
        job = reopened.get_job("portrait-job")
        self.assertEqual(job["output_format"], "portrait")
        self.assertFalse(job["captions_enabled"])

    def test_initialize_migrates_legacy_feedback_table(self) -> None:
        legacy_path = self.root / "legacy.db"
        with closing(sqlite3.connect(legacy_path)) as connection:
            connection.execute(
                """
                CREATE TABLE jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    phase TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    target_duration INTEGER NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    clip_filename TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(job_id, clip_filename)
                )
                """
            )
            connection.commit()

        legacy_store = JobStore(legacy_path)
        legacy_store.initialize()
        with legacy_store.connection() as connection:
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(feedback)"
                ).fetchall()
            }
            job_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(jobs)"
                ).fetchall()
            }
        self.assertTrue(
            {"features_json", "score", "profile_version"}.issubset(columns)
        )
        self.assertTrue(
            {
                "output_format",
                "captions_enabled",
                "content_hash",
            }.issubset(job_columns)
        )

    def test_capacity_is_enforced_transactionally(self) -> None:
        self.create_job("job-1", capacity=1)
        second_source = self.root / "job-2.mp4"
        second_source.write_bytes(b"fixture")
        created = self.store.create_job(
            {
                "id": "job-2",
                "filename": second_source.name,
                "source_path": str(second_source),
                "target_duration": 30,
            },
            capacity=1,
        )
        self.assertFalse(created)

    def test_processing_job_recovers_after_restart(self) -> None:
        self.create_job()
        self.store.claim_next_job()
        recovery = self.store.recover_incomplete()
        job = self.store.get_job("job-1")
        self.assertEqual(recovery["recovered"], 1)
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["phase"], "服务重启，任务已恢复")

    def test_missing_source_fails_recovery(self) -> None:
        source = self.create_job()
        self.store.claim_next_job()
        source.unlink()
        recovery = self.store.recover_incomplete()
        job = self.store.get_job("job-1")
        self.assertEqual(recovery["failed"], 1)
        self.assertEqual(job["status"], "failed")

    def test_queued_job_can_be_cancelled(self) -> None:
        self.create_job()
        job = self.store.request_cancel("job-1")
        self.assertEqual(job["status"], "cancelled")
        self.assertTrue(job["cancel_requested"])
        self.assertIn("source_path", job)
        self.assertIsNone(self.store.claim_next_job())

    def test_feedback_is_upserted_and_aggregated(self) -> None:
        self.create_job()
        self.store.update_job(
            "job-1",
            status="completed",
            progress=100,
            phase="成片已就绪",
            result={"clips": [{"filename": "job-1-01.mp4"}]},
        )
        self.store.save_feedback("job-1", "job-1-01.mp4", "keep", None)
        self.store.save_feedback("job-1", "job-1-01.mp4", "reject", "切点太早")
        job = self.store.get_job("job-1")
        stats = self.store.feedback_stats()
        self.assertEqual(len(job["feedback"]), 1)
        self.assertEqual(job["feedback"][0]["rating"], "reject")
        self.assertEqual(stats, {"keep": 0, "reject": 1, "total": 1, "keep_rate": 0.0})

    def test_active_job_cannot_be_deleted(self) -> None:
        self.create_job()
        with self.assertRaises(RuntimeError):
            self.store.delete_job("job-1")

    def test_performance_stats_aggregate_completed_results(self) -> None:
        for index, (total_ms, throughput, cache_hit) in enumerate(
            ((100.0, 4.0, False), (300.0, 2.0, True)),
            start=1,
        ):
            job_id = f"perf-{index}"
            self.create_job(job_id)
            self.store.update_job(
                job_id,
                status="completed",
                result={
                    "clips": [],
                    "performance": {
                        "total_ms": total_ms,
                        "source_seconds_per_processing_second": throughput,
                    },
                    "cache": {"hit": cache_hit},
                },
            )

        stats = self.store.performance_stats()
        self.assertEqual(stats["sample_count"], 2)
        self.assertEqual(stats["median_total_ms"], 200.0)
        self.assertEqual(stats["p95_total_ms"], 300.0)
        self.assertEqual(stats["median_throughput"], 3.0)
        self.assertEqual(stats["cache_hit_count"], 1)
        self.assertEqual(stats["cache_hit_rate"], 0.5)

    def test_artifact_references_include_active_uploads_and_result_clips(self) -> None:
        active_source = self.create_job("active")
        self.create_job("completed")
        self.store.update_job(
            "completed",
            status="completed",
            result={"clips": [{"filename": "completed-01.mp4"}]},
        )

        references = self.store.artifact_references()
        self.assertIn(
            str(active_source.resolve()),
            references["upload_paths"],
        )
        self.assertNotIn(
            str((self.root / "completed.mp4").resolve()),
            references["upload_paths"],
        )
        self.assertEqual(
            references["clip_filenames"],
            {"completed-01.mp4"},
        )

    def test_expired_terminal_jobs_are_listed_without_active_jobs(self) -> None:
        self.create_job("expired")
        self.store.update_job("expired", status="failed", error="fixture")
        self.create_job("active")
        with self.store.connection() as connection:
            connection.execute(
                "UPDATE jobs SET updated_at = ? WHERE id IN (?, ?)",
                ("2000-01-01T00:00:00+00:00", "expired", "active"),
            )

        expired = self.store.list_expired_terminal_jobs(
            "2020-01-01T00:00:00+00:00"
        )
        self.assertEqual([job["id"] for job in expired], ["expired"])
        self.assertIn("source_path", expired[0])

    def test_reusable_job_requires_matching_parameters_and_model(self) -> None:
        source = self.root / "cached.mp4"
        source.write_bytes(b"fixture")
        self.assertTrue(
            self.store.create_job(
                {
                    "id": "cached",
                    "filename": source.name,
                    "source_path": str(source),
                    "content_hash": "sha256-fixture",
                    "target_duration": 30,
                    "output_format": "portrait",
                    "captions_enabled": True,
                },
                capacity=4,
            )
        )
        self.store.update_job(
            "cached",
            status="completed",
            result={
                "scoring_profile": {"version": 3},
                "cache": {"processing_signature": "pipeline-v1"},
                "clips": [{"filename": "cached-01.mp4"}],
            },
        )

        reusable = self.store.find_reusable_job(
            content_hash="sha256-fixture",
            target_duration=30,
            output_format="portrait",
            captions_enabled=True,
            scoring_profile_version=3,
            processing_signature="pipeline-v1",
            exclude_job_id="new-job",
        )
        self.assertEqual(reusable["id"], "cached")
        self.assertIsNone(
            self.store.find_reusable_job(
                content_hash="sha256-fixture",
                target_duration=30,
                output_format="portrait",
                captions_enabled=True,
                scoring_profile_version=4,
                processing_signature="pipeline-v1",
                exclude_job_id="new-job",
            )
        )
        self.assertIsNone(
            self.store.find_reusable_job(
                content_hash="sha256-fixture",
                target_duration=30,
                output_format="square",
                captions_enabled=True,
                scoring_profile_version=3,
                processing_signature="pipeline-v1",
                exclude_job_id="new-job",
            )
        )
        self.assertIsNone(
            self.store.find_reusable_job(
                content_hash="sha256-fixture",
                target_duration=30,
                output_format="portrait",
                captions_enabled=True,
                scoring_profile_version=3,
                processing_signature="pipeline-v2",
                exclude_job_id="new-job",
            )
        )


if __name__ == "__main__":
    unittest.main()
