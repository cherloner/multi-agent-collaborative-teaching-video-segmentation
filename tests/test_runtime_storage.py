import os
import tempfile
import time
import unittest
from pathlib import Path

from job_store import JobStore
from runtime_storage import (
    RuntimeStorage,
    StorageCapacityError,
    run_storage_maintenance,
)


class RuntimeStorageTests(unittest.TestCase):
    def make_storage(
        self,
        root: Path,
        *,
        max_bytes: int = 1024 * 1024,
    ) -> RuntimeStorage:
        runtime = root / "runtime"
        return RuntimeStorage(
            runtime,
            runtime / "uploads",
            runtime / "clips",
            max_bytes=max_bytes,
            min_free_bytes=0,
        )

    def test_snapshot_and_quota_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = self.make_storage(Path(temp_dir), max_bytes=10)
            (storage.upload_dir / "source.mp4").write_bytes(b"1234")
            (storage.clip_dir / "clip.mp4").write_bytes(b"123")

            snapshot = storage.snapshot()
            self.assertEqual(snapshot["runtime_bytes"], 7)
            self.assertEqual(snapshot["available_budget_bytes"], 3)
            self.assertTrue(storage.can_allocate(3))
            with self.assertRaises(StorageCapacityError):
                storage.ensure_capacity(4)

    def test_orphan_cleanup_preserves_references_and_outside_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            storage = self.make_storage(root)
            active_upload = storage.upload_dir / "active.mp4"
            orphan_upload = storage.upload_dir / "orphan.mp4"
            referenced_clip = storage.clip_dir / "referenced.mp4"
            orphan_clip = storage.clip_dir / "orphan.mp4"
            outside = root / "outside.txt"
            for path in (
                active_upload,
                orphan_upload,
                referenced_clip,
                orphan_clip,
                outside,
            ):
                path.write_bytes(b"fixture")
                os.utime(path, (time.time() - 7200, time.time() - 7200))

            cleanup = storage.cleanup_orphans(
                {
                    "upload_paths": {str(active_upload.resolve())},
                    "clip_filenames": {referenced_clip.name},
                },
                grace_seconds=3600,
            )

            self.assertEqual(cleanup["removed_files"], 2)
            self.assertTrue(active_upload.exists())
            self.assertTrue(referenced_clip.exists())
            self.assertFalse(orphan_upload.exists())
            self.assertFalse(orphan_clip.exists())
            self.assertTrue(outside.exists())

            storage.remove_job_artifacts(
                {
                    "source_path": str(outside),
                    "result": {
                        "clips": [{"filename": "../../outside.txt"}],
                    },
                }
            )
            self.assertTrue(outside.exists())

    def test_retention_removes_only_expired_terminal_job_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            storage = self.make_storage(root)
            store = JobStore(root / "jobs.db")
            store.initialize()

            for job_id in ("expired", "fresh"):
                source = storage.upload_dir / f"{job_id}.mp4"
                clip = storage.clip_dir / f"{job_id}-01.mp4"
                source.write_bytes(b"source")
                clip.write_bytes(b"clip")
                self.assertTrue(
                    store.create_job(
                        {
                            "id": job_id,
                            "filename": source.name,
                            "source_path": str(source),
                            "target_duration": 30,
                        },
                        capacity=4,
                    )
                )
                store.update_job(
                    job_id,
                    status="completed",
                    result={"clips": [{"filename": clip.name}]},
                )

            with store.connection() as connection:
                connection.execute(
                    "UPDATE jobs SET updated_at = ? WHERE id = ?",
                    ("2000-01-01T00:00:00+00:00", "expired"),
                )

            maintenance = run_storage_maintenance(
                store,
                storage,
                retention_days=30,
                orphan_grace_hours=-1,
            )

            self.assertEqual(maintenance["expired_jobs_removed"], 1)
            self.assertIsNone(store.get_job("expired"))
            self.assertFalse((storage.upload_dir / "expired.mp4").exists())
            self.assertFalse((storage.clip_dir / "expired-01.mp4").exists())
            self.assertIsNotNone(store.get_job("fresh"))
            self.assertTrue((storage.upload_dir / "fresh.mp4").exists())
            self.assertTrue((storage.clip_dir / "fresh-01.mp4").exists())


if __name__ == "__main__":
    unittest.main()
