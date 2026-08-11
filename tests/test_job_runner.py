import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from feedback_calibration import DEFAULT_WEIGHTS
from job_store import JobStore
from runtime_storage import RuntimeStorage
from process_control import run_process
from server import JobRunner, PROCESSING_SIGNATURE


class JobRunnerTests(unittest.TestCase):
    def test_processing_job_honors_cooperative_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "runner-cancel.mp4"
            source.write_bytes(b"fixture")
            clips = root / "clips"
            clips.mkdir()
            store = JobStore(root / "jobs.db")
            store.initialize()
            active_profile = store.save_scoring_profile(
                {
                    "status": "active",
                    "source": "test",
                    "weights": dict(DEFAULT_WEIGHTS),
                    "sample_count": 12,
                },
                activate=True,
            )
            self.assertTrue(
                store.create_job(
                    {
                        "id": "runner-cancel",
                        "filename": source.name,
                        "source_path": str(source),
                        "target_duration": 30,
                    },
                    capacity=2,
                )
            )

            first_stage_finished = threading.Event()
            continue_processing = threading.Event()
            received_profile = {}

            def fake_process(
                _source,
                _clips,
                _job_id,
                _duration,
                progress,
                **_kwargs,
            ):
                received_profile.update(_kwargs["scoring_profile"])
                progress(20, "第一阶段")
                first_stage_finished.set()
                continue_processing.wait(timeout=5)
                progress(40, "第二阶段")
                return {"clips": []}

            runner = JobRunner(store, worker_count=1, clip_dir=clips)
            with patch("server.process_video", side_effect=fake_process):
                runner.start()
                self.assertTrue(first_stage_finished.wait(timeout=5))
                store.request_cancel("runner-cancel")
                continue_processing.set()
                for _ in range(100):
                    job = store.get_job("runner-cancel")
                    if job["status"] == "cancelled":
                        break
                    time.sleep(0.02)
                else:
                    self.fail("runner did not cancel the processing job")
                runner.stop()

            self.assertFalse(source.exists())
            self.assertEqual(store.get_job("runner-cancel")["phase"], "任务已取消")
            self.assertEqual(received_profile["version"], active_profile["version"])

    def test_processing_job_fails_when_cooperative_budget_is_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "runner-timeout.mp4"
            source.write_bytes(b"fixture")
            clips = root / "clips"
            clips.mkdir()
            store = JobStore(root / "jobs.db")
            store.initialize()
            self.assertTrue(
                store.create_job(
                    {
                        "id": "runner-timeout",
                        "filename": source.name,
                        "source_path": str(source),
                        "target_duration": 30,
                    },
                    capacity=2,
                )
            )

            def fake_process(
                _source,
                _clips,
                _job_id,
                _duration,
                progress,
                **_kwargs,
            ):
                progress(20, "第一阶段")
                time.sleep(0.05)
                progress(40, "第二阶段")
                return {"clips": []}

            runner = JobRunner(
                store,
                worker_count=1,
                clip_dir=clips,
                job_timeout_seconds=0.02,
            )
            with patch("server.process_video", side_effect=fake_process):
                runner.start()
                for _ in range(100):
                    job = store.get_job("runner-timeout")
                    if job["status"] == "failed":
                        break
                    time.sleep(0.02)
                else:
                    self.fail("runner did not enforce the processing budget")
                runner.stop()

            job = store.get_job("runner-timeout")
            self.assertEqual(job["error"], "任务处理超时，已清理临时文件")
            self.assertFalse(source.exists())

    def test_identical_completed_job_reuses_independent_clip_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = root / "runtime"
            uploads = runtime / "uploads"
            clips = runtime / "clips"
            storage = RuntimeStorage(
                runtime,
                uploads,
                clips,
                max_bytes=128 * 1024 * 1024,
                min_free_bytes=0,
            )
            store = JobStore(root / "jobs.db")
            store.initialize()

            cached_source = uploads / "cache-source.mp4"
            cached_source.write_bytes(b"same-content")
            cached_clip = clips / "cache-source-01.mp4"
            cached_clip.write_bytes(b"rendered-clip")
            self.assertTrue(
                store.create_job(
                    {
                        "id": "cache-source",
                        "filename": cached_source.name,
                        "source_path": str(cached_source),
                        "content_hash": "same-sha256",
                        "target_duration": 30,
                        "output_format": "source",
                        "captions_enabled": True,
                    },
                    capacity=4,
                )
            )
            store.update_job(
                "cache-source",
                status="completed",
                result={
                    "metadata": {"duration": 60.0},
                    "scoring_profile": {"version": 0},
                    "cache": {
                        "hit": False,
                        "processing_signature": PROCESSING_SIGNATURE,
                    },
                    "clips": [
                        {
                            "filename": cached_clip.name,
                            "url": f"/generated/{cached_clip.name}",
                            "render": {"caption_count": 0},
                        }
                    ],
                },
            )

            target_source = uploads / "cache-target.mp4"
            target_source.write_bytes(b"same-content")
            self.assertTrue(
                store.create_job(
                    {
                        "id": "cache-target",
                        "filename": target_source.name,
                        "source_path": str(target_source),
                        "content_hash": "same-sha256",
                        "target_duration": 30,
                        "output_format": "source",
                        "captions_enabled": True,
                    },
                    capacity=4,
                )
            )

            runner = JobRunner(
                store,
                worker_count=1,
                clip_dir=clips,
                storage=storage,
            )
            with patch("server.process_video") as process_mock:
                runner.start()
                for _ in range(100):
                    job = store.get_job("cache-target")
                    if job["status"] == "completed":
                        break
                    time.sleep(0.02)
                else:
                    self.fail("runner did not complete the cached job")
                runner.stop()
                process_mock.assert_not_called()

            result = store.get_job("cache-target")["result"]
            copied_clip = clips / "cache-target-01.mp4"
            self.assertTrue(result["cache"]["hit"])
            self.assertEqual(result["cache"]["source_job_id"], "cache-source")
            self.assertEqual(result["clips"][0]["filename"], copied_clip.name)
            self.assertEqual(copied_clip.read_bytes(), cached_clip.read_bytes())
            self.assertTrue(cached_clip.exists())
            self.assertFalse(target_source.exists())
            self.assertGreater(
                result["performance"][
                    "source_seconds_per_processing_second"
                ],
                0,
            )

    def test_cancellation_terminates_an_active_external_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "external-cancel.mp4"
            source.write_bytes(b"fixture")
            process_started = root / "process-started.txt"
            clips = root / "clips"
            clips.mkdir()
            store = JobStore(root / "jobs.db")
            store.initialize()
            self.assertTrue(
                store.create_job(
                    {
                        "id": "external-cancel",
                        "filename": source.name,
                        "source_path": str(source),
                        "target_duration": 30,
                    },
                    capacity=2,
                )
            )

            def fake_process(
                _source,
                _clips,
                _job_id,
                _duration,
                progress,
                **_kwargs,
            ):
                progress(20, "启动外部媒体进程")
                run_process(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import pathlib,time;"
                            f"pathlib.Path({str(process_started)!r})"
                            ".write_text('started');"
                            "time.sleep(10)"
                        ),
                    ],
                    timeout=20,
                    check=True,
                )
                return {"clips": []}

            runner = JobRunner(store, worker_count=1, clip_dir=clips)
            with patch("server.process_video", side_effect=fake_process):
                runner.start()
                for _ in range(100):
                    if process_started.exists():
                        break
                    time.sleep(0.02)
                else:
                    self.fail("external process did not start")
                cancel_started = time.monotonic()
                store.request_cancel("external-cancel")
                for _ in range(100):
                    job = store.get_job("external-cancel")
                    if job["status"] == "cancelled":
                        break
                    time.sleep(0.02)
                else:
                    self.fail("active external process was not cancelled")
                cancel_elapsed = time.monotonic() - cancel_started
                runner.stop()

            self.assertLess(cancel_elapsed, 2.0)
            self.assertFalse(source.exists())
            self.assertEqual(
                store.get_job("external-cancel")["phase"],
                "任务已取消",
            )

    def test_job_budget_terminates_an_active_external_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "external-timeout.mp4"
            source.write_bytes(b"fixture")
            clips = root / "clips"
            clips.mkdir()
            store = JobStore(root / "jobs.db")
            store.initialize()
            self.assertTrue(
                store.create_job(
                    {
                        "id": "external-timeout",
                        "filename": source.name,
                        "source_path": str(source),
                        "target_duration": 30,
                    },
                    capacity=2,
                )
            )

            def fake_process(
                _source,
                _clips,
                _job_id,
                _duration,
                progress,
                **_kwargs,
            ):
                progress(20, "启动长时媒体进程")
                run_process(
                    [
                        sys.executable,
                        "-c",
                        "import time; time.sleep(10)",
                    ],
                    timeout=20,
                    check=True,
                )
                return {"clips": []}

            runner = JobRunner(
                store,
                worker_count=1,
                clip_dir=clips,
                job_timeout_seconds=0.2,
            )
            started = time.monotonic()
            with patch("server.process_video", side_effect=fake_process):
                runner.start()
                for _ in range(150):
                    job = store.get_job("external-timeout")
                    if job["status"] == "failed":
                        source_exists_at_terminal = source.exists()
                        break
                    time.sleep(0.02)
                else:
                    self.fail("external process did not honor the job budget")
                runner.stop()

            self.assertLess(time.monotonic() - started, 2.0)
            self.assertEqual(
                store.get_job("external-timeout")["error"],
                "任务处理超时，已清理临时文件",
            )
            self.assertFalse(source_exists_at_terminal)
            self.assertFalse(source.exists())

    def test_shutdown_requeues_job_and_preserves_source_for_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "shutdown-resume.mp4"
            source.write_bytes(b"fixture")
            process_started = root / "shutdown-process-started.txt"
            clips = root / "clips"
            clips.mkdir()
            store = JobStore(root / "jobs.db")
            store.initialize()
            self.assertTrue(
                store.create_job(
                    {
                        "id": "shutdown-resume",
                        "filename": source.name,
                        "source_path": str(source),
                        "target_duration": 30,
                    },
                    capacity=2,
                )
            )

            def blocking_process(
                _source,
                _clips,
                _job_id,
                _duration,
                progress,
                **_kwargs,
            ):
                progress(30, "外部进程运行中")
                run_process(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import pathlib,time;"
                            f"pathlib.Path({str(process_started)!r})"
                            ".write_text('started');"
                            "time.sleep(10)"
                        ),
                    ],
                    timeout=20,
                    check=True,
                )
                return {"clips": []}

            runner = JobRunner(store, worker_count=1, clip_dir=clips)
            with patch("server.process_video", side_effect=blocking_process):
                runner.start()
                for _ in range(100):
                    if process_started.exists():
                        break
                    time.sleep(0.02)
                else:
                    self.fail("external process did not start")
                stop_started = time.monotonic()
                runner.stop()
                stop_elapsed = time.monotonic() - stop_started

            interrupted = store.get_job("shutdown-resume")
            self.assertLess(stop_elapsed, 2.0)
            self.assertEqual(interrupted["status"], "queued")
            self.assertEqual(interrupted["phase"], "服务停止，等待恢复")
            self.assertTrue(source.exists())

            resumed_runner = JobRunner(
                store,
                worker_count=1,
                clip_dir=clips,
            )
            with patch(
                "server.process_video",
                return_value={"clips": []},
            ):
                resumed_runner.start()
                for _ in range(100):
                    resumed = store.get_job("shutdown-resume")
                    if resumed["status"] == "completed":
                        break
                    time.sleep(0.02)
                else:
                    self.fail("requeued job did not resume after restart")
                resumed_runner.stop()

            self.assertEqual(resumed_runner.recovery["recovered"], 1)
            self.assertFalse(source.exists())


if __name__ == "__main__":
    unittest.main()
