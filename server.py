from __future__ import annotations

import json
import mimetypes
import os
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import monotonic, perf_counter
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cutpoint_pipeline import process_video
from feedback_calibration import (
    default_scoring_profile,
    refresh_scoring_profile,
    scoring_model_status,
)
from job_store import JobStore
from process_control import POLL_INTERVAL_SECONDS, controlled_processes
from semantic_signals import semantic_provider_status
from speaker_signals import speaker_provider_status
from runtime_storage import (
    RuntimeStorage,
    StorageCapacityError,
    run_storage_maintenance,
)


ROOT = Path(__file__).resolve().parent


def _build_processing_signature() -> str:
    digest = sha256(b"CUTPOINT-processing-cache-v1")
    source_names = (
        "server.py",
        "cutpoint_pipeline.py",
        "render_profiles.py",
        "audio_quality.py",
        "semantic_signals.py",
        "speaker_signals.py",
        "visual_signals.py",
        "feedback_calibration.py",
        "process_control.py",
        "requirements.txt",
        "requirements-semantic.txt",
        "requirements-speaker.txt",
    )
    for name in source_names:
        path = ROOT / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    cache_environment = {
        key: os.getenv(key, "")
        for key in (
            "CUTPOINT_ASR_PROVIDER",
            "CUTPOINT_WHISPER_MODEL",
            "CUTPOINT_WHISPER_COMPUTE_TYPE",
            "CUTPOINT_OCR_PROVIDER",
            "CUTPOINT_OCR_INTERVAL_SECONDS",
            "CUTPOINT_TESSERACT_LANG",
            "CUTPOINT_DIARIZATION_PROVIDER",
            "CUTPOINT_PYANNOTE_MODEL",
        )
    }
    for executable in ("ffmpeg", "ffprobe"):
        executable_path = shutil.which(executable)
        if not executable_path:
            continue
        try:
            stat = Path(executable_path).stat()
            cache_environment[executable] = {
                "path": executable_path,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        except OSError:
            cache_environment[executable] = {"path": executable_path}
    digest.update(
        json.dumps(
            cache_environment,
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
    )
    return digest.hexdigest()[:24]


PROCESSING_SIGNATURE = _build_processing_signature()
RUNTIME_DIR = ROOT / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
CLIP_DIR = RUNTIME_DIR / "clips"
DATABASE_PATH = Path(
    os.getenv("CUTPOINT_DB_PATH", str(RUNTIME_DIR / "cutpoint.db"))
)
MAX_UPLOAD_BYTES = int(os.getenv("CUTPOINT_MAX_UPLOAD_MB", "2048")) * 1024 * 1024
MAX_WORKERS = min(max(int(os.getenv("CUTPOINT_WORKERS", "2")), 1), 4)
MAX_PENDING_JOBS = max(
    int(os.getenv("CUTPOINT_MAX_PENDING_JOBS", "8")),
    MAX_WORKERS,
)
JOB_TIMEOUT_SECONDS = max(
    float(os.getenv("CUTPOINT_JOB_TIMEOUT_SECONDS", "3600")),
    1.0,
)
RUNTIME_MAX_BYTES = max(
    int(float(os.getenv("CUTPOINT_RUNTIME_MAX_GB", "20")) * 1024**3),
    1,
)
MIN_FREE_BYTES = max(
    int(float(os.getenv("CUTPOINT_MIN_FREE_GB", "1")) * 1024**3),
    0,
)
RETENTION_DAYS = int(os.getenv("CUTPOINT_RETENTION_DAYS", "30"))
ORPHAN_GRACE_HOURS = float(
    os.getenv("CUTPOINT_ORPHAN_GRACE_HOURS", "24")
)
ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
FFMPEG_READY = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CLIP_DIR.mkdir(parents=True, exist_ok=True)
mimetypes.add_type("image/webp", ".webp")


class JobCancelled(RuntimeError):
    pass


class JobBudgetExceeded(RuntimeError):
    pass


class JobInterrupted(RuntimeError):
    pass


class FeedbackRequest(BaseModel):
    clip_filename: str = Field(min_length=1, max_length=160)
    rating: Literal["keep", "reject"]
    reason: str | None = Field(default=None, max_length=300)


class JobRunner:
    def __init__(
        self,
        store: JobStore,
        worker_count: int,
        clip_dir: Path,
        *,
        storage: RuntimeStorage | None = None,
        job_timeout_seconds: float = JOB_TIMEOUT_SECONDS,
    ):
        self.store = store
        self.worker_count = worker_count
        self.clip_dir = clip_dir
        self.storage = storage
        self.job_timeout_seconds = max(float(job_timeout_seconds), 0.001)
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.active_workers = 0
        self.active_lock = threading.Lock()
        self.recovery = {"recovered": 0, "failed": 0}
        self.maintenance = {"status": "not_run"}

    def start(self) -> None:
        if self.threads:
            return
        self.recovery = self.store.recover_incomplete()
        self.stop_event.clear()
        for index in range(self.worker_count):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"cutpoint-worker-{index + 1}",
                daemon=True,
            )
            worker.start()
            self.threads.append(worker)
        self.notify()

    def stop(self) -> None:
        self.stop_event.set()
        self.wake_event.set()
        for worker in self.threads:
            worker.join(timeout=5)
        self.threads = []

    def notify(self) -> None:
        self.wake_event.set()

    def stats(self) -> dict:
        with self.active_lock:
            active_workers = self.active_workers
        return {
            "workers": self.worker_count,
            "active_workers": active_workers,
            "active_jobs": self.store.count_active(),
            "capacity": MAX_PENDING_JOBS,
            "job_timeout_seconds": self.job_timeout_seconds,
            "external_process_control": {
                "mode": "hard",
                "poll_interval_ms": round(
                    POLL_INTERVAL_SECONDS * 1000
                ),
            },
            "recovery": self.recovery,
            "maintenance": self.maintenance,
            "performance": self.store.performance_stats(),
        }

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            job = self.store.claim_next_job()
            if not job:
                self.wake_event.wait(timeout=0.5)
                self.wake_event.clear()
                continue
            with self.active_lock:
                self.active_workers += 1
            try:
                self._run_job(job)
            finally:
                with self.active_lock:
                    self.active_workers -= 1

    def _run_job(self, job: dict) -> None:
        job_id = job["id"]
        source_path = Path(job["source_path"])
        deadline = monotonic() + self.job_timeout_seconds
        preserve_source = False
        try:
            self._check_cancel(job_id)
            self._check_shutdown()
            self._check_budget(deadline)
            scoring_profile = (
                self.store.active_scoring_profile()
                or default_scoring_profile()
            )
            reusable = self.store.find_reusable_job(
                content_hash=job.get("content_hash"),
                target_duration=int(job["target_duration"]),
                output_format=job["output_format"],
                captions_enabled=bool(job["captions_enabled"]),
                scoring_profile_version=int(
                    scoring_profile.get("version", 0)
                ),
                processing_signature=PROCESSING_SIGNATURE,
                exclude_job_id=job_id,
            )
            cache_bytes = self._reusable_artifact_bytes(reusable)
            if reusable and cache_bytes is None:
                reusable = None
            if self.storage:
                source_size = source_path.stat().st_size
                estimated_outputs = (
                    cache_bytes
                    if reusable and cache_bytes is not None
                    else max(source_size * 2, 16 * 1024 * 1024)
                )
                self.storage.ensure_capacity(estimated_outputs)

            def report(progress: int, phase: str) -> None:
                self._check_cancel(job_id)
                self._check_shutdown()
                self._check_budget(deadline)
                self.store.update_job(job_id, progress=progress, phase=phase)

            def check_process_control() -> None:
                self._check_cancel(job_id)
                self._check_shutdown()
                self._check_budget(deadline)

            with controlled_processes(check_process_control):
                result = None
                if reusable:
                    report(72, "复用相同视频的成片结果")
                    result = self._clone_cached_result(reusable, job_id)
                if result is None:
                    result = process_video(
                        source_path,
                        self.clip_dir,
                        job_id,
                        int(job["target_duration"]),
                        report,
                        scoring_profile=scoring_profile,
                        output_format=job["output_format"],
                        captions_enabled=bool(job["captions_enabled"]),
                    )
                    result["cache"] = {
                        "hit": False,
                        "processing_signature": PROCESSING_SIGNATURE,
                    }
            self._check_cancel(job_id)
            self._check_shutdown()
            self._check_budget(deadline)
            if not self._remove_source(source_path):
                raise RuntimeError("无法清理上传源文件")
            self.store.update_job(
                job_id,
                status="completed",
                progress=100,
                phase="成片已就绪",
                result=result,
                error=None,
            )
        except JobInterrupted:
            preserve_source = True
            outputs_clean = self._cleanup_outputs(job_id)
            self.store.update_job(
                job_id,
                status="queued",
                progress=4,
                phase="服务停止，等待恢复",
                error=(
                    None
                    if outputs_clean
                    else "服务停止时部分成片清理失败"
                ),
            )
        except JobCancelled:
            outputs_clean = self._cleanup_outputs(job_id)
            source_clean = self._remove_source(source_path)
            self.store.update_job(
                job_id,
                status="cancelled",
                phase="任务已取消",
                cancel_requested=True,
                error=(
                    None
                    if outputs_clean and source_clean
                    else "任务已取消，但部分临时文件清理失败"
                ),
            )
        except Exception as exc:
            outputs_clean = self._cleanup_outputs(job_id)
            source_clean = self._remove_source(source_path)
            error_message = self._safe_error(exc)
            if not outputs_clean or not source_clean:
                error_message = (
                    f"{error_message}；部分临时文件清理失败"
                )[:500]
            self.store.update_job(
                job_id,
                status="failed",
                phase="任务失败",
                error=error_message,
            )
        finally:
            if not preserve_source:
                self._remove_source(source_path)

    def _remove_source(self, source_path: Path) -> bool:
        try:
            if self.storage:
                self.storage.remove_job_artifacts(
                    {"source_path": str(source_path), "result": {}}
                )
            else:
                source_path.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def _check_cancel(self, job_id: str) -> None:
        if self.store.is_cancel_requested(job_id):
            raise JobCancelled(job_id)

    def _check_shutdown(self) -> None:
        if self.stop_event.is_set():
            raise JobInterrupted("worker shutdown")

    @staticmethod
    def _check_budget(deadline: float) -> None:
        if monotonic() > deadline:
            raise JobBudgetExceeded("任务超过允许的处理时长")

    def _cleanup_outputs(self, job_id: str) -> bool:
        cleaned = True
        for output_path in self.clip_dir.glob(f"{job_id}-*.mp4"):
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                cleaned = False
        return cleaned

    def _reusable_artifact_bytes(self, reusable: dict | None) -> int | None:
        if not reusable:
            return None
        clips = reusable.get("result", {}).get("clips", [])
        if not clips:
            return None
        total = 0
        for clip in clips:
            filename = Path(str(clip.get("filename", ""))).name
            candidate = (self.clip_dir / filename).resolve()
            try:
                candidate.relative_to(self.clip_dir.resolve())
                if not candidate.is_file():
                    return None
                total += candidate.stat().st_size
            except (AttributeError, OSError, ValueError):
                return None
        return total

    def _clone_cached_result(
        self,
        reusable: dict,
        job_id: str,
    ) -> dict | None:
        started = perf_counter()
        result = deepcopy(reusable["result"])
        copied_paths: list[Path] = []
        copy_timings = []
        try:
            for index, clip in enumerate(result["clips"], start=1):
                source_name = Path(str(clip["filename"])).name
                source = (self.clip_dir / source_name).resolve()
                source.relative_to(self.clip_dir.resolve())
                if not source.is_file():
                    raise FileNotFoundError(source)
                target = self.clip_dir / f"{job_id}-{index:02d}.mp4"
                copy_started = perf_counter()
                shutil.copy2(source, target)
                copy_ms = round((perf_counter() - copy_started) * 1000, 2)
                copied_paths.append(target)
                copy_timings.append(copy_ms)
                clip["filename"] = target.name
                clip["url"] = f"/generated/{target.name}"
                clip.setdefault("render", {})["elapsed_ms"] = copy_ms
                clip["render"]["reused"] = True
        except (KeyError, OSError, TypeError, ValueError):
            for copied_path in copied_paths:
                copied_path.unlink(missing_ok=True)
            return None

        elapsed_seconds = perf_counter() - started
        total_ms = max(round(elapsed_seconds * 1000, 2), 0.01)
        source_duration = float(result.get("metadata", {}).get("duration", 0))
        throughput = (
            source_duration / elapsed_seconds
            if elapsed_seconds > 0 and source_duration > 0
            else None
        )
        result["performance"] = {
            "analysis_total_ms": 0.0,
            "analysis_realtime_factor": None,
            "stages_ms": {},
            "render_total_ms": round(sum(copy_timings), 2),
            "render_clips_ms": copy_timings,
            "total_ms": total_ms,
            "source_seconds_per_processing_second": (
                round(throughput, 3)
                if throughput is not None
                else None
            ),
        }
        result["cache"] = {
            "hit": True,
            "source_job_id": reusable["id"],
            "processing_signature": PROCESSING_SIGNATURE,
        }
        return result

    @staticmethod
    def _safe_error(error: Exception) -> str:
        if isinstance(error, FileNotFoundError):
            return "媒体处理程序不可用或源文件已丢失"
        if isinstance(error, JobBudgetExceeded):
            return "任务处理超时，已清理临时文件"
        if isinstance(error, StorageCapacityError):
            return "运行目录空间不足，请删除旧任务或提高存储配额"
        message = str(error).strip()
        return message[:500] if message else error.__class__.__name__


store = JobStore(DATABASE_PATH)
storage = RuntimeStorage(
    RUNTIME_DIR,
    UPLOAD_DIR,
    CLIP_DIR,
    max_bytes=RUNTIME_MAX_BYTES,
    min_free_bytes=MIN_FREE_BYTES,
)
runner = JobRunner(
    store,
    MAX_WORKERS,
    CLIP_DIR,
    storage=storage,
    job_timeout_seconds=JOB_TIMEOUT_SECONDS,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    store.initialize()
    try:
        runner.maintenance = {
            "status": "ok",
            **run_storage_maintenance(
                store,
                storage,
                retention_days=RETENTION_DAYS,
                orphan_grace_hours=ORPHAN_GRACE_HOURS,
            ),
        }
    except Exception as exc:
        runner.maintenance = {
            "status": "error",
            "error": JobRunner._safe_error(exc),
        }
    runner.start()
    try:
        yield
    finally:
        runner.stop()


app = FastAPI(
    title="CUTPOINT local API",
    version="0.7.0",
    lifespan=lifespan,
)
app.mount("/generated", StaticFiles(directory=CLIP_DIR), name="generated")
app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok" if FFMPEG_READY else "degraded",
        "engine": "ffmpeg",
        "ready": FFMPEG_READY,
        "queue": runner.stats(),
        "database": "sqlite",
        "storage": storage.snapshot(),
        "semantic": semantic_provider_status(),
        "speaker": speaker_provider_status(),
        "ranking_model": scoring_model_status(store),
    }


@app.post("/api/jobs", status_code=202)
async def create_job(
    video: UploadFile = File(...),
    target_duration: int = Form(30),
    output_format: Literal[
        "source",
        "landscape",
        "portrait",
        "square",
    ] = Form("source"),
    captions_enabled: bool = Form(True),
) -> dict:
    if not FFMPEG_READY:
        raise HTTPException(status_code=503, detail="本机未找到 FFmpeg 或 FFprobe")
    if store.count_active() >= MAX_PENDING_JOBS:
        raise HTTPException(status_code=429, detail="本机任务队列已满，请稍后再试")

    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="请上传 MP4、MOV、MKV、WebM 或 M4V 视频")
    if not 15 <= target_duration <= 60:
        raise HTTPException(status_code=422, detail="目标时长必须在 15–60 秒之间")

    job_id = uuid.uuid4().hex[:12]
    source_path = UPLOAD_DIR / f"{job_id}{suffix}"
    total = 0
    content_digest = sha256()
    baseline_usage = storage.runtime_usage_bytes()
    try:
        with source_path.open("wb") as destination:
            while chunk := await video.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="视频超过本地服务允许的大小")
                try:
                    storage.ensure_capacity(
                        total,
                        current_usage_bytes=baseline_usage,
                    )
                except StorageCapacityError as exc:
                    raise HTTPException(
                        status_code=507,
                        detail=str(exc),
                    ) from exc
                content_digest.update(chunk)
                destination.write(chunk)
    except Exception:
        source_path.unlink(missing_ok=True)
        raise
    finally:
        await video.close()

    if total == 0:
        source_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="视频文件为空")

    created = store.create_job(
        {
            "id": job_id,
            "filename": Path(video.filename or "video").name,
            "source_path": str(source_path),
            "content_hash": content_digest.hexdigest(),
            "target_duration": target_duration,
            "output_format": output_format,
            "captions_enabled": captions_enabled,
            "progress": 4,
            "phase": "视频已接收",
        },
        MAX_PENDING_JOBS,
    )
    if not created:
        source_path.unlink(missing_ok=True)
        raise HTTPException(status_code=429, detail="本机任务队列已满，请稍后再试")

    runner.notify()
    return store.get_job(job_id)


@app.get("/api/jobs")
def list_jobs(limit: int = 20) -> dict:
    return {"jobs": store.list_jobs(limit)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = store.request_cancel(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    source_path = job.get("source_path")
    if job["status"] == "cancelled" and source_path:
        storage.remove_job_artifacts(job)
    runner.notify()
    return store.get_job(job_id)


@app.post("/api/jobs/{job_id}/feedback")
def save_feedback(job_id: str, payload: FeedbackRequest) -> dict:
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="只能评价已完成任务")
    clip_filenames = {
        clip["filename"] for clip in job.get("result", {}).get("clips", [])
    }
    if payload.clip_filename not in clip_filenames:
        raise HTTPException(status_code=422, detail="候选片段不属于该任务")
    reason = payload.reason.strip() if payload.reason else None
    feedback = store.save_feedback(
        job_id,
        payload.clip_filename,
        payload.rating,
        reason or None,
    )
    calibration = refresh_scoring_profile(store)
    return {
        "feedback": feedback,
        "stats": store.feedback_stats(),
        "calibration": calibration,
    }


@app.get("/api/feedback/stats")
def feedback_stats() -> dict:
    return store.feedback_stats()


@app.get("/api/feedback/model")
def feedback_model() -> dict:
    return scoring_model_status(store)


@app.post("/api/feedback/calibrate")
def calibrate_feedback_model() -> dict:
    return refresh_scoring_profile(store, force=True)


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> None:
    try:
        job = store.delete_job(job_id)
    except RuntimeError:
        raise HTTPException(status_code=409, detail="请先取消正在处理的任务") from None
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    storage.remove_job_artifacts(job)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/styles.css")
def stylesheet() -> FileResponse:
    return FileResponse(ROOT / "styles.css", media_type="text/css")


@app.get("/script.js")
def javascript() -> FileResponse:
    return FileResponse(ROOT / "script.js", media_type="text/javascript")


if __name__ == "__main__":
    port = int(os.getenv("CUTPOINT_PORT", "4173"))
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=False)
