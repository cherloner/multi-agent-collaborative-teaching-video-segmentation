from __future__ import annotations

import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


class StorageCapacityError(RuntimeError):
    pass


class RuntimeStorage:
    def __init__(
        self,
        runtime_dir: Path,
        upload_dir: Path,
        clip_dir: Path,
        *,
        max_bytes: int,
        min_free_bytes: int,
    ):
        self.runtime_dir = runtime_dir.resolve()
        self.upload_dir = upload_dir.resolve()
        self.clip_dir = clip_dir.resolve()
        self.max_bytes = max(int(max_bytes), 1)
        self.min_free_bytes = max(int(min_free_bytes), 0)
        for directory in (self.runtime_dir, self.upload_dir, self.clip_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if not self._is_inside(self.runtime_dir, self.upload_dir):
            raise ValueError("upload directory must be inside runtime directory")
        if not self._is_inside(self.runtime_dir, self.clip_dir):
            raise ValueError("clip directory must be inside runtime directory")

    @staticmethod
    def _is_inside(base: Path, candidate: Path) -> bool:
        try:
            candidate.resolve().relative_to(base.resolve())
            return True
        except ValueError:
            return False

    def _directory_size(self, directory: Path) -> int:
        total = 0
        for item in directory.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                # A worker may finish or remove an artifact while health is
                # collecting its snapshot. A transient race should not make
                # the health endpoint fail.
                continue
        return total

    def runtime_usage_bytes(self) -> int:
        return self._directory_size(self.upload_dir) + self._directory_size(
            self.clip_dir
        )

    def snapshot(self) -> dict:
        upload_bytes = self._directory_size(self.upload_dir)
        clip_bytes = self._directory_size(self.clip_dir)
        disk = shutil.disk_usage(self.runtime_dir)
        runtime_bytes = upload_bytes + clip_bytes
        budget_remaining = max(self.max_bytes - runtime_bytes, 0)
        disk_remaining = max(disk.free - self.min_free_bytes, 0)
        return {
            "status": (
                "ok"
                if budget_remaining > 0 and disk_remaining > 0
                else "full"
            ),
            "runtime_bytes": runtime_bytes,
            "upload_bytes": upload_bytes,
            "clip_bytes": clip_bytes,
            "max_bytes": self.max_bytes,
            "disk_free_bytes": disk.free,
            "min_free_bytes": self.min_free_bytes,
            "available_budget_bytes": min(
                budget_remaining,
                disk_remaining,
            ),
        }

    def can_allocate(
        self,
        expected_bytes: int,
        *,
        current_usage_bytes: int | None = None,
    ) -> bool:
        expected = max(int(expected_bytes), 0)
        usage = (
            self.runtime_usage_bytes()
            if current_usage_bytes is None
            else max(int(current_usage_bytes), 0)
        )
        disk_free = shutil.disk_usage(self.runtime_dir).free
        return (
            usage + expected <= self.max_bytes
            and disk_free - expected >= self.min_free_bytes
        )

    def ensure_capacity(
        self,
        expected_bytes: int,
        *,
        current_usage_bytes: int | None = None,
    ) -> None:
        if not self.can_allocate(
            expected_bytes,
            current_usage_bytes=current_usage_bytes,
        ):
            raise StorageCapacityError(
                "运行目录空间预算不足，请删除旧任务或调整存储配额"
            )

    def _safe_file(self, base: Path, candidate: Path) -> Path | None:
        resolved = candidate.resolve()
        if resolved == base or not self._is_inside(base, resolved):
            return None
        return resolved

    def remove_job_artifacts(self, job: dict) -> dict:
        removed_files = 0
        removed_bytes = 0
        source_path = job.get("source_path")
        candidates = []
        if source_path:
            safe_source = self._safe_file(
                self.upload_dir,
                Path(source_path),
            )
            if safe_source:
                candidates.append(safe_source)
        for clip in job.get("result", {}).get("clips", []):
            filename = Path(str(clip.get("filename", ""))).name
            if not filename:
                continue
            safe_clip = self._safe_file(
                self.clip_dir,
                self.clip_dir / filename,
            )
            if safe_clip:
                candidates.append(safe_clip)

        for candidate in candidates:
            if not candidate.is_file():
                continue
            size = candidate.stat().st_size
            candidate.unlink()
            removed_files += 1
            removed_bytes += size
        return {
            "removed_files": removed_files,
            "removed_bytes": removed_bytes,
        }

    def cleanup_orphans(
        self,
        references: dict,
        *,
        grace_seconds: float,
        now: float | None = None,
    ) -> dict:
        reference_uploads = {
            str(Path(value).resolve())
            for value in references.get("upload_paths", set())
        }
        reference_clips = {
            Path(value).name
            for value in references.get("clip_filenames", set())
        }
        threshold = (time.time() if now is None else now) - max(
            float(grace_seconds),
            0.0,
        )
        removed_files = 0
        removed_bytes = 0
        for directory, is_referenced in (
            (
                self.upload_dir,
                lambda path: str(path.resolve()) in reference_uploads,
            ),
            (
                self.clip_dir,
                lambda path: path.name in reference_clips,
            ),
        ):
            for candidate in directory.rglob("*"):
                safe_candidate = self._safe_file(directory, candidate)
                try:
                    should_keep = (
                        safe_candidate is None
                        or not safe_candidate.is_file()
                        or is_referenced(safe_candidate)
                        or safe_candidate.stat().st_mtime > threshold
                    )
                except OSError:
                    continue
                if should_keep:
                    continue
                try:
                    size = safe_candidate.stat().st_size
                    safe_candidate.unlink()
                except OSError:
                    continue
                removed_files += 1
                removed_bytes += size
        return {
            "removed_files": removed_files,
            "removed_bytes": removed_bytes,
        }


def run_storage_maintenance(
    store,
    storage: RuntimeStorage,
    *,
    retention_days: int,
    orphan_grace_hours: float,
) -> dict:
    expired_removed = 0
    artifact_files = 0
    artifact_bytes = 0
    if retention_days >= 0:
        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(days=retention_days)
        ).isoformat(timespec="seconds")
        while expired_jobs := store.list_expired_terminal_jobs(cutoff):
            for job in expired_jobs:
                deleted = store.delete_job(job["id"])
                if not deleted:
                    continue
                removal = storage.remove_job_artifacts(deleted)
                expired_removed += 1
                artifact_files += removal["removed_files"]
                artifact_bytes += removal["removed_bytes"]

    if orphan_grace_hours >= 0:
        orphan_cleanup = storage.cleanup_orphans(
            store.artifact_references(),
            grace_seconds=orphan_grace_hours * 3600,
        )
    else:
        orphan_cleanup = {"removed_files": 0, "removed_bytes": 0}
    return {
        "expired_jobs_removed": expired_removed,
        "artifact_files_removed": artifact_files,
        "artifact_bytes_removed": artifact_bytes,
        "orphan_files_removed": orphan_cleanup["removed_files"],
        "orphan_bytes_removed": orphan_cleanup["removed_bytes"],
    }
