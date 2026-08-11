from __future__ import annotations

import os
import signal
import subprocess
import threading
from contextlib import contextmanager
from time import monotonic
from typing import Callable, Iterator, Sequence


POLL_INTERVAL_SECONDS = 0.1
_local = threading.local()


@contextmanager
def controlled_processes(check: Callable[[], None]) -> Iterator[None]:
    previous = getattr(_local, "check", None)
    _local.check = check
    try:
        yield
    finally:
        if previous is None:
            try:
                delattr(_local, "check")
            except AttributeError:
                pass
        else:
            _local.check = previous


def _check_control() -> None:
    check = getattr(_local, "check", None)
    if check is not None:
        check()


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
                creationflags=getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0,
                ),
            )
            process.wait(timeout=0.5)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=0.5)
            return
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except OSError:
                pass
    try:
        process.terminate()
        process.wait(timeout=0.5)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.kill()
        process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pass


def run_process(
    command: Sequence[str | os.PathLike[str]],
    *,
    capture_output: bool = False,
    text: bool = False,
    encoding: str | None = None,
    errors: str | None = None,
    timeout: float | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    _check_control()
    normalized_command = [str(value) for value in command]
    stdout_target = subprocess.PIPE if capture_output else None
    stderr_target = subprocess.PIPE if capture_output else None
    popen_options = {
        "stdout": stdout_target,
        "stderr": stderr_target,
        "text": text,
    }
    if text:
        popen_options["encoding"] = encoding
        popen_options["errors"] = errors
    if os.name == "nt":
        popen_options["creationflags"] = getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0,
        )
    else:
        popen_options["start_new_session"] = True

    process = subprocess.Popen(normalized_command, **popen_options)
    started = monotonic()
    stdout = stderr = None
    try:
        while True:
            _check_control()
            remaining = (
                None
                if timeout is None
                else timeout - (monotonic() - started)
            )
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(
                    normalized_command,
                    timeout,
                )
            wait_slice = min(
                POLL_INTERVAL_SECONDS,
                remaining,
            ) if remaining is not None else POLL_INTERVAL_SECONDS
            try:
                stdout, stderr = process.communicate(timeout=wait_slice)
                break
            except subprocess.TimeoutExpired:
                continue
        _check_control()
    except BaseException:
        _terminate(process)
        try:
            stdout, stderr = process.communicate(timeout=0.5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        raise

    completed = subprocess.CompletedProcess(
        normalized_command,
        process.returncode,
        stdout,
        stderr,
    )
    if check and completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode,
            normalized_command,
            output=stdout,
            stderr=stderr,
        )
    return completed
