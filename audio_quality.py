from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

from process_control import run_process


AUDIO_TARGET_LUFS = -16.0
AUDIO_TARGET_LRA = 11.0
AUDIO_TRUE_PEAK_DB = -1.5
AUDIO_SAMPLE_RATE = 48000
AUDIO_FADE_IN_SECONDS = 0.08
AUDIO_FADE_OUT_SECONDS = 0.12


def build_audio_filter(duration: float) -> str:
    fade_out_start = max(0.0, float(duration) - AUDIO_FADE_OUT_SECONDS)
    return (
        "[0:a:0]"
        f"loudnorm=I={AUDIO_TARGET_LUFS:g}:"
        f"LRA={AUDIO_TARGET_LRA:g}:TP={AUDIO_TRUE_PEAK_DB:g},"
        f"aresample={AUDIO_SAMPLE_RATE}:async=1:first_pts=0,"
        f"afade=t=in:st=0:d={AUDIO_FADE_IN_SECONDS:g},"
        f"afade=t=out:st={fade_out_start:.3f}:"
        f"d={AUDIO_FADE_OUT_SECONDS:g}"
        "[aout]"
    )


def audio_processing_summary(has_audio: bool) -> dict:
    if not has_audio:
        return {
            "status": "not_present",
            "normalized": False,
        }
    return {
        "status": "processed",
        "normalized": True,
        "target_lufs": AUDIO_TARGET_LUFS,
        "true_peak_db": AUDIO_TRUE_PEAK_DB,
        "sample_rate": AUDIO_SAMPLE_RATE,
        "fade_in_seconds": AUDIO_FADE_IN_SECONDS,
        "fade_out_seconds": AUDIO_FADE_OUT_SECONDS,
    }


def _float_value(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp_duration(value: object) -> float | None:
    if not isinstance(value, str) or ":" not in value:
        return None
    try:
        hours, minutes, seconds = value.split(":", 2)
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (TypeError, ValueError):
        return None


def _stream_duration(stream: dict) -> float | None:
    return _float_value(stream.get("duration")) or _timestamp_duration(
        stream.get("tags", {}).get("DURATION")
    )


def inspect_rendered_media(
    media_path: Path,
    *,
    expected_duration: float,
    expected_width: int,
    expected_height: int,
    expects_audio: bool,
    duration_tolerance: float = 0.25,
    sync_tolerance: float = 0.12,
) -> dict:
    result = run_process(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=60,
    )
    payload = json.loads(result.stdout)
    video = next(
        (
            stream
            for stream in payload.get("streams", [])
            if stream.get("codec_type") == "video"
        ),
        None,
    )
    audio = next(
        (
            stream
            for stream in payload.get("streams", [])
            if stream.get("codec_type") == "audio"
        ),
        None,
    )
    format_duration = _float_value(payload.get("format", {}).get("duration"))
    video_duration = _stream_duration(video or {}) or format_duration
    audio_duration = _stream_duration(audio or {}) or (
        format_duration if audio else None
    )
    actual_duration = format_duration or video_duration
    duration_error = (
        abs(actual_duration - expected_duration)
        if actual_duration is not None
        else None
    )
    av_sync_delta = (
        abs(video_duration - audio_duration)
        if video_duration is not None and audio_duration is not None
        else None
    )
    video_start = _float_value((video or {}).get("start_time")) or 0.0
    audio_start = (
        _float_value((audio or {}).get("start_time"))
        if audio
        else None
    )
    start_delta = (
        abs(video_start - audio_start)
        if audio_start is not None
        else None
    )

    checks = {
        "video_present": video is not None,
        "dimensions": bool(
            video
            and int(video.get("width", 0)) == expected_width
            and int(video.get("height", 0)) == expected_height
        ),
        "duration": (
            duration_error is not None
            and duration_error <= duration_tolerance
        ),
        "audio_presence": (audio is not None) == bool(expects_audio),
        "av_duration_sync": (
            not expects_audio
            or (
                av_sync_delta is not None
                and av_sync_delta <= sync_tolerance
            )
        ),
        "av_start_sync": (
            not expects_audio
            or (
                start_delta is not None
                and start_delta <= sync_tolerance
            )
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "duration_seconds": (
            round(actual_duration, 4)
            if actual_duration is not None
            else None
        ),
        "duration_error_seconds": (
            round(duration_error, 4)
            if duration_error is not None
            else None
        ),
        "av_sync_delta_seconds": (
            round(av_sync_delta, 4)
            if av_sync_delta is not None
            else None
        ),
        "av_start_delta_seconds": (
            round(start_delta, 4)
            if start_delta is not None
            else None
        ),
        "audio_sample_rate": (
            int(audio["sample_rate"])
            if audio and audio.get("sample_rate")
            else None
        ),
    }


def measure_integrated_loudness(media_path: Path) -> dict:
    result = run_process(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(media_path),
            "-vn",
            "-af",
            (
                f"loudnorm=I={AUDIO_TARGET_LUFS:g}:"
                f"LRA={AUDIO_TARGET_LRA:g}:TP={AUDIO_TRUE_PEAK_DB:g}:"
                "print_format=json"
            ),
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    matches = re.findall(
        r"\{\s*\"input_i\".*?\}",
        result.stderr,
        flags=re.DOTALL,
    )
    if not matches:
        raise RuntimeError("FFmpeg 未返回响度测量结果")
    payload = json.loads(matches[-1])
    integrated = _float_value(payload.get("input_i"))
    true_peak = _float_value(payload.get("input_tp"))
    return {
        "integrated_lufs": integrated,
        "true_peak_db": true_peak,
        "target_error_lu": (
            round(abs(integrated - AUDIO_TARGET_LUFS), 4)
            if integrated is not None
            else None
        ),
    }


def _rms(samples: list[int]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def measure_edge_fades(
    media_path: Path,
    *,
    sample_rate: int = 8000,
) -> dict:
    result = run_process(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=180,
    )
    samples = [
        value[0]
        for value in struct.iter_unpack("<h", result.stdout)
    ]
    edge_size = max(round(sample_rate * 0.03), 1)
    body_start = min(round(sample_rate * 0.25), len(samples))
    body_end = min(round(sample_rate * 0.75), len(samples))
    body = samples[body_start:body_end]
    body_rms = _rms(body)
    start_ratio = _rms(samples[:edge_size]) / body_rms if body_rms else 0.0
    end_ratio = _rms(samples[-edge_size:]) / body_rms if body_rms else 0.0
    return {
        "start_rms_ratio": round(start_ratio, 4),
        "end_rms_ratio": round(end_ratio, 4),
        "effectiveness": round(
            max(0.0, 1.0 - max(start_ratio, end_ratio)),
            4,
        ),
    }
