from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

from audio_quality import (
    audio_processing_summary,
    build_audio_filter,
    inspect_rendered_media,
)
from feedback_calibration import (
    FEATURE_NAMES,
    public_scoring_profile,
    score_features,
)
from process_control import run_process
from render_profiles import (
    ExportProfile,
    build_ass_subtitles,
    build_video_filter,
    resolve_export_profile,
)
from semantic_signals import (
    TimedText,
    extract_semantic_signals,
    semantic_window_metrics,
)
from speaker_signals import (
    SpeakerTurn,
    extract_speaker_signals,
    speaker_window_metrics,
)
from visual_signals import (
    VisualSample,
    extract_visual_samples,
    signature_similarity,
    window_signature,
)


ProgressCallback = Callable[[int, str], None]

SCENE_RE = re.compile(r"pts_time:([0-9.]+)")
SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")
BLACK_START_RE = re.compile(r"black_start:([0-9.]+)")
BLACK_END_RE = re.compile(r"black_end:([0-9.]+)")
VISUAL_SIMILARITY_THRESHOLD = 0.94


def _timed_call(timings: dict[str, float], name: str, callback):
    started = perf_counter()
    try:
        return callback()
    finally:
        timings[name] = round((perf_counter() - started) * 1000, 2)


@dataclass(frozen=True)
class SilenceRange:
    start: float
    end: float


@dataclass(frozen=True)
class BlackRange:
    start: float
    end: float


def _run(command: list[str], timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    return run_process(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=True,
    )


def probe_video(video_path: Path) -> dict:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(video_path),
        ]
    )
    payload = json.loads(result.stdout)
    video_stream = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if not video_stream:
        raise ValueError("文件中没有可识别的视频轨道")

    audio_stream = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "audio"),
        None,
    )
    duration = float(
        payload.get("format", {}).get("duration")
        or video_stream.get("duration")
        or 0
    )
    if duration <= 0:
        raise ValueError("无法读取视频时长")

    fps_raw = video_stream.get("avg_frame_rate", "0/1")
    numerator, _, denominator = fps_raw.partition("/")
    fps = float(numerator) / max(float(denominator or 1), 1)

    return {
        "duration": round(duration, 3),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "fps": round(fps, 2),
        "video_codec": video_stream.get("codec_name") or "unknown",
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "has_audio": audio_stream is not None,
    }


def detect_scene_changes(video_path: Path) -> list[float]:
    result = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(video_path),
            "-vf",
            "select='gt(scene,0.12)',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    return sorted({round(float(value), 3) for value in SCENE_RE.findall(result.stderr)})


def detect_silences(video_path: Path, duration: float) -> list[SilenceRange]:
    result = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(video_path),
            "-af",
            "silencedetect=n=-35dB:d=0.45",
            "-vn",
            "-f",
            "null",
            "-",
        ]
    )
    starts = [float(value) for value in SILENCE_START_RE.findall(result.stderr)]
    ends = [float(value) for value in SILENCE_END_RE.findall(result.stderr)]
    ranges: list[SilenceRange] = []
    for index, start in enumerate(starts):
        end = ends[index] if index < len(ends) else duration
        if end > start:
            ranges.append(SilenceRange(round(start, 3), round(min(end, duration), 3)))
    return ranges


def detect_black_ranges(video_path: Path, duration: float) -> list[BlackRange]:
    result = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(video_path),
            "-vf",
            "blackdetect=d=0.3:pix_th=0.10",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    starts = [float(value) for value in BLACK_START_RE.findall(result.stderr)]
    ends = [float(value) for value in BLACK_END_RE.findall(result.stderr)]
    ranges: list[BlackRange] = []
    for index, start in enumerate(starts):
        end = ends[index] if index < len(ends) else duration
        if end > start:
            ranges.append(BlackRange(round(start, 3), round(min(end, duration), 3)))
    return ranges


def range_overlap(
    start: float,
    end: float,
    ranges: Sequence[SilenceRange | BlackRange],
) -> float:
    overlap = 0.0
    for item in ranges:
        overlap += max(0.0, min(end, item.end) - max(start, item.start))
    return overlap


def select_diverse_candidates(
    candidates: list[dict],
    target_duration: float,
    limit: int,
    visual_samples: list[VisualSample] | None = None,
    visual_similarity_threshold: float = VISUAL_SIMILARITY_THRESHOLD,
    diagnostics: dict | None = None,
    scoring_profile: dict | None = None,
) -> list[dict]:
    samples = visual_samples or []
    selected: list[dict] = []
    selected_signatures: list[tuple[bytes, ...]] = []
    temporal_filtered = 0
    visual_filtered = 0
    considered_ids: set[int] = set()
    remaining = list(candidates)
    signature_cache: dict[int, tuple[bytes, ...]] = {}

    def visual_adjusted_score(
        candidate: dict,
        signature: tuple[bytes, ...],
        uniqueness: float | None,
    ) -> tuple[float, dict | None, list[str]]:
        features = candidate.get("features")
        available_features = list(
            candidate.get("available_features", [])
        )
        adjusted_score = float(candidate["score"])
        if (
            signature
            and uniqueness is not None
            and isinstance(features, dict)
            and available_features
        ):
            features = {
                **features,
                "visual_uniqueness": uniqueness,
            }
            if "visual_uniqueness" not in available_features:
                available_features.append("visual_uniqueness")
            adjusted_score = score_features(
                features,
                scoring_profile,
                available_features,
            )
        return adjusted_score, features, available_features

    while remaining and len(selected) < limit:
        ranked = []
        for candidate in remaining:
            overlaps_existing = any(
                max(
                    0.0,
                    min(candidate["end"], chosen["end"])
                    - max(candidate["start"], chosen["start"]),
                )
                >= target_duration * 0.2
                for chosen in selected
            )
            if overlaps_existing:
                temporal_filtered += 1
                continue
            cache_key = id(candidate)
            signature = signature_cache.get(cache_key)
            if signature is None:
                signature = window_signature(
                    candidate["start"],
                    candidate["end"],
                    samples,
                )
                signature_cache[cache_key] = signature
            upper_score, _, _ = visual_adjusted_score(
                candidate,
                signature,
                1.0 if signature else None,
            )
            ranked.append((upper_score, candidate, signature))
        ranked.sort(
            key=lambda item: (
                item[0],
                float(item[1]["score"]),
                -float(item[1]["start"]),
            ),
            reverse=True,
        )

        viable = []
        deferred = []
        best_score = float("-inf")
        for ranked_index, (
            upper_score,
            candidate,
            signature,
        ) in enumerate(ranked):
            if viable and upper_score < best_score - 1e-12:
                deferred.extend(
                    item[1]
                    for item in ranked[ranked_index:]
                )
                break
            considered_ids.add(id(candidate))
            max_similarity = max(
                (
                    signature_similarity(
                        signature,
                        selected_signature,
                    )
                    for selected_signature in selected_signatures
                ),
                default=0.0,
            )
            if (
                signature
                and selected_signatures
                and max_similarity >= visual_similarity_threshold
            ):
                visual_filtered += 1
                continue

            uniqueness = (
                max(0.0, 1.0 - max_similarity)
                if signature
                else None
            )
            (
                adjusted_score,
                features,
                available_features,
            ) = visual_adjusted_score(
                candidate,
                signature,
                uniqueness,
            )
            viable.append(
                (
                    adjusted_score,
                    candidate,
                    signature,
                    max_similarity,
                    uniqueness,
                    features,
                    available_features,
                )
            )
            best_score = max(best_score, adjusted_score)

        if not viable:
            break
        chosen = max(
            viable,
            key=lambda item: (
                item[0],
                float(item[1]["score"]),
                -float(item[1]["start"]),
            ),
        )
        (
            adjusted_score,
            candidate,
            signature,
            max_similarity,
            uniqueness,
            features,
            available_features,
        ) = chosen
        candidate["score"] = round(adjusted_score, 3)
        if isinstance(features, dict):
            candidate["features"] = {
                name: round(float(value), 4)
                for name, value in features.items()
            }
            candidate["available_features"] = available_features

        if signature:
            candidate["visual"] = {
                "status": "active",
                "sample_count": len(signature),
                "max_similarity": round(max_similarity, 4),
                "uniqueness": round(float(uniqueness), 4),
            }
            candidate["quality"]["visual_uniqueness"] = round(
                float(uniqueness),
                4,
            )
            candidate["evidence"].append(
                "视觉去重基准"
                if not selected_signatures
                else f"与已选画面相似度 {max_similarity:.0%}"
            )
        else:
            candidate["visual"] = {
                "status": "unavailable",
                "sample_count": 0,
                "max_similarity": None,
                "uniqueness": None,
            }
            candidate["quality"]["visual_uniqueness"] = None

        selected.append(candidate)
        selected_signatures.append(signature)
        remaining = [
            item[1]
            for item in viable
            if item[1] is not candidate
        ] + deferred

    if diagnostics is not None:
        diagnostics.update(
            {
                "candidate_count": len(candidates),
                "considered": len(considered_ids),
                "selected": len(selected),
                "temporal_filtered": temporal_filtered,
                "visual_filtered": visual_filtered,
                "visual_similarity_threshold": visual_similarity_threshold,
            }
        )
    return sorted(selected, key=lambda item: item["start"])


def propose_clips(
    duration: float,
    target_duration: int,
    scene_times: list[float],
    silences: list[SilenceRange],
    black_ranges: list[BlackRange] | None = None,
    semantic_spans: list[TimedText] | None = None,
    limit: int = 3,
    scoring_profile: dict | None = None,
    visual_samples: list[VisualSample] | None = None,
    visual_similarity_threshold: float = VISUAL_SIMILARITY_THRESHOLD,
    diagnostics: dict | None = None,
    speaker_turns: list[SpeakerTurn] | None = None,
) -> list[dict]:
    black_ranges = black_ranges or []
    semantic_spans = semantic_spans or []
    speaker_turns = speaker_turns or []
    target = min(max(float(target_duration), 15.0), 60.0, duration)
    if duration <= target:
        silence_ratio = range_overlap(0.0, duration, silences) / max(duration, 1.0)
        black_ratio = range_overlap(0.0, duration, black_ranges) / max(duration, 1.0)
        semantic = semantic_window_metrics(0.0, duration, semantic_spans)
        speaker = speaker_window_metrics(0.0, duration, speaker_turns)
        features = {
            "speech_ratio": max(0.0, 1.0 - silence_ratio),
            "clean_frame_ratio": max(0.0, 1.0 - black_ratio),
            "boundary_alignment": 1.0,
            "scene_density": min(len(scene_times) / 3.0, 1.0),
            "semantic_score": semantic["score"],
            "visual_uniqueness": 0.0,
            "speaker_boundary_score": speaker["boundary_score"],
        }
        available_features = list(FEATURE_NAMES)
        available_features.remove("visual_uniqueness")
        if not semantic_spans:
            available_features.remove("semantic_score")
        if not speaker_turns:
            available_features.remove("speaker_boundary_score")
        score = score_features(features, scoring_profile, available_features)
        evidence = ["完整视频"]
        if semantic["span_count"]:
            evidence.append(f"文本证据 {semantic['span_count']} 条")
        if speaker["turn_count"]:
            evidence.append(
                f"说话人 {speaker['speaker_count']} 位 / "
                f"轮次 {speaker['turn_count']} 段"
            )
        clips = [
            {
                "start": 0.0,
                "end": round(duration, 3),
                "duration": round(duration, 3),
                "score": round(score, 3),
                "evidence": evidence,
                "features": {
                    name: round(value, 4)
                    for name, value in features.items()
                },
                "available_features": available_features,
                "semantic": semantic,
                "speaker": speaker,
                "quality": {
                    "speech_ratio": round(1.0 - silence_ratio, 3),
                    "clean_frame_ratio": round(1.0 - black_ratio, 3),
                    "semantic_coverage": semantic["coverage"],
                    "speaker_boundary_score": (
                        speaker["boundary_score"]
                        if speaker_turns
                        else None
                    ),
                    "warnings": _quality_warnings(silence_ratio, black_ratio),
                },
            }
        ]
        return select_diverse_candidates(
            clips,
            target,
            limit,
            visual_samples,
            visual_similarity_threshold,
            diagnostics,
            scoring_profile,
        )

    boundary_points = [0.0, duration, *scene_times]
    for silence in silences:
        boundary_points.extend((silence.start, silence.end))
    for black_range in black_ranges:
        boundary_points.extend((black_range.start, black_range.end))
    for span in semantic_spans:
        boundary_points.extend((span.start, span.end))
    for turn in speaker_turns:
        boundary_points.extend((turn.start, turn.end))

    starts = {0.0, max(0.0, duration - target)}
    for boundary in boundary_points:
        starts.add(max(0.0, min(boundary, duration - target)))
        starts.add(max(0.0, min(boundary - target, duration - target)))

    regular_step = max(target * 0.8, 12.0)
    starts.update(
        min(duration - target, regular_step * index)
        for index in range(1, math.ceil(duration / regular_step))
    )

    candidates: list[dict] = []
    for start in sorted(starts):
        end = start + target
        scenes_inside = sum(start + 0.5 < scene < end - 0.5 for scene in scene_times)
        silence_ratio = range_overlap(start, end, silences) / max(target, 1.0)
        black_ratio = range_overlap(start, end, black_ranges) / max(target, 1.0)
        speech_ratio = max(0.0, 1.0 - silence_ratio)
        clean_frame_ratio = max(0.0, 1.0 - black_ratio)
        scene_score = min(scenes_inside / 3.0, 1.0)
        start_aligned = any(abs(start - boundary) <= 1.0 for boundary in boundary_points)
        end_aligned = any(abs(end - boundary) <= 1.0 for boundary in boundary_points)
        boundary_score = (float(start_aligned) + float(end_aligned)) / 2.0
        semantic = semantic_window_metrics(start, end, semantic_spans)
        speaker = speaker_window_metrics(start, end, speaker_turns)
        if speaker_turns:
            boundary_score = (
                0.65 * boundary_score
                + 0.35 * speaker["boundary_score"]
            )
        features = {
            "speech_ratio": speech_ratio,
            "clean_frame_ratio": clean_frame_ratio,
            "boundary_alignment": boundary_score,
            "scene_density": scene_score,
            "semantic_score": semantic["score"],
            "visual_uniqueness": 0.0,
            "speaker_boundary_score": speaker["boundary_score"],
        }
        available_features = list(FEATURE_NAMES)
        available_features.remove("visual_uniqueness")
        if not semantic_spans:
            available_features.remove("semantic_score")
        if not speaker_turns:
            available_features.remove("speaker_boundary_score")
        score = score_features(features, scoring_profile, available_features)
        evidence = [f"{scenes_inside} 个场景变化"]
        evidence.append(f"有效声音 {speech_ratio:.0%}")
        evidence.append(f"有效画面 {clean_frame_ratio:.0%}")
        if semantic["span_count"]:
            evidence.append(
                f"文本证据 {semantic['span_count']} 条 / 覆盖 {semantic['coverage']:.0%}"
            )
        if speaker["turn_count"]:
            evidence.append(
                f"说话人 {speaker['speaker_count']} 位 / "
                f"轮次 {speaker['turn_count']} 段"
            )
        candidates.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(target, 3),
                "score": round(score, 3),
                "evidence": evidence,
                "features": {
                    name: round(value, 4)
                    for name, value in features.items()
                },
                "available_features": available_features,
                "semantic": semantic,
                "speaker": speaker,
                "quality": {
                    "speech_ratio": round(speech_ratio, 3),
                    "clean_frame_ratio": round(clean_frame_ratio, 3),
                    "semantic_coverage": semantic["coverage"],
                    "speaker_boundary_score": (
                        speaker["boundary_score"]
                        if speaker_turns
                        else None
                    ),
                    "warnings": _quality_warnings(silence_ratio, black_ratio),
                },
            }
        )

    return select_diverse_candidates(
        candidates,
        target,
        limit,
        visual_samples,
        visual_similarity_threshold,
        diagnostics,
        scoring_profile,
    )


def _quality_warnings(silence_ratio: float, black_ratio: float) -> list[str]:
    warnings = []
    if silence_ratio > 0.35:
        warnings.append("静音占比较高")
    if black_ratio > 0.08:
        warnings.append("包含明显黑帧")
    return warnings


def render_clip(
    video_path: Path,
    output_path: Path,
    start: float,
    duration: float,
    export_profile: ExportProfile,
    semantic_spans: Sequence[object] | None = None,
    captions_enabled: bool = True,
    has_audio: bool = True,
    verify_output: bool = True,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subtitle_path = output_path.with_suffix(".captions.ass")
    caption_count = 0
    caption_warning = None

    def render_command(active_subtitle_path: Path | None) -> list[str]:
        filter_graph = build_video_filter(
            export_profile,
            active_subtitle_path,
        )
        if has_audio:
            filter_graph += f";{build_audio_filter(duration)}"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-filter_complex",
            filter_graph,
            "-map",
            "[vout]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
        ]
        if has_audio:
            command.extend(
                [
                    "-map",
                    "[aout]",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "160k",
                    "-ar",
                    "48000",
                ]
            )
        else:
            command.append("-an")
        command.extend(
            [
                "-avoid_negative_ts",
                "make_zero",
                "-max_muxing_queue_size",
                "1024",
            "-movflags",
            "+faststart",
            str(output_path),
            ]
        )
        return command

    try:
        if captions_enabled and semantic_spans:
            subtitle_text, caption_count = build_ass_subtitles(
                semantic_spans,
                start,
                start + duration,
                export_profile,
            )
            if caption_count:
                subtitle_path.write_text(subtitle_text, encoding="utf-8")
        try:
            _run(
                render_command(
                    subtitle_path if caption_count else None,
                )
            )
        except Exception as caption_error:
            if not caption_count:
                raise
            output_path.unlink(missing_ok=True)
            _run(render_command(None))
            caption_count = 0
            caption_warning = (
                "字幕烧录失败，已保留无字幕成片："
                f"{str(caption_error).strip()[:160]}"
            )
        delivery_check = (
            inspect_rendered_media(
                output_path,
                expected_duration=duration,
                expected_width=export_profile.width,
                expected_height=export_profile.height,
                expects_audio=has_audio,
            )
            if verify_output
            else {"passed": True, "checks": {"verification": True}}
        )
        if not delivery_check["passed"]:
            failed_checks = [
                name
                for name, passed in delivery_check["checks"].items()
                if not passed
            ]
            raise RuntimeError(
                "成片质量检查失败："
                + "、".join(failed_checks)
            )
        return {
            **export_profile.to_dict(),
            "captions_enabled": bool(captions_enabled),
            "caption_count": caption_count,
            "caption_warning": caption_warning,
            "audio": audio_processing_summary(has_audio),
            "delivery_check": delivery_check,
        }
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        subtitle_path.unlink(missing_ok=True)


def analyze_video(
    video_path: Path,
    target_duration: int,
    progress: ProgressCallback | None = None,
    semantic_spans: list[TimedText] | None = None,
    scoring_profile: dict | None = None,
    visual_samples: list[VisualSample] | None = None,
    speaker_turns: list[SpeakerTurn] | None = None,
) -> dict:
    analysis_started = perf_counter()
    stage_timings: dict[str, float] = {}
    report = progress or (lambda _progress, _phase: None)
    report(14, "读取媒体信息")
    metadata = _timed_call(
        stage_timings,
        "probe",
        lambda: probe_video(video_path),
    )

    report(28, "检测场景变化")
    scenes = _timed_call(
        stage_timings,
        "scene_detection",
        lambda: detect_scene_changes(video_path),
    )

    silences: list[SilenceRange] = []
    if metadata["has_audio"]:
        report(42, "分析声音边界")
        silences = _timed_call(
            stage_timings,
            "silence_detection",
            lambda: detect_silences(video_path, metadata["duration"]),
        )
    else:
        stage_timings["silence_detection"] = 0.0

    report(54, "检测黑帧与无效画面")
    black_ranges = _timed_call(
        stage_timings,
        "black_detection",
        lambda: detect_black_ranges(video_path, metadata["duration"]),
    )

    report(60, "提取字幕与语义文本")
    semantic_started = perf_counter()
    if semantic_spans is None:
        semantic_extraction = extract_semantic_signals(
            video_path,
            metadata["duration"],
        )
        semantic_spans = semantic_extraction.spans
        semantic_providers = semantic_extraction.providers
        semantic_warnings = semantic_extraction.warnings
    else:
        semantic_providers = {"injected": "active" if semantic_spans else "empty"}
        semantic_warnings = []
    stage_timings["semantic_extraction"] = round(
        (perf_counter() - semantic_started) * 1000,
        2,
    )

    report(62, "识别说话人轮次与对话边界")
    speaker_started = perf_counter()
    if speaker_turns is None:
        if metadata["has_audio"]:
            speaker_extraction = extract_speaker_signals(
                video_path,
                metadata["duration"],
            )
            speaker_turns = speaker_extraction.turns
            speaker_providers = speaker_extraction.providers
            speaker_warnings = speaker_extraction.warnings
        else:
            speaker_turns = []
            speaker_providers = {"diarization": "no_audio"}
            speaker_warnings = []
    else:
        speaker_providers = {
            "injected": "active" if speaker_turns else "empty"
        }
        speaker_warnings = []
    stage_timings["speaker_diarization"] = round(
        (perf_counter() - speaker_started) * 1000,
        2,
    )

    report(65, "提取视觉指纹并检查重复内容")
    visual_started = perf_counter()
    if visual_samples is None:
        visual_extraction = extract_visual_samples(
            video_path,
            metadata["duration"],
        )
        visual_samples = visual_extraction.samples
        visual_status = visual_extraction.status
        visual_interval = visual_extraction.interval_seconds
        visual_warnings = visual_extraction.warnings
    else:
        visual_status = "active" if visual_samples else "empty"
        visual_interval = None
        visual_warnings = []
    stage_timings["visual_fingerprints"] = round(
        (perf_counter() - visual_started) * 1000,
        2,
    )

    report(68, "融合多模态时间轴证据")
    selection_diagnostics: dict = {}
    clips = _timed_call(
        stage_timings,
        "candidate_ranking",
        lambda: propose_clips(
            metadata["duration"],
            target_duration,
            scenes,
            silences,
            black_ranges,
            semantic_spans,
            scoring_profile=scoring_profile,
            visual_samples=visual_samples,
            diagnostics=selection_diagnostics,
            speaker_turns=speaker_turns,
        ),
    )
    analysis_total_ms = round(
        (perf_counter() - analysis_started) * 1000,
        2,
    )
    realtime_factor = (
        metadata["duration"] / (analysis_total_ms / 1000)
        if analysis_total_ms > 0
        else None
    )
    return {
        "metadata": metadata,
        "performance": {
            "analysis_total_ms": analysis_total_ms,
            "analysis_realtime_factor": (
                round(realtime_factor, 3)
                if realtime_factor is not None
                else None
            ),
            "stages_ms": stage_timings,
        },
        "scoring_profile": public_scoring_profile(scoring_profile),
        "signals": {
            "scene_changes": len(scenes),
            "scene_times": scenes,
            "silence_ranges": len(silences),
            "silence_timeline": [
                {"start": item.start, "end": item.end} for item in silences
            ],
            "black_ranges": len(black_ranges),
            "black_timeline": [
                {"start": item.start, "end": item.end} for item in black_ranges
            ],
            "semantic_spans": len(semantic_spans),
            "semantic_timeline": [span.to_dict() for span in semantic_spans],
            "semantic_providers": semantic_providers,
            "semantic_warnings": semantic_warnings,
            "speaker_turns": len(speaker_turns),
            "speaker_count": len(
                {turn.speaker for turn in speaker_turns}
            ),
            "speaker_timeline": [
                turn.to_dict() for turn in speaker_turns
            ],
            "speaker_providers": speaker_providers,
            "speaker_warnings": speaker_warnings,
            "visual_samples": len(visual_samples),
            "visual_interval_seconds": visual_interval,
            "visual_provider": visual_status,
            "visual_warnings": visual_warnings,
            "selection": selection_diagnostics,
        },
        "clips": clips,
    }


def process_video(
    video_path: Path,
    output_dir: Path,
    job_id: str,
    target_duration: int,
    progress: ProgressCallback,
    scoring_profile: dict | None = None,
    output_format: str = "source",
    captions_enabled: bool = True,
) -> dict:
    processing_started = perf_counter()
    analysis = analyze_video(
        video_path,
        target_duration,
        progress,
        scoring_profile=scoring_profile,
    )
    clips = analysis["clips"]
    export_profile = resolve_export_profile(
        output_format,
        analysis["metadata"]["width"],
        analysis["metadata"]["height"],
    )

    rendered_clips = []
    captions_rendered = 0
    render_warnings = []
    render_timings = []
    for index, clip in enumerate(clips, start=1):
        progress(
            70 + round(index / max(len(clips), 1) * 26),
            f"生成{export_profile.label}短片 {index}/{len(clips)}",
        )
        filename = f"{job_id}-{index:02d}.mp4"
        output_path = output_dir / filename
        render_started = perf_counter()
        render_details = render_clip(
            video_path,
            output_path,
            clip["start"],
            clip["duration"],
            export_profile,
            analysis["signals"]["semantic_timeline"],
            captions_enabled,
            has_audio=analysis["metadata"]["has_audio"],
        )
        render_elapsed_ms = round(
            (perf_counter() - render_started) * 1000,
            2,
        )
        render_timings.append(render_elapsed_ms)
        render_details["elapsed_ms"] = render_elapsed_ms
        captions_rendered += render_details["caption_count"]
        if render_details["caption_warning"]:
            render_warnings.append(render_details["caption_warning"])
        rendered_clips.append(
            {
                **clip,
                "quality": {
                    **clip["quality"],
                    "warnings": [
                        *clip["quality"]["warnings"],
                        *(
                            [render_details["caption_warning"]]
                            if render_details["caption_warning"]
                            else []
                        ),
                    ],
                },
                "filename": filename,
                "url": f"/generated/{filename}",
                "render": render_details,
            }
        )

    progress(98, "完成质量检查")
    total_ms = round((perf_counter() - processing_started) * 1000, 2)
    throughput = (
        analysis["metadata"]["duration"] / (total_ms / 1000)
        if total_ms > 0
        else None
    )
    return {
        "metadata": analysis["metadata"],
        "export_profile": {
            **export_profile.to_dict(),
            "captions_enabled": bool(captions_enabled),
            "captions_rendered": captions_rendered,
            "warnings": render_warnings,
        },
        "scoring_profile": analysis["scoring_profile"],
        "performance": {
            **analysis["performance"],
            "render_total_ms": round(sum(render_timings), 2),
            "render_clips_ms": render_timings,
            "total_ms": total_ms,
            "source_seconds_per_processing_second": (
                round(throughput, 3)
                if throughput is not None
                else None
            ),
        },
        "signals": analysis["signals"],
        "clips": rendered_clips,
    }
