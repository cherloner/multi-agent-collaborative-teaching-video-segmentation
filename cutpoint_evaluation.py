from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from typing import Iterable

from audio_quality import (
    measure_edge_fades,
    measure_integrated_loudness,
)
from cutpoint_pipeline import (
    analyze_video,
    probe_video,
    propose_clips,
    render_clip,
)
from feedback_calibration import FEATURE_NAMES, calibrate_feedback
from render_profiles import resolve_export_profile
from speaker_signals import SpeakerTurn
from visual_signals import frame_similarity


Interval = tuple[float, float]


def boundary_metrics(
    predicted: list[float],
    expected: list[float],
    tolerance: float = 0.75,
) -> dict:
    unmatched = set(range(len(expected)))
    errors: list[float] = []
    for value in sorted(predicted):
        candidates = [
            (abs(value - expected[index]), index)
            for index in unmatched
            if abs(value - expected[index]) <= tolerance
        ]
        if not candidates:
            continue
        error, matched_index = min(candidates)
        unmatched.remove(matched_index)
        errors.append(error)

    matches = len(errors)
    precision = matches / len(predicted) if predicted else float(not expected)
    recall = matches / len(expected) if expected else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matches": matches,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_error_seconds": round(mean(errors), 4) if errors else None,
    }


def interval_iou(left: Interval, right: Interval) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def _merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[list[float]] = []
    for start, end in sorted((float(start), float(end)) for start, end in intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def timeline_iou(predicted: list[Interval], expected: list[Interval]) -> float:
    left = _merge_intervals(predicted)
    right = _merge_intervals(expected)
    left_duration = sum(end - start for start, end in left)
    right_duration = sum(end - start for start, end in right)
    intersection = 0.0
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        intersection += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    union = left_duration + right_duration - intersection
    return round(intersection / union, 4) if union > 0 else 1.0


def highlight_metrics(
    clips: list[dict],
    expected: list[Interval],
    minimum_iou: float = 0.45,
) -> dict:
    predicted = [(float(clip["start"]), float(clip["end"])) for clip in clips]
    best_for_expected = [
        max((interval_iou(reference, candidate) for candidate in predicted), default=0.0)
        for reference in expected
    ]
    best_for_predicted = [
        max((interval_iou(candidate, reference) for reference in expected), default=0.0)
        for candidate in predicted
    ]
    hits = sum(value >= minimum_iou for value in best_for_expected)
    useful = sum(value >= minimum_iou for value in best_for_predicted)
    return {
        "expected_hits": hits,
        "expected_count": len(expected),
        "recall": round(hits / len(expected), 4) if expected else 1.0,
        "precision": round(useful / len(predicted), 4) if predicted else float(not expected),
        "mean_best_iou": round(mean(best_for_expected), 4) if best_for_expected else 1.0,
        "best_ious": [round(value, 4) for value in best_for_expected],
    }


def clip_quality_metrics(
    clips: list[dict],
    target_duration: int,
    source_duration: float | None = None,
) -> dict:
    if not clips:
        return {
            "speech_ratio": 0.0,
            "clean_frame_ratio": 0.0,
            "semantic_coverage": 0.0,
            "visual_uniqueness": 0.0,
            "duration_compliance": 0.0,
            "warning_rate": 1.0,
        }
    visual_values = [
        clip["quality"].get("visual_uniqueness")
        for clip in clips
        if clip["quality"].get("visual_uniqueness") is not None
    ]
    return {
        "speech_ratio": round(
            mean(clip["quality"]["speech_ratio"] for clip in clips),
            4,
        ),
        "clean_frame_ratio": round(
            mean(clip["quality"]["clean_frame_ratio"] for clip in clips),
            4,
        ),
        "semantic_coverage": round(
            mean(clip["quality"].get("semantic_coverage", 0.0) for clip in clips),
            4,
        ),
        "visual_uniqueness": round(
            mean(visual_values),
            4,
        )
        if visual_values
        else 0.0,
        "duration_compliance": round(
            mean(
                abs(float(clip["duration"]) - target_duration) <= 0.25
                or (
                    source_duration is not None
                    and source_duration < target_duration
                    and abs(float(clip["duration"]) - source_duration) <= 0.25
                )
                for clip in clips
            ),
            4,
        ),
        "warning_rate": round(
            mean(bool(clip["quality"]["warnings"]) for clip in clips),
            4,
        ),
    }


def build_deterministic_fixture(
    output_path: Path,
    include_subtitles: bool = True,
) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("生成评测素材需要 FFmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subtitle_path = output_path.with_suffix(".srt") if include_subtitles else None
    if subtitle_path:
        subtitle_path.write_text(
            """1
00:00:01,000 --> 00:00:08,000
第一段核心结论：增长达到 42%，关键原因是留存改善。

2
00:00:10,000 --> 00:00:18,000
方法与数据都支持这个判断。

3
00:00:25,500 --> 00:00:32,000
第二段关键发现：场景切换后有效信息仍然连续。

4
00:00:34,000 --> 00:00:42,000
最终建议是保留完整语句并避开静音边界。
""",
            encoding="utf-8",
        )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-t",
        "20",
        "-i",
        "color=c=0x8b1e3f:s=320x180:r=24",
        "-f",
        "lavfi",
        "-t",
        "20",
        "-i",
        "sine=frequency=440:sample_rate=44100",
        "-f",
        "lavfi",
        "-t",
        "5",
        "-i",
        "color=c=0x163a5f:s=320x180:r=24",
        "-f",
        "lavfi",
        "-t",
        "5",
        "-i",
        "anullsrc=r=44100:cl=mono",
        "-f",
        "lavfi",
        "-t",
        "20",
        "-i",
        "color=c=0x276749:s=320x180:r=24",
        "-f",
        "lavfi",
        "-t",
        "20",
        "-i",
        "sine=frequency=880:sample_rate=44100",
        "-f",
        "lavfi",
        "-t",
        "5",
        "-i",
        "color=c=black:s=320x180:r=24",
        "-f",
        "lavfi",
        "-t",
        "5",
        "-i",
        "anullsrc=r=44100:cl=mono",
        *(["-i", str(subtitle_path)] if subtitle_path else []),
        "-filter_complex",
        "[0:v][1:a][2:v][3:a][4:v][5:a][6:v][7:a]"
        "concat=n=4:v=1:a=1[v][a]",
        "-map",
        "[v]",
        "-map",
        "[a]",
        *(["-map", "8:0"] if subtitle_path else []),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        *(["-c:s", "mov_text"] if subtitle_path else []),
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, timeout=120)
    finally:
        if subtitle_path:
            subtitle_path.unlink(missing_ok=True)


def _ranges_from_payload(values: list[dict]) -> list[Interval]:
    return [(float(item["start"]), float(item["end"])) for item in values]


def evaluate_case(case: dict, manifest_dir: Path) -> dict:
    video_path = (manifest_dir / case["video"]).resolve()
    fixture_missing = not video_path.exists() or video_path.stat().st_size == 0
    if fixture_missing and case.get("fixture") in {
        "deterministic-v1",
        "deterministic-v2",
    }:
        build_deterministic_fixture(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"评测视频不存在: {video_path}")

    target_duration = int(case.get("target_duration", 30))
    analysis = analyze_video(video_path, target_duration)
    expected = case["expected"]
    signals = analysis["signals"]
    metrics = {
        "scene_boundaries": boundary_metrics(
            signals["scene_times"],
            [float(value) for value in expected.get("scene_times", [])],
            float(case.get("boundary_tolerance", 0.75)),
        ),
        "silence_iou": timeline_iou(
            _ranges_from_payload(signals["silence_timeline"]),
            [tuple(value) for value in expected.get("silence_ranges", [])],
        ),
        "black_iou": timeline_iou(
            _ranges_from_payload(signals["black_timeline"]),
            [tuple(value) for value in expected.get("black_ranges", [])],
        ),
        "semantic_iou": timeline_iou(
            [
                (float(item["start"]), float(item["end"]))
                for item in signals["semantic_timeline"]
            ],
            [tuple(value) for value in expected.get("semantic_ranges", [])],
        ),
        "highlights": highlight_metrics(
            analysis["clips"],
            [tuple(value) for value in expected.get("highlights", [])],
            float(case.get("minimum_highlight_iou", 0.45)),
        ),
        "clip_quality": clip_quality_metrics(
            analysis["clips"],
            target_duration,
            analysis["metadata"]["duration"],
        ),
        "analysis_realtime_factor": float(
            analysis.get("performance", {}).get(
                "analysis_realtime_factor",
                0.0,
            )
            or 0.0
        ),
    }
    return {
        "id": case["id"],
        "video": str(video_path),
        "metrics": metrics,
        "observed": {
            "signals": signals,
            "clips": analysis["clips"],
            "performance": analysis.get("performance", {}),
        },
    }


def evaluate_feedback_experiment(experiment: dict | None) -> dict | None:
    if not experiment:
        return None
    repeats = max(int(experiment.get("repeats", 8)), 1)
    examples = []
    for index in range(repeats):
        for rating in ("keep", "reject"):
            examples.append(
                {
                    "job_id": f"feedback-fixture-{index % 4}",
                    "clip_filename": f"{rating}-{index}.mp4",
                    "rating": rating,
                    "features": experiment[rating],
                    "available_features": list(FEATURE_NAMES),
                }
            )
    profile = calibrate_feedback(examples)
    baseline = profile.get("baseline") or {}
    candidate = profile.get("candidate") or {}
    return {
        "status": profile["status"],
        "sample_count": profile["sample_count"],
        "baseline_pairwise_accuracy": baseline.get("pairwise_accuracy", 0.0),
        "candidate_pairwise_accuracy": candidate.get(
            "pairwise_accuracy",
            0.0,
        ),
        "pairwise_accuracy_gain": candidate.get("accuracy_gain", 0.0),
        "weights": profile["weights"],
    }


def evaluate_render_experiment(
    experiment: dict | None,
    manifest_dir: Path,
) -> dict | None:
    if not experiment:
        return None
    video_path = (manifest_dir / experiment["video"]).resolve()
    if not video_path.exists() or video_path.stat().st_size == 0:
        if experiment.get("fixture") in {"deterministic-v1", "deterministic-v2"}:
            build_deterministic_fixture(video_path)
        else:
            raise FileNotFoundError(video_path)
    output_path = (manifest_dir / experiment["output"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = probe_video(video_path)
    profile = resolve_export_profile(
        experiment.get("output_format", "landscape"),
        metadata["width"],
        metadata["height"],
    )
    duration = float(experiment.get("duration", 5.0))
    details = render_clip(
        video_path,
        output_path,
        float(experiment.get("start", 0.0)),
        duration,
        profile,
        captions_enabled=False,
        has_audio=metadata["has_audio"],
    )
    delivery = details["delivery_check"]
    loudness = (
        measure_integrated_loudness(output_path)
        if metadata["has_audio"]
        else None
    )
    fades = (
        measure_edge_fades(output_path)
        if metadata["has_audio"]
        else None
    )
    duration_error = delivery.get("duration_error_seconds") or 0.0
    sync_delta = delivery.get("av_sync_delta_seconds") or 0.0
    loudness_error = (loudness or {}).get("target_error_lu") or 0.0
    return {
        "output": str(output_path),
        "delivery": delivery,
        "loudness": loudness,
        "fades": fades,
        "scores": {
            "render_delivery_pass": float(delivery["passed"]),
            "render_duration_accuracy": round(
                max(0.0, 1.0 - duration_error / 0.25),
                4,
            ),
            "av_sync_accuracy": round(
                max(0.0, 1.0 - sync_delta / 0.12),
                4,
            ),
            "audio_loudness_accuracy": round(
                max(0.0, 1.0 - loudness_error),
                4,
            ),
            "audio_fade_effectiveness": (
                fades["effectiveness"] if fades else 1.0
            ),
        },
    }


def evaluate_speaker_experiment(
    experiment: dict | None,
) -> dict | None:
    if not experiment:
        return None
    turns = [
        SpeakerTurn(
            float(item[0]),
            float(item[1]),
            str(item[2]),
        )
        for item in experiment["turns"]
    ]
    clips = propose_clips(
        float(experiment["duration"]),
        int(experiment["target_duration"]),
        [],
        [],
        limit=int(experiment.get("limit", 1)),
        speaker_turns=turns,
    )
    expected = tuple(float(value) for value in experiment["expected_clip"])
    selected = (
        (float(clips[0]["start"]), float(clips[0]["end"]))
        if clips
        else None
    )
    boundary_scores = [
        clip["quality"].get("speaker_boundary_score")
        for clip in clips
        if clip["quality"].get("speaker_boundary_score") is not None
    ]
    return {
        "selected_clip": selected,
        "expected_clip": expected,
        "speaker_boundary_protection": round(
            mean(boundary_scores),
            4,
        )
        if boundary_scores
        else 0.0,
        "speaker_expected_cut_accuracy": float(selected == expected),
    }


def evaluate_visual_robustness_experiment(
    experiment: dict | None,
) -> dict | None:
    if not experiment:
        return None
    size = 12
    original_values = []
    for y in range(size):
        for x in range(size):
            original_values.extend(
                (
                    min(
                        255,
                        30
                        + x * 12
                        + (80 if 3 <= x <= 8 and 2 <= y <= 9 else 0),
                    ),
                    min(
                        255,
                        25
                        + y * 14
                        + (
                            60
                            if (x - 4) ** 2 + (y - 7) ** 2 < 10
                            else 0
                        ),
                    ),
                    min(255, 35 + ((x * 3 + y * 5) % 7) * 18),
                )
            )
    original = bytes(original_values)
    scales = tuple(
        float(value)
        for value in experiment.get("channel_scales", [0.82, 0.94, 1.08])
    )
    offsets = tuple(
        float(value)
        for value in experiment.get("channel_offsets", [24, 10, -4])
    )
    crop_margin = min(
        max(int(experiment.get("crop_margin", 1)), 0),
        2,
    )
    crop_size = size - crop_margin * 2
    transformed_values = []
    for output_y in range(size):
        source_y = crop_margin + round(
            output_y * (crop_size - 1) / (size - 1)
        )
        for output_x in range(size):
            source_x = crop_margin + round(
                output_x * (crop_size - 1) / (size - 1)
            )
            source_offset = (source_y * size + source_x) * 3
            for channel, value in enumerate(
                original[source_offset : source_offset + 3]
            ):
                transformed_values.append(
                    max(
                        0,
                        min(
                            255,
                            round(
                                value * scales[channel]
                                + offsets[channel]
                            ),
                        ),
                    )
                )
    transformed = bytes(transformed_values)
    unrelated = bytes(
        (x * 19 + y * 7 + channel * 83) % 256
        for y in range(size)
        for x in range(size)
        for channel in range(3)
    )
    duplicate_similarity = frame_similarity(original, transformed)
    unrelated_similarity = frame_similarity(original, unrelated)
    return {
        "crop_margin": crop_margin,
        "channel_scales": scales,
        "channel_offsets": offsets,
        "duplicate_similarity": round(duplicate_similarity, 4),
        "unrelated_similarity": round(unrelated_similarity, 4),
        "scores": {
            "perceptual_duplicate_similarity": round(
                duplicate_similarity,
                4,
            ),
            "perceptual_similarity_margin": round(
                duplicate_similarity - unrelated_similarity,
                4,
            ),
        },
    }


def run_evaluation(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = [
        evaluate_case(case, manifest_path.parent)
        for case in manifest.get("cases", [])
    ]
    if not cases:
        raise ValueError("评测清单中没有 case")

    feedback_experiment = evaluate_feedback_experiment(
        manifest.get("feedback_experiment")
    )
    render_experiment = evaluate_render_experiment(
        manifest.get("render_experiment"),
        manifest_path.parent,
    )
    speaker_experiment = evaluate_speaker_experiment(
        manifest.get("speaker_experiment")
    )
    visual_robustness_experiment = (
        evaluate_visual_robustness_experiment(
            manifest.get("visual_robustness_experiment")
        )
    )
    aggregate = {
        "scene_f1": round(mean(case["metrics"]["scene_boundaries"]["f1"] for case in cases), 4),
        "silence_iou": round(mean(case["metrics"]["silence_iou"] for case in cases), 4),
        "black_iou": round(mean(case["metrics"]["black_iou"] for case in cases), 4),
        "semantic_iou": round(
            mean(case["metrics"]["semantic_iou"] for case in cases),
            4,
        ),
        "highlight_recall": round(mean(case["metrics"]["highlights"]["recall"] for case in cases), 4),
        "highlight_mean_iou": round(
            mean(case["metrics"]["highlights"]["mean_best_iou"] for case in cases),
            4,
        ),
        "speech_ratio": round(
            mean(case["metrics"]["clip_quality"]["speech_ratio"] for case in cases),
            4,
        ),
        "clean_frame_ratio": round(
            mean(case["metrics"]["clip_quality"]["clean_frame_ratio"] for case in cases),
            4,
        ),
        "semantic_coverage": round(
            mean(case["metrics"]["clip_quality"]["semantic_coverage"] for case in cases),
            4,
        ),
        "visual_uniqueness": round(
            mean(
                case["metrics"]["clip_quality"]["visual_uniqueness"]
                for case in cases
            ),
            4,
        ),
        "analysis_realtime_factor": round(
            mean(
                case["metrics"]["analysis_realtime_factor"]
                for case in cases
            ),
            4,
        ),
    }
    if feedback_experiment:
        aggregate.update(
            {
                "feedback_pairwise_accuracy": round(
                    feedback_experiment["candidate_pairwise_accuracy"],
                    4,
                ),
                "feedback_pairwise_accuracy_gain": round(
                    feedback_experiment["pairwise_accuracy_gain"],
                    4,
                ),
            }
        )
    if render_experiment:
        aggregate.update(render_experiment["scores"])
    if speaker_experiment:
        aggregate.update(
            {
                "speaker_boundary_protection": speaker_experiment[
                    "speaker_boundary_protection"
                ],
                "speaker_expected_cut_accuracy": speaker_experiment[
                    "speaker_expected_cut_accuracy"
                ],
            }
        )
    if visual_robustness_experiment:
        aggregate.update(visual_robustness_experiment["scores"])
    gates = manifest.get("gates", {})
    gate_results = {
        name: {
            "actual": aggregate.get(name),
            "minimum": minimum,
            "passed": aggregate.get(name) is not None and aggregate[name] >= minimum,
        }
        for name, minimum in gates.items()
    }
    return {
        "version": manifest.get("version", 1),
        "case_count": len(cases),
        "aggregate": aggregate,
        "gates": gate_results,
        "passed": all(result["passed"] for result in gate_results.values()),
        "feedback_experiment": feedback_experiment,
        "render_experiment": render_experiment,
        "speaker_experiment": speaker_experiment,
        "visual_robustness_experiment": visual_robustness_experiment,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 CUTPOINT 可重复质量评测")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluation/manifest.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runtime/evaluation/report.json"),
    )
    parser.add_argument("--strict", action="store_true", help="质量门槛失败时返回非零状态")
    args = parser.parse_args()

    report = run_evaluation(args.manifest.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    print(f"quality gates: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"report: {args.report.resolve()}")
    return 1 if args.strict and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
