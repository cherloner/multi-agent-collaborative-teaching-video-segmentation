import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from cutpoint_evaluation import build_deterministic_fixture
from cutpoint_pipeline import select_diverse_candidates
from visual_signals import (
    VisualSample,
    extract_visual_samples,
    frame_similarity,
    signature_similarity,
    window_signature,
)


def solid_descriptor(red: int, green: int, blue: int) -> bytes:
    return bytes([red, green, blue] * (12 * 12))


def patterned_descriptor() -> bytes:
    values = []
    for y in range(12):
        for x in range(12):
            values.extend(
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
    return bytes(values)


def crop_and_grade(descriptor: bytes) -> bytes:
    transformed = []
    scales = (0.82, 0.94, 1.08)
    offsets = (24, 10, -4)
    for output_y in range(12):
        source_y = 1 + round(output_y * 9 / 11)
        for output_x in range(12):
            source_x = 1 + round(output_x * 9 / 11)
            offset = (source_y * 12 + source_x) * 3
            for channel, value in enumerate(
                descriptor[offset : offset + 3]
            ):
                transformed.append(
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
    return bytes(transformed)


def candidate(start: float, score: float) -> dict:
    return {
        "start": start,
        "end": start + 15,
        "duration": 15,
        "score": score,
        "evidence": [],
        "quality": {"warnings": []},
    }


def feedback_candidate(start: float, speech: float) -> dict:
    item = candidate(start, speech)
    item["features"] = {
        "speech_ratio": speech,
        "clean_frame_ratio": 0.5,
        "boundary_alignment": 0.5,
        "scene_density": 0.5,
        "semantic_score": 0.5,
        "visual_uniqueness": 0.0,
        "speaker_boundary_score": 0.0,
    }
    item["available_features"] = [
        "speech_ratio",
        "clean_frame_ratio",
        "boundary_alignment",
        "scene_density",
        "semantic_score",
    ]
    return item


class VisualSimilarityTests(unittest.TestCase):
    def test_identical_frames_are_more_similar_than_different_colors(self) -> None:
        red = solid_descriptor(140, 20, 40)
        green = solid_descriptor(20, 150, 70)
        self.assertEqual(frame_similarity(red, red), 1.0)
        self.assertLess(frame_similarity(red, green), 0.75)

    def test_similarity_survives_center_crop_and_color_grade(self) -> None:
        original = patterned_descriptor()
        transformed = crop_and_grade(original)
        unrelated = bytes(
            (x * 19 + y * 7 + channel * 83) % 256
            for y in range(12)
            for x in range(12)
            for channel in range(3)
        )
        duplicate_score = frame_similarity(original, transformed)
        unrelated_score = frame_similarity(original, unrelated)
        self.assertGreaterEqual(duplicate_score, 0.94)
        self.assertGreater(duplicate_score - unrelated_score, 0.5)

    def test_duplicate_window_is_filtered_before_unique_window(self) -> None:
        red = solid_descriptor(140, 20, 40)
        green = solid_descriptor(20, 150, 70)
        samples = [
            VisualSample(time, red)
            for time in (1, 5, 9, 21, 25, 29)
        ] + [
            VisualSample(time, green)
            for time in (41, 45, 49)
        ]
        diagnostics = {}
        selected = select_diverse_candidates(
            [
                candidate(0, 0.95),
                candidate(20, 0.90),
                candidate(40, 0.85),
            ],
            target_duration=15,
            limit=3,
            visual_samples=samples,
            diagnostics=diagnostics,
        )
        self.assertEqual([item["start"] for item in selected], [0, 40])
        self.assertEqual(diagnostics["visual_filtered"], 1)
        self.assertGreater(
            selected[1]["quality"]["visual_uniqueness"],
            0.25,
        )

    def test_feedback_weight_can_prefer_a_more_unique_candidate(self) -> None:
        original = patterned_descriptor()
        unrelated = bytes(
            (x * 19 + y * 7 + channel * 83) % 256
            for y in range(12)
            for x in range(12)
            for channel in range(3)
        )
        similar = bytes(
            round(original_value * 0.75 + unrelated_value * 0.25)
            for original_value, unrelated_value in zip(
                original,
                unrelated,
            )
        )
        samples = [
            VisualSample(time, original)
            for time in (1, 5, 9)
        ] + [
            VisualSample(time, similar)
            for time in (21, 25, 29)
        ] + [
            VisualSample(time, unrelated)
            for time in (41, 45, 49)
        ]
        selected = select_diverse_candidates(
            [
                feedback_candidate(0, 1.0),
                feedback_candidate(20, 0.95),
                feedback_candidate(40, 0.85),
            ],
            target_duration=15,
            limit=2,
            visual_samples=samples,
            scoring_profile={
                "weights": {
                    "speech_ratio": 0.35,
                    "clean_frame_ratio": 0.01,
                    "boundary_alignment": 0.01,
                    "scene_density": 0.01,
                    "semantic_score": 0.01,
                    "visual_uniqueness": 0.60,
                    "speaker_boundary_score": 0.01,
                }
            },
        )
        self.assertEqual(
            [item["start"] for item in selected],
            [0, 40],
        )
        self.assertIn(
            "visual_uniqueness",
            selected[1]["available_features"],
        )
        self.assertGreater(
            selected[1]["features"]["visual_uniqueness"],
            0.5,
        )


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg visual sampling test")
class VisualExtractionTests(unittest.TestCase):
    def test_single_decode_produces_reusable_window_signatures(self) -> None:
        fixture = Path("runtime/evaluation/deterministic-v2.mp4")
        if not fixture.exists() or fixture.stat().st_size == 0:
            build_deterministic_fixture(fixture)
        extraction = extract_visual_samples(fixture, 50)
        self.assertEqual(extraction.status, "active")
        self.assertEqual(len(extraction.samples), 25)

        first = window_signature(0, 15, extraction.samples)
        second = window_signature(25, 40, extraction.samples)
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertLess(signature_similarity(first, second), 0.8)

    def test_extraction_failure_degrades_without_raising(self) -> None:
        extraction = extract_visual_samples(
            Path("runtime/evaluation/video-does-not-exist.mp4"),
            30,
        )
        self.assertEqual(extraction.status, "failed")
        self.assertFalse(extraction.samples)
        self.assertTrue(extraction.warnings)

    def test_real_video_crop_and_grade_remains_a_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = root / "original.mp4"
            transformed = root / "transformed.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=320x180:rate=15:duration=4",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-y",
                    str(original),
                ],
                check=True,
                timeout=60,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(original),
                    "-vf",
                    (
                        "crop=iw*0.84:ih*0.84:(iw-ow)/2:(ih-oh)/2,"
                        "scale=320:180:flags=area,"
                        "eq=brightness=0.07:contrast=0.92:saturation=1.08"
                    ),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-y",
                    str(transformed),
                ],
                check=True,
                timeout=60,
            )
            original_samples = extract_visual_samples(
                original,
                4,
                interval_seconds=1,
            ).samples
            transformed_samples = extract_visual_samples(
                transformed,
                4,
                interval_seconds=1,
            ).samples
            similarity = signature_similarity(
                window_signature(0, 4, original_samples),
                window_signature(0, 4, transformed_samples),
            )
            self.assertGreaterEqual(similarity, 0.94)


if __name__ == "__main__":
    unittest.main()
