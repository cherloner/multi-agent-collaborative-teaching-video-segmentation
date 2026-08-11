import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cutpoint_evaluation import build_deterministic_fixture
from cutpoint_pipeline import probe_video, render_clip
from render_profiles import (
    FIXED_PROFILES,
    build_ass_subtitles,
    build_video_filter,
    resolve_export_profile,
)


class ExportProfileTests(unittest.TestCase):
    def test_fixed_profiles_have_expected_dimensions_and_safe_margins(self) -> None:
        expected = {
            "landscape": (1280, 720, "16:9"),
            "portrait": (720, 1280, "9:16"),
            "square": (1080, 1080, "1:1"),
        }
        for key, (width, height, ratio) in expected.items():
            profile = resolve_export_profile(key, 320, 180)
            self.assertEqual((profile.width, profile.height), (width, height))
            self.assertEqual(profile.aspect_ratio, ratio)
            self.assertGreater(profile.caption_margin_v, height * 0.08)

    def test_source_profile_preserves_even_source_dimensions(self) -> None:
        profile = resolve_export_profile("source", 1919, 1079)
        self.assertEqual((profile.width, profile.height), (1918, 1078))
        self.assertFalse(profile.fixed_canvas)

    def test_ass_subtitles_clip_timing_and_ignore_ocr_text(self) -> None:
        profile = FIXED_PROFILES["portrait"]
        content, count = build_ass_subtitles(
            [
                {
                    "start": 8,
                    "end": 12,
                    "text": "保留落在片段内的字幕文本",
                    "source": "subtitle:zh",
                },
                {
                    "start": 13,
                    "end": 16,
                    "text": "画面角标不应烧录",
                    "source": "ocr:tesseract",
                },
            ],
            clip_start=10,
            clip_end=20,
            profile=profile,
        )
        self.assertEqual(count, 1)
        self.assertIn("Dialogue: 0,0:00:00.00,0:00:02.00", content)
        self.assertIn(f",{profile.caption_margin_v},1", content)
        self.assertNotIn("画面角标", content)

    def test_fixed_canvas_filter_uses_blurred_background_and_safe_overlay(self) -> None:
        graph = build_video_filter(FIXED_PROFILES["portrait"])
        self.assertIn("boxblur", graph)
        self.assertIn("force_original_aspect_ratio=decrease", graph)
        self.assertIn("overlay=(W-w)/2:(H-h)/2", graph)

    def test_caption_filter_failure_retries_without_subtitles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "fallback.mp4"
            calls = []

            def fake_run(command, timeout=1800):
                calls.append(command)
                if len(calls) == 1:
                    raise RuntimeError("ass filter unavailable")
                Path(command[-1]).write_bytes(b"fallback")

            with patch("cutpoint_pipeline._run", side_effect=fake_run):
                details = render_clip(
                    root / "source.mp4",
                    output,
                    start=0,
                    duration=3,
                    export_profile=FIXED_PROFILES["portrait"],
                    semantic_spans=[
                        {
                            "start": 0,
                            "end": 2,
                            "text": "需要降级的字幕",
                            "source": "subtitle:zh",
                        }
                    ],
                    captions_enabled=True,
                    verify_output=False,
                )

            self.assertEqual(len(calls), 2)
            self.assertEqual(details["caption_count"], 0)
            self.assertIn("字幕烧录失败", details["caption_warning"])
            self.assertTrue(output.exists())


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg render profile test")
class RenderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = Path("runtime/evaluation/deterministic-v2.mp4")
        if not cls.fixture.exists() or cls.fixture.stat().st_size == 0:
            build_deterministic_fixture(cls.fixture)

    def test_all_fixed_profiles_render_to_declared_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for key, profile in FIXED_PROFILES.items():
                output = root / f"{key}.mp4"
                details = render_clip(
                    self.fixture,
                    output,
                    start=0,
                    duration=0.6,
                    export_profile=profile,
                    captions_enabled=False,
                )
                metadata = probe_video(output)
                self.assertEqual(
                    (metadata["width"], metadata["height"]),
                    (profile.width, profile.height),
                )
                self.assertEqual(details["caption_count"], 0)

    def test_portrait_render_burns_generated_caption_and_cleans_ass_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "portrait-caption.mp4"
            details = render_clip(
                self.fixture,
                output,
                start=0,
                duration=2.5,
                export_profile=FIXED_PROFILES["portrait"],
                semantic_spans=[
                    {
                        "start": 0.5,
                        "end": 2.2,
                        "text": "字幕位于竖屏安全区",
                        "source": "subtitle:zh",
                    }
                ],
                captions_enabled=True,
            )
            self.assertEqual(details["caption_count"], 1)
            self.assertTrue(output.exists())
            self.assertFalse(output.with_suffix(".captions.ass").exists())


if __name__ == "__main__":
    unittest.main()
