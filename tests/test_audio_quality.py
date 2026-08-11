import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from audio_quality import (
    AUDIO_SAMPLE_RATE,
    build_audio_filter,
    inspect_rendered_media,
    measure_edge_fades,
    measure_integrated_loudness,
)
from cutpoint_evaluation import build_deterministic_fixture
from cutpoint_pipeline import render_clip
from render_profiles import FIXED_PROFILES, resolve_export_profile


class AudioFilterTests(unittest.TestCase):
    def test_filter_declares_loudness_rate_and_edge_fades(self) -> None:
        graph = build_audio_filter(15)
        self.assertIn("loudnorm=I=-16", graph)
        self.assertIn(f"aresample={AUDIO_SAMPLE_RATE}", graph)
        self.assertIn("afade=t=in:st=0:d=0.08", graph)
        self.assertIn("afade=t=out:st=14.880:d=0.12", graph)


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "FFmpeg audio quality test",
)
class AudioQualityIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = Path("runtime/evaluation/deterministic-v2.mp4")
        if not cls.fixture.exists() or cls.fixture.stat().st_size == 0:
            build_deterministic_fixture(cls.fixture)

    def test_render_normalizes_loudness_fades_edges_and_passes_sync_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "audio-quality.mp4"
            details = render_clip(
                self.fixture,
                output,
                start=0,
                duration=5,
                export_profile=FIXED_PROFILES["landscape"],
                captions_enabled=False,
                has_audio=True,
            )
            loudness = measure_integrated_loudness(output)
            fades = measure_edge_fades(output)

            self.assertTrue(details["delivery_check"]["passed"])
            self.assertEqual(
                details["delivery_check"]["audio_sample_rate"],
                AUDIO_SAMPLE_RATE,
            )
            self.assertLessEqual(
                details["delivery_check"]["av_sync_delta_seconds"],
                0.05,
            )
            self.assertLessEqual(loudness["target_error_lu"], 0.5)
            self.assertGreaterEqual(fades["effectiveness"], 0.75)

            rejected = inspect_rendered_media(
                output,
                expected_duration=5,
                expected_width=720,
                expected_height=1280,
                expects_audio=True,
            )
            self.assertFalse(rejected["passed"])
            self.assertFalse(rejected["checks"]["dimensions"])

    def test_video_without_audio_renders_and_passes_absence_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "silent.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=navy:s=160x90:r=24",
                    "-t",
                    "1",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source),
                ],
                check=True,
                timeout=60,
            )
            output = root / "silent-output.mp4"
            details = render_clip(
                source,
                output,
                start=0,
                duration=1,
                export_profile=resolve_export_profile("source", 160, 90),
                captions_enabled=False,
                has_audio=False,
            )
            self.assertTrue(details["delivery_check"]["passed"])
            self.assertEqual(details["audio"]["status"], "not_present")
            self.assertIsNone(
                details["delivery_check"]["audio_sample_rate"]
            )


if __name__ == "__main__":
    unittest.main()
