import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cutpoint_evaluation import build_deterministic_fixture
from cutpoint_pipeline import (
    BlackRange,
    SilenceRange,
    analyze_video,
    propose_clips,
)
from semantic_signals import (
    FasterWhisperExtractor,
    TesseractOCRExtractor,
    TimedText,
    parse_srt,
    semantic_window_metrics,
)


class SubtitleParsingTests(unittest.TestCase):
    def test_parse_srt_cleans_markup_and_keeps_timing(self) -> None:
        spans = parse_srt(
            """1
00:00:01,250 --> 00:00:03,750
<i>关键结论</i> 42%

2
00:00:04.000 --> 00:00:06.500
{\\an8}下一句话
""",
            source="subtitle:zh",
        )
        self.assertEqual(len(spans), 2)
        self.assertEqual((spans[0].start, spans[0].end), (1.25, 3.75))
        self.assertEqual(spans[0].text, "关键结论 42%")
        self.assertEqual(spans[1].text, "下一句话")


class SemanticMetricTests(unittest.TestCase):
    def test_window_metrics_reward_coverage_and_complete_spans(self) -> None:
        spans = [
            TimedText(1, 6, "第一条完整语义证据", "subtitle:zh"),
            TimedText(8, 12, "第二条完整语义证据", "subtitle:zh"),
        ]
        metrics = semantic_window_metrics(0, 15, spans)
        self.assertEqual(metrics["span_count"], 2)
        self.assertGreater(metrics["coverage"], 0.5)
        self.assertEqual(metrics["completeness"], 1.0)
        self.assertIn("第一条", metrics["preview"])

    def test_semantics_change_top_ranked_candidate(self) -> None:
        spans = [
            TimedText(
                26,
                39,
                "第二段包含连续、完整且信息密集的关键结论与数据证据",
                "subtitle:zh",
            )
        ]
        clips = propose_clips(
            50,
            15,
            [20, 25, 45],
            [SilenceRange(20, 25), SilenceRange(45, 50)],
            [BlackRange(45, 50)],
            semantic_spans=spans,
            limit=1,
        )
        self.assertEqual((clips[0]["start"], clips[0]["end"]), (25, 40.0))
        self.assertGreater(clips[0]["quality"]["semantic_coverage"], 0.8)


class FasterWhisperAdapterTests(unittest.TestCase):
    def test_adapter_normalizes_injected_model_segments(self) -> None:
        segments = [
            SimpleNamespace(
                start=1.0,
                end=4.5,
                text="  一条识别文本  ",
                avg_logprob=-0.1,
            )
        ]

        class FakeModel:
            def transcribe(self, _path, **_kwargs):
                return segments, SimpleNamespace(language="zh")

        def factory(*_args, **_kwargs):
            return FakeModel()

        extractor = FasterWhisperExtractor(
            model_name="fixture",
            model_factory=factory,
        )
        spans = extractor.extract(Path("not-read.mp4"), 10)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].text, "一条识别文本")
        self.assertEqual(spans[0].source, "asr:faster-whisper")
        self.assertGreater(spans[0].confidence, 0.8)


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg OCR sampling test")
class TesseractAdapterTests(unittest.TestCase):
    def test_adapter_samples_frames_with_injected_ocr_function(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "ocr-fixture.mp4"
            build_deterministic_fixture(video_path)

            def fake_ocr(frame_path: Path) -> str:
                return f"画面文本 {frame_path.stem}"

            extractor = TesseractOCRExtractor(
                interval_seconds=10,
                ocr_function=fake_ocr,
            )
            spans = extractor.extract(video_path, 50)
            self.assertEqual(len(spans), 5)
            self.assertEqual((spans[0].start, spans[0].end), (0, 10))
            self.assertEqual(spans[0].source, "ocr:tesseract")
            self.assertIn("frame-000001", spans[0].text)

    def test_pipeline_falls_back_when_video_has_no_semantic_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "no-subtitles.mp4"
            build_deterministic_fixture(video_path, include_subtitles=False)
            analysis = analyze_video(video_path, 15)
            self.assertEqual(analysis["signals"]["semantic_spans"], 0)
            self.assertEqual(
                analysis["signals"]["semantic_providers"]["embedded_subtitles"],
                "empty",
            )
            self.assertTrue(analysis["clips"])
            self.assertTrue(
                all(
                    clip["quality"]["semantic_coverage"] == 0
                    for clip in analysis["clips"]
                )
            )


if __name__ == "__main__":
    unittest.main()
