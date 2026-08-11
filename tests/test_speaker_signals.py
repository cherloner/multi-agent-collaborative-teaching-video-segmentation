import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cutpoint_pipeline import propose_clips
from speaker_signals import (
    PyannoteDiarizationExtractor,
    SpeakerTurn,
    extract_speaker_signals,
    speaker_window_metrics,
)


class FakeAnnotation:
    def itertracks(self, yield_label=False):
        self.yield_label = yield_label
        yield SimpleNamespace(start=0.0, end=4.0), "track-1", "SPEAKER_00"
        yield SimpleNamespace(start=4.1, end=8.0), "track-2", "SPEAKER_00"
        yield SimpleNamespace(start=8.0, end=13.5), "track-3", "SPEAKER_01"


class PyannoteAdapterTests(unittest.TestCase):
    def test_adapter_normalizes_and_merges_injected_turns(self) -> None:
        annotation = FakeAnnotation()

        def factory(_model_name, _token):
            return lambda _path: annotation

        extractor = PyannoteDiarizationExtractor(
            model_name="fixture",
            pipeline_factory=factory,
        )
        turns = extractor.extract(Path("not-read.mp4"), 12)
        self.assertTrue(annotation.yield_label)
        self.assertEqual(len(turns), 2)
        self.assertEqual(
            (turns[0].start, turns[0].end, turns[0].speaker),
            (0.0, 8.0, "SPEAKER_00"),
        )
        self.assertEqual(turns[1].end, 12)

    def test_provider_off_is_a_clean_empty_result(self) -> None:
        with patch.dict(
            os.environ,
            {"CUTPOINT_DIARIZATION_PROVIDER": "off"},
        ):
            result = extract_speaker_signals(Path("not-read.mp4"), 30)
        self.assertFalse(result.turns)
        self.assertEqual(result.providers["diarization"], "off")
        self.assertFalse(result.warnings)

    def test_missing_dependency_or_token_degrades_with_warning(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CUTPOINT_DIARIZATION_PROVIDER": "pyannote",
                "CUTPOINT_HF_TOKEN": "",
                "HF_TOKEN": "",
            },
        ):
            result = extract_speaker_signals(Path("not-read.mp4"), 30)
        self.assertFalse(result.turns)
        self.assertEqual(
            result.providers["diarization"],
            "unavailable",
        )
        self.assertTrue(result.warnings)


class SpeakerBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.turns = [
            SpeakerTurn(0, 10, "A"),
            SpeakerTurn(10, 25, "B"),
            SpeakerTurn(25, 40, "A"),
            SpeakerTurn(40, 55, "B"),
            SpeakerTurn(55, 60, "A"),
        ]

    def test_metrics_penalize_cutting_inside_active_turns(self) -> None:
        aligned = speaker_window_metrics(10, 25, self.turns)
        mid_turn = speaker_window_metrics(2, 12, self.turns)
        self.assertEqual(aligned["boundary_score"], 1.0)
        self.assertEqual(mid_turn["boundary_score"], 0.0)
        self.assertEqual(aligned["speaker_count"], 1)

    def test_turn_boundaries_change_best_candidate(self) -> None:
        clips = propose_clips(
            60,
            15,
            [],
            [],
            limit=1,
            speaker_turns=self.turns,
        )
        self.assertEqual((clips[0]["start"], clips[0]["end"]), (10, 25.0))
        self.assertEqual(
            clips[0]["quality"]["speaker_boundary_score"],
            1.0,
        )
        self.assertEqual(
            clips[0]["features"]["speaker_boundary_score"],
            1.0,
        )
        self.assertIn(
            "speaker_boundary_score",
            clips[0]["available_features"],
        )
        self.assertIn("说话人 1 位", " · ".join(clips[0]["evidence"]))


if __name__ == "__main__":
    unittest.main()
