import shutil
import unittest
from pathlib import Path

from cutpoint_evaluation import run_evaluation


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "FFmpeg integration test",
)
class EvaluationIntegrationTests(unittest.TestCase):
    def test_deterministic_quality_gates_pass(self) -> None:
        report = run_evaluation(Path("evaluation/manifest.json").resolve())
        self.assertTrue(report["passed"], report["gates"])
        self.assertEqual(report["aggregate"]["scene_f1"], 1.0)
        self.assertEqual(report["aggregate"]["highlight_recall"], 1.0)
        self.assertGreaterEqual(report["aggregate"]["clean_frame_ratio"], 0.95)
        self.assertGreaterEqual(
            report["aggregate"]["analysis_realtime_factor"],
            1.0,
        )
        self.assertGreaterEqual(
            report["aggregate"]["perceptual_duplicate_similarity"],
            0.94,
        )
        self.assertGreaterEqual(
            report["aggregate"]["perceptual_similarity_margin"],
            0.5,
        )
        performance = report["cases"][0]["observed"]["performance"]
        self.assertGreater(performance["analysis_total_ms"], 0)
        self.assertTrue(
            {
                "probe",
                "scene_detection",
                "silence_detection",
                "black_detection",
                "semantic_extraction",
                "speaker_diarization",
                "visual_fingerprints",
                "candidate_ranking",
            }.issubset(performance["stages_ms"])
        )


if __name__ == "__main__":
    unittest.main()
