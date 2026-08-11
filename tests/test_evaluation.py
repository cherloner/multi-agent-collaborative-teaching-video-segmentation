import unittest

from cutpoint_evaluation import (
    boundary_metrics,
    clip_quality_metrics,
    highlight_metrics,
    interval_iou,
    timeline_iou,
)


class BoundaryMetricTests(unittest.TestCase):
    def test_boundary_matching_is_one_to_one(self) -> None:
        result = boundary_metrics([9.8, 10.1, 20.2], [10.0, 20.0], tolerance=0.5)
        self.assertEqual(result["matches"], 2)
        self.assertAlmostEqual(result["precision"], 2 / 3, places=4)
        self.assertEqual(result["recall"], 1.0)


class IntervalMetricTests(unittest.TestCase):
    def test_interval_iou(self) -> None:
        self.assertAlmostEqual(interval_iou((0, 10), (5, 15)), 1 / 3)

    def test_timeline_iou_merges_overlaps(self) -> None:
        self.assertEqual(timeline_iou([(0, 5), (4, 10)], [(0, 10)]), 1.0)


class HighlightMetricTests(unittest.TestCase):
    def test_highlight_recall_counts_expected_intervals(self) -> None:
        clips = [{"start": 0, "end": 15}, {"start": 30, "end": 45}]
        result = highlight_metrics(clips, [(0, 20), (25, 45)], minimum_iou=0.45)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["precision"], 1.0)

    def test_clip_quality_aggregates_ratios(self) -> None:
        clips = [
            {
                "duration": 15,
                "quality": {
                    "speech_ratio": 1.0,
                    "clean_frame_ratio": 1.0,
                    "warnings": [],
                },
            },
            {
                "duration": 15,
                "quality": {
                    "speech_ratio": 0.8,
                    "clean_frame_ratio": 0.9,
                    "warnings": ["包含明显黑帧"],
                },
            },
        ]
        result = clip_quality_metrics(clips, 15)
        self.assertEqual(result["speech_ratio"], 0.9)
        self.assertEqual(result["clean_frame_ratio"], 0.95)
        self.assertEqual(result["warning_rate"], 0.5)

    def test_short_source_duration_is_compliant(self) -> None:
        clips = [
            {
                "duration": 12,
                "quality": {
                    "speech_ratio": 1.0,
                    "clean_frame_ratio": 1.0,
                    "warnings": [],
                },
            }
        ]
        result = clip_quality_metrics(clips, 15, source_duration=12)
        self.assertEqual(result["duration_compliance"], 1.0)


if __name__ == "__main__":
    unittest.main()
