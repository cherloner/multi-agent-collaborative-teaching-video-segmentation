import unittest

from cutpoint_pipeline import BlackRange, SilenceRange, propose_clips


class ProposeClipsTests(unittest.TestCase):
    def test_short_video_returns_complete_video(self) -> None:
        clips = propose_clips(12.5, 30, [], [])
        self.assertEqual(clips[0]["start"], 0.0)
        self.assertEqual(clips[0]["end"], 12.5)
        self.assertEqual(clips[0]["evidence"], ["完整视频"])

    def test_target_duration_is_clamped(self) -> None:
        clips = propose_clips(180, 90, [20, 45, 80, 130], [])
        self.assertTrue(clips)
        self.assertTrue(all(15 <= clip["duration"] <= 60 for clip in clips))

    def test_speech_heavy_clip_scores_higher(self) -> None:
        clips = propose_clips(
            120,
            30,
            [12, 18, 24, 72],
            [SilenceRange(60, 90)],
        )
        self.assertTrue(clips)
        self.assertGreater(clips[0]["score"], 0.5)

    def test_candidate_selection_avoids_black_and_silent_ranges(self) -> None:
        clips = propose_clips(
            50,
            15,
            [20, 25, 45],
            [SilenceRange(20, 25), SilenceRange(45, 50)],
            [BlackRange(45, 50)],
        )
        self.assertEqual(
            [(clip["start"], clip["end"]) for clip in clips],
            [(0.0, 15.0), (25, 40.0)],
        )
        self.assertTrue(all(clip["quality"]["speech_ratio"] == 1.0 for clip in clips))
        self.assertTrue(all(not clip["quality"]["warnings"] for clip in clips))


if __name__ == "__main__":
    unittest.main()
