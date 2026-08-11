import tempfile
import unittest
from pathlib import Path

from feedback_calibration import (
    FEATURE_NAMES,
    calibrate_feedback,
    refresh_scoring_profile,
    score_features,
)
from job_store import JobStore


def feature_vector(
    *,
    speech: float,
    semantic: float,
    visual: float = 0.5,
    speaker: float = 0.5,
) -> dict:
    return {
        "speech_ratio": speech,
        "clean_frame_ratio": 0.85,
        "boundary_alignment": 0.6,
        "scene_density": 0.4,
        "semantic_score": semantic,
        "visual_uniqueness": visual,
        "speaker_boundary_score": speaker,
    }


def synthetic_examples() -> list[dict]:
    examples = []
    for index in range(8):
        examples.append(
            {
                "job_id": f"job-{index % 4}",
                "clip_filename": f"keep-{index}.mp4",
                "rating": "keep",
                "features": feature_vector(speech=0.42, semantic=0.95),
                "available_features": list(FEATURE_NAMES),
            }
        )
        examples.append(
            {
                "job_id": f"job-{index % 4}",
                "clip_filename": f"reject-{index}.mp4",
                "rating": "reject",
                "features": feature_vector(speech=0.96, semantic=0.05),
                "available_features": list(FEATURE_NAMES),
            }
        )
    return examples


class CalibrationTests(unittest.TestCase):
    def test_insufficient_feedback_never_activates_profile(self) -> None:
        profile = calibrate_feedback(synthetic_examples()[:6])
        self.assertEqual(profile["status"], "insufficient_data")
        self.assertFalse(profile["active"])

    def test_feedback_can_correct_a_misranked_default_signal(self) -> None:
        profile = calibrate_feedback(synthetic_examples())
        self.assertEqual(profile["status"], "active")
        self.assertTrue(profile["active"])
        self.assertGreater(
            profile["weights"]["semantic_score"],
            profile["weights"]["speech_ratio"],
        )
        self.assertGreater(
            profile["candidate"]["pairwise_accuracy"],
            profile["baseline"]["pairwise_accuracy"],
        )

    def test_unavailable_semantic_signal_is_removed_and_renormalized(self) -> None:
        score = score_features(
            {
                "speech_ratio": 1.0,
                "clean_frame_ratio": 0.0,
                "boundary_alignment": 0.0,
                "scene_density": 0.0,
                "semantic_score": 1.0,
            },
            available_features=(
                "speech_ratio",
                "clean_frame_ratio",
                "boundary_alignment",
                "scene_density",
            ),
        )
        self.assertAlmostEqual(score, 0.54, places=3)

    def test_legacy_five_feature_samples_remain_trainable(self) -> None:
        legacy_examples = []
        legacy_features = FEATURE_NAMES[:5]
        for item in synthetic_examples():
            legacy_examples.append(
                {
                    **item,
                    "features": {
                        name: item["features"][name]
                        for name in legacy_features
                    },
                    "available_features": list(legacy_features),
                }
            )
        profile = calibrate_feedback(legacy_examples)
        self.assertEqual(profile["sample_count"], len(legacy_examples))
        self.assertNotEqual(profile["status"], "insufficient_data")

    def test_feedback_can_learn_visual_and_speaker_preferences(self) -> None:
        examples = []
        for index in range(8):
            common = {
                "speech_ratio": 0.75,
                "clean_frame_ratio": 0.9,
                "boundary_alignment": 0.7,
                "scene_density": 0.5,
                "semantic_score": 0.7,
            }
            for rating, value in (("keep", 0.95), ("reject", 0.05)):
                examples.append(
                    {
                        "job_id": f"preference-{index % 4}",
                        "clip_filename": f"{rating}-{index}.mp4",
                        "rating": rating,
                        "features": {
                            **common,
                            "visual_uniqueness": value,
                            "speaker_boundary_score": value,
                        },
                        "available_features": list(FEATURE_NAMES),
                    }
                )
        profile = calibrate_feedback(examples)
        self.assertEqual(profile["status"], "active")
        learned_weight = (
            profile["weights"]["visual_uniqueness"]
            + profile["weights"]["speaker_boundary_score"]
        )
        default_weight = 0.12
        self.assertGreater(learned_weight, default_weight)
        self.assertGreater(
            profile["candidate"]["mean_margin"],
            profile["baseline"]["mean_margin"],
        )


class CalibrationPersistenceTests(unittest.TestCase):
    def test_feedback_snapshots_train_and_persist_active_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JobStore(root / "jobs.db")
            store.initialize()
            examples = synthetic_examples()
            for job_index in range(4):
                job_id = f"job-{job_index}"
                source = root / f"{job_id}.mp4"
                source.write_bytes(b"fixture")
                self.assertTrue(
                    store.create_job(
                        {
                            "id": job_id,
                            "filename": source.name,
                            "source_path": str(source),
                            "target_duration": 30,
                        },
                        capacity=4,
                    )
                )
                clips = []
                for item in [
                    value for value in examples if value["job_id"] == job_id
                ]:
                    clips.append(
                        {
                            "filename": item["clip_filename"],
                            "score": 0.5,
                            "features": item["features"],
                            "available_features": item["available_features"],
                        }
                    )
                store.update_job(
                    job_id,
                    status="completed",
                    progress=100,
                    phase="completed",
                    result={
                        "scoring_profile": {"version": 0},
                        "clips": clips,
                    },
                )
                for item in [
                    value for value in examples if value["job_id"] == job_id
                ]:
                    store.save_feedback(
                        job_id,
                        item["clip_filename"],
                        item["rating"],
                        None,
                    )

            training = store.feedback_training_examples()
            self.assertEqual(training["total"], 16)
            self.assertEqual(training["skipped"], 0)
            profile = refresh_scoring_profile(store)
            self.assertEqual(profile["status"], "active")
            self.assertEqual(store.active_scoring_profile()["version"], profile["version"])

            reopened = JobStore(root / "jobs.db")
            reopened.initialize()
            self.assertEqual(
                reopened.active_scoring_profile()["dataset_signature"],
                profile["dataset_signature"],
            )


if __name__ == "__main__":
    unittest.main()
