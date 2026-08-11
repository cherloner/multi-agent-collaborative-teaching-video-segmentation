from __future__ import annotations

import hashlib
import json
from bisect import bisect_left, bisect_right
from statistics import mean
from typing import Iterable


FEATURE_NAMES = (
    "speech_ratio",
    "clean_frame_ratio",
    "boundary_alignment",
    "scene_density",
    "semantic_score",
    "visual_uniqueness",
    "speaker_boundary_score",
)

# The original five-signal profile keeps 88% of the prior relative weighting.
# Visual uniqueness and speaker-boundary protection each receive a conservative
# 6% prior and can be adjusted by feedback when those signals are available.
DEFAULT_WEIGHTS = {
    "speech_ratio": 0.38016,
    "clean_frame_ratio": 0.15488,
    "boundary_alignment": 0.09856,
    "scene_density": 0.07040,
    "semantic_score": 0.17600,
    "visual_uniqueness": 0.06000,
    "speaker_boundary_score": 0.06000,
}

MIN_FEEDBACK_SAMPLES = 12
MIN_CLASS_SAMPLES = 4
MIN_DISTINCT_JOBS = 3


def default_scoring_profile() -> dict:
    return {
        "version": 0,
        "status": "default",
        "active": True,
        "source": "built-in",
        "weights": dict(DEFAULT_WEIGHTS),
        "sample_count": 0,
        "keep_count": 0,
        "reject_count": 0,
    }


def _safe_feature(value: object) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def normalized_weights(
    weights: dict[str, float] | None,
    available_features: Iterable[str] | None = None,
) -> dict[str, float]:
    source = weights if weights is not None else DEFAULT_WEIGHTS
    available = set(
        FEATURE_NAMES
        if available_features is None
        else available_features
    )
    values = {
        name: max(float(source.get(name, 0.0)), 0.0)
        for name in FEATURE_NAMES
        if name in available
    }
    total = sum(values.values())
    if total <= 0:
        values = {
            name: DEFAULT_WEIGHTS[name]
            for name in FEATURE_NAMES
            if name in available
        }
        total = sum(values.values())
    return {name: value / total for name, value in values.items()}


def score_features(
    features: dict[str, float],
    profile: dict | None = None,
    available_features: Iterable[str] | None = None,
) -> float:
    weights = normalized_weights(
        (profile or {}).get("weights"),
        available_features,
    )
    return sum(
        weights[name] * _safe_feature(features.get(name, 0.0))
        for name in weights
    )


def public_scoring_profile(profile: dict | None) -> dict:
    source = profile or default_scoring_profile()
    weights = normalized_weights(source.get("weights"))
    return {
        "version": int(source.get("version", 0)),
        "status": source.get("status", "default"),
        "source": source.get("source", "built-in"),
        "weights": {
            name: round(weights.get(name, 0.0), 4)
            for name in FEATURE_NAMES
        },
        "sample_count": int(source.get("sample_count", 0)),
    }


def _dataset_signature(examples: list[dict]) -> str:
    payload = [
        {
            "job_id": item.get("job_id"),
            "clip_filename": item.get("clip_filename"),
            "rating": item.get("rating"),
            "features": {
                name: round(_safe_feature(item.get("features", {}).get(name)), 6)
                for name in FEATURE_NAMES
            },
            "available_features": sorted(
                item.get("available_features") or FEATURE_NAMES
            ),
            "updated_at": item.get("updated_at"),
        }
        for item in sorted(
            examples,
            key=lambda value: (
                str(value.get("job_id", "")),
                str(value.get("clip_filename", "")),
            ),
        )
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _prepare_examples(
    examples: list[dict],
) -> list[tuple[str, tuple[float, ...], tuple[int, ...]]]:
    feature_indexes = {
        name: index
        for index, name in enumerate(FEATURE_NAMES)
    }
    prepared = []
    for item in examples:
        available_indexes = tuple(
            feature_indexes[name]
            for name in item.get("available_features", FEATURE_NAMES)
            if name in feature_indexes
        )
        prepared.append(
            (
                item["rating"],
                tuple(
                    _safe_feature(item["features"].get(name, 0.0))
                    for name in FEATURE_NAMES
                ),
                available_indexes,
            )
        )
    return prepared


def _ranking_metrics_prepared(
    examples: list[tuple[str, tuple[float, ...], tuple[int, ...]]],
    weights: tuple[float, ...],
) -> dict:
    keep_scores = []
    reject_scores = []
    for rating, features, available_indexes in examples:
        weight_total = sum(weights[index] for index in available_indexes)
        if weight_total <= 0:
            score = 0.0
        else:
            score = (
                sum(
                    weights[index] * features[index]
                    for index in available_indexes
                )
                / weight_total
            )
        if rating == "keep":
            keep_scores.append(score)
        else:
            reject_scores.append(score)
    reject_scores.sort()
    comparisons = len(keep_scores) * len(reject_scores)
    if not comparisons:
        return {"pairwise_accuracy": 0.5, "mean_margin": 0.0}

    wins = ties = 0
    for score in keep_scores:
        wins += bisect_left(reject_scores, score)
        ties += bisect_right(reject_scores, score) - bisect_left(
            reject_scores,
            score,
        )
    accuracy = (wins + 0.5 * ties) / comparisons
    margin = mean(keep_scores) - mean(reject_scores)
    return {
        "pairwise_accuracy": round(accuracy, 6),
        "mean_margin": round(margin, 6),
    }


def _weight_candidates() -> Iterable[dict[str, float]]:
    # A deterministic 5% simplex grid keeps calibration reproducible and
    # prevents any signal from being completely removed or dominating >60%.
    yield dict(DEFAULT_WEIGHTS)
    units = 20
    minimum = 1
    maximum = 12

    def compositions(
        remaining: int,
        feature_index: int,
        values: tuple[int, ...] = (),
    ) -> Iterable[tuple[int, ...]]:
        slots_left = len(FEATURE_NAMES) - feature_index
        if slots_left == 1:
            if minimum <= remaining <= maximum:
                yield (*values, remaining)
            return
        lower = max(
            minimum,
            remaining - maximum * (slots_left - 1),
        )
        upper = min(
            maximum,
            remaining - minimum * (slots_left - 1),
        )
        for value in range(lower, upper + 1):
            yield from compositions(
                remaining - value,
                feature_index + 1,
                (*values, value),
            )

    for values in compositions(units, 0):
        yield {
            name: value / units
            for name, value in zip(FEATURE_NAMES, values)
        }


def calibrate_feedback(
    examples: list[dict],
    *,
    min_samples: int = MIN_FEEDBACK_SAMPLES,
    min_class_samples: int = MIN_CLASS_SAMPLES,
    min_distinct_jobs: int = MIN_DISTINCT_JOBS,
) -> dict:
    usable = []
    for item in examples:
        features = item.get("features")
        if (
            item.get("rating") not in {"keep", "reject"}
            or not isinstance(features, dict)
        ):
            continue
        declared = item.get("available_features")
        available = [
            name
            for name in (
                declared
                if declared is not None
                else features.keys()
            )
            if name in FEATURE_NAMES and name in features
        ]
        if len(available) < 3:
            continue
        usable.append(
            {
                **item,
                "available_features": available,
            }
        )
    keep_count = sum(item["rating"] == "keep" for item in usable)
    reject_count = len(usable) - keep_count
    distinct_jobs = len({item.get("job_id") for item in usable if item.get("job_id")})
    signature = _dataset_signature(usable)
    base = {
        "source": "feedback-calibration",
        "dataset_signature": signature,
        "sample_count": len(usable),
        "keep_count": keep_count,
        "reject_count": reject_count,
        "distinct_jobs": distinct_jobs,
        "minimums": {
            "samples": min_samples,
            "per_class": min_class_samples,
            "distinct_jobs": min_distinct_jobs,
        },
    }

    if (
        len(usable) < min_samples
        or keep_count < min_class_samples
        or reject_count < min_class_samples
        or distinct_jobs < min_distinct_jobs
    ):
        return {
            **base,
            "status": "insufficient_data",
            "active": False,
            "weights": dict(DEFAULT_WEIGHTS),
            "baseline": None,
            "candidate": None,
        }

    prepared_examples = _prepare_examples(usable)
    default_weight_vector = tuple(
        DEFAULT_WEIGHTS[name]
        for name in FEATURE_NAMES
    )
    baseline = _ranking_metrics_prepared(
        prepared_examples,
        default_weight_vector,
    )
    best_weights = dict(DEFAULT_WEIGHTS)
    best_metrics = baseline
    best_objective = float("-inf")
    for weights in _weight_candidates():
        metrics = _ranking_metrics_prepared(
            prepared_examples,
            tuple(weights[name] for name in FEATURE_NAMES),
        )
        distance = sum(
            abs(weights[name] - DEFAULT_WEIGHTS[name])
            for name in FEATURE_NAMES
        )
        objective = (
            metrics["pairwise_accuracy"]
            + 0.18 * metrics["mean_margin"]
            - 0.035 * distance
        )
        if objective > best_objective + 1e-12:
            best_objective = objective
            best_weights = weights
            best_metrics = metrics

    accuracy_gain = (
        best_metrics["pairwise_accuracy"] - baseline["pairwise_accuracy"]
    )
    margin_gain = best_metrics["mean_margin"] - baseline["mean_margin"]
    improved = accuracy_gain >= 0.02 or (
        accuracy_gain >= -1e-9 and margin_gain >= 0.025
    )
    status = "active" if improved else "no_improvement"
    return {
        **base,
        "status": status,
        "active": improved,
        "weights": {
            name: round(best_weights[name], 4)
            for name in FEATURE_NAMES
        },
        "baseline": baseline,
        "candidate": {
            **best_metrics,
            "accuracy_gain": round(accuracy_gain, 6),
            "margin_gain": round(margin_gain, 6),
        },
    }


def refresh_scoring_profile(store, *, force: bool = False) -> dict:
    examples = store.feedback_training_examples()
    candidate = calibrate_feedback(examples["examples"])
    candidate["skipped_samples"] = examples["skipped"]
    latest = store.latest_scoring_profile()
    if (
        not force
        and latest
        and latest.get("dataset_signature") == candidate["dataset_signature"]
    ):
        return latest
    return store.save_scoring_profile(
        candidate,
        activate=candidate["status"] == "active",
    )


def scoring_model_status(store) -> dict:
    return {
        "active": store.active_scoring_profile() or default_scoring_profile(),
        "latest_attempt": store.latest_scoring_profile(),
    }
