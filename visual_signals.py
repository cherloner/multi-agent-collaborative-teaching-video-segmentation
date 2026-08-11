from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from math import isqrt, sqrt
from pathlib import Path
from statistics import mean

from process_control import run_process


DEFAULT_INTERVAL_SECONDS = 2.0
FINGERPRINT_WIDTH = 12
FINGERPRINT_HEIGHT = 12


@dataclass(frozen=True)
class VisualSample:
    time: float
    descriptor: bytes


@dataclass(frozen=True)
class VisualExtraction:
    samples: list[VisualSample]
    interval_seconds: float
    status: str
    warnings: list[str]


def extract_visual_samples(
    video_path: Path,
    duration: float,
    *,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    width: int = FINGERPRINT_WIDTH,
    height: int = FINGERPRINT_HEIGHT,
) -> VisualExtraction:
    if not shutil.which("ffmpeg"):
        return VisualExtraction(
            [],
            interval_seconds,
            "unavailable",
            ["视觉指纹需要 FFmpeg"],
        )

    safe_interval = min(max(float(interval_seconds), 0.5), 10.0)
    frame_size = width * height * 3
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        (
            f"fps=1/{safe_interval:g},"
            f"scale={width}:{height}:flags=area,format=rgb24"
        ),
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        result = run_process(
            command,
            capture_output=True,
            check=True,
            timeout=600,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        message = str(exc).strip() or exc.__class__.__name__
        return VisualExtraction(
            [],
            safe_interval,
            "failed",
            [f"视觉指纹提取失败：{message[:180]}"],
        )

    complete_bytes = len(result.stdout) - (len(result.stdout) % frame_size)
    samples = [
        VisualSample(
            time=min(
                max(duration - 0.001, 0.0),
                index * safe_interval + safe_interval / 2.0,
            ),
            descriptor=result.stdout[offset : offset + frame_size],
        )
        for index, offset in enumerate(range(0, complete_bytes, frame_size))
    ]
    return VisualExtraction(
        samples,
        safe_interval,
        "active" if samples else "empty",
        [],
    )


def _raw_similarity(left: bytes, right: bytes) -> float:
    if not left or len(left) != len(right):
        return 0.0
    mean_difference = sum(
        abs(left_value - right_value)
        for left_value, right_value in zip(left, right)
    ) / len(left)
    return max(0.0, 1.0 - mean_difference / 255.0)


def _decode_luma(descriptor: bytes) -> tuple[list[float], int, int] | None:
    if not descriptor or len(descriptor) % 3:
        return None
    pixel_count = len(descriptor) // 3
    width = isqrt(pixel_count)
    if width * width != pixel_count:
        return None
    luma = [
        (
            descriptor[offset] * 0.299
            + descriptor[offset + 1] * 0.587
            + descriptor[offset + 2] * 0.114
        )
        for offset in range(0, len(descriptor), 3)
    ]
    return luma, width, width


def _contrast(values: list[float]) -> float:
    if not values:
        return 0.0
    center = mean(values)
    return sqrt(mean((value - center) ** 2 for value in values))


def _correlation(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator <= 1e-9:
        return 0.0
    return max(
        0.0,
        min(
            1.0,
            sum(
                left_value * right_value
                for left_value, right_value in zip(
                    left_centered,
                    right_centered,
                )
            )
            / denominator,
        ),
    )


def _shifted_values(
    left: list[float],
    right: list[float],
    width: int,
    height: int,
    dx: int,
    dy: int,
) -> tuple[list[float], list[float]]:
    left_values = []
    right_values = []
    x_start = max(0, -dx)
    x_end = min(width, width - dx)
    y_start = max(0, -dy)
    y_end = min(height, height - dy)
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            left_values.append(left[y * width + x])
            right_values.append(right[(y + dy) * width + x + dx])
    return left_values, right_values


def _best_aligned_correlation(
    left: list[float],
    right: list[float],
    width: int,
    height: int,
) -> float:
    return max(
        _correlation(
            *_shifted_values(left, right, width, height, dx, dy)
        )
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    )


def _edge_map(
    luma: list[float],
    width: int,
    height: int,
) -> tuple[list[float], int, int]:
    edge_width = max(width - 1, 0)
    edge_height = max(height - 1, 0)
    edges = []
    for y in range(edge_height):
        for x in range(edge_width):
            origin = luma[y * width + x]
            horizontal = luma[y * width + x + 1] - origin
            vertical = luma[(y + 1) * width + x] - origin
            edges.append(sqrt(horizontal**2 + vertical**2))
    return edges, edge_width, edge_height


def _center_crop_variants(
    values: list[float],
    width: int,
    height: int,
) -> list[list[float]]:
    variants = [values]
    max_margin = min(2, (min(width, height) - 3) // 2)
    for margin in range(1, max_margin + 1):
        crop_width = width - margin * 2
        crop_height = height - margin * 2
        resized = []
        for output_y in range(height):
            source_y = margin + round(
                output_y * (crop_height - 1) / max(height - 1, 1)
            )
            for output_x in range(width):
                source_x = margin + round(
                    output_x * (crop_width - 1) / max(width - 1, 1)
                )
                resized.append(values[source_y * width + source_x])
        variants.append(resized)
    return variants


def _normalized_histogram(
    values: list[float],
    bins: int = 12,
) -> list[float]:
    if not values:
        return [0.0] * bins
    center = mean(values)
    spread = _contrast(values)
    if spread <= 1e-9:
        histogram = [0.0] * bins
        histogram[bins // 2] = 1.0
        return histogram
    counts = [0] * bins
    for value in values:
        normalized = min(max((value - center) / spread, -3.0), 3.0)
        index = min(int((normalized + 3.0) / 6.0 * bins), bins - 1)
        counts[index] += 1
    return [count / len(values) for count in counts]


def _histogram_intersection(
    left: list[float],
    right: list[float],
) -> float:
    return sum(
        min(left_value, right_value)
        for left_value, right_value in zip(left, right)
    )


@lru_cache(maxsize=8192)
def frame_similarity(left: bytes, right: bytes) -> float:
    if not left or len(left) != len(right):
        return 0.0
    if left == right:
        return 1.0
    raw_similarity = _raw_similarity(left, right)
    left_frame = _decode_luma(left)
    right_frame = _decode_luma(right)
    if not left_frame or not right_frame:
        return raw_similarity
    left_luma, width, height = left_frame
    right_luma, right_width, right_height = right_frame
    if (width, height) != (right_width, right_height):
        return raw_similarity

    # Flat frames contain no structural evidence. Keep the RGB comparison so
    # two unrelated title cards or solid colors are not treated as duplicates.
    if min(_contrast(left_luma), _contrast(right_luma)) < 3.0:
        return raw_similarity

    left_variants = _center_crop_variants(left_luma, width, height)
    right_variants = _center_crop_variants(right_luma, width, height)
    structure = max(
        _best_aligned_correlation(
            left_variant,
            right_variant,
            width,
            height,
        )
        for left_variant in left_variants
        for right_variant in right_variants
    )
    left_edge_variants = [
        _edge_map(variant, width, height)[0]
        for variant in left_variants
    ]
    right_edge_variants = [
        _edge_map(variant, width, height)[0]
        for variant in right_variants
    ]
    edge_width = width - 1
    edge_height = height - 1
    edge_structure = max(
        _best_aligned_correlation(
            left_edges,
            right_edges,
            edge_width,
            edge_height,
        )
        for left_edges in left_edge_variants
        for right_edges in right_edge_variants
    ) if edge_width and edge_height else 0.0
    distribution = max(
        (
            _histogram_intersection(
                _normalized_histogram(left_variant),
                _normalized_histogram(right_variant),
            )
            + _histogram_intersection(
                _normalized_histogram(left_edges),
                _normalized_histogram(right_edges),
            )
        )
        / 2.0
        for left_variant, left_edges in zip(
            left_variants,
            left_edge_variants,
        )
        for right_variant, right_edges in zip(
            right_variants,
            right_edge_variants,
        )
    )
    score = (
        structure * 0.51
        + edge_structure * 0.24
        + distribution * 0.15
        + raw_similarity * 0.10
    )
    return max(0.0, min(score, 1.0))


def window_signature(
    start: float,
    end: float,
    samples: list[VisualSample],
) -> tuple[bytes, ...]:
    inside = [
        sample.descriptor
        for sample in samples
        if start <= sample.time < end
    ]
    if inside or not samples:
        return tuple(inside)
    midpoint = (start + end) / 2.0
    nearest = min(samples, key=lambda sample: abs(sample.time - midpoint))
    return (nearest.descriptor,)


@lru_cache(maxsize=1024)
def signature_similarity(
    left: tuple[bytes, ...],
    right: tuple[bytes, ...],
) -> float:
    if not left or not right:
        return 0.0

    def directed(
        source: tuple[bytes, ...],
        target: tuple[bytes, ...],
    ) -> float:
        return mean(
            max(frame_similarity(frame, candidate) for candidate in target)
            for frame in source
        )

    return (directed(left, right) + directed(right, left)) / 2.0
