from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from process_control import run_process


SRT_TIMING_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{3})"
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
ASS_TAG_RE = re.compile(r"\{\\[^}]+\}")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TimedText:
    start: float
    end: float
    text: str
    source: str
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SemanticExtraction:
    spans: list[TimedText]
    providers: dict[str, str]
    warnings: list[str]


def _run(command: list[str], timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    return run_process(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=True,
    )


def _parse_timestamp(value: str) -> float:
    normalized = value.replace(",", ".")
    hours, minutes, seconds = normalized.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _clean_text(value: str) -> str:
    cleaned = ASS_TAG_RE.sub("", HTML_TAG_RE.sub("", value))
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def parse_srt(payload: str, source: str = "subtitle") -> list[TimedText]:
    spans: list[TimedText] = []
    blocks = re.split(r"\r?\n\s*\r?\n", payload.strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next(
            (index for index, line in enumerate(lines) if SRT_TIMING_RE.search(line)),
            None,
        )
        if timing_index is None:
            continue
        match = SRT_TIMING_RE.search(lines[timing_index])
        text = _clean_text(" ".join(lines[timing_index + 1 :]))
        if not text:
            continue
        start = _parse_timestamp(match.group("start"))
        end = _parse_timestamp(match.group("end"))
        if end <= start:
            continue
        spans.append(
            TimedText(
                start=round(start, 3),
                end=round(end, 3),
                text=text,
                source=source,
                confidence=1.0,
            )
        )
    return spans


class EmbeddedSubtitleExtractor:
    name = "embedded_subtitles"

    def extract(self, video_path: Path, _duration: float) -> list[TimedText]:
        probe = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "s",
                "-show_entries",
                "stream=index,codec_name:stream_tags=language,title",
                "-of",
                "json",
                str(video_path),
            ],
            timeout=60,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if not streams:
            return []
        stream = streams[0]
        language = stream.get("tags", {}).get("language", "und")
        source = f"subtitle:{language}"
        extracted = _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-map",
                f"0:{stream['index']}",
                "-f",
                "srt",
                "-",
            ]
        )
        return parse_srt(extracted.stdout, source)


class FasterWhisperExtractor:
    name = "faster_whisper"

    def __init__(
        self,
        model_name: str | None = None,
        model_factory: Callable | None = None,
    ):
        self.model_name = model_name or os.getenv(
            "CUTPOINT_WHISPER_MODEL",
            "small",
        )
        self.model_factory = model_factory

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def extract(self, video_path: Path, _duration: float) -> list[TimedText]:
        if self.model_factory is None:
            if not self.available():
                raise RuntimeError("未安装 faster-whisper")
            from faster_whisper import WhisperModel

            factory = WhisperModel
        else:
            factory = self.model_factory

        model = factory(
            self.model_name,
            device=os.getenv("CUTPOINT_WHISPER_DEVICE", "auto"),
            compute_type=os.getenv("CUTPOINT_WHISPER_COMPUTE_TYPE", "int8"),
        )
        segments, _info = model.transcribe(
            str(video_path),
            vad_filter=True,
            word_timestamps=False,
        )
        spans = []
        for segment in segments:
            text = _clean_text(str(segment.text))
            if not text or segment.end <= segment.start:
                continue
            average_log_probability = float(
                getattr(segment, "avg_logprob", -0.2)
            )
            confidence = min(max(math.exp(average_log_probability), 0.0), 1.0)
            spans.append(
                TimedText(
                    start=round(float(segment.start), 3),
                    end=round(float(segment.end), 3),
                    text=text,
                    source="asr:faster-whisper",
                    confidence=round(confidence, 3),
                )
            )
        return spans


class TesseractOCRExtractor:
    name = "tesseract_ocr"

    def __init__(
        self,
        interval_seconds: float | None = None,
        ocr_function: Callable | None = None,
    ):
        self.interval_seconds = interval_seconds or float(
            os.getenv("CUTPOINT_OCR_INTERVAL_SECONDS", "5")
        )
        self.ocr_function = ocr_function

    @staticmethod
    def available() -> bool:
        return (
            importlib.util.find_spec("pytesseract") is not None
            and importlib.util.find_spec("PIL") is not None
            and shutil.which("tesseract") is not None
        )

    def extract(self, video_path: Path, duration: float) -> list[TimedText]:
        if self.ocr_function is None:
            if not self.available():
                raise RuntimeError("未安装 Tesseract OCR 运行环境")
            import pytesseract
            from PIL import Image

            def read_image(path: Path) -> str:
                with Image.open(path) as image:
                    return pytesseract.image_to_string(
                        image,
                        lang=os.getenv("CUTPOINT_TESSERACT_LANG", "chi_sim+eng"),
                    )

        else:
            read_image = self.ocr_function

        interval = max(self.interval_seconds, 1.0)
        spans: list[TimedText] = []
        with tempfile.TemporaryDirectory(prefix="cutpoint-ocr-") as temp_dir:
            frame_pattern = str(Path(temp_dir) / "frame-%06d.png")
            _run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video_path),
                    "-vf",
                    f"fps=1/{interval},scale=960:-2",
                    "-vsync",
                    "vfr",
                    frame_pattern,
                ]
            )
            previous_text = ""
            for index, frame_path in enumerate(
                sorted(Path(temp_dir).glob("frame-*.png"))
            ):
                text = _clean_text(str(read_image(frame_path)))
                if not text or text == previous_text:
                    continue
                start = min(index * interval, duration)
                end = min(start + interval, duration)
                if end > start:
                    spans.append(
                        TimedText(
                            start=round(start, 3),
                            end=round(end, 3),
                            text=text,
                            source="ocr:tesseract",
                            confidence=0.7,
                        )
                    )
                previous_text = text
        return spans


def extract_semantic_signals(
    video_path: Path,
    duration: float,
) -> SemanticExtraction:
    spans: list[TimedText] = []
    providers: dict[str, str] = {}
    warnings: list[str] = []

    extractors: list[object] = [EmbeddedSubtitleExtractor()]
    asr_provider = os.getenv("CUTPOINT_ASR_PROVIDER", "off").lower()
    ocr_provider = os.getenv("CUTPOINT_OCR_PROVIDER", "off").lower()
    if asr_provider == "faster-whisper":
        extractors.append(FasterWhisperExtractor())
    elif asr_provider not in {"off", "embedded"}:
        warnings.append(f"未知 ASR 提供器: {asr_provider}")
    if ocr_provider == "tesseract":
        extractors.append(TesseractOCRExtractor())
    elif ocr_provider != "off":
        warnings.append(f"未知 OCR 提供器: {ocr_provider}")

    for extractor in extractors:
        try:
            extracted = extractor.extract(video_path, duration)
            spans.extend(extracted)
            providers[extractor.name] = "active" if extracted else "empty"
        except Exception as exc:
            providers[extractor.name] = "unavailable"
            warnings.append(f"{extractor.name}: {str(exc)[:180]}")

    unique = {
        (
            round(span.start, 3),
            round(span.end, 3),
            span.text,
            span.source,
        ): span
        for span in spans
        if 0 <= span.start < span.end <= duration + 0.5
    }
    return SemanticExtraction(
        spans=sorted(unique.values(), key=lambda item: (item.start, item.end)),
        providers=providers,
        warnings=warnings,
    )


def semantic_provider_status() -> dict:
    asr_provider = os.getenv("CUTPOINT_ASR_PROVIDER", "off").lower()
    ocr_provider = os.getenv("CUTPOINT_OCR_PROVIDER", "off").lower()
    return {
        "embedded_subtitles": {
            "enabled": True,
            "available": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
        },
        "asr": {
            "provider": asr_provider,
            "enabled": asr_provider == "faster-whisper",
            "available": FasterWhisperExtractor.available(),
        },
        "ocr": {
            "provider": ocr_provider,
            "enabled": ocr_provider == "tesseract",
            "available": TesseractOCRExtractor.available(),
        },
    }


def semantic_window_metrics(
    start: float,
    end: float,
    spans: list[TimedText],
) -> dict:
    duration = max(end - start, 0.001)
    relevant = [
        span
        for span in spans
        if min(end, span.end) - max(start, span.start) > 0
    ]
    if not relevant:
        return {
            "score": 0.0,
            "coverage": 0.0,
            "density": 0.0,
            "completeness": 0.0,
            "span_count": 0,
            "sources": [],
            "preview": "",
        }

    intervals = sorted(
        (max(start, span.start), min(end, span.end)) for span in relevant
    )
    merged: list[list[float]] = []
    for interval_start, interval_end in intervals:
        if not merged or interval_start > merged[-1][1]:
            merged.append([interval_start, interval_end])
        else:
            merged[-1][1] = max(merged[-1][1], interval_end)
    covered_seconds = sum(item_end - item_start for item_start, item_end in merged)
    coverage = min(covered_seconds / duration, 1.0)
    weighted_characters = sum(
        len(span.text.replace(" ", "")) * span.confidence for span in relevant
    )
    density = min(weighted_characters / max(duration * 3.0, 1.0), 1.0)
    complete = sum(
        span.start >= start - 0.25 and span.end <= end + 0.25
        for span in relevant
    )
    completeness = complete / len(relevant)
    score = 0.45 * coverage + 0.35 * density + 0.20 * completeness
    preview = " ".join(span.text for span in relevant)
    return {
        "score": round(score, 4),
        "coverage": round(coverage, 4),
        "density": round(density, 4),
        "completeness": round(completeness, 4),
        "span_count": len(relevant),
        "sources": sorted({span.source for span in relevant}),
        "preview": preview[:160],
    }
