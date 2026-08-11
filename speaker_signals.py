from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str
    confidence: float = 1.0
    source: str = "diarization:pyannote"

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker,
            "confidence": round(self.confidence, 4),
            "source": self.source,
        }


@dataclass(frozen=True)
class SpeakerExtraction:
    turns: list[SpeakerTurn]
    providers: dict[str, str]
    warnings: list[str]


class PyannoteDiarizationExtractor:
    def __init__(
        self,
        *,
        model_name: str | None = None,
        token: str | None = None,
        device: str | None = None,
        pipeline_factory=None,
    ):
        self.model_name = (
            model_name
            or os.getenv("CUTPOINT_PYANNOTE_MODEL")
            or DEFAULT_PYANNOTE_MODEL
        )
        self.token = token or os.getenv("CUTPOINT_HF_TOKEN") or os.getenv(
            "HF_TOKEN"
        )
        self.device = (
            device
            or os.getenv("CUTPOINT_DIARIZATION_DEVICE")
            or "auto"
        )
        self.pipeline_factory = pipeline_factory

    @staticmethod
    def dependency_available() -> bool:
        if importlib.util.find_spec("pyannote") is None:
            return False
        return importlib.util.find_spec("pyannote.audio") is not None

    def available(self) -> bool:
        return bool(
            self.pipeline_factory
            or (self.dependency_available() and self.token)
        )

    def _load_pipeline(self):
        if self.pipeline_factory:
            return self.pipeline_factory(self.model_name, self.token)
        if not self.dependency_available():
            raise RuntimeError("未安装 pyannote.audio")
        if not self.token:
            raise RuntimeError("未配置 CUTPOINT_HF_TOKEN")

        from pyannote.audio import Pipeline

        try:
            pipeline = Pipeline.from_pretrained(
                self.model_name,
                token=self.token,
            )
        except TypeError:
            pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.token,
            )
        if self.device != "auto":
            import torch

            pipeline.to(torch.device(self.device))
        return pipeline

    def extract(
        self,
        video_path: Path,
        duration: float,
    ) -> list[SpeakerTurn]:
        pipeline = self._load_pipeline()
        output = pipeline(str(video_path))
        annotation = getattr(output, "speaker_diarization", output)
        raw_turns = []
        for segment, _track, speaker in annotation.itertracks(
            yield_label=True
        ):
            start = max(0.0, float(segment.start))
            end = min(float(duration), float(segment.end))
            if end - start < 0.1:
                continue
            raw_turns.append(
                SpeakerTurn(
                    start,
                    end,
                    str(speaker),
                )
            )

        merged: list[SpeakerTurn] = []
        for turn in sorted(raw_turns, key=lambda item: (item.start, item.end)):
            if (
                merged
                and merged[-1].speaker == turn.speaker
                and turn.start - merged[-1].end <= 0.25
            ):
                previous = merged[-1]
                merged[-1] = SpeakerTurn(
                    previous.start,
                    max(previous.end, turn.end),
                    previous.speaker,
                    min(previous.confidence, turn.confidence),
                    previous.source,
                )
            else:
                merged.append(turn)
        return merged


def extract_speaker_signals(
    video_path: Path,
    duration: float,
) -> SpeakerExtraction:
    provider = os.getenv(
        "CUTPOINT_DIARIZATION_PROVIDER",
        "off",
    ).strip().lower()
    if provider in {"", "off", "none"}:
        return SpeakerExtraction([], {"diarization": "off"}, [])
    if provider != "pyannote":
        return SpeakerExtraction(
            [],
            {"diarization": "unsupported"},
            [f"不支持的说话人提供器：{provider}"],
        )

    extractor = PyannoteDiarizationExtractor()
    if not extractor.available():
        reasons = []
        if not extractor.dependency_available():
            reasons.append("未安装 pyannote.audio")
        if not extractor.token:
            reasons.append("未配置 CUTPOINT_HF_TOKEN")
        return SpeakerExtraction(
            [],
            {"diarization": "unavailable"},
            ["说话人分离未启用：" + "；".join(reasons)],
        )
    try:
        turns = extractor.extract(video_path, duration)
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        return SpeakerExtraction(
            [],
            {"diarization": "failed"},
            [f"说话人分离失败：{message[:220]}"],
        )
    return SpeakerExtraction(
        turns,
        {"diarization": "active" if turns else "empty"},
        [],
    )


def speaker_provider_status() -> dict:
    provider = os.getenv(
        "CUTPOINT_DIARIZATION_PROVIDER",
        "off",
    ).strip().lower()
    extractor = PyannoteDiarizationExtractor()
    enabled = provider == "pyannote"
    return {
        "provider": provider or "off",
        "enabled": enabled,
        "available": bool(enabled and extractor.available()),
        "dependency_available": extractor.dependency_available(),
        "token_configured": bool(extractor.token),
        "model": extractor.model_name if enabled else None,
    }


def _safe_boundary(
    timestamp: float,
    turns: list[SpeakerTurn],
    tolerance: float = 0.35,
) -> float:
    for turn in turns:
        if turn.start + tolerance < timestamp < turn.end - tolerance:
            return 0.0
    return 1.0


def speaker_window_metrics(
    start: float,
    end: float,
    turns: list[SpeakerTurn],
) -> dict:
    overlapping = [
        turn
        for turn in turns
        if min(end, turn.end) > max(start, turn.start)
    ]
    overlap_duration = sum(
        max(0.0, min(end, turn.end) - max(start, turn.start))
        for turn in overlapping
    )
    duration = max(end - start, 0.001)
    start_safe = _safe_boundary(start, turns)
    end_safe = _safe_boundary(end, turns)
    return {
        "available": bool(turns),
        "boundary_score": round((start_safe + end_safe) / 2.0, 4),
        "start_safe": bool(start_safe),
        "end_safe": bool(end_safe),
        "turn_count": len(overlapping),
        "speaker_count": len({turn.speaker for turn in overlapping}),
        "coverage": round(min(overlap_duration / duration, 1.0), 4),
    }
