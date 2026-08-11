from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ExportProfile:
    key: str
    label: str
    width: int
    height: int
    caption_margin_v: int
    font_size: int
    fixed_canvas: bool = True

    @property
    def aspect_ratio(self) -> str:
        divisor = _greatest_common_divisor(self.width, self.height)
        return f"{self.width // divisor}:{self.height // divisor}"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "caption_safe_margin_bottom": self.caption_margin_v,
        }


FIXED_PROFILES = {
    "landscape": ExportProfile(
        "landscape",
        "横屏 16:9",
        1280,
        720,
        72,
        36,
    ),
    "portrait": ExportProfile(
        "portrait",
        "竖屏 9:16",
        720,
        1280,
        220,
        56,
    ),
    "square": ExportProfile(
        "square",
        "方形 1:1",
        1080,
        1080,
        120,
        48,
    ),
}
SUPPORTED_OUTPUT_FORMATS = ("source", *FIXED_PROFILES)


def _greatest_common_divisor(left: int, right: int) -> int:
    while right:
        left, right = right, left % right
    return max(left, 1)


def _even(value: int) -> int:
    return max(2, int(value) // 2 * 2)


def resolve_export_profile(
    key: str,
    source_width: int,
    source_height: int,
) -> ExportProfile:
    if key in FIXED_PROFILES:
        return FIXED_PROFILES[key]
    if key != "source":
        raise ValueError(f"unsupported output format: {key}")
    width = _even(source_width)
    height = _even(source_height)
    return ExportProfile(
        "source",
        "保持原画幅",
        width,
        height,
        max(18, round(height * 0.09)),
        min(max(18, round(height * 0.05)), 64),
        fixed_canvas=False,
    )


def escape_filter_path(path: Path) -> str:
    return (
        path.resolve()
        .as_posix()
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def build_video_filter(
    profile: ExportProfile,
    subtitle_path: Path | None = None,
) -> str:
    width = profile.width
    height = profile.height
    if profile.fixed_canvas:
        graph = (
            f"[0:v]split=2[background][foreground];"
            f"[background]scale={width}:{height}:"
            "force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=24:2[blurred];"
            f"[foreground]scale={width}:{height}:"
            "force_original_aspect_ratio=decrease[sharp];"
            "[blurred][sharp]overlay=(W-w)/2:(H-h)/2,"
            "setsar=1[composed]"
        )
    else:
        graph = (
            f"[0:v]scale={width}:{height}:flags=lanczos,"
            "setsar=1[composed]"
        )

    if subtitle_path:
        escaped_path = escape_filter_path(subtitle_path)
        graph += f";[composed]ass=filename='{escaped_path}'[vout]"
    else:
        graph += ";[composed]null[vout]"
    return graph


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _span_value(span: object, name: str, default=None):
    if isinstance(span, dict):
        return span.get(name, default)
    return getattr(span, name, default)


def _caption_text(text: str, max_characters: int) -> str:
    cleaned = " ".join(str(text).replace("\n", " ").split())
    cleaned = (
        cleaned.replace("\\", "＼")
        .replace("{", "｛")
        .replace("}", "｝")
    )
    lines = textwrap.wrap(
        cleaned,
        width=max_characters,
        break_long_words=True,
        break_on_hyphens=False,
    )
    return r"\N".join(lines[:2])


def build_ass_subtitles(
    spans: Iterable[object],
    clip_start: float,
    clip_end: float,
    profile: ExportProfile,
) -> tuple[str, int]:
    max_characters = 16 if profile.key == "portrait" else 24
    events = []
    seen = set()
    for span in spans:
        source = str(_span_value(span, "source", ""))
        if source.startswith("ocr:"):
            continue
        start = max(float(_span_value(span, "start", 0.0)), clip_start)
        end = min(float(_span_value(span, "end", 0.0)), clip_end)
        text = _caption_text(
            str(_span_value(span, "text", "")),
            max_characters,
        )
        if end - start < 0.25 or not text:
            continue
        relative_start = start - clip_start
        relative_end = end - clip_start
        key = (round(relative_start, 2), round(relative_end, 2), text)
        if key in seen:
            continue
        seen.add(key)
        events.append(
            "Dialogue: 0,"
            f"{_ass_timestamp(relative_start)},"
            f"{_ass_timestamp(relative_end)},"
            f"Default,,0,0,0,,{text}"
        )

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {profile.width}
PlayResY: {profile.height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,{profile.font_size},&H00FFFFFF,&H000000FF,&H00101018,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,70,70,{profile.caption_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    body = "\n".join(events)
    return header + body + ("\n" if body else ""), len(events)
