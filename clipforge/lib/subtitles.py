"""ASS-субтитры с покадровой подсветкой активного слова (karaoke-style).

Принцип:
  * текст разбивается на «строки» по N слов;
  * каждая строка — отдельное событие Dialogue с полным набором слов;
  * каждое слово получает \\transition-блок {\kf ms}col1,col2:word col3\col1,
    что даёт плавную «прокраску» слова в момент его произнесения;
  * тайминги слов берутся из Whisper word-timestamps.

Формат ASS требует \n для переноса строк внутри события — экранируем аккуратно.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .transcribe import Word

log = logging.getLogger("clipforge")


@dataclass
class Line:
    start: float
    end: float
    words: list[Word]


def ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def group_lines(words: list[Word], max_words: int = 4,
                gap_break: float = 0.6) -> list[Line]:
    """Группировка слов в карусельные строки субтитров."""
    lines: list[Line] = []
    cur: list[Word] = []
    for w in words:
        if cur and (w.start - cur[-1].end > gap_break or len(cur) >= max_words):
            lines.append(Line(cur[0].start, cur[-1].end, cur))
            cur = []
        cur.append(w)
    if cur:
        lines.append(Line(cur[0].start, cur[-1].end, cur))
    # склейка слишком коротких строк (<0.7с) со следующей — иначе мельтешение
    merged: list[Line] = []
    for ln in lines:
        if merged and (ln.end - ln.start < 0.7 or ln.end - merged[-1].start > 3.4):
            prev = merged[-1]
            if ln.end - ln.start < 0.7:
                merged[-1] = Line(prev.start, max(prev.end, ln.end),
                                  prev.words + ln.words)
                continue
        merged.append(ln)
    return merged


PUNCT_RE = re.compile(r"^[\s\.,!?:;\-–—«»\"'()\[\]…]+|[\s\.,!?:;\-–—«»\"'()\[\]…]+$")


def clean_word(t: str) -> str:
    return PUNCT_RE.sub("", t) or t


def escape_ass(s: str) -> str:
    return (s.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
             .replace("\r", " ").replace("\n", " "))


def build_karaoke_events(lines: list[Line], cfg) -> list[str]:
    events: list[str] = []
    base, acc = cfg.base_color, cfg.accent_color
    for ln in lines:
        parts = []
        prev_end = ln.start
        for w in ln.words:
            txt = escape_ass(clean_word(w.text))
            if not txt:
                continue
            # межсловная пауза — обычным цветом
            gap_ms = int((w.start - prev_end) * 1000)
            if gap_ms > 40:
                parts.append(f"{{\\k{gap_ms // 10}}} ")
            dur_ms = max(80, int((w.end - w.start) * 1000))
            if cfg.sub_highlight_active:
                parts.append(f"{{\\kf{dur_ms}\\c{acc}}}{txt}")
            else:
                parts.append(f"{{\\k{dur_ms}}}{txt}")
            prev_end = w.end
        if not parts:
            continue
        text = "".join(parts) + f"{{\\N\\c{base}}}"
        st, en = ass_time(ln.start), ass_time(max(ln.end, ln.start + 0.3))
        events.append(f"Dialogue: 0,{st},{en},Default,,0,0,0,,{text}")
    return events


HEADER_TMPL = """[Script Info]
; ClipForge karaoke subtitles
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00101010,&H7F000000,-1,0,0,0,100,100,0.5,0,1,{outline},2.5,2,60,60,{marginv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


HOOK_STYLE_TMPL = """Style: Hook,{font},{hook_size},&H00FFFFFF,&H000000FF,&H00000000,&HE0000000,-1,0,0,0,100,100,1.5,0,1,6,3,8,40,40,{hook_marginv},1
"""


def write_ass(path: str, lines: list[Line], cfg, hook_text: str | None = None,
              clip_start: float = 0.0) -> None:
    """Пишет .ass файл. Тайминги уже абсолютные (время исходника) — при
    рендере мы сдвигаем их на -clip_start через \\an и shift в вызове ffmpeg?
    Нет: проще сдвинуть здесь — все клипы рендерятся с обнулённым PTS."""
    header = HEADER_TMPL.format(
        w=cfg.out_w, h=cfg.out_h, font=cfg.sub_font, size=cfg.sub_font_size,
        outline=cfg.sub_outline, marginv=cfg.sub_margin_v)
    if hook_text:
        header += HOOK_STYLE_TMPL.format(
            font=cfg.sub_font, hook_size=int(cfg.sub_font_size * 1.15),
            hook_marginv=int(cfg.out_h * 0.16))
    events: list[str] = []
    if hook_text and cfg.hook_overlay:
        ht = escape_ass(hook_text)
        st = ass_time(max(0.0, lines[0].start - clip_start - 0.35)
                      if lines else 0.0)
        en = ass_time(float(cfg.hook_overlay_seconds))
        # анимация: въезд снизу + лёгкий pop
        anim = ("{\\fad(120,220)\\t(0,180,\\fscx112\\fscy112)"
                "\\t(180,320,\\fscx100\\fscy100)}")
        events.append(f"Dialogue: 1,{st},{en},Hook,,0,0,0,,{anim}{ht}\\N")
    for ln in lines:
        shifted = Line(max(0.0, ln.start - clip_start),
                       max(0.1, ln.end - clip_start), ln.words)
        evs = build_karaoke_events([shifted], cfg)
        events.extend(evs)
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(header + "\n".join(events) + "\n")
