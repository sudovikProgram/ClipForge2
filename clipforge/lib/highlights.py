"""Поиск и скоринг «интересных моментов».

Алгоритм:
 1. Из транскрипта строится последовательность фраз (сегментов Whisper).
 2. Скользящее окно из N фраз образует кандидатный клип [start,end].
    Границы «прилипают» к тишине (VAD), чтобы не резать слово пополам.
 3. Каждый кандидат получает оценку 0..1 по шести признакам:
      hook     — сила первой фразы (вопрос/интрига/хук-слова, «в лоб» без воды);
      energy   — эмоциональность (громкость + вариативность + темп речи);
      density  — информационная плотность (символов в секунду речи);
      arc      — завершённая мысль (есть «развязка»: маркеры вывода в конце);
      visual   — визуальная активность (движение + смена сцен);
      retention— отсутствие провалов (длинной тишины/монотонности внутри).
 4. Жадный отбор нетопчущихся клипов с лёгким «перегревом» начала:
    клип может начинаться на ~1.5с раньше первой фразы — это даёт breathing room.
"""
from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field, asdict

import numpy as np

from .transcribe import Segment, Word

log = logging.getLogger("clipforge")


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    words: list[Word]
    scores: dict = field(default_factory=dict)
    total: float = 0.0
    index: int = -1

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self):
        d = asdict(self)
        d["duration"] = round(self.duration, 2)
        return d


# -----------------------------------------------------------------------------
# Текстовые признаки
# -----------------------------------------------------------------------------
_Q_RE = re.compile(r"[?？]")
_EXCL_RE = re.compile(r"[!！]")
_NUM_RE = re.compile(r"\d[\d.,]*|процент|percent|млн|млрд|\bbillion\b|\bmillion\b")
_YOU_RE = re.compile(r"\b(you|your|ты|вы|твой|ваш|тебе|вам)\b", re.I)
_SECOND_PERSON_OPEN = re.compile(
    r"^\s*(you|do you|did you|are you|can you|tell me|imagine|what|why|how|"
    r"ты|вы|что|почему|как|представь|запомни|слушай|смотрите)", re.I)
_FILLER_RE = re.compile(
    r"^\s*(um+|uh+|э-э|ну|это|значит|как бы|типа|so|well|okay|ok|so yeah|"
    r"и так далее|в общем|короче)\b[\W]*", re.I | re.I)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def strip_leading_filler(text: str) -> str:
    """Убираем «эээ/ну/so» в начале — они убивают хук."""
    t = text
    prev = None
    while prev != t:
        prev = t
        t = _FILLER_RE.sub("", t).lstrip(" ,.-…")
    return t or text


def hook_score(first_text: str, cfg) -> float:
    """0..1: насколько первая фраза цепляет."""
    t = norm(first_text)
    if not t:
        return 0.0
    s = 0.0
    if _Q_RE.search(t):
        s += 0.30
    if _EXCL_RE.search(t):
        s += 0.12
    if _SECOND_PERSON_OPEN.match(t):
        s += 0.22
    elif _YOU_RE.search(t):
        s += 0.12
    hits = sum(1 for kw in cfg.hook_words if kw in t)
    s += min(0.34, hits * 0.12)
    n_words = len(t.split())
    # Короткая первая фраза = быстрый хук; слишком длинная = вода
    if 2 <= n_words <= 9:
        s += 0.18
    elif n_words <= 16:
        s += 0.08
    if _NUM_RE.search(t):
        s += 0.06          # конкретика/цифры усиливают
    return min(1.0, s)


def closing_score(last_text: str, cfg) -> float:
    """Есть ли в финале ощущение развязки, а не обрубленной фразы."""
    t = norm(last_text)
    s = 0.0
    if any(m in t for m in cfg.closing_markers):
        s += 0.4
    if t.endswith((".", "!", "?", "…")) or _EXCL_RE.search(t) or _Q_RE.search(t):
        s += 0.3
    if _YOU_RE.search(t):
        s += 0.1
    return min(1.0, s)


# -----------------------------------------------------------------------------
# Аудио/визуальные признаки
# -----------------------------------------------------------------------------
def energy_score(audio: dict, a: float, b: float) -> float:
    db = audio["db"]
    if db.size == 0:
        return 0.5
    i0 = max(0, int(a * 1000 / 20))
    i1 = min(len(db), int(b * 1000 / 20))
    if i1 - i0 < 5:
        return 0.3
    seg = db[i0:i1]
    loud_p95 = audio["loud_p95"]
    quiet = audio["quiet_p20"]
    span = max(6.0, loud_p95 - quiet)
    # уровень громкости относительно общего максимума файла
    lvl = float(np.percentile(seg, 85))
    rel = (lvl - quiet) / span                      # 0..~1
    dyn = float(np.std(seg))                        # выразительность
    dyn_n = min(1.0, dyn / 6.0)
    return float(np.clip(0.55 * rel + 0.45 * dyn_n, 0, 1))


def speech_ratio(audio: dict, a: float, b: float) -> float:
    m = audio["speech_mask"]
    if m.size == 0:
        return 0.0
    i0 = max(0, int(a * 1000 / 20))
    i1 = min(len(m), int(b * 1000 / 20))
    if i1 <= i0:
        return 0.0
    return float(m[i0:i1].mean())


def longest_gap_in_silence(audio: dict, a: float, b: float) -> float:
    """Самая длинная тишина внутри окна (сек) — индикатор провала удержания."""
    worst = 0.0
    for s, e in audio["silences"]:
        if e <= a or s >= b:
            continue
        worst = max(worst, min(e, b) - max(s, a))
    return worst


def visual_score2(times: np.ndarray, motion: np.ndarray, scenes: list[float],
                  a: float, b: float) -> float:
    m = 0.5
    if times.size:
        sel = motion[(times >= a) & (times <= b)]
        if sel.size:
            m = float(np.clip(np.percentile(sel, 75), 0, 1))
    ns = sum(1 for p in scenes if a <= p <= b)
    sc = min(1.0, ns / 3.0)
    return float(np.clip(0.65 * m + 0.35 * sc, 0, 1))


# -----------------------------------------------------------------------------
# Генерация кандидатов
# -----------------------------------------------------------------------------
def snap_to_silence(audio: dict, t: float, direction: str, pad: float) -> float:
    """Прилипание границы к ближайшей тишине (не дальше 2.5с)."""
    sil = audio["silences"]
    best, bestd = t, 2.5
    for s, e in sil:
        mid = (s + e) / 2
        if direction == "left" and mid <= t + 0.3:
            d = t - mid
        elif direction == "right" and mid >= t - 0.3:
            d = mid - t
        else:
            continue
        if d < bestd:
            best, bestd = mid, d
    if direction == "left":
        return max(0.0, best - pad)
    return best + pad


def build_candidates(segments: list[Segment], audio: dict, cfg,
                     times: np.ndarray, motion: np.ndarray,
                     scenes: list[float]) -> list[Candidate]:
    if not segments:
        return []
    cands: list[Candidate] = []
    n = len(segments)

    # Размер окна подбираем: сколько фраз влезает в target_clip_seconds
    for window in (2, 3, 4, 5, 6):
        for i in range(n - window + 1):
            grp = segments[i:i + window]
            raw_start = grp[0].start - 0.6          # запас до первой фразы
            raw_end = grp[-1].end + 0.35
            start = snap_to_silence(audio, raw_start, "left", cfg.silence_pad)
            end = snap_to_silence(audio, raw_end, "right", cfg.silence_pad)
            dur = end - start
            if dur < cfg.min_clip_seconds or dur > cfg.max_clip_seconds:
                continue
            # не допускаем склейку посреди слова: стартуем не позже начала слова
            first_word_start = min((w.start for s in grp for w in s.words),
                                   default=grp[0].start)
            if start > first_word_start - 0.1:
                start = max(raw_start, first_word_start - 0.45)
            last_word_end = max((w.end for s in grp for w in s.words),
                                default=grp[-1].end)
            if end < last_word_end + 0.1:
                end = raw_end + 0.3
            dur = end - start
            if not (cfg.min_clip_seconds <= dur <= cfg.max_clip_seconds):
                continue

            all_words = [w for s in grp for w in s.words] or \
                        [Word(tw, 0, 0) for tw in (" ".join(s.text for s in grp)).split()]
            text = " ".join(norm(s.text) for s in grp).strip()
            first_txt = strip_leading_filler(grp[0].text)
            if not first_txt:
                continue

            # --- признаки ---
            hook = hook_score(first_txt, cfg)
            ener = energy_score(audio, start, end)
            sr = speech_ratio(audio, start, end)
            chars = len(re.sub(r"\s", "", text))
            density = float(np.clip(chars / dur / 22.0, 0, 1))   # ~22 зн/с = отлично
            arc = closing_score(grp[-1].text, cfg)
            vis = visual_score2(times, motion, scenes, start, end)
            gap = longest_gap_in_silence(audio, start, end)
            retention = float(np.clip(1.0 - max(0.0, gap - 0.8) / 4.0, 0, 1))
            # речь должна занимать большую часть — иначе «пусто»
            retention *= np.clip(sr / 0.55, 0.35, 1.0)

            # штраф за сильное отклонение от целевой длины
            len_pen = 1.0 - min(0.5, abs(dur - cfg.target_clip_seconds) /
                                cfg.target_clip_seconds * 0.55)

            total = (cfg.w_hook * hook + cfg.w_energy * ener +
                     cfg.w_density * density + cfg.w_arc * arc +
                     cfg.w_visual * vis + cfg.w_retention * retention)
            total *= len_pen

            cands.append(Candidate(
                start=round(start, 3), end=round(end, 3),
                text=" ".join(s.text.strip() for s in grp),
                words=all_words,
                scores={"hook": round(hook, 3), "energy": round(ener, 3),
                        "density": round(density, 3), "arc": round(arc, 3),
                        "visual": round(vis, 3), "retention": round(float(retention), 3)},
                total=round(total, 4),
            ))

    log.info("Кандидатов построено: %d", len(cands))
    return cands


def deduplicate(cands: list[Candidate], iou_thresh: float = 0.45,
                keep_per_region: int = 1) -> list[Candidate]:
    """Жадный отбор: убираем сильно пересекающиеся кандидатов, лучшие первые."""
    ranked = sorted(cands, key=lambda c: c.total, reverse=True)
    chosen: list[Candidate] = []
    for c in ranked:
        clash = False
        for k in chosen:
            inter = max(0.0, min(c.end, k.end) - max(c.start, k.start))
            union = max(c.end, k.end) - min(c.start, k.start)
            if union > 0 and inter / union > iou_thresh:
                clash = True
                break
        if not clash:
            chosen.append(c)
    chosen.sort(key=lambda c: c.start)
    for idx, c in enumerate(chosen):
        c.index = idx
    return chosen


def select_clips(cands: list[Candidate], max_clips: int) -> list[Candidate]:
    """Топ-N по итоговому баллу (возвращаем в хронологическом порядке)."""
    top = sorted(cands, key=lambda c: c.total, reverse=True)[:max_clips]
    top.sort(key=lambda c: c.start)
    for idx, c in enumerate(top):
        c.index = idx
    return top
