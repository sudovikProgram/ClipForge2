"""Аудио-анализ: тишина/речь (VAD), громкость, энергия по времени.

Всё считается из WAV 16 кГц, без тяжёлых зависимостей (numpy обязателен).
Используется для:
  * поиска «словесных» границ клипов (стык на тишине, а не посреди слова);
  * оценки эмоциональной энергии фрагментов;
  * генерации sidechain-маски для дакинга музыки под речь.
"""
from __future__ import annotations

import logging
import wave

import numpy as np

log = logging.getLogger("clipforge")

FRAME_MS = 20.0


def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        raise RuntimeError(f"Ожидается 16-битный WAV, получен {sw*8}-бит: {path}")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def frame_energy(samples: np.ndarray, sr: int, frame_ms: float = FRAME_MS) -> np.ndarray:
    """RMS-энергия по кадрам (дБ)."""
    flen = max(1, int(sr * frame_ms / 1000.0))
    n_frames = len(samples) // flen
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)
    frames = samples[: n_frames * flen].reshape(n_frames, flen)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
    db = 20.0 * np.log10(rms)
    return db


def speech_mask(db: np.ndarray, *, floor_db: float = -45.0,
                offset_db: float = 12.0) -> np.ndarray:
    """Адаптивный детектор речи: кадр «голосовой», если громче (шумовой пол - offset).

    floor_db — абсолютный порог тишины; offset_db — над шумовым полом.
    """
    if db.size == 0:
        return np.zeros(0, dtype=bool)
    noise_floor = np.percentile(db, 20)
    thr = max(noise_floor + offset_db, floor_db)
    mask = db > thr
    # Морфология: склеиваем короткие разрывы (<120мс), убираем вспышки (<80мс)
    mask = _close_gaps(mask, int(120 / FRAME_MS))
    mask = _drop_short(mask, int(80 / FRAME_MS))
    return mask


def _close_gaps(mask: np.ndarray, frames: int) -> np.ndarray:
    m = mask.copy()
    i = 0
    n = len(m)
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if j < n and j - i <= frames and i > 0:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


def _drop_short(mask: np.ndarray, frames: int) -> np.ndarray:
    m = mask.copy()
    i = 0
    n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if j - i < frames:
                m[i:j] = False
            i = j
        else:
            i += 1
    return m


def silence_intervals(mask: np.ndarray, frame_ms: float = FRAME_MS,
                      min_silence: float = 0.25) -> list[tuple[float, float]]:
    """Список интервалов тишины [start,end] в секундах длиной >= min_silence."""
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            s, e = i * frame_ms / 1000.0, j * frame_ms / 1000.0
            if e - s >= min_silence:
                out.append((s, e))
            i = j
        else:
            i += 1
    return out


def find_cut_point(silences: list[tuple[float, float]], after_t: float,
                   before_t: float | None = None) -> float:
    """Ближайшая точка «чистого» монтажа после after_t (центр тишины).

    Если тишины нет — возвращает after_t (монтируем по слову, это приемлемо).
    """
    best = None
    for s, e in silences:
        mid = (s + e) / 2.0
        if mid >= after_t - 0.05 and (before_t is None or mid <= before_t):
            best = mid
            break
    return best if best is not None else after_t


def energy_curve(db: np.ndarray, window_s: float = 2.0,
                 frame_ms: float = FRAME_MS) -> np.ndarray:
    """Сглаженная кривая громкости (для графика и скоринга)."""
    if db.size == 0:
        return db
    win = max(1, int(window_s * 1000 / frame_ms))
    kernel = np.ones(win) / win
    return np.convolve(db, kernel, mode="same")


def analyze(wav_path: str) -> dict:
    """Полный аудио-профиль файла.

    Возвращает dict:
      sr, duration, db (покадрово), speech_mask, silences, energy,
      loud_p95, quiet_p20 — нормировочные константы.
    """
    samples, sr = read_wav_mono(wav_path)
    db = frame_energy(samples, sr)
    mask = speech_mask(db)
    silences = silence_intervals(mask)
    energy = energy_curve(db)
    loud = float(np.percentile(db, 95)) if db.size else -45.0
    quiet = float(np.percentile(db, 20)) if db.size else -60.0
    dur = len(samples) / sr
    log.info("Аудио: %.1fс, речь %.0f%%, silence-gaps %d",
             dur, 100 * mask.mean(), len(silences))
    return {
        "sr": sr, "duration": dur, "db": db, "speech_mask": mask,
        "silences": silences, "energy": energy,
        "loud_p95": loud, "quiet_p20": quiet,
    }
