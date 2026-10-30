"""Визуальный анализ: сцены (ffmpeg scdet), движение, лица.

Используется для:
  * smart-crop в вертикальный формат (центр интереса по кадрам);
  * оценки визуальной активности момента;
  * «визуальных» границ клипов (смена сцены = естественный монтажный стык).

Лица: opencv (если установлен) → mediapipe → без детектора (только движение).
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass

import numpy as np

from .utils import run

log = logging.getLogger("clipforge")


# -----------------------------------------------------------------------------
# Сцены через ffmpeg scdet
# -----------------------------------------------------------------------------
def detect_scenes(video: str, threshold: float = 15.0, max_points: int = 4000,
                  duration: float | None = None) -> list[float]:
    """Точки смены сцен в секундах через ffmpeg scdet.

    scdet работает со шкалой метаданных YDIF ~0..255; типовые пороги 10..30
    (меньше — чувствительнее). Значения 0..1 из старых конфигов автоматически
    конвертируются в шкалу scdet (thr*100), чтобы обратная совместимость не ломалась.

    Примечание: -t ДО -i ограничивает чтение входа — быстро на многочасовых файлах.
    """
    thr = threshold * 100.0 if 0 < threshold <= 1 else float(threshold)
    cmd = [
        os.environ.get("CLIPFORGE_FFMPEG", "ffmpeg"), "-hide_banner", "-loglevel", "info",
    ]
    if duration:
        cmd += ["-t", f"{duration:.2f}"]
    cmd += [
        "-i", video,
        "-vf", f"scdet=threshold={thr:.2f}", "-an", "-f", "null", "-",
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, errors="replace", timeout=7200)
        out = p.stderr
    except subprocess.TimeoutExpired:
        log.warning("scdet: таймаут — считаем видео одной сценой")
        return []
    # scdet печатает строки вида:
    #   [scdet @ ..] lavfi.scd.score: 23.4, lavfi.scd.time: 30
    # либо (другие сборки): ... pts_time:12.345
    pts = [float(m) for m in re.findall(r"lavfi\.scd\.time:\s*([0-9.]+)", out)]
    pts += [float(m) for m in re.findall(r"pts_time:([0-9.]+)", out)]
    pts = sorted(set(round(t, 2) for t in pts))
    if len(pts) > max_points:
        step = len(pts) // max_points + 1
        pts = pts[::step]
    log.info("Сцены: найдено %s точек смены (threshold=%.2f)", len(pts), threshold)
    return pts


# -----------------------------------------------------------------------------
# Движение: покадровые метрики через ffmpeg select+metadata
# -----------------------------------------------------------------------------
@dataclass
class MotionTrack:
    times: np.ndarray          # (N,) секунды
    motion: np.ndarray         # (N,) 0..1 нормированная межкадровая разница
    cx: np.ndarray             # (N,) горизонтальный центр масс движения 0..1
    cy: np.ndarray             # (N,) вертикальный центр масс движения 0..1
    width: int
    height: int

    def sample(self, t: float) -> tuple[float, float, float, float]:
        """(motion, cx, cy, ok) в момент t (nearest)."""
        if len(self.times) == 0:
            return 0.0, 0.5, 0.5, False
        i = int(np.searchsorted(self.times, t))
        i = min(max(i, 0), len(self.times) - 1)
        return float(self.motion[i]), float(self.cx[i]), float(self.cy[i]), True


def measure_motion(video: str, fps_probe: float = 2.0, max_frames: int = 8000,
                   analysis_w: int = 96) -> MotionTrack:
    """Оценка движения и «центра интереса» по малым покадровым данным.

    Метод: scale к 96px + tblend difference + signalstats (YDIF) даёт энергию
    движения; для центра интереса используем метаданные crop-detect на
    упрощённой модели: средневзвешенная позиция яркостных «активных» окон.
    Реализация максимально лёгкая, чтобы работать на многочасовом видео.
    """
    # Шаг 1: базовые метрики YDIF/YAVG по кадрам
    cmd = [
        os.environ.get("CLIPFORGE_FFMPEG", "ffmpeg"), "-hide_banner", "-loglevel", "info",
        "-i", video,
        "-vf", f"fps={fps_probe},scale={analysis_w}:-2,format=gray,"
               f"tblend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YDIF:file=-",
        "-an", "-f", "null", "-",
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, errors="replace", timeout=7200)
        raw = p.stdout
    except subprocess.TimeoutExpired:
        log.warning("motion: таймаут — движение не учтено")
        return MotionTrack(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), 0, 0)

    times: list[float] = []
    ydif: list[float] = []
    cur_t: float | None = None
    for line in raw.splitlines():
        m = re.match(r"frame:(\d+)\s+pts:(\d+)\s+pts_time:([0-9.]+)", line)
        if m:
            cur_t = float(m.group(3))
            continue
        m = re.search(r"lavfi\.signalstats\.YDIF=([0-9.]+)", line)
        if m and cur_t is not None:
            times.append(cur_t)
            ydif.append(float(m.group(1)))
            cur_t = None

    if not times:
        return MotionTrack(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), 0, 0)
    t_arr = np.asarray(times)
    d_arr = np.asarray(ydif)
    dmax = np.percentile(d_arr, 98) or 1.0
    motion = np.clip(d_arr / dmax, 0, 1)

    # Шаг 2: центр интереса по «bright pixels» (эвристика без ML):
    # берём кадры с высоким движением/яркостью и оцениваем centroid по tile-сетке.
    cx, cy = _interest_centroids(video, t_arr, fps_probe)

    n = min(len(t_arr), max_frames)
    if len(t_arr) > max_frames:
        idx = np.linspace(0, len(t_arr) - 1, max_frames).astype(int)
        t_arr, motion, cx, cy = t_arr[idx], motion[idx], cx[idx], cy[idx]

    # Проброс NaN → интерполяция
    good = ~np.isnan(cx)
    if good.sum() >= 2:
        cx = np.interp(t_arr, t_arr[good], cx[good])
        cy = np.interp(t_arr, t_arr[good], cy[good])
    else:
        cx = np.full_like(t_arr, 0.5)
        cy = np.full_like(t_arr, 0.5)

    return MotionTrack(t_arr, motion, cx, cy, 0, 0)


def _interest_centroids(video: str, times: np.ndarray, fps_probe: float):
    """Centroid яркого/контрастного содержимого через blackframe-free подход:
    scale tiny + cropdetect на динамическом пороге слишком медлителен, поэтому
    используем geq-нет: просто читаем малые кадры через pipe и считаем в numpy."""
    w = 64
    h = 64
    cmd = [
        os.environ.get("CLIPFORGE_FFMPEG", "ffmpeg"), "-hide_banner", "-loglevel", "error",
        "-i", video,
        "-vf", f"fps={fps_probe},scale={w}:{h},format=gray",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           timeout=7200)
        buf = np.frombuffer(p.stdout, dtype=np.uint8)
    except Exception as e:
        log.debug("centroid недоступен: %s", e)
        return np.full(len(times), np.nan), np.full(len(times), np.nan)
    frames = buf.reshape((-1, h, w)).astype(np.float32) / 255.0
    n = min(len(frames), len(times))
    cx = np.full(len(times), np.nan)
    cy = np.full(len(times), np.nan)
    gx = np.linspace(0, 1, w, dtype=np.float32)
    gy = np.linspace(0, 1, h, dtype=np.float32)
    for i in range(n):
        img = frames[i]
        # вес = локальный контраст (отклонение от среднего) + яркость
        wgt = np.abs(img - img.mean()) * 0.7 + img * 0.3
        s = wgt.sum()
        if s <= 1e-6:
            cx[i], cy[i] = 0.5, 0.5
            continue
        cx[i] = float((wgt.sum(axis=0) * gx).sum() / s)
        cy[i] = float((wgt.sum(axis=1) * gy).sum() / s)
    return cx, cy


# -----------------------------------------------------------------------------
# Лица (опционально)
# -----------------------------------------------------------------------------
class FaceDetector:
    """Портретный детектор с деградацией: cv2 → mediapipe → None."""

    def __init__(self):
        self.backend = None
        self._cascade = None
        try:
            import cv2
            c = cv2.CascadeClassifier(cv2.data.haarcascades +
                                       "haarcascade_frontalface_default.xml")
            if not c.empty():
                self.backend = "cv2"
                self._cascade = c
                self._cv2 = cv2
                return
        except Exception:
            pass
        try:
            from mediapipe.face_detection import FaceDetection  # type: ignore
            self.backend = "mediapipe"
            self._md = FaceDetection.from_model_name("short_range") \
                if hasattr(FaceDetection, "from_model_name") else None
        except Exception:
            pass

    def faces(self, frame_gray: np.ndarray) -> list[tuple[float, float, float]]:
        """Список лиц (cx, cy, size) в нормированных координатах 0..1."""
        if self.backend == "cv2":
            rects = self._cascade.detectMultiScale(
                frame_gray, scaleFactor=1.15, minNeighbors=5,
                minSize=(int(frame_gray.shape[0] * 0.08),) * 2)
            H, W = frame_gray.shape[:2]
            out = []
            for x, y, w, h in rects:
                out.append(((x + w / 2) / W, (y + h / 2) / H, max(w, h) / max(W, H)))
            return out
        return []

    @property
    def available(self) -> bool:
        return self.backend is not None


def face_track(video: str, motion: MotionTrack, fps_probe: float = 2.0,
               max_samples: int = 600, src_w: int = 0, src_h: int = 0) -> np.ndarray | None:
    """Покадровый трек лиц: time, fx, fy, fw (нормированные координаты).

    src_w/src_h — размеры исходного видео (для точного расчёта height в pipe).
    Возвращает (N,4) или None, если детектор недоступен/лиц нет.
    """
    det = FaceDetector()
    if not det.available:
        log.info("Детектор лиц недоступен — smart-crop по движению/яркости.")
        return None
    if src_w <= 0 or src_h <= 0:
        log.info("Неизвестны размеры видео — пропускаем детекцию лиц.")
        return None
    w = 320
    h = max(2, int(round(w * src_h / src_w / 2)) * 2)
    cmd = [
        os.environ.get("CLIPFORGE_FFMPEG", "ffmpeg"), "-hide_banner", "-loglevel", "error",
        "-i", video,
        "-vf", f"fps={fps_probe},scale={w}:{h},format=gray",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           timeout=7200)
        buf = np.frombuffer(p.stdout, dtype=np.uint8)
    except Exception:
        return None
    frame_px = w * h
    if len(buf) < frame_px:
        return None
    est_frames = len(buf) // frame_px
    frames = buf[: est_frames * frame_px].reshape(est_frames, h, w)
    step = max(1, est_frames // max_samples)
    rows = []
    for i in range(0, est_frames, step):
        fs = det.faces(frames[i])
        t = i / fps_probe
        if fs:
            # самое большое лицо = главный герой
            fx, fy, fw = max(fs, key=lambda a: a[2])
            rows.append((t, fx, fy, fw))
    if not rows:
        log.info("Лица не обнаружены — кадрирование по контенту.")
        return None
    log.info("Лица: %s сэмплов с детекцией", len(rows))
    return np.asarray(rows, dtype=np.float32)
