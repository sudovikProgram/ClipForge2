"""Smart-crop: горизонтально-динамичный вертикальный кадр 9:16 из исходника.

Кадрирование меняется ТОЛЬКО по горизонтали (ширина crop фиксирована =
H_src*9/16, высота = H_src). Это исключает «прыжки» по вертикали и даёт
плавное панорамирование — стандартный приём face-tracking шортсов.

Траектория x(t) считается Python'ом по трекам лиц/движения, сглаживается EMA
и ограничивается скоростью поворота; затем экспортируется в expression-файл
для ffmpeg `crop=x='expr'`.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("clipforge")


def _ema(vals: np.ndarray, alpha: float) -> np.ndarray:
    out = vals.copy()
    for i in range(1, len(out)):
        out[i] = alpha * vals[i] + (1 - alpha) * out[i - 1]
    return out


def _limit_speed(vals: np.ndarray, max_step: float) -> np.ndarray:
    out = vals.copy()
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        if abs(d) > max_step:
            out[i] = out[i - 1] + np.sign(d) * max_step
    return out


def compute_crop_path(src_w: int, src_h: int, clip_start: float, clip_end: float,
                      fps: float, *, motion_times: np.ndarray | None = None,
                      motion_cx: np.ndarray | None = None,
                      motion_cy: np.ndarray | None = None,
                      faces: np.ndarray | None = None,
                      face_weight: float = 1.6, motion_weight: float = 0.6,
                      smooth: float = 0.06, mode: str = "auto") -> str | None:
    """Возвращает expression-строку для параметра x фильтра crop.

    None → использовать статичный центр (x='(iw-ow)/2').
    """
    if mode == "center":
        return None
    crop_w = src_h * 9 / 16.0
    if crop_w >= src_w:
        # Источник уже «уже», чем 9:16 — горизонтального люфта нет
        return None
    max_x = src_w - crop_w
    n = max(2, int(round((clip_end - clip_start) * fps)))
    ts = np.linspace(clip_start, clip_end, n)

    targets = np.full(n, np.nan)

    if faces is not None and len(faces):
        ft, fx = faces[:, 0], faces[:, 1]
        sel = (ft >= clip_start - 1) & (ft <= clip_end + 1)
        if sel.sum() >= 2:
            f_interp = np.interp(ts, ft[sel], fx[sel])
            # лицо держим чуть выше центра кадра по горизонтали — ок
            targets = f_interp * src_w

    if motion_times is not None and motion_cx is not None and len(motion_times):
        sel = (motion_times >= clip_start - 1) & (motion_times <= clip_end + 1)
        if sel.sum() >= 2:
            m_interp = np.interp(ts, motion_times[sel], motion_cx[sel]) * src_w
            if np.isnan(targets).all():
                targets = m_interp
            else:
                good = ~np.isnan(targets)
                blended = np.where(
                    good,
                    (face_weight * np.nan_to_num(targets) +
                     motion_weight * m_interp) / (face_weight + motion_weight),
                    m_interp)
                targets = blended

    if np.isnan(targets).all():
        return None
    # интерполяция пропусков (нет детекции в части окон)
    good = ~np.isnan(targets)
    targets = np.interp(ts, ts[good], targets[good])

    # смешиваем с центром, чтобы не уезжать на самый край
    center = src_w / 2.0
    pull = 0.85                                   # насколько доверяем треку
    targets = center + (targets - center) * pull

    # Ограничение: лицо не должно прижиматься к краям вертикального окна.
    # Если цель выходит за «мёртвую зону» — сдвигаем окно так, чтобы цель
    # была на margin от края; иначе центрируем окно на цели.
    margin = crop_w * 0.18
    xs = np.where(
        targets < margin,
        0.0,                                   # упираемся в левый край
        np.where(targets > src_w - margin,
                 max_x,                        # упираемся в правый край
                 targets - crop_w / 2.0))      # центр на цели
    xs = np.clip(xs, 0, max_x)

    xs = _ema(xs, smooth)
    # плавность движения: максимум ~5% ширины кадра за кадр @30fps
    max_step = crop_w * 0.05
    xs = _limit_speed(xs, max_step)
    xs = np.clip(xs, 0, max_x)

    # Ключевые точки: адаптивное прореживание (не больше ~90 точек, монотонные времена)
    step_pts = max(1, int(fps / 4))
    while n // step_pts > 90:
        step_pts += 1
    kp_t = list(ts[::step_pts])
    kp_x = list(xs[::step_pts])
    # гарантия строгой монотонности времён ключей
    keep_t, keep_x = [kp_t[0]], [kp_x[0]]
    for t_, x_ in zip(kp_t[1:], kp_x[1:]):
        if t_ > keep_t[-1] + 0.05:
            keep_t.append(t_); keep_x.append(x_)
    kp_t, kp_x = keep_t, keep_x
    if abs(kp_x[-1] - xs[-1]) > 1.0:
        kp_t.append(ts[-1])
        kp_x.append(xs[-1])

    # Строим вложенные if(lt(T_abs,t_k), lerp(x_{k-1}->x_k, frac), ...)
    # T здесь — абсолютное время исходника (мы режем через -ss/-to ДО фильтра?
    # Нет: trim внутри графа, поэтому T остаётся абсолютным).
    def lerp(i: int) -> str:
        t0, t1 = kp_t[i - 1], kp_t[i]
        x0, x1 = kp_x[i - 1], kp_x[i]
        span = max(1e-3, t1 - t0)
        return (f"({x0:.1f}+({x1:.1f}-{x0:.1f})"
                f"*clip((T-{t0:.3f})/{span:.3f},0,1))")

    expr = f"{kp_x[0]:.1f}"
    for i in range(1, len(kp_t)):
        expr = f"if(lt(T,{kp_t[i]:.3f}),{lerp(i)},{expr})"
    expr = f"if(lt(T,{kp_t[0]:.3f}),{kp_x[0]:.1f},{expr})"
    if expr.count("(") != expr.count(")") or len(expr) > 20000:
        log.warning("crop path: слишком сложное выражение — упрощаем до 2 точек")
        if len(kp_t) >= 2:
            x0, x1 = kp_x[0], kp_x[-1]
            t0, t1 = kp_t[0], kp_t[-1]
            span = max(1e-3, t1 - t0)
            expr = (f"if(lt(T,{t0:.3f}),{x0:.1f},"
                    f"if(lt(T,{t1:.3f}),({x0:.1f}+({x1:.1f}-{x0:.1f})"
                    f"*clip((T-{t0:.3f})/{span:.3f},0,1)),{x1:.1f}))")
        else:
            return None
    return expr
