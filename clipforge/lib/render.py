"""Рендер финального вертикального ролика: ffmpeg-граф за один проход.

Делает:
  * точный cut клипа (-ss/-to до входа — кадр-точное перекодирование);
  * smart-crop в 9:16 (горизонтальное панорамирование по лицам/движению) + scale;
  * лёгкая «соцсетевая» цветокоррекция + резкость;
  * ASS-субтитры с karaoke-подсветкой + хук-оверлей;
  * прогресс-бар удержания внизу кадра;
  * аудио: шумоподавление → усиление голоса → R128 loudnorm (-14 LUFS) → fade-in;
  * опциональная фоновая музыка с дакингом под речь (sidechaincompress);
  * +faststart для мгновенного стриминга при загрузке на YouTube/TikTok.
"""
from __future__ import annotations

import logging
import os

from .utils import run

log = logging.getLogger("clipforge")


def _ffmpeg_bin() -> str:
    return os.environ.get("CLIPFORGE_FFMPEG", "ffmpeg")


# -----------------------------------------------------------------------------
# Видео-цепочка (применяется к потоку [v])
# -----------------------------------------------------------------------------
def build_video_filterchain(*, src_w: int, src_h: int, cfg,
                            crop_expr: str | None, ass_path: str | None,
                            duration: float) -> str:
    parts: list[str] = []

    # 1. Кадрирование 9:16
    if crop_expr:
        cw = int(src_h * 9 / 16) // 2 * 2
        ch = src_h // 2 * 2
        expr = crop_expr.replace("'", "")
        parts.append(f"crop=w={cw}:h={ch}:x='{expr}':y=0")
    else:
        # центральный статичный кроп (для вертикальных источников — без изменений)
        parts.append("crop=w='min(iw,ih*9/16)':h='min(ih,iw*16/9)'")

    parts.append(f"scale={cfg.out_w}:{cfg.out_h}:flags=lanczos")
    parts.append("setsar=1")

    # 2. Цвет: чуть контрастнее/насыщеннее — стандарт восприятия ленты
    parts.append("eq=contrast=1.06:saturation=1.18:brightness=0.012")
    parts.append("unsharp=5:5:0.5:5:5:0.0")

    # 3. Прогресс-бар (поверх видео, ниже субтитров не мешает)
    if cfg.progress_bar:
        bar_y = cfg.out_h - 12
        parts.append(
            f"drawbox=x=0:y={bar_y}:w=iw:h=6:color=white@0.18:t=fill,"
            f"drawbox=x=0:y={bar_y}:w='iw*t/{duration:.2f}':h=6:color=red@0.9:t=fill"
        )

    # 4. Субтитры и хук-оверлей (libass)
    if cfg.subtitles and ass_path:
        p = os.path.abspath(ass_path).replace("\\", "/").replace(":", "\\:")
        parts.append(f"subtitles=filename='{p}'")

    # 5. Отладочные safe-зоны UI TikTok/Shorts
    if cfg.safe_zones:
        parts.append(
            f"drawbox=x=0:y={cfg.out_h - 420}:w=iw:h=420:color=green@0.15:t=fill,"
            f"drawbox=x=0:y=0:w=iw:h=220:color=blue@0.12:t=fill")

    return ",".join(parts)


# -----------------------------------------------------------------------------
# Аудио-цепочки
# -----------------------------------------------------------------------------
def build_voice_chain(cfg) -> str:
    a: list[str] = []
    if cfg.enable_denoise:
        a.append("afftdn=nr=10:nf=-32:tn=1")
    a.append(f"volume={cfg.voice_boost_db}dB")
    a.append(f"loudnorm=I={cfg.loudnorm_target_i}:TP=-1.5:LRA=11")
    a.append("afade=t=in:st=0:d=0.10")
    return ",".join(a)


def build_music_chain(cfg, speech_windows: list[tuple[float, float]],
                      duration: float) -> str:
    """Музыка: низкие частоты убраны (освободаем место голосу), дакинг по
    интервалам речи через volume-автоматизацию + мягкий фейд аутта."""
    a = ["highpass=f=120", "lowpass=f=9500"]
    gain_lin = 10 ** (cfg.music_gain_db / 20.0)
    duck_lin = 10 ** ((cfg.music_gain_db + cfg.duck_music_db) / 20.0)
    conds = "+".join(f"between(t,{max(0.0, s):.2f},{min(duration, e):.2f})"
                     for s, e in speech_windows) or "0"
    expr = f"{gain_lin:.4f}-({gain_lin - duck_lin:.4f})*({conds})"
    a.append(f"volume=volume='{expr}':eval=frame")
    a.append(f"afade=t=in:st=0:d=0.4,afade=t=out:st={max(0.0, duration - 0.7):.2f}:d=0.7")
    return ",".join(a)


# -----------------------------------------------------------------------------
# Основной рендер
# -----------------------------------------------------------------------------
def render_clip(video: str, out_path: str, *, start: float, end: float,
                src_w: int, src_h: int, cfg,
                crop_expr: str | None, ass_path: str | None,
                music_track: str | None = None,
                speech_times: list[tuple[float, float]] | None = None,
                title: str | None = None,
                threads: int = 0) -> None:
    """Рендер одного вертикального клипа одним ffmpeg-проходом."""
    duration = end - start
    vf = build_video_filterchain(src_w=src_w, src_h=src_h, cfg=cfg,
                                 crop_expr=crop_expr, ass_path=ass_path,
                                 duration=duration)

    cmd = [_ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", video]

    has_music = bool(music_track and os.path.exists(music_track))
    voice = build_voice_chain(cfg)

    if has_music:
        win = [(s - start, e - start) for s, e in (speech_times or [])]
        music = build_music_chain(cfg, win, duration)
        graph = (
            f"[0:v]{vf}[vout];"
            f"[0:a]{voice}[a0];"
            f"[1:a]{music}[am];"
            f"[am][a0]sidechaincompress=threshold=0.05:ratio=6:attack=20:"
            f"release=320:makeup=1[amus];"
            f"[a0][amus]amix=inputs=2:duration=first:dropout_transition=0:"
            f"normalize=0[aout]"
        )
        cmd += ["-stream_loop", "-1", "-i", music_track]
    else:
        graph = f"[0:v]{vf}[vout];[0:a]{voice}[aout]"

    cmd += [
        "-filter_complex", graph,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.video_crf),
        "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", str(cfg.fps),
        "-g", str(cfg.fps * 2),
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
    ]
    if title:
        cmd += ["-metadata", f"title={title}"]
    cmd += ["-metadata", "comment=ClipForge auto-shorts"]
    if threads:
        cmd += ["-threads", str(threads)]
    cmd += ["-t", f"{duration:.3f}", out_path]

    log.debug("ffmpeg: %s", " ".join(cmd))
    try:
        run(cmd, timeout=7200)
    except RuntimeError as e:
        if not has_music:
            raise
        # Fallback: без sidechain (некоторые сборки ffmpeg его не имеют)
        log.warning("рендер со звуковой цепью упал (%s) — упрощаем audio-graph", e)
        graph = (
            f"[0:v]{vf}[vout];"
            f"[0:a]{voice}[a0];"
            f"[1:a]{build_music_chain(cfg, [(s - start, e - start) for s, e in (speech_times or [])], duration)}[am];"
            f"[a0][am]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
        cmd2 = [c for c in cmd]
        i = cmd2.index("-filter_complex")
        cmd2[i + 1] = graph
        run(cmd2, timeout=7200)

    size_mb = os.path.getsize(out_path) / 1e6
    log.info("Готово: %s (%.1fс, %.1f МБ)", os.path.basename(out_path),
             duration, size_mb)


def render_preview_frame(video: str, out_jpg: str, t: float, cfg,
                         crop_expr: str | None, src_w: int, src_h: int,
                         ass_path: str | None) -> None:
    """Один кадр-превью (для отладки кадрирования/субтитров)."""
    vf = build_video_filterchain(src_w=src_w, src_h=src_h, cfg=cfg,
                                 crop_expr=None, ass_path=ass_path,
                                 duration=1)
    cmd = [_ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1",
           "-vf", vf, out_jpg]
    try:
        run(cmd, timeout=120)
    except Exception:
        pass
