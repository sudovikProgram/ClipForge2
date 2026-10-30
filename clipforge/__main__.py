"""ClipForge — пайплайн: длинное видео → готовые Shorts/TikTok.

Использование:
    python -m clipforge input.mp4 -n 5 -o output/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .config import Config
from .lib import audio as audiolib
from .lib import crop as croplib
from .lib import highlights as hllib
from .lib import metadata as mdlib
from .lib import render as rdelib
from .lib import subtitles as sublib
from .lib import vision
from .lib.transcribe import transcribe, save_segments, load_segments
from .lib.utils import (TempDir, extract_audio, fmt_ts, probe_media,
                        setup_logging, VIDEO_EXTS)

log = logging.getLogger("clipforge")


# -----------------------------------------------------------------------------
# Кэш анализа (дорогая транскрипция не повторяется при перерендере)
# -----------------------------------------------------------------------------
def _cache_dir(video: str) -> Path:
    stem = Path(video).stem
    d = Path(".clipforge_cache") / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def analyze_video(video: str, cfg: Config, workdir: str,
                  use_cache: bool = True, force: bool = False) -> dict:
    """Полный анализ источника: транскрипт + аудио-профиль + сцены + движение + лица."""
    cache = _cache_dir(video)
    info = probe_media(video)
    prof: dict = {}

    # --- Транскрипция -------------------------------------------------------
    seg_path = cache / "segments.json"
    wav16 = os.path.join(workdir, "audio16.wav")
    ext_seg = os.environ.get("CLIPFORGE_SEGMENTS")  # готовый транскрипт (обход музыки/шума)
    if ext_seg and Path(ext_seg).exists() and not force:
        log.info("Загружаю внешний транскрипт: %s", ext_seg)
        segments = load_segments(ext_seg)
        shutil.copy2(ext_seg, seg_path)
        cached_wav = cache / "audio16.wav"
        if cached_wav.exists():
            wav16 = str(cached_wav)
    elif use_cache and seg_path.exists() and not force:
        log.info("Кэш: транскрипт загружен из %s", seg_path)
        segments = load_segments(str(seg_path))
        cached_wav = cache / "audio16.wav"
        if cached_wav.exists():
            wav16 = str(cached_wav)
    else:
        t0 = time.time()
        log.info("[1/5] Извлечение аудио…")
        extract_audio(video, wav16, sr=16000)
        shutil.copy(wav16, cache / "audio16.wav")
        log.info("[2/5] Транскрипция (модель %s)… это самая долгая часть",
                 cfg.whisper_model)
        segments = transcribe(str(cache / "audio16.wav"), cfg.whisper_model,
                              cfg.language)
        save_segments(segments, str(seg_path))
        log.info("Транскрипция заняла %.1fс → %d сегментов",
                 time.time() - t0, len(segments))
        wav16 = str(cache / "audio16.wav")
    prof["segments"] = segments

    # --- Аудио-профиль ------------------------------------------------------
    log.info("[3/5] Аудио-анализ (VAD/энергия)…")
    prof["audio"] = audiolib.analyze(wav16)

    # --- Сцены ---------------------------------------------------------------
    scenes_p = cache / "scenes.json"
    if use_cache and scenes_p.exists() and not force:
        scenes = json.loads(scenes_p.read_text())
    else:
        log.info("[4/5] Детекция смены сцен…")
        scenes = vision.detect_scenes(video, cfg.scene_threshold,
                                      duration=info["duration"])
        scenes_p.write_text(json.dumps(scenes))
    prof["scenes"] = scenes

    # --- Движение + центр интереса ------------------------------------------
    mot_p = cache / "motion.npz"
    if use_cache and mot_p.exists() and not force:
        z = np.load(mot_p)
        motion = {"times": z["t"], "motion": z["m"],
                  "cx": z["cx"], "cy": z["cy"]}
    else:
        log.info("[5/5] Анализ движения и центра интереса…")
        mt = vision.measure_motion(video)
        motion = {"times": mt.times, "motion": mt.motion,
                  "cx": mt.cx, "cy": mt.cy}
        np.savez_compressed(mot_p, t=mt.times, m=mt.motion,
                            cx=mt.cx, cy=mt.cy)
    prof["motion"] = motion

    # --- Лица ----------------------------------------------------------------
    faces_p = cache / "faces.npy"
    if use_cache and faces_p.exists() and not force:
        arr = np.load(faces_p)
        faces = arr if arr.size else None
    elif cfg.enable_face_detect:
        ft = vision.face_track(video, None, src_w=info["width"],
                               src_h=info["height"])
        np.save(faces_p, ft if ft is not None else np.zeros((0, 4)))
        faces = ft
    else:
        faces = None
    prof["faces"] = faces
    prof["info"] = info
    return prof


# -----------------------------------------------------------------------------
# Основной конвейер
# -----------------------------------------------------------------------------
def process(video: str, cfg: Config, out_dir: str, max_clips: int,
            music: str | None = None, no_llm: bool = False,
            preview: bool = False) -> list[dict]:
    video = str(Path(video).expanduser().resolve())
    if not os.path.exists(video):
        raise FileNotFoundError(f"Видео не найдено: {video}")
    info = probe_media(video)
    if not info["has_video"]:
        raise RuntimeError("В файле нет видеопотока")
    log.info("Источник: %s (%dx%d, %.1f мин)", os.path.basename(video),
             info["width"], info["height"], info["duration"] / 60)

    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    with TempDir() as wd:
        prof = analyze_video(video, cfg, wd)
        segments, audio = prof["segments"], prof["audio"]
        scenes, motion, faces = prof["scenes"], prof["motion"], prof["faces"]

        if not segments:
            log.error("Речь не обнаружена — клипы невозможны. "
                      "Попробуйте --model medium или проверьте звуковую дорожку.")
            return []

        cands = hllib.build_candidates(segments, audio, cfg,
                                       motion["times"], motion["motion"], scenes)
        uniq = hllib.deduplicate(cands)
        clips = hllib.select_clips(uniq, max_clips)
        if not clips:
            log.error("Ни один кандидат не прошёл по длительности "
                      "(min=%.0fs max=%.0fs). Ослабьте лимиты.",
                      cfg.min_clip_seconds, cfg.max_clip_seconds)
            return []
        log.info("Отобрано %d клипов из %d уникальных кандидатов",
                 len(clips), len(uniq))

        report: list[dict] = []
        for i, c in enumerate(clips, 1):
            log.info("--- Клип %d/%d: %s → %s (%.1fс, score=%.3f) ---",
                     i, len(clips), fmt_ts(c.start), fmt_ts(c.end),
                     c.duration, c.total)

            meta = mdlib.build_metadata(c.text, Path(video).stem, cfg,
                                        use_llm=not no_llm)

            crop_expr = croplib.compute_crop_path(
                info["width"], info["height"], c.start, c.end, cfg.fps,
                motion_times=motion["times"], motion_cx=motion["cx"],
                motion_cy=motion["cy"], faces=faces,
                face_weight=cfg.face_weight, motion_weight=cfg.motion_weight,
                smooth=cfg.smooth_crop, mode=cfg.crop_mode)

            words = [w for w in c.words if c.start <= w.start <= c.end]
            lines = sublib.group_lines(words, cfg.sub_max_words)
            ass_path = None
            if cfg.subtitles and lines:
                ass_path = os.path.join(out_dir, f"clip{i:02d}_subs.ass")
                sublib.write_ass(ass_path, lines, cfg,
                                 hook_text=meta["hook_overlay"],
                                 clip_start=c.start)

            speech_windows = [(w.start, w.end) for w in words] or \
                [(s.start, s.end) for s in segments if c.start <= s.start <= c.end]

            out_file = os.path.join(out_dir, f"clip{i:02d}.mp4")
            try:
                rdelib.render_clip(
                    video, out_file, start=c.start, end=c.end,
                    src_w=info["width"], src_h=info["height"], cfg=cfg,
                    crop_expr=crop_expr, ass_path=ass_path,
                    music_track=music, speech_times=speech_windows,
                    title=meta["title"])
            except Exception as e:
                log.error("Клип %d не отрендерен: %s", i, e)
                continue

            if preview:
                rdelib.render_preview_frame(
                    video, os.path.join(out_dir, f"clip{i:02d}_preview.jpg"),
                    c.start + 1.0, cfg, crop_expr,
                    info["width"], info["height"], ass_path)

            entry = {
                "file": out_file,
                "source_range": [round(c.start, 2), round(c.end, 2)],
                "duration": round(c.duration, 2),
                "score": c.total,
                "scores_detail": c.scores,
                "transcript": c.text,
                **meta,
            }
            report.append(entry)
            (Path(out_dir) / f"clip{i:02d}.json").write_text(
                json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

        index = {
            "source": video,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_sec": round(time.time() - t_start, 1),
            "clips": report,
        }
        (Path(out_dir) / "report.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown_report(index, Path(out_dir) / "REPORT.md")

    log.info("✅ Готово: %d роликов за %.1f мин. Папка: %s",
             len(report), (time.time() - t_start) / 60, out_dir)
    return report


def write_markdown_report(index: dict, path: Path) -> None:
    lines = [f"# ClipForge: отчёт по «{os.path.basename(index['source'])}»\n",
             f"Сгенерировано: {index['generated_at']} • "
             f"роликов: {len(index['clips'])} • время: {index['runtime_sec']}с\n"]
    for i, c in enumerate(index["clips"], 1):
        s, e = c["source_range"]
        lines += [
            f"## {i}. {c['title']}",
            f"*Файл:* `{os.path.basename(c['file'])}` · "
            f"*из источника:* {fmt_ts(s)}–{fmt_ts(e)} ({c['duration']}с) · "
            f"*рейтинг:* {c['score']:.2f} {c['scores_detail']}",
            "",
            f"**Хук на экран:** {c['hook_overlay']}",
            "",
            f"**Описание:**\n{c['description']}",
            "",
            "**Теги:** " + ", ".join(c["tags"]),
            "",
            f"**Транскрипт:** {c['transcript'][:300]}…",
            "", "---", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="clipforge",
        description="ClipForge: длинные видео → увлекательные Shorts/TikTok "
                    "(вертикаль 9:16, karaoke-субтитры, хуки, метаданные).")
    ap.add_argument("video", help="Исходное длинное видео (mp4/mkv/mov/...)")
    ap.add_argument("-n", "--clips", type=int, default=5,
                    help="сколько клипов сделать (по умолчанию 5)")
    ap.add_argument("-o", "--out", default="output", help="папка результата")
    ap.add_argument("--model", default=None,
                    help="модель Whisper: tiny|base|small|medium|large-v3")
    ap.add_argument("--lang", default=None,
                    help="язык речи (ru/en/…), автоопределение если пусто")
    ap.add_argument("--music", default=None,
                    help="файл фоновой музыки (mp3/wav) — добавляется с дакингом")
    ap.add_argument("--crop", choices=["auto", "center", "smart"], default=None,
                    help="режим кадрирования: center = статичный центр, "
                         "smart/auto = слежение за лицами/движением")
    ap.add_argument("--min-len", type=float, default=None)
    ap.add_argument("--max-len", type=float, default=None)
    ap.add_argument("--no-subtitles", action="store_true")
    ap.add_argument("--no-llm", action="store_true",
                    help="не использовать LLM для заголовков (только эвристики)")
    ap.add_argument("--preview", action="store_true",
                    help="сохранять JPG-превью кадра каждого клипа")
    ap.add_argument("--force", action="store_true", help="игнорировать кэш анализа")
    ap.add_argument("--config", default=None, help="JSON с настройками")
    ap.add_argument("--save-config", metavar="PATH", default=None,
                    help="сохранить текущие настройки в JSON и выйти")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(args.verbose)
    cfg = Config()
    if args.config:
        cfg.apply_json(args.config)
    if args.model:
        cfg.whisper_model = args.model
    if args.lang:
        cfg.language = args.lang
    if args.crop:
        cfg.crop_mode = args.crop
    if args.min_len:
        cfg.min_clip_seconds = args.min_len
    if args.max_len:
        cfg.max_clip_seconds = args.max_len
    if args.no_subtitles:
        cfg.subtitles = False
    if args.save_config:
        cfg.to_json(args.save_config)
        print("Конфигурация сохранена:", args.save_config)
        return 0

    ext = Path(args.video).suffix.lower()
    if ext and ext not in VIDEO_EXTS:
        log.warning("Необычное расширение '%s' — пробуем открыть", ext)

    try:
        report = process(args.video, cfg, args.out, args.clips,
                         music=args.music, no_llm=args.no_llm,
                         preview=args.preview)
    except KeyboardInterrupt:
        log.error("Прервано пользователем")
        return 130
    except Exception as e:
        log.error("Ошибка: %s", e, exc_info=args.verbose)
        return 1

    for c in report:
        print(f"\n📌 {os.path.basename(c['file'])}  [{c['source_range'][0]}s–"
              f"{c['source_range'][1]}s]\n   {c['title']}\n"
              f"   хук: {c['hook_overlay']}  |  рейтинг {c['score']:.2f}")
    return 0 if report else 1


if __name__ == "__main__":
    sys.exit(main())
