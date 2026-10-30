"""Интеграционный тест без Whisper: мок транскрипта + реальные модули.
Проверяет: аудио-анализ, кандидатов, скоринг, дедуп, crop-expr, ASS, ffmpeg-рендер."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.abspath("."))

from clipforge.config import Config
from clipforge.lib import audio as audiolib, vision, highlights as hllib
from clipforge.lib import crop as croplib, subtitles as sublib, metadata as mdlib
from clipforge.lib import render as rdelib
from clipforge.lib.transcribe import Segment, Word, save_segments, load_segments
from clipforge.lib.utils import extract_audio, TempDir

cfg = Config()
cfg.whisper_model = "tiny"
video = "tests/long_test.mp4"

with TempDir() as wd:
    wav = os.path.join(wd, "a.wav")
    extract_audio(video, wav)
    prof = audiolib.analyze(wav)
    assert prof["duration"] > 170, prof
    print("audio OK:", len(prof["silences"]), "silence gaps")

    # мок «речи»: фразы каждые 18с по 12с (совпадают с огибающей фикстуры)
    segments = []
    texts = [
        "Why nobody tells you this about money? Here is the secret.",
        "Wait, stop scrolling — this mistake costs everyone a fortune.",
        "The truth about productivity: you always waste your mornings.",
        "Imagine what happens if you never check your phone again.",
        "This is the craziest reveal of the entire interview.",
        "So remember: that's why successful people wake up at five.",
    ]
    for i, t in enumerate(texts):
        s0 = i * 18 + 2.0
        words = t.split()
        per = 12.0 / max(1, len(words))
        ws = [Word(w, s0 + j * per, s0 + (j + 0.9) * per) for j, w in enumerate(words)]
        segments.append(Segment(s0, s0 + 12.0, t, ws))

    scenes = vision.detect_scenes(video, threshold=0.30, duration=prof["duration"])
    print("scenes OK:", len(scenes), scenes[:8])
    mt = vision.measure_motion(video, fps_probe=2.0)
    print("motion OK:", len(mt.times), "samples, cx mean", round(float(mt.cx.mean()), 3))

    cands = hllib.build_candidates(segments, prof, cfg, mt.times, mt.motion, scenes)
    print("candidates:", len(cands))
    assert cands, "no candidates!"
    uniq = hllib.deduplicate(cands)
    clips = hllib.select_clips(uniq, 3)
    assert clips
    c = clips[0]
    print("clip:", c.start, c.end, round(c.total, 3), c.scores)
    assert 12 <= c.duration <= 58
    assert 0 <= c.total <= 1

    # crop path
    expr = croplib.compute_crop_path(1280, 720, c.start, c.end, cfg.fps,
                                     motion_times=mt.times, motion_cx=mt.cx,
                                     motion_cy=mt.cy, faces=None, mode="auto")
    if expr is not None:
        assert expr.count("(") == expr.count(")")
        print("crop expr len:", len(expr))

    # метаданные (без LLM)
    meta = mdlib.build_metadata(c.text, "long_test", cfg, use_llm=False)
    assert meta["title"] and meta["hook_overlay"] and meta["tags"]
    print("meta:", meta["title"], "|", meta["hook_overlay"])

    # субтитры
    words = [w for w in c.words if c.start <= w.start <= c.end]
    lines = sublib.group_lines(words, cfg.sub_max_words)
    ass = os.path.join(wd, "subs.ass")
    sublib.write_ass(ass, lines, cfg, hook_text=meta["hook_overlay"], clip_start=c.start)
    content = open(ass, encoding="utf-8-sig").read()
    assert "Dialogue" in content and "\\kf" in content
    print("ass OK:", content.count("Dialogue"), "events")

    # рендер
    out = os.path.join(wd, "clip_test.mp4")
    speech_windows = [(w.start, w.end) for w in words]
    rdelib.render_clip(video, out, start=c.start, end=c.end, src_w=1280, src_h=720,
                       cfg=cfg, crop_expr=expr, ass_path=ass,
                       music_track=None, speech_times=speech_windows,
                       title=meta["title"])
    from clipforge.lib.utils import probe_media
    p = probe_media(out)
    assert p["width"] == 1080 and p["height"] == 1920, p
    assert abs(p["duration"] - c.duration) < 0.6, (p["duration"], c.duration)
    assert p["has_audio"]
    print("render OK:", p)

print("\nALL PIPELINE TESTS PASSED ✅")
