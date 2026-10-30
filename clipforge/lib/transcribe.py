"""Транскрипция видео через faster-whisper (предпочтительно) или openai-whisper.

Результат нормализуется в единый формат: список сегментов со словами и таймингами.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict

log = logging.getLogger("clipforge")


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d


def _device_and_compute() -> tuple[str, str]:
    try:
        import torch
        if torch.cuda.is_available():
            # int8 требует CUDA >= 12; безопасно пробуем float16 на GPU
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _backend_faster_whisper(wav: str, model_name: str, language: str | None,
                            initial_prompt: str | None) -> list[Segment]:
    from faster_whisper import WhisperModel
    device, compute = _device_and_compute()
    log.info("faster-whisper: модель=%s device=%s compute=%s", model_name, device, compute)
    model = WhisperModel(model_name, device=device, compute_type=compute)
    seg_iter, info = model.transcribe(
        wav,
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 350},
        beam_size=5,
        initial_prompt=initial_prompt,
    )
    log.info("Язык: %s (p=%.2f)", info.language, info.language_probability)
    segments: list[Segment] = []
    for s in seg_iter:
        words = [Word(w.word.strip(), w.start, w.end, w.probability or 1.0)
                 for w in (s.words or []) if w.word and w.word.strip()]
        text = s.text.strip()
        if not text:
            continue
        segments.append(Segment(round(s.start, 3), round(s.end, 3), text, words))
        if len(segments) % 50 == 0:
            log.debug("...%d сегментов (%.0fс)", len(segments), s.end)
    return segments


def _backend_openai_whisper(wav: str, model_name: str, language: str | None,
                            initial_prompt: str | None) -> list[Segment]:
    import whisper
    device = "cuda" if _device_and_compute()[0] == "cuda" else "cpu"
    log.info("openai-whisper: модель=%s device=%s (медленнее; для скорости поставьте faster-whisper)",
             model_name, device)
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(
        wav, language=language, word_timestamps=True,
        verbose=False, initial_prompt=initial_prompt,
    )
    segments: list[Segment] = []
    for s in result["segments"]:
        words = [Word(str(w["word"]).strip(), float(w["start"]), float(w["end"]),
                      float(w.get("probability", 1.0)))
                 for w in s.get("words", []) if str(w.get("word", "")).strip()]
        segments.append(Segment(round(float(s["start"]), 3), round(float(s["end"]), 3),
                                str(s["text"]).strip(), words))
    return segments


def transcribe(wav_path: str, model_name: str = "small", language: str | None = None,
               initial_prompt: str | None = None) -> list[Segment]:
    """Транскрибирует WAV-файл. Автоматически выбирает лучший доступный бэкенд."""
    try:
        import faster_whisper  # noqa: F401
        backend = _backend_faster_whisper
    except ImportError:
        try:
            import whisper  # noqa: F401
            backend = _backend_openai_whisper
        except ImportError:
            raise RuntimeError(
                "Не найдено ни one из: faster-whisper, openai-whisper. "
                "Установите: pip install faster-whisper"
            )
    segments = backend(wav_path, model_name, language, initial_prompt)
    if not segments:
        log.warning("Транскрипция не вернула текст — возможно, в аудио нет речи.")
    return segments


def save_segments(segments: list[Segment], path: str) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in segments], f, ensure_ascii=False, indent=1)


def load_segments(path: str) -> list[Segment]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Segment(s["start"], s["end"], s["text"],
                    [Word(**w) for w in s.get("words", [])]) for s in raw]
