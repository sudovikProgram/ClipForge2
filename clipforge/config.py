"""Конфигурация ClipForge: пути, параметры анализа и рендеринга.

Все настройки можно переопределить JSON-файлом (--config) и флагами CLI.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, fields


# -----------------------------------------------------------------------------
# Пути к бинарникам (можно задать через переменные окружения)
# -----------------------------------------------------------------------------
FFMPEG = os.environ.get("CLIPFORGE_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("CLIPFORGE_FFPROBE", "ffprobe")


@dataclass
class Config:
    # ---- Анализ --------------------------------------------------------
    whisper_model: str = "small"          # tiny|base|small|medium|large-v3
    transcribe_device: str = "auto"       # auto|cpu|cuda
    language: str | None = None           # None = автоопределение
    scene_threshold: float = 15.0         # порог scdet (шкала YDIF ~0..255), ниже = больше сцен; 0..1 конвертируется автоматически
    max_clip_seconds: float = 58.0        # жёсткий лимит длительности шортса
    min_clip_seconds: float = 12.0        # минимум для осмысленного хука
    target_clip_seconds: float = 34.0     # целевая «идеальная» длина
    silence_pad: float = 0.25             # запас тишины по краям склейки
    sentence_gap: float = 0.35            # разрыв между сегментами = граница фразы

    # ---- Скоринг -------------------------------------------------------
    w_hook: float = 0.30                  # вес силы первой фразы (хук)
    w_energy: float = 0.15                # эмоциональность речи/звука
    w_density: float = 0.15               # «плотность контента» (слов в секунду)
    w_arc: float = 0.20                   # наличие завершённой мысли (арка)
    w_visual: float = 0.10                # визуальная активность (сцены/движение)
    w_retention: float = 0.10             # отсутствие длинных провалов внутри

    # Хук-слова, повышающие оценку начала фрагмента (мультиязычные)
    hook_words: list[str] = field(default_factory=lambda: [
        # EN
        "secret", "crazy", "insane", "never", "always", "why", "how", "what if",
        "nobody", "everyone", "stop", "wait", "listen", "imagine", "truth",
        "mistake", "warning", "reveal", "shocking", "you won't believe", "here's",
        "the problem", "that's why", "actually", "realize", "money", "free",
        # RU
        "секрет", "шуточн", "безумн", "никогда", "всегда", "почему", "как",
        "никто", "все", "стой", "подожди", "представь", "правда", "ошибк",
        "внимание", "шок", "реальн", "на самом деле", "проблем", "деньги",
        "бесплатн", "важно", "главное", "смотрите", "слушайте", "тайна",
        "раскрыва", "вы не поверите", "оказывает",
    ])
    # Слова-«маркеры завершения мысли» — хороший конец клипа
    closing_markers: list[str] = field(default_factory=lambda: [
        "and that's", "so", "that's why", "in conclusion", "remember", "bottom line",
        "поэтому", "в итоге", "и вот", "запомни", "вывод", "собственно",
    ])

    # ---- Рендер --------------------------------------------------------
    out_w: int = 1080
    out_h: int = 1920
    fps: int = 30
    video_crf: int = 20
    preset: str = "veryfast"
    crop_mode: str = "auto"               # auto|center|smart
    face_weight: float = 1.6              # приоритет лиц при кадрировании
    motion_weight: float = 0.6            # приоритет движения
    smooth_crop: float = 0.06             # сглаживание траектории кадра (EMA alpha)

    # Звук
    enable_denoise: bool = True
    loudnorm_target_i: float = -14.0      # цель R128 для соцсетей
    duck_music_db: float = -13.0          # насколько приглушать музыку под речь
    music_gain_db: float = -7.0           # базовая громкость музыки
    voice_boost_db: float = 2.0           # усиление голоса в миксе

    # Субтитры
    subtitles: bool = True
    sub_font: str = "DejaVu Sans"
    sub_font_size: int = 64               # относительно высоты 1920
    sub_max_words: int = 4                # слов на строку карусельных субтитров
    sub_highlight_active: bool = True     # активное слово — цветом (karaoke)
    sub_outline: int = 4
    sub_margin_v: int = 300               # отступ снизу (px)
    accent_color: str = "&H0000E5FF"      # ASS BGR: ярко-жёлтый
    base_color: str = "&H00FFFFFF"        # белый

    # Оверлеи
    hook_overlay: bool = True             # крупный заголовок-хук в начале
    hook_overlay_seconds: float = 2.6
    progress_bar: bool = True
    safe_zones: bool = False              # показать зоны UI TikTok/YT (для отладки)

    # Импорты из Python-библиотек (опциональны)
    enable_face_detect: bool = True
    enable_speech_emotion: bool = True

    def apply_json(self, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        names = {f.name for f in fields(self)}
        for k, v in data.items():
            if k in names:
                setattr(self, k, v)
        return self

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
