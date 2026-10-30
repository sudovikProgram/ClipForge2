"""Генерация публикационных метаданных: заголовок, хук-оверлей, описание, теги.

Эвристики работают поверх транскрипта; при наличии LLM-ключа (OpenAI-совместимого)
используется он — качество заголовков кратно выше. Без сети всё считается офлайн.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request

log = logging.getLogger("clipforge")

_STOP_RU = {"это", "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
            "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
            "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
            "меня", "еще", "нет", "о", "из", "ему", "теперь", "when", "what", "how",
            "the", "and", "that", "for", "with", "you", "this", "they", "have", "are"}


def keywords(text: str, k: int = 6) -> list[str]:
    words = re.findall(r"[А-Яа-яЁёA-Za-z0-9\-']{4,}", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w in _STOP_RU or len(w) < 5:
            continue
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:k]
    return [w for w, _ in top]


def make_hook_overlay(candidate_text: str, max_chars: int = 60) -> str:
    """Крупный текст-хук в начале ролика: первая «цепляющая» фраза."""
    first = candidate_text.split(".")[0].split("?")[0].split("!")[0].strip()
    first = re.sub(r"^\W+", "", first)
    if len(first) > max_chars:
        cut = first[:max_chars].rsplit(" ", 1)[0]
        first = cut + "…"
    if not first:
        first = "Смотрите до конца"
    return first.upper() if len(first) < 34 else first.capitalize()


def make_title(candidate_text: str, lang_hint: str = "ru") -> str:
    """Заголовок ≤ 70 символов с эмодзи-крючком (без кликбейта-мусора)."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", candidate_text) if s.strip()]
    base = None
    for s in sents:
        if 20 <= len(s) <= 70:
            base = s
            break
    if base is None and sents:
        base = sents[0][:70]
    if base is None:
        base = "Интересный момент"
    base = base.rstrip(".!?…")
    hooks = ["🔥", "😱", "🤯", "❗️", "💡", "👀"]
    low = base.lower()
    if any(k in low for k in ("секрет", "тайна", "почему", "как ", "why", "secret", "how")):
        emoji = "💡"
    elif any(k in low for k in ("шок", "никогда", "ужас", "wow", "crazy", "insane")):
        emoji = "🤯"
    else:
        emoji = hooks[hash(base) % len(hooks)]
    return f"{base} {emoji}"


def make_description(candidate_text: str, tags: list[str], source_name: str) -> str:
    body = candidate_text.strip()
    if len(body) > 480:
        body = body[:480].rsplit(" ", 1)[0] + "…"
    tagline = " ".join("#" + t.replace(" ", "") for t in tags[:8])
    return (f"{body}\n\n"
            f"Нарезано автоматически ClipForge из «{source_name}».\n"
            f"{tagline}\n\n#Shorts #TikTok")


def make_tags(candidate_text: str, cfg) -> list[str]:
    kw = keywords(candidate_text, 6)
    fixed = ["shorts", "tiktok", "reels", "viral", "interesting"]
    out = []
    for t in kw + fixed:
        t = t.lower().strip(",.;:")
        if t and t not in out:
            out.append(t)
    return out[:12]


# -----------------------------------------------------------------------------
# LLM-улучшение (опционально): OpenAI-совместимый API
# -----------------------------------------------------------------------------
def llm_polish(text: str, lang: str = "ru") -> dict | None:
    """Если заданы OPENAI_API_KEY (+необязательно OPENAI_BASE_URL/MODEL),
    генерируем профессиональные заголовок/хук/теги через LLM."""
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CLIPFORGE_LLM_KEY")
    if not key:
        return None
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("CLIPFORGE_LLM_MODEL", "gpt-4o-mini")
    prompt = (
        f"Ты — SMM-продюсер вертикальных видео. Из транскрипта момента сделай JSON: "
        f'{{"title": "<до 80 симв., цепляющий, без кликбейт-клише>", '
        f'"hook_overlay": "<до 7 слов, КРУПНЫМ текстом на экран>", '
        f'"description": "<2 предложения>", "tags": ["...5..10 тегов без #"]}}. '
        f"Язык ответа: {lang}. Транскрипт:\n" + text[:2000]
    )
    try:
        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "response_format": {"type": "json_object"},
            }).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=40) as r:
            data = json.loads(r.read())
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        log.warning("LLM polish недоступен: %s — используем эвристики", e)
        return None


def build_metadata(candidate_text: str, source_name: str, cfg,
                   use_llm: bool = True) -> dict:
    meta = {
        "hook_overlay": make_hook_overlay(candidate_text),
        "title": make_title(candidate_text),
        "description": make_description(candidate_text,
                                        make_tags(candidate_text, cfg),
                                        source_name),
        "tags": make_tags(candidate_text, cfg),
    }
    if use_llm:
        polished = llm_polish(candidate_text)
        if polished:
            for k in meta:
                v = polished.get(k)
                if v:
                    meta[k] = v
    return meta
