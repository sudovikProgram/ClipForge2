"""Общие утилиты: запуск ffmpeg/ffprobe, логирование, временные файлы."""
from __future__ import annotations

import logging
import subprocess
import tempfile
import os
from pathlib import Path

log = logging.getLogger("clipforge")

FFMPEG = os.environ.get("CLIPFORGE_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("CLIPFORGE_FFPROBE", "ffprobe")

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".ts"}


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def run(cmd: list[str], *, quiet: bool = True, check: bool = True,
        timeout: float | None = None) -> subprocess.CompletedProcess:
    """Запуск команды с понятной диагностикой ошибок."""
    log.debug("exec: %s", " ".join(cmd[:12]) + (" ..." if len(cmd) > 12 else ""))
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            check=check,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "")[-1500:]
        raise RuntimeError(
            f"Команда завершилась с ошибкой ({e.returncode}): {cmd[0]}\n{tail}"
        ) from None
    except FileNotFoundError:
        raise RuntimeError(f"Не найден бинарник: {cmd[0]} (установите ffmpeg)") from None


def probe_media(path: str | Path) -> dict:
    """Метаданные медиафайла: длительность, наличие видео/аудио, размеры, FPS."""
    cmd = [
        os.environ.get("CLIPFORGE_FFPROBE", "ffprobe"),
        "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    import json
    out = run(cmd).stdout
    data = json.loads(out)
    info = {
        "duration": float(data["format"].get("duration", 0) or 0),
        "width": 0, "height": 0, "fps": 30.0,
        "has_video": False, "has_audio": False,
    }
    for st in data.get("streams", []):
        if st["codec_type"] == "video" and not info["has_video"]:
            info["has_video"] = True
            info["width"] = int(st.get("width", 0))
            info["height"] = int(st.get("height", 0))
            fr = st.get("avg_frame_rate", "30/1")
            try:
                num, den = fr.split("/")
                info["fps"] = float(num) / float(den) if float(den) else 30.0
            except Exception:
                info["fps"] = 30.0
        elif st["codec_type"] == "audio":
            info["has_audio"] = True
    return info


def extract_audio(video: str | Path, wav_path: str | Path, sr: int = 16000) -> None:
    """Извлечение моно-аудио для Whisper/анализа."""
    run([
        os.environ.get("CLIPFORGE_FFMPEG", "ffmpeg"), "-y",
        "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr),
        "-c:a", "pcm_s16le", str(wav_path),
    ])


class TempDir:
    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="clipforge_")
        return self.path

    def __exit__(self, *exc):
        import shutil
        shutil.rmtree(self.path, ignore_errors=True)


def fmt_ts(t: float) -> str:
    """Часы:минуты:секунды для человекочитаемых отчётов."""
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
