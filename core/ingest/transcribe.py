"""Transkription von Lektionsaufzeichnungen (Audio/Video) mit faster-whisper.

Wichtig (technische Grenze): Ein LLM kann kein Audio. ASR und Analyse sind zwei
getrennte, aufeinander aufbauende Schritte:
    Audio --(Whisper)--> Text --(Ollama)--> Analyse
Das Modell wird lazy geladen und im Prozess gecached (Ladezeit ~10-60 s).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterator

from .. import config


class TranscriptionError(RuntimeError):
    pass


_MODEL_CACHE: dict[tuple[str, str], object] = {}

# Ab dieser Groesse wird vor der Transkription die Tonspur extrahiert.
AUDIO_EXTRACT_THRESHOLD_MB = 150
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def media_duration(path: Path) -> float | None:
    """Dauer in Sekunden via ffprobe (fuer die Fortschrittsanzeige)."""
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL,   # sonst SIGTTIN im Hintergrundbetrieb
        )
        return float(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return None


def extract_audio(path: Path, target_dir: Path | None = None,
                  progress: Callable[[float, str], None] | None = None) -> Path:
    """Extrahiert die Tonspur als 16 kHz Mono-WAV – das Format, das Whisper intern nutzt.

    Eine 700-MB-Bildschirmaufnahme schrumpft so auf wenige hundert MB reines Audio,
    ohne dass die Erkennungsqualitaet leidet: Whisper resampled ohnehin auf 16 kHz Mono.

    WICHTIG – zwei Fallstricke, die hier bewusst behandelt werden:
    * `-nostdin` + stdin=DEVNULL: ffmpeg liest sonst vom Terminal. Laeuft die App als
      Hintergrundprozess, bekommt ffmpeg dabei SIGTTIN und wird vom OS *angehalten*
      (ps STAT "T") – es haengt dann ewig bei 0 % CPU.
    * `-progress pipe:1`: liefert maschinenlesbaren Fortschritt, den wir streamen.
      Gleichzeitig wird stderr geleert, sonst laeuft der Pipe-Puffer voll und blockiert.
    """
    if not ffmpeg_available():
        raise TranscriptionError("ffmpeg fehlt. Installation: `brew install ffmpeg`")
    target_dir = target_dir or config.TRANSCRIPT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / f"{path.stem}.16k.wav"
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        if progress:
            progress(1.0, "Tonspur bereits vorhanden")
        return out

    total = media_duration(path) or 0.0
    tmp = out.with_suffix(".part.wav")
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-progress", "pipe:1", "-nostats",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(tmp),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    try:
        for line in proc.stdout or []:            # "out_time_us=12345678"
            if line.startswith("out_time_us=") and total and progress:
                try:
                    done = int(line.split("=", 1)[1]) / 1_000_000
                except ValueError:
                    continue
                progress(min(done / total, 1.0),
                         f"Tonspur {_hhmmss(done)} / {_hhmmss(total)}")
        proc.wait(timeout=60)
    except Exception:
        proc.kill()
        raise
    finally:
        stderr = (proc.stderr.read() if proc.stderr else "") or ""

    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise TranscriptionError(
            f"Audio-Extraktion fehlgeschlagen: {stderr.strip()[:300] or 'unbekannter Fehler'}"
        )
    tmp.replace(out)   # erst nach Erfolg umbenennen -> kein halbes Ergebnis im Cache
    return out


def model_cached(model_size: str) -> bool:
    """Liegen die Gewichte lokal? Sonst wuerde der erste Lauf einen Download starten."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    return any(hub.glob(f"models--Systran--faster-whisper-{model_size}/snapshots/*/model.bin"))


def _load_model(model_size: str, compute_type: str):
    key = (model_size, compute_type)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper ist nicht installiert. Installation: `pip install faster-whisper` "
            "(ausserhalb des VPN)."
        ) from exc

    # Fail-fast: Ohne lokale Gewichte wuerde faster-whisper einen Download starten,
    # der im Firmen-VPN minutenlang im Timeout haengt. Lieber sofort sagen, was zu tun ist.
    allow_download = os.environ.get("STUDYHUB_ALLOW_WHISPER_DOWNLOAD") == "1"
    if not model_cached(model_size) and not allow_download:
        raise TranscriptionError(
            f"Whisper-Modell «{model_size}» ist lokal nicht vorhanden. "
            "Einmalig ausserhalb des Firmen-VPN laden: "
            f"`./.venv/bin/python scripts/fetch_whisper.py {model_size}` "
            "– danach laeuft die Transkription vollstaendig offline."
        )
    try:
        # Apple Silicon: CPU + int8 ist der verlaessliche Pfad (kein CUDA vorhanden).
        # cpu_threads auf die Performance-Kerne setzen – der Default (4) laesst
        # auf einem M-Chip viel Leistung liegen.
        model = WhisperModel(
            model_size, device="cpu", compute_type=compute_type,
            cpu_threads=config.WHISPER_THREADS,
            num_workers=1,
            local_files_only=not allow_download,
        )
    except Exception as exc:
        raise TranscriptionError(
            f"Whisper-Modell «{model_size}» konnte nicht geladen werden ({str(exc)[:120]}). "
            f"Abhilfe: `./.venv/bin/python scripts/fetch_whisper.py {model_size}` ausserhalb des VPN."
        ) from exc
    _MODEL_CACHE[key] = model
    return model


def _hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def transcribe(
    path: Path,
    model_size: str | None = None,
    language: str | None = None,
    compute_type: str | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[str, dict]:
    """-> (markdown_transkript, meta).

    Ablauf: (1) Tonspur extrahieren, (2) Whisper laden, (3) segmentweise transkribieren.
    Der Fortschritt deckt alle drei Phasen ab (0-10 % / 10-15 % / 15-100 %).

    Das Transkript bekommt alle ~2 Minuten einen Zeitanker ("## 00:12:00"), damit
    spaeter auf Stellen der Aufzeichnung verwiesen werden kann. Die Datei wird NICHT
    zerschnitten: faster-whisper liest streamend, Schnitte zerreissen Woerter.
    """
    if not ffmpeg_available():
        raise TranscriptionError("ffmpeg fehlt. Installation: `brew install ffmpeg`")

    def phase(lo: float, hi: float) -> Callable[[float, str], None]:
        def report(frac: float, info: str) -> None:
            if progress:
                progress(lo + (hi - lo) * max(0.0, min(frac, 1.0)), info)
        return report

    source = path
    size_mb = path.stat().st_size / 1024 / 1024 if path.exists() else 0
    is_video = path.suffix.lower() in VIDEO_SUFFIXES
    if is_video or size_mb > AUDIO_EXTRACT_THRESHOLD_MB:
        source = extract_audio(path, progress=phase(0.0, 0.10))

    size = model_size or config.WHISPER_MODEL
    if progress:
        progress(0.10, f"lade Whisper «{size}»")
    model = _load_model(size, compute_type or config.WHISPER_COMPUTE)
    lang = (language if language is not None else config.WHISPER_LANGUAGE) or None

    options = dict(
        language=lang,
        vad_filter=True,                   # Stille/Pausen ueberspringen -> schneller & sauberer
        vad_parameters={"min_silence_duration_ms": 700},
        beam_size=config.WHISPER_BEAM_SIZE,
        condition_on_previous_text=False,  # verhindert Wiederholungsschleifen bei Vorlesungen
    )

    batched = False
    engine = model
    if config.WHISPER_BATCH_SIZE > 1:
        try:  # faster-whisper >= 1.1: bis zu 4x schneller durch Batch-Verarbeitung
            from faster_whisper import BatchedInferencePipeline

            engine = BatchedInferencePipeline(model=model)
            options["batch_size"] = config.WHISPER_BATCH_SIZE
            batched = True
        except Exception:
            engine = model                 # aelteres faster-whisper -> normaler Pfad

    if progress:
        progress(0.15, f"transkribiere ({'batched' if batched else 'sequentiell'})")
    segments, info = engine.transcribe(str(source), **options)
    total = getattr(info, "duration", None) or media_duration(source) or 0.0
    report = phase(0.15, 1.0)

    next_anchor = 0.0
    last_end = 0.0
    started = time.monotonic()
    blocks: list[list[str]] = []            # je Zeitanker ein Absatz-Block
    for seg in _iter(segments):
        if seg.start >= next_anchor or not blocks:
            blocks.append([f"## {_hhmmss(seg.start)}"])
            next_anchor = seg.start + 120
        blocks[-1].append(seg.text.strip())
        last_end = seg.end
        if total:
            speed = seg.end / max(time.monotonic() - started, 0.1)
            eta = (total - seg.end) / speed if speed > 0 else 0
            report(seg.end / total,
                   f"{_hhmmss(seg.end)} / {_hhmmss(total)} · {speed:.1f}x · "
                   f"noch ~{_hhmmss(eta)}")

    # Jeder Zeitblock wird ein eigener Absatz (Anker-Zeile + Fliesstext), Bloecke durch
    # Leerzeilen getrennt. Erst das macht das Transkript fuer library.chunk_text() teilbar.
    paragraphs = [f"{block[0]}\n{' '.join(block[1:]).strip()}" for block in blocks if len(block) > 1]
    text = "\n\n".join(paragraphs).strip()
    if not text:
        raise TranscriptionError("Keine Sprache erkannt (stille Aufnahme oder falsches Format?).")
    elapsed = time.monotonic() - started
    meta = {
        "duration_s": round(total or last_end, 1),
        "duration": _hhmmss(total or last_end),
        "language": getattr(info, "language", lang or "?"),
        "language_probability": round(getattr(info, "language_probability", 0.0), 2),
        "whisper_model": size,
        "source_size_mb": round(size_mb, 1),
        "audio_extracted": source != path,
        "batched": batched,
        "transcribe_s": round(elapsed, 1),
        "speed_factor": round((total or last_end) / max(elapsed, 0.1), 1),
    }
    return text, meta


def _iter(segments) -> Iterator:
    """faster-whisper liefert einen lazy Generator – Fehler treten erst hier auf."""
    try:
        yield from segments
    except Exception as exc:  # pragma: no cover
        raise TranscriptionError(f"Transkription fehlgeschlagen: {exc}") from exc

