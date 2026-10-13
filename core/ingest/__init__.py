"""Ingest-Pipeline: Upload -> Text -> Material-Bibliothek.

Eine einzige Eintrittsfunktion (`ingest_file`) entscheidet anhand der Dateiendung,
ob extrahiert (Dokument) oder transkribiert (Audio/Video) wird.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .. import config, library
from . import documents, transcribe

IngestError = (documents.ExtractionError, transcribe.TranscriptionError)


def save_upload(name: str, data: bytes) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(c for c in name if c.isalnum() or c in "._- ").strip() or "upload"
    path = config.UPLOAD_DIR / f"{stamp}_{safe}"
    path.write_bytes(data)
    return path


def resolve_local_path(raw: str) -> Path:
    """Pfadeingabe der UI robust aufloesen (Tilde, Anfuehrungszeichen, escapte Leerzeichen).

    Erlaubt es, grosse Vorlesungsaufzeichnungen zu verarbeiten, ohne sie durch den
    Browser-Upload zu schleusen – die Datei wird an Ort und Stelle gelesen.

    Prueft den Lesezugriff sofort: macOS schuetzt Downloads/Dokumente/Schreibtisch per
    TCC. Ohne diese Pruefung schlaegt erst der Hintergrund-Job nach Minuten fehl.
    """
    text = raw.strip().strip('"').strip("'").replace("\\ ", " ")
    path = Path(text).expanduser()
    if not path.exists():
        raise documents.ExtractionError(f"Datei nicht gefunden: {path}")
    if not path.is_file():
        raise documents.ExtractionError(f"Kein regulaerer Dateipfad: {path}")
    suffix = path.suffix.lower()
    if suffix not in (config.DOC_SUFFIXES | config.AUDIO_SUFFIXES):
        raise documents.ExtractionError(f"Dateityp {suffix} wird nicht unterstuetzt.")
    try:
        with open(path, "rb") as handle:
            handle.read(64)
    except PermissionError as exc:
        raise documents.ExtractionError(
            f"macOS verweigert den Zugriff auf «{path.parent.name}». "
            "Zwei Loesungen:\n"
            "1. Systemeinstellungen → Datenschutz & Sicherheit → Festplattenvollzugriff → "
            "Terminal (bzw. die App, aus der du run.sh startest) aktivieren und Terminal neu starten.\n"
            "2. Oder die Datei in einen unkritischen Ordner verschieben, "
            "z. B. `mv \"{}\" ~/Studium/`".format(path)
        ) from exc
    except OSError as exc:
        raise documents.ExtractionError(f"Datei nicht lesbar: {exc}") from exc
    return path


def ingest_file(
    path: Path,
    original_name: str | None = None,
    whisper_model: str | None = None,
    language: str | None = None,
    progress: Callable[[float, str], None] | None = None,
    project_id: str = "",
    session_id: str = "",
) -> library.Material:
    """Extrahiert bzw. transkribiert eine Datei und legt sie scoped in der Bibliothek ab.

    Der Scope (project_id/session_id) bestimmt, welche Sessions das Material spaeter sehen.
    """
    name = original_name or path.name
    suffix = path.suffix.lower()

    if suffix in config.AUDIO_SUFFIXES:
        text, meta = transcribe.transcribe(
            path, model_size=whisper_model, language=language, progress=progress
        )
        kind, source = "transcript", f"faster-whisper({meta['whisper_model']})"
        # Transkript zusaetzlich als eigenstaendige Datei ablegen (wiederverwendbar)
        (config.TRANSCRIPT_DIR / f"{path.stem}.md").write_text(text, encoding="utf-8")
    elif suffix in config.DOC_SUFFIXES:
        if progress:
            progress(0.3, "extrahiere Text")
        text, kind, meta = documents.extract(path)
        source = f"extract({suffix.lstrip('.')})"
    else:
        raise documents.ExtractionError(
            f"Dateityp {suffix} wird nicht unterstuetzt. "
            f"Unterstuetzt: {', '.join(sorted(config.DOC_SUFFIXES | config.AUDIO_SUFFIXES))}"
        )

    if progress:
        progress(1.0, "fertig")
    header = f"# {name}\n\n"
    return library.add(
        name=name, text=header + text, kind=kind, source=source, meta=meta,
        project_id=project_id, session_id=session_id,
    )


__all__ = ["ingest_file", "save_upload", "resolve_local_path", "IngestError",
           "documents", "transcribe"]

