"""Zentrale Konfiguration & Pfade.

Alles laeuft lokal: keine Cloud-Calls, keine Telemetrie.
Ueberschreibbar via Umgebungsvariablen (z. B. OLLAMA_HOST).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Ollama ---
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
# /api/chat ist der korrekte Endpunkt fuer das messages-Array.
# /api/generate kennt KEINE messages und hat kein Rollen-Konzept -> nicht verwenden.
CHAT_ENDPOINT = f"{OLLAMA_HOST}/api/chat"
TAGS_ENDPOINT = f"{OLLAMA_HOST}/api/tags"
SHOW_ENDPOINT = f"{OLLAMA_HOST}/api/show"
EMBED_ENDPOINT = f"{OLLAMA_HOST}/api/embed"

REQUEST_TIMEOUT = (10, 600)  # (connect, read) – lange Reads wegen langsamer lokaler Modelle

# --- Ablage ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("STUDYHUB_DATA", BASE_DIR / "data"))
SESSION_DIR = DATA_DIR / "sessions"
MATERIAL_DIR = DATA_DIR / "materials"   # extrahierter Text (.md/.txt)
UPLOAD_DIR = DATA_DIR / "uploads"       # Originaldateien
TRANSCRIPT_DIR = DATA_DIR / "transcripts"

for _d in (SESSION_DIR, MATERIAL_DIR, UPLOAD_DIR, TRANSCRIPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Kontext-Budget ---
# Default-Fenster, das an Ollama gesendet wird (num_ctx). Wird pro Session gesetzt.
DEFAULT_NUM_CTX = 8192
# Ab diesem Anteil des Budgets wird die Memory-Compression ausgeloest.
COMPRESSION_TRIGGER = 0.75
# Anteil des Budgets, der fuer die Antwort reserviert bleibt.
RESPONSE_RESERVE = 0.20
# So viele der juengsten Nachrichten werden NIE zusammengefasst.
KEEP_RECENT_MESSAGES = 6

# --- Transkription ---
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")  # tiny|base|small|medium|large-v3
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "")  # "" = auto, sonst "de"/"en"

# Performance-Kerne nutzen: der Default von ctranslate2 (4 Threads) verschenkt
# auf einem M-Chip viel Leistung. os.cpu_count() zaehlt auch die Effizienzkerne,
# die hier kaum beitragen – deshalb ~70 %.
WHISPER_THREADS = int(os.environ.get("WHISPER_THREADS", max(4, int((os.cpu_count() or 8) * 0.7))))
# Batch-Verarbeitung (faster-whisper >= 1.1): bis zu 4x schneller. 1 = deaktiviert.
WHISPER_BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))
# beam_size 1 ist deutlich schneller als 5, bei Vorlesungen kaum schlechter.
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))

# --- Hintergrund-Jobs ---
# Gleichzeitig laufende Transkriptionen/Zusammenfassungen. Mehr als 2 bringt auf
# CPU nichts, da Whisper bereits alle Kerne auslastet.
MAX_PARALLEL_JOBS = int(os.environ.get("STUDYHUB_MAX_JOBS", "2"))
JOB_DIR = DATA_DIR / "jobs"
JOB_DIR.mkdir(parents=True, exist_ok=True)
JOB_RETENTION_H = 48  # abgeschlossene Jobs so lange in der Liste behalten

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".m4v", ".webm", ".ogg", ".flac", ".aac"}
DOC_SUFFIXES = {".pdf", ".pptx", ".docx", ".txt", ".md", ".csv", ".html", ".htm", ".vtt", ".srt"}

# --- Personas (Selective Context Loading: pro Thema eine eigene Session) ---
PERSONAS: dict[str, str] = {
    "Akademischer Tutor": (
        "Du bist ein akademischer Tutor an einer Hochschule. Antworte praezise, strukturiert und "
        "fachlich korrekt. Erklaere Fachbegriffe beim ersten Auftreten. Nutze Gliederungen und "
        "Beispiele. Wenn dir Informationen fehlen, sage das explizit, statt zu raten."
    ),
    "Pruefungs-Coach": (
        "Du bist ein Pruefungs-Coach. Du erzeugst aus dem bereitgestellten Material Lernfragen, "
        "Flashcards (Format 'F: ... / A: ...') und Merksaetze. Du bewertest meine Antworten kritisch "
        "und zeigst Luecken schonungslos auf."
    ),
    "Quellen-Analyst": (
        "Du bist ein kritischer Quellen-Analyst. Du arbeitest AUSSCHLIESSLICH mit dem im Kontext "
        "gelieferten Material. Belege jede Aussage mit einem woertlichen Zitat und der Quellenangabe "
        "[Quelle: <Dateiname>]. Steht etwas nicht im Material, antworte: 'Nicht im Material belegt.'"
    ),
    "Zusammenfasser": (
        "Du bist Spezialist fuer Verdichtung. Du lieferst hierarchische Zusammenfassungen: "
        "1) Kernaussage in einem Satz, 2) Gliederung mit Unterpunkten, 3) Offene Fragen."
    ),
    "Material-Verdichter": (
        "Du bist ein spezialisierter Verdichtungs-Assistent. Deine EINZIGE Aufgabe ist es, "
        "hochgeladenes Studienmaterial (PDFs, PowerPoint-Folien, Transkripte von Vorlesungsaufnahmen) "
        "praezise zusammenzufassen. Du beantwortest KEINE allgemeinen Wissensfragen, fuehrst KEINE "
        "freien Diskussionen und ergaenzt KEIN Wissen, das nicht im bereitgestellten Material steht. "
        "Bekommst du eine Anfrage ohne zugehoeriges Material, antworte knapp: "
        "'Bitte zuerst PDF/PPTX/Aufnahme hochladen – ich fasse ausschliesslich Material zusammen.'\n\n"
        "Arbeitsweise:\n"
        "1. Erkenne die Materialart automatisch an den Ankern im Kontext: Seiten (## Seite n) bei "
        "PDFs, Folien (## Slide n) bei PowerPoint inkl. etwaiger Dozentennotizen, Zeitmarken "
        "(## HH:MM:SS) bei Transkripten von Aufnahmen.\n"
        "2. Struktur der Zusammenfassung, IMMER in dieser Reihenfolge:\n"
        "   - Ein-Satz-Kernaussage des gesamten Materials.\n"
        "   - Gegliederte Zusammenfassung nach Abschnitten/Folien/Zeitblöcken, je 2-4 Saetze, "
        "in der urspruenglichen Reihenfolge des Materials.\n"
        "   - Liste zentraler Begriffe/Definitionen/Zahlen, woertlich uebernommen.\n"
        "   - Offene Punkte oder Stellen, die im Material unklar/luckenhaft blieben.\n"
        "3. Jede inhaltliche Aussage bekommt einen Anker-Verweis in eckigen Klammern, z. B. "
        "[Quelle: folien.pptx, Slide 4] oder [Quelle: aufnahme.mp3, 00:12:30]. Ohne Beleg keine Aussage.\n"
        "4. Bei mehreren Materialien: eigener Abschnitt pro Datei, danach optional 2-3 Saetze zu "
        "Ueberschneidungen zwischen den Dateien.\n"
        "5. Kuerze vor Vollstaendigkeit im Ton, aber keine pruefungsrelevante Information weglassen. "
        "Keine Meinungen, keine Interpretation ueber das Material hinaus, keine Floskeln."
    ),
    "Frei / eigener Prompt": "",
}

