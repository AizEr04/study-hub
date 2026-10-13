#!/usr/bin/env bash
# Installiert alle Abhängigkeiten des Study Hubs.
# WICHTIG: Ausserhalb des Firmen-VPN ausführen (PyPI ist im VPN blockiert).
#
#   ./setup.sh
#
set -euo pipefail
cd "$(dirname "$0")"

# --- 1) Python-Version prüfen ------------------------------------------------
PYTHON="$(command -v python3 || true)"
if [ -z "$PYTHON" ]; then
  echo "❌ python3 nicht gefunden. Installation: https://www.python.org/downloads/"
  exit 1
fi
PY_VER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="${PY_VER%%.*}"; PY_MINOR="${PY_VER##*.}"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  echo "❌ Python $PY_VER ist zu alt – benötigt wird >= 3.10."
  exit 1
fi
echo "▶ Python $PY_VER gefunden."

# --- 2) ffmpeg (für Audio/Video-Transkription) -------------------------------
if ! command -v ffmpeg > /dev/null; then
  echo "⚠  ffmpeg fehlt – Audio/Video kann nicht transkribiert werden."
  if command -v brew > /dev/null; then
    echo "▶ Installiere ffmpeg via Homebrew…"
    brew install ffmpeg
  else
    echo "   Bitte manuell installieren: brew install ffmpeg"
  fi
else
  echo "▶ ffmpeg gefunden."
fi

# --- 3) Virtuelle Umgebung ---------------------------------------------------
if [ ! -d .venv ]; then
  echo "▶ Erstelle .venv…"
  "$PYTHON" -m venv .venv
fi
PIP="./.venv/bin/pip"

echo "▶ Aktualisiere pip…"
"$PIP" install --upgrade pip

# --- 4) Abhängigkeiten installieren ------------------------------------------
echo "▶ Installiere requirements.txt…"
"$PIP" install -r requirements.txt

# --- 5) Verifikation ----------------------------------------------------------
echo "▶ Prüfe Installation…"
./.venv/bin/python - <<'EOF'
import importlib, sys
mods = {
    "streamlit": "streamlit", "requests": "requests", "pypdf": "pypdf",
    "pptx": "python-pptx", "docx": "python-docx", "bs4": "beautifulsoup4",
    "tiktoken": "tiktoken", "faster_whisper": "faster-whisper",
}
missing = []
for mod, pkg in mods.items():
    try:
        importlib.import_module(mod)
        print(f"  ✅ {pkg}")
    except ImportError:
        missing.append(pkg)
        print(f"  ❌ {pkg}")
if missing:
    print("\nFehlende Pakete: " + ", ".join(missing))
    sys.exit(1)
print("\n🎉 Alles installiert.")
EOF

# --- 6) Whisper-Modell vorladen (nur hier moeglich – im VPN ist huggingface.co blockiert) ---
echo ""
echo "▶ Whisper-Modell bereitstellen (für Vorlesungsaufzeichnungen)…"
./.venv/bin/python scripts/fetch_whisper.py "${WHISPER_MODEL:-small}" || \
  echo "⚠  Whisper-Modell konnte nicht geladen werden – später nachholen mit:
     ./.venv/bin/python scripts/fetch_whisper.py small"

# --- 7) Ollama-Hinweis --------------------------------------------------------
if curl -s --max-time 2 http://localhost:11434/api/tags > /dev/null; then
  echo "▶ Ollama läuft."
else
  echo "⚠  Ollama antwortet nicht – vor dem Start ausführen: ollama serve"
fi

echo ""
echo "Fertig. Start mit:  ./run.sh   (→ http://localhost:8501)"

