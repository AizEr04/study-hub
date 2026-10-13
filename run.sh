#!/usr/bin/env bash
# Startet den Study Hub. Legt beim ersten Lauf die venv an.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "▶ Erstelle virtuelle Umgebung (benötigt Internetzugang / ggf. VPN aus)…"
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

if ! curl -s --max-time 2 http://localhost:11434/api/tags > /dev/null; then
  echo "⚠  Ollama antwortet nicht – starte es in einem zweiten Terminal mit: ollama serve"
fi

exec ./.venv/bin/streamlit run app.py --server.port "${PORT:-8501}" --browser.gatherUsageStats false

