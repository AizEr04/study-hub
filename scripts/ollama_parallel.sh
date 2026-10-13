#!/usr/bin/env bash
# Ollama fuer parallele Sessions konfigurieren.
#
#   ./scripts/ollama_parallel.sh [anzahl] [modelle_im_speicher]
#
# Ohne diese Einstellung serialisiert Ollama die Anfragen: Session B wartet,
# bis Session A fertig geantwortet hat. Mit OLLAMA_NUM_PARALLEL koennen
# mehrere Sessions gleichzeitig mit demselben Modell arbeiten.
#
# Achtung Speicher: Jede parallele Anfrage belegt zusaetzlichen KV-Cache.
# Faustregel bei gemma4:26b (~18 GB) und num_ctx 8192: 2-3 parallel sind realistisch.
set -euo pipefail

PARALLEL="${1:-3}"
MAX_MODELS="${2:-2}"

echo "▶ Setze OLLAMA_NUM_PARALLEL=$PARALLEL, OLLAMA_MAX_LOADED_MODELS=$MAX_MODELS"
launchctl setenv OLLAMA_NUM_PARALLEL "$PARALLEL"
launchctl setenv OLLAMA_MAX_LOADED_MODELS "$MAX_MODELS"
# Modell nach dem letzten Turn laenger im RAM halten -> kein Nachladen (30-60 s) pro Session
launchctl setenv OLLAMA_KEEP_ALIVE "30m"

echo "▶ Starte Ollama neu, damit die Werte greifen…"
if pgrep -x "Ollama" > /dev/null; then
  osascript -e 'quit app "Ollama"' 2>/dev/null || true
  sleep 3
  open -a Ollama
  sleep 4
else
  pkill -x ollama 2>/dev/null || true
  sleep 2
  (nohup ollama serve > /tmp/ollama.log 2>&1 &)
  sleep 4
fi

if curl -s --max-time 5 http://localhost:11434/api/tags > /dev/null; then
  echo "✅ Ollama laeuft mit $PARALLEL parallelen Slots."
  echo "   Pruefen: launchctl getenv OLLAMA_NUM_PARALLEL"
else
  echo "⚠️  Ollama antwortet noch nicht – ggf. manuell starten: ollama serve"
fi

