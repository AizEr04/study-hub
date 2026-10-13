"""Whisper-Modellgewichte vorab herunterladen.

    ./.venv/bin/python scripts/fetch_whisper.py [small medium ...]

MUSS ausserhalb des Firmen-VPN laufen – die Gewichte kommen von huggingface.co.
Danach funktioniert die Transkription vollstaendig offline.

Groessen (Download / RAM / Tempo bei 90-Min-Vorlesung auf Apple Silicon CPU):
    tiny      75 MB   ~1 GB    ~8 min    grobe Stichworte
    base     145 MB   ~1 GB   ~12 min
    small    484 MB   ~2 GB   ~25 min    guter Kompromiss (Default)
    medium   1.5 GB   ~5 GB   ~60 min    fuer Fachvokabular/schlechten Ton
    large-v3 3.1 GB   ~10 GB  ~120 min   maximale Qualitaet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config  # noqa: E402

SIZES = ["tiny", "base", "small", "medium", "large-v3"]


def cached(size: str) -> bool:
    """Liegen die Gewichte bereits im HuggingFace-Cache?"""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    return any(hub.glob(f"models--Systran--faster-whisper-{size}/snapshots/*/model.bin"))


def main() -> int:
    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        print("Bereits lokal verfuegbar:")
        for size in SIZES:
            print(f"  {'✅' if cached(size) else '⬇️ '} {size}")
        return 0

    wanted = [a for a in sys.argv[1:] if not a.startswith("-")] or [config.WHISPER_MODEL]
    unknown = [w for w in wanted if w not in SIZES]
    if unknown:
        print(f"❌ Unbekannte Groesse: {', '.join(unknown)}\n   Erlaubt: {', '.join(SIZES)}")
        return 1

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("❌ faster-whisper fehlt. Zuerst `./setup.sh` ausserhalb des VPN ausfuehren.")
        return 1

    os.environ["STUDYHUB_ALLOW_WHISPER_DOWNLOAD"] = "1"  # hier ist der Download erwuenscht
    print("\nWhisper-Modelle bereitstellen")
    failed = []
    for size in wanted:
        if cached(size):
            print(f"  ✅ {size} · bereits im Cache")
            continue
        print(f"  ⏳ {size} · lade herunter (einmalig, kann einige Minuten dauern)…")
        try:
            WhisperModel(size, device="cpu", compute_type=config.WHISPER_COMPUTE)
            print(f"  ✅ {size} · bereit")
        except Exception as exc:  # Netzwerk, VPN, Plattenplatz …
            failed.append(size)
            print(f"  ❌ {size} · {str(exc)[:160]}")

    if failed:
        print(
            "\n⚠️  Download fehlgeschlagen. Haeufigste Ursache: Firmen-VPN blockiert "
            "huggingface.co.\n   VPN trennen und erneut ausfuehren."
        )
        return 1

    print("\n🎉 Fertig – Transkription laeuft ab jetzt vollstaendig offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




