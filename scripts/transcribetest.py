"""End-to-End-Test der Hintergrund-Transkription (mit echter Sprache).

    ./.venv/bin/python scripts/transcribetest.py

Erzeugt per macOS `say` eine kurze deutsche Sprachaufnahme, packt sie in ein
Video (prueft damit auch die Tonspur-Extraktion) und laesst sie als Hintergrund-Job
transkribieren. Zeigt dabei, dass der Hauptthread frei bleibt.

Benoetigt ein lokal vorhandenes Whisper-Modell (`scripts/fetch_whisper.py small`).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
TEST_DATA = os.environ.setdefault("STUDYHUB_DATA", tempfile.mkdtemp(prefix="studyhub-trtest-"))

from core import config, jobs as jobmod, library, session_store, tasks  # noqa: E402
from core.ingest.transcribe import model_cached  # noqa: E402

OK, FAIL = "  ✅", "  ❌"
TEXT = (
    "Willkommen zur ersten Lektion in Information Security Fundamentals. "
    "Wir behandeln heute die Schutzziele der Informationssicherheit. "
    "Vertraulichkeit bedeutet, dass Daten nur für Berechtigte zugänglich sind. "
    "Integrität stellt sicher, dass Daten nicht unbemerkt verändert werden. "
    "Verfügbarkeit garantiert, dass Systeme funktionieren, wenn man sie braucht."
)
failures = 0


def check(label: str, condition: bool, info: str = "") -> None:
    global failures
    print(f"{OK if condition else FAIL} {label}{(' · ' + info) if info else ''}")
    if not condition:
        failures += 1


def build_media(tmp: Path) -> Path | None:
    """Sprachaufnahme -> Video (damit auch die Audio-Extraktion getestet wird)."""
    aiff, video = tmp / "speech.aiff", tmp / "ISF_Vorlesung.mp4"
    if not shutil.which("say"):
        return None
    subprocess.run(["say", "-v", "Anna", "-o", str(aiff), TEXT],
                   check=True, stdin=subprocess.DEVNULL, capture_output=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(aiff),
         "-f", "lavfi", "-i", "color=c=blue:s=320x240", "-shortest",
         "-c:v", "libx264", "-c:a", "aac", str(video)],
        check=True, stdin=subprocess.DEVNULL, capture_output=True,
    )
    return video


def main() -> int:
    print(f"\nTranskriptions-Test (isolierte Daten: {TEST_DATA})")
    print(f"  Threads={config.WHISPER_THREADS} · Batch={config.WHISPER_BATCH_SIZE} "
          f"· Beam={config.WHISPER_BEAM_SIZE}")

    if not model_cached("small"):
        print("\n⚠️  Whisper-Modell «small» fehlt lokal – Test übersprungen.\n"
              "   Ausserhalb des VPN: ./.venv/bin/python scripts/fetch_whisper.py small")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="media-"))
    try:
        video = build_media(tmp)
        if video is None:
            print("⚠️  `say` nicht verfügbar – Test übersprungen.")
            return 0
        check("Testvideo erzeugt", video.exists(), f"{video.stat().st_size / 1024:.0f} KB")

        session = session_store.create(
            "gemma4:latest", "Akademischer Tutor",
            config.PERSONAS["Akademischer Tutor"], "ISF Woche 1",
        )
        manager = jobmod.JobManager(workers=2)
        job = manager.submit(
            "transcribe", video.name, tasks.ingest_task,
            session_id=session.id, session_title=session.title,
            path=str(video), display_name=video.name,
            whisper_model="small", language="de",
        )

        print("\n  Fortschritt (Hauptthread bleibt frei):")
        t0, seen, ticks = time.time(), set(), 0
        while job.active and time.time() - t0 < 300:
            key = f"{job.progress:.2f}"
            if key not in seen:
                seen.add(key)
                print(f"    [{time.time() - t0:5.1f}s] {job.progress * 100:5.1f}%  {job.message}")
            ticks += 1
            time.sleep(0.3)

        check("Job erfolgreich", job.status == jobmod.DONE, job.status or job.error[:80])
        check("Hauptthread war nutzbar", ticks > 5, f"{ticks} Schleifendurchläufe parallel")
        check("Fortschritt war abgestuft", len(seen) >= 3, f"{len(seen)} Stufen")
        if job.status != jobmod.DONE:
            print("   ", job.error[:300])
            return 1

        mat = library.get(job.result["material_id"])
        text = mat.text().lower()
        check("Material angelegt", mat is not None)
        check("Tonspur extrahiert", bool(mat.meta.get("audio_extracted")))
        check("Batching aktiv", bool(mat.meta.get("batched")),
              f"{mat.meta.get('speed_factor')}x Echtzeit")
        check("Sprache erkannt", mat.meta.get("language") == "de")
        check("Zeitanker vorhanden", "## 00:" in mat.text())
        hits = [w for w in ("sicherheit", "vertraulichkeit", "integrität", "daten") if w in text]
        check("Inhalt korrekt transkribiert", len(hits) >= 2, ", ".join(hits))
        check("Der Session zugeordnet",
              mat.id in (session_store.load(session.id) or session).material_ids)

        print("\n  --- Transkript (Auszug) ---")
        print("  " + mat.text()[:300].replace("\n", "\n  "))
        print(f"\n  Kennzahlen: {mat.meta.get('duration')} Audio in "
              f"{mat.meta.get('transcribe_s')}s = {mat.meta.get('speed_factor')}x Echtzeit")
        est = 90 * 60 / max(mat.meta.get("speed_factor", 1), 0.1) / 60
        print(f"  Hochgerechnet auf 90 Min Vorlesung: ca. {est:.0f} Minuten")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'🎉 Transkription arbeitet korrekt.' if not failures else f'⚠️  {failures} Fehler.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        if "studyhub-trtest-" in TEST_DATA:
            shutil.rmtree(TEST_DATA, ignore_errors=True)
    raise SystemExit(code)

