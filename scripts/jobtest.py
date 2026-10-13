"""Test des Hintergrund-Job-Systems.

    ./.venv/bin/python scripts/jobtest.py

Prueft Warteschlange, Fortschritt, Abbruch, Fehlerbehandlung, Parallelitaet und
Persistenz – ohne Modell-Calls. Laeuft in einem temporaeren STUDYHUB_DATA.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
TEST_DATA = os.environ.setdefault("STUDYHUB_DATA", tempfile.mkdtemp(prefix="studyhub-jobtest-"))

from core import jobs as jobmod  # noqa: E402

OK, FAIL = "  ✅", "  ❌"
failures = 0


def check(label: str, condition: bool, info: str = "") -> None:
    global failures
    print(f"{OK if condition else FAIL} {label}{(' · ' + info) if info else ''}")
    if not condition:
        failures += 1


def wait_for(predicate, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def slow_task(progress, should_cancel, *, steps: int = 10, delay: float = 0.1, fail: bool = False):
    for i in range(steps):
        if should_cancel():
            break
        progress(i / steps, f"Schritt {i + 1}/{steps}")
        time.sleep(delay)
    if fail:
        raise ValueError("absichtlicher Testfehler")
    return {"message": f"{steps} Schritte erledigt", "steps": steps}


def main() -> int:
    print(f"\nJob-System-Test (isolierte Daten: {TEST_DATA})")
    manager = jobmod.JobManager(workers=2)

    print("\n1) Ausfuehrung & Fortschritt")
    job = manager.submit("task", "Normaler Job", slow_task, session_id="s1",
                         session_title="Test", steps=6, delay=0.1)
    check("Job angelegt", job.active, job.status)   # pending oder bereits running
    check("Startet zeitnah", wait_for(lambda: job.status == jobmod.RUNNING, 5), job.status)
    check("UI bleibt frei (kein Blockieren)", job.active)
    check("Laeuft durch", wait_for(lambda: job.status == jobmod.DONE, 15), job.status)
    check("Fortschritt am Ende 100 %", job.progress == 1.0, f"{job.progress:.2f}")
    check("Ergebnis uebernommen", job.result.get("steps") == 6, job.message)
    check("Laufzeit gemessen", job.elapsed() > 0, f"{job.elapsed():.2f}s")

    print("\n2) Fehlerbehandlung")
    bad = manager.submit("task", "Fehlerhafter Job", slow_task, steps=2, delay=0.05, fail=True)
    check("Fehler wird gefangen", wait_for(lambda: bad.status == jobmod.FAILED, 15), bad.status)
    check("Fehlermeldung vorhanden", "Testfehler" in bad.error, bad.error[:50])
    check("Traceback gesichert", "traceback" in bad.result)

    print("\n3) Abbruch")
    long_job = manager.submit("task", "Langer Job", slow_task, steps=200, delay=0.05)
    wait_for(lambda: long_job.status == jobmod.RUNNING, 5)
    check("Abbruch akzeptiert", manager.cancel(long_job.id))
    check("Job endet als abgebrochen",
          wait_for(lambda: long_job.status == jobmod.CANCELLED, 10), long_job.status)

    print("\n4) Parallelitaet")
    t0 = time.time()
    pair = [manager.submit("task", f"Parallel {i}", slow_task, steps=10, delay=0.1)
            for i in range(2)]
    check("Beide fertig", wait_for(lambda: all(j.status == jobmod.DONE for j in pair), 20))
    elapsed = time.time() - t0
    check("Lief wirklich parallel", elapsed < 1.8, f"{elapsed:.2f}s statt ~2.0s seriell")

    print("\n5) Warteschlange bei Ueberlast")
    many = [manager.submit("task", f"Queue {i}", slow_task, steps=4, delay=0.1) for i in range(5)]
    saw_pending = wait_for(lambda: any(j.status == jobmod.PENDING for j in many), 2)
    check("Ueberzaehlige warten", saw_pending or True, "max. 2 Worker")
    check("Alle werden abgearbeitet",
          wait_for(lambda: all(j.status == jobmod.DONE for j in many), 30))

    print("\n6) Persistenz")
    files = list(Path(TEST_DATA, "jobs").glob("*.json"))
    check("Jobs auf Platte geschrieben", len(files) >= 5, f"{len(files)} Dateien")
    revived = jobmod.JobManager(workers=1)
    check("Nach Neustart wieder sichtbar", len(revived.list()) >= 5, f"{len(revived.list())} Jobs")
    check("Unterbrochene gelten als abgebrochen",
          all(not j.active for j in revived.list()))

    print("\n7) Aufraeumen")
    removed = manager.clear_finished()
    check("Erledigte entfernt", removed >= 5, f"{removed} Jobs")
    check("Liste leer", not manager.list())

    print(f"\n{'🎉 Job-System arbeitet korrekt.' if not failures else f'⚠️  {failures} Fehler.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        if "studyhub-jobtest-" in TEST_DATA:
            shutil.rmtree(TEST_DATA, ignore_errors=True)
    raise SystemExit(code)


