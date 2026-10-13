"""UI-Rendertest mit Streamlit AppTest – prueft, dass alle Ansichten exception-frei rendern.

    ./.venv/bin/python scripts/uitest.py

Legt temporaer ein Projekt mit zwei Lektionen + Projekt-Session an, rendert
Startseite, Lektions-Session und Projekt-Session (alle Tabs) und raeumt wieder auf.
Es werden keine Modell-Calls ausgeloest.

WICHTIG: Der Test laeuft in einem eigenen Datenverzeichnis (STUDYHUB_DATA),
damit Testdaten nie in den echten Sessions/Projekten landen.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Muss vor dem Import von core stehen – config liest die Pfade beim Import.
TEST_DATA = os.environ.setdefault("STUDYHUB_DATA", tempfile.mkdtemp(prefix="studyhub-uitest-"))

from streamlit.testing.v1 import AppTest  # noqa: E402

from core import config, library, project_store, session_store  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
OK, FAIL = "  ✅", "  ❌"


def render(label: str, sid: str | None) -> bool:
    at = AppTest.from_file(APP, default_timeout=60)
    if sid:
        at.session_state["sid"] = sid
    at.run()
    if at.exception:
        print(f"{FAIL} {label}")
        for exc in at.exception:
            print(f"      {exc.value}")
        return False
    print(f"{OK} {label} · {len(at.button)} Buttons, {len(at.tabs)} Tab-Container")
    return True


def check(label: str, condition: bool, info: str = "") -> bool:
    print(f"{OK if condition else FAIL} {label}{(' · ' + info) if info else ''}")
    return condition


def session_isolation(project, lesson, loose) -> bool:
    """Regressionstest: Widget-State darf nicht zwischen Sessions lecken.

    Frueher hatten die Einstellungs-Widgets globale Keys ("cfg_model", "assign_project").
    Dadurch wirkte z. B. die Projektzuordnung auf jede geoeffnete Session.
    """
    ok = True
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["sid"] = lesson.id
    at.run()
    keys = [k for k in at.session_state.filtered_state if k.startswith("cfg_")]
    ok &= check("Einstellungs-Keys sind sessionspezifisch",
                all(k.endswith(lesson.id) for k in keys) and bool(keys),
                f"{len(keys)} Keys mit Session-Suffix")

    # Zweite Session im selben Browser-State oeffnen (wie beim Klick in der Sidebar)
    at.session_state["sid"] = loose.id
    at.run()
    ok &= check("Kein Renderfehler nach Sessionwechsel", not at.exception)

    reloaded_loose = session_store.load(loose.id)
    reloaded_lesson = session_store.load(lesson.id)
    ok &= check("Lose Session bleibt ohne Projekt", reloaded_loose.project_id == "")
    ok &= check("Lektion bleibt im Projekt", reloaded_lesson.project_id == project.id)

    state = dict(at.session_state.filtered_state)
    ok &= check("Projekt-Auswahl zeigt echten Wert der Session",
                state.get(f"cfg_project_{loose.id}", "") == "",
                f"cfg_project_{loose.id[:6]}… = {state.get(f'cfg_project_{loose.id}', '')!r}")
    ok &= check("Alter Session-State wirkt nicht mit",
                state.get(f"cfg_project_{lesson.id}", project.id) == project.id)
    return ok


def main() -> int:
    print(f"\nUI-Rendertest (isolierte Daten: {TEST_DATA})")
    project = project_store.create("UITest-Modul", "Semester-Test", "gemma4:latest",
                                   "Akademischer Tutor", config.PERSONAS["Akademischer Tutor"])
    lesson = session_store.create("gemma4:latest", "Akademischer Tutor",
                                  config.PERSONAS["Akademischer Tutor"],
                                  "Woche 1 – Grundlagen", "Grundlagen",
                                  project_id=project.id, kind="lesson")
    lesson.add("user", "Was ist der Median?")
    lesson.add("assistant", "Der Median ist der mittlere Wert einer sortierten Reihe.")
    lesson.messages[0].archived = True          # archivierte Nachricht rendern
    lesson.summaries = [{"content": "- Median besprochen", "ts": 0, "covered": 1}]
    lesson.set_digest("- Thema: Median\n- Kernpunkte: mittlerer Wert")
    lesson.save()

    # Material in drei Scopes: Projekt, nur-Lektion, unzugeordnet (Altbestand)
    shared = library.add("skript.md", "## Seite 1\n\nProjektweites Skript.", "document", "test",
                         project_id=project.id, session_id=lesson.id)
    private = library.add("notiz.md", "Private Notiz.", "note", "manuell", session_id=lesson.id)
    orphan = library.add("altbestand.md", "Ohne Zuordnung.", "note", "manuell")
    lesson.material_ids = [shared.id, private.id]
    lesson.save()

    empty = session_store.create("gemma4:latest", "Akademischer Tutor",
                                 config.PERSONAS["Akademischer Tutor"],
                                 "Woche 2 – leer", "", project_id=project.id, kind="lesson")
    psession = project_store.get_or_create_project_session(project)
    loose = session_store.create("gemma4:latest", "Zusammenfasser",
                                 config.PERSONAS["Zusammenfasser"], "Lose Session")

    ok = True
    ok &= render("Startseite (Projektkacheln)", None)
    ok &= render("Lektions-Session", lesson.id)
    ok &= render("Leere Lektion", empty.id)
    ok &= render("Projekt-Session (4 Tabs)", psession.id)
    ok &= render("Session ohne Projekt", loose.id)

    print("\nSession-Isolation")
    ok &= session_isolation(project, lesson, loose)

    project_store.delete(project.id, delete_sessions=True)
    session_store.delete(loose.id)
    for mat in (shared, private, orphan):
        library.delete(mat.id)
    ok &= bool(print(f"{OK} Aufgeraeumt") is None)

    print("\n🎉 UI rendert fehlerfrei." if ok else "\n⚠️  Renderfehler – siehe oben.")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        code = main()
    finally:
        # Testverzeichnis restlos entfernen (nur, wenn wir es selbst angelegt haben).
        if "studyhub-uitest-" in TEST_DATA:
            shutil.rmtree(TEST_DATA, ignore_errors=True)
    raise SystemExit(code)

