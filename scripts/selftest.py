"""End-to-End-Selbsttest ohne UI.

    ./.venv/bin/python scripts/selftest.py [--chat]

Prueft: Ollama-Verbindung, Ingest (TXT/PPTX/VTT), Chunking, Retrieval,
Token-Budget, Payload-Aufbau, Projekte/Scope und optional einen echten Streaming-Call.

WICHTIG: Laeuft in einem eigenen Datenverzeichnis (STUDYHUB_DATA), damit Testdaten
nie in den echten Sessions/Projekten landen.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Muss vor dem Import von core stehen – config liest die Pfade beim Import.
TEST_DATA = os.environ.setdefault("STUDYHUB_DATA", tempfile.mkdtemp(prefix="studyhub-selftest-"))

from core import config, context_manager as ctx, library, ollama_client, project_store, session_store  # noqa: E402
from core.ingest import ingest_file  # noqa: E402

TMP = config.DATA_DIR / "_selftest"
TMP.mkdir(parents=True, exist_ok=True)
OK, FAIL = "  ✅", "  ❌"


def check(label: str, condition: bool, info: str = "") -> bool:
    print(f"{OK if condition else FAIL} {label}{(' · ' + info) if info else ''}")
    return condition


def make_txt() -> Path:
    p = TMP / "vorlesung.txt"
    p.write_text(
        "## Seite 1\n\nHypothesentests pruefen eine Nullhypothese H0 gegen eine Alternative H1.\n\n"
        "## Seite 2\n\nDer p-Wert ist die Wahrscheinlichkeit, unter H0 ein mindestens so extremes "
        "Ergebnis zu beobachten. Signifikanzniveau alpha ist typischerweise 0.05.\n\n"
        "## Seite 3\n\nDer Fehler 1. Art bedeutet, H0 faelschlich zu verwerfen. "
        "Der Fehler 2. Art bedeutet, H0 faelschlich beizubehalten.\n",
        encoding="utf-8",
    )
    return p


def make_pptx() -> Path | None:
    try:
        from pptx import Presentation
        from pptx.util import Inches
    except ImportError:
        return None
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Konfidenzintervalle"
    slide.placeholders[1].text = "95%-KI ueberdeckt den wahren Parameter in 95% der Stichproben."
    slide.notes_slide.notes_text_frame.text = "Pruefungsrelevant: Interpretation des KI."
    slide2 = prs.slides.add_slide(prs.slide_layouts[5])
    slide2.shapes.title.text = "Teststaerke"
    slide2.shapes.add_textbox(Inches(1), Inches(2), Inches(6), Inches(1)).text_frame.text = (
        "Power = 1 - beta, steigt mit Stichprobengroesse."
    )
    p = TMP / "slides.pptx"
    prs.save(str(p))
    return p


def make_vtt() -> Path:
    p = TMP / "lektion.vtt"
    p.write_text(
        "WEBVTT\n\n1\n00:00:01.000 --> 00:00:05.000\nWillkommen zur Lektion ueber Hypothesentests.\n\n"
        "2\n00:00:05.000 --> 00:00:09.000\nWir betrachten heute den p-Wert im Detail.\n",
        encoding="utf-8",
    )
    return p


def main() -> int:
    failures = 0
    print(f"\nSelbsttest (isolierte Daten: {TEST_DATA})")
    print("\n1) Ollama")
    online = ollama_client.is_online()
    failures += not check("Server erreichbar", online, config.OLLAMA_HOST)
    if not online:
        print("\nOllama starten: `ollama serve`")
        return 1
    models = [m["name"] for m in ollama_client.list_models()]
    failures += not check("Modelle gefunden", bool(models), ", ".join(models))
    model = models[0]
    max_ctx = ollama_client.model_context_length(model)
    check("Kontextfenster ermittelt", bool(max_ctx), f"{model} → {max_ctx}")

    print("\n2) Ingest")
    materials = []
    for maker, name in ((make_txt, "TXT"), (make_pptx, "PPTX"), (make_vtt, "VTT")):
        path = maker()
        if path is None:
            print(f"{FAIL} {name} uebersprungen (Paket fehlt)")
            continue
        try:
            mat = ingest_file(path)
            materials.append(mat)
            check(f"{name} → Material", mat.n_chars > 0, f"{mat.kind}, ~{mat.n_tokens_est} Tok")
        except Exception as exc:
            failures += 1
            print(f"{FAIL} {name}: {exc}")
    failures += not check("Notizen aus PPTX uebernommen",
                          any("Notizen" in m.text() for m in materials if m.kind == "slides"))

    print("\n3) Chunking & Retrieval")
    chunks = [c for m in materials for c in library.chunk_text(m, target_chars=200, overlap=0)]
    failures += not check("Chunks erzeugt", len(chunks) > 1, f"{len(chunks)} Stueck")
    top = ctx.rank_chunks(chunks, "Was ist der Fehler 1. Art?", top_k=2)
    failures += not check("Retrieval trifft Thema", any("Fehler 1" in c.text for c in top),
                          top[0].text[:60].replace("\n", " ") if top else "")

    # Regression: eine grosse Aufzeichnung (ein Block ohne Absaetze) muss trotzdem in
    # viele kleine Chunks zerfallen UND im Payload landen. Frueher -> 1 Riesen-Chunk,
    # der das Material-Budget sprengte und komplett verworfen wurde (Material fehlte).
    big_text = "## 00:00:00\n" + (
        "Vertraulichkeit bedeutet dass Daten nur fuer Berechtigte zugaenglich sind. "
        "Integritaet stellt sicher dass Daten nicht unbemerkt veraendert werden. "
    ) * 400
    big = library.add("Vorlesung.mp4", big_text, "transcript", "test-whisper")
    big_chunks = library.chunk_text(big)
    biggest = max(ctx.estimate_tokens(c.text) for c in big_chunks)
    failures += not check("Grosse Aufzeichnung wird gechunkt", len(big_chunks) >= 5,
                          f"{len(big_chunks)} Chunks, groesster {biggest} Tok")
    failures += not check("Kein Chunk sprengt uebliches Budget", biggest <= 1500,
                          f"groesster {biggest} Tok")

    big_sess = session_store.create(model, "Akademischer Tutor",
                                    config.PERSONAS["Akademischer Tutor"], "Aufzeichnung")
    big_sess.material_ids = [big.id]
    big_sess.num_ctx = 8192
    big_sess.add("user", "Fasse die Vorlesungsaufzeichnung zusammen.")
    big_sess.save()
    _, big_report = ctx.build_payload(big_sess, "Fasse die Vorlesungsaufzeichnung zusammen.")
    failures += not check("Grosse Aufzeichnung landet im Kontext",
                          bool(big_report["material"]["used"]),
                          f"Modus {big_report['material']['mode']}, "
                          f"{len(big_report['material']['used'])} Quellen, "
                          f"{big_report['material']['tokens']} Tok")
    failures += not check("Aufzeichnung bleibt im Budget",
                          big_report["prompt_tokens_est"] <= big_report["budget"].usable,
                          f"{big_report['prompt_tokens_est']} / {big_report['budget'].usable}")
    session_store.delete(big_sess.id)
    library.delete(big.id)

    print("\n4) Session & Kontext-Budget")
    session = session_store.create(model, "Akademischer Tutor",
                                   config.PERSONAS["Akademischer Tutor"], "Selftest", "Statistik")
    session.material_ids = [m.id for m in materials]
    session.num_ctx = min(8192, max_ctx or 8192)
    session.add("user", "Erklaere den p-Wert.")
    session.save()
    payload, report = ctx.build_payload(session)
    failures += not check("Payload gebaut", payload[0]["role"] == "system", f"{len(payload)} Nachrichten")
    failures += not check("Material im Kontext", bool(report["material"]["used"]),
                          f"Modus {report['material']['mode']}: {report['material']['used']}")
    failures += not check("Budget eingehalten",
                          report["prompt_tokens_est"] <= report["budget"].usable,
                          f"{report['prompt_tokens_est']} / {report['budget'].usable} Tokens")

    session.num_ctx = 2048  # erzwingt Retrieval statt Volltext
    payload_small, report_small = ctx.build_payload(session)
    failures += not check("Kleines Fenster → Retrieval-Modus",
                          report_small["material"]["mode"] in ("retrieval", "full"),
                          report_small["material"]["mode"])
    failures += not check("Auch klein im Budget",
                          report_small["prompt_tokens_est"] <= report_small["budget"].usable,
                          f"{report_small['prompt_tokens_est']} / {report_small['budget'].usable}")
    session.num_ctx = min(8192, max_ctx or 8192)

    print("\n5) Projekte & Gesamtkontext")
    project = project_store.create("Selftest-Projekt", "Semester-Test", model, "Akademischer Tutor",
                                   config.PERSONAS["Akademischer Tutor"])
    lessons = []
    for week, (topic, mat) in enumerate(zip(["Deskriptive Statistik", "Hypothesentests"],
                                            materials[:2]), start=1):
        lesson = session_store.create(model, "Akademischer Tutor",
                                      config.PERSONAS["Akademischer Tutor"],
                                      f"Woche {week} – {topic}", topic,
                                      project_id=project.id, kind="lesson")
        lesson.material_ids = [mat.id]
        # Material projektweit freigeben – nur so erbt es die Projekt-Session.
        library.update(mat.id, project_id=project.id, session_id=lesson.id)
        lesson.add("user", f"Erklaere {topic}.")
        lesson.add("assistant", f"{topic} umfasst die Kernidee X und das Beispiel Y. " * 5)
        lesson.set_digest(f"- Thema: {topic}\n- Kernpunkte: Kernidee X, Beispiel Y\n- Offene Fragen: keine")
        lesson.save()
        lessons.append(lesson)

    failures += not check("Lektionen zugeordnet", len(project_store.sessions_of(project.id, "lesson")) == 2)
    psession = project_store.get_or_create_project_session(project)
    failures += not check("Projekt-Session erstellt", psession.kind == "project", psession.title)
    failures += not check("Idempotent (keine Duplikate)",
                          project_store.get_or_create_project_session(project).id == psession.id)

    inherited = ctx.effective_material_ids(psession)
    failures += not check("Material aus Lektionen geerbt", len(inherited) == 2, f"{len(inherited)} Materialien")

    psession.num_ctx = min(8192, max_ctx or 8192)
    psession.save()
    p_payload, p_report = ctx.build_payload(psession, "Wie haengen die Wochen zusammen?")
    dig = p_report["digests"]
    failures += not check("Semester-Gedaechtnis im Payload", len(dig["lessons"]) == 2,
                          ", ".join(dig["lessons"]))
    failures += not check("Digest-Block als system-Nachricht",
                          any("SEMESTER-GEDAECHTNIS" in m["content"] for m in p_payload))
    failures += not check("Projekt-Budget eingehalten",
                          p_report["prompt_tokens_est"] <= p_report["budget"].usable,
                          f"{p_report['prompt_tokens_est']} / {p_report['budget'].usable}")
    failures += not check("Digest-Budget reserviert", p_report["budget"].digest_max > 0,
                          f"{p_report['budget'].digest_max} Tokens")

    # Stale-Erkennung: neue Nachricht in einer Lektion -> Digest veraltet
    lessons[0].add("user", "Und wie war das mit dem Median?")
    lessons[0].save()
    failures += not check("Veralteter Digest erkannt", lessons[0].digest_stale())
    _, p_report2 = ctx.build_payload(psession, "Median?")
    failures += not check("Veralteter Digest wird gemeldet", lessons[0].title in p_report2["digests"]["stale"])

    # Enges Fenster -> relevanteste Lektionen gewinnen
    psession.num_ctx = 2048
    psession.save()
    _, p_report3 = ctx.build_payload(psession, "Hypothesentests")
    failures += not check("Enges Fenster bleibt im Budget",
                          p_report3["prompt_tokens_est"] <= p_report3["budget"].usable,
                          f"{p_report3['prompt_tokens_est']} / {p_report3['budget'].usable}")
    psession.num_ctx = min(8192, max_ctx or 8192)
    psession.save()

    print("\n6) Scope: Material bleibt im Projekt bzw. in der Session")
    # Material der Lektionen gehoert dem Projekt -> muss projektweit sichtbar sein
    for mat in materials[:2]:
        library.update(mat.id, project_id=project.id, session_id=lessons[0].id)
    failures += not check("Projekt-Material in Lektion sichtbar",
                          len(library.visible_for(lessons[1])) >= 2,
                          f"{len(library.visible_for(lessons[1]))} sichtbar")

    # Privates Material einer Lektion darf NICHT im Gesamtkontext landen
    private = library.add("privat.md", "Geheime Notiz nur fuer Woche 1.", "note", "manuell",
                          project_id="", session_id=lessons[0].id)
    lessons[0].material_ids.append(private.id)
    lessons[0].save()
    failures += not check("Privates Material in eigener Lektion sichtbar",
                          library.is_visible(private, lessons[0]))
    failures += not check("Privates Material NICHT in anderer Lektion",
                          not library.is_visible(private, lessons[1]))
    failures += not check("Privates Material NICHT im Gesamtkontext",
                          private.id not in ctx.effective_material_ids(psession))

    # Alleinstehende Session sieht nichts vom Projekt
    solo = session_store.create(model, "Zusammenfasser", config.PERSONAS["Zusammenfasser"], "Solo")
    failures += not check("Alleinstehende Session sieht kein Projekt-Material",
                          not library.visible_for(solo), f"{len(library.visible_for(solo))} sichtbar")
    solo_mat = library.add("solo.md", "Nur fuer die Solo-Session.", "note", "manuell",
                           project_id="", session_id=solo.id)
    solo.material_ids.append(solo_mat.id)
    solo.save()
    failures += not check("Eigenes Material der Solo-Session sichtbar",
                          [m.id for m in library.visible_for(solo)] == [solo_mat.id])

    print("\n7) Session nachtraeglich zuordnen / herausloesen")
    result = project_store.assign(solo, project.id)
    solo = session_store.load(solo.id)
    failures += not check("Session ins Projekt verschoben", solo.project_id == project.id,
                          f"{result['moved']} Material(ien) umgehaengt")
    failures += not check("Material wanderte ins Projekt",
                          (library.get(solo_mat.id) or solo_mat).project_id == project.id)
    failures += not check("Nun auch in anderen Lektionen sichtbar",
                          library.is_visible(library.get(solo_mat.id), lessons[1]))
    failures += not check("Gesamtkontext erbt es jetzt",
                          solo_mat.id in ctx.effective_material_ids(session_store.load(psession.id)))

    project_store.assign(solo, "")
    solo = session_store.load(solo.id)
    failures += not check("Session wieder herausgeloest", solo.project_id == "")
    failures += not check("Material wieder sessionprivat",
                          (library.get(solo_mat.id) or solo_mat).project_id == ""
                          and not library.is_visible(library.get(solo_mat.id), lessons[1]))

    print("\n8) Material bearbeiten")
    library.save_text(solo_mat.id, "Korrigierter Text mit Umlauten: Prüfung, Übung.")
    edited = library.get(solo_mat.id)
    failures += not check("Text gespeichert", "Prüfung" in edited.text())
    failures += not check("Tokenzahl neu berechnet", edited.n_tokens_est != solo_mat.n_tokens_est,
                          f"{solo_mat.n_tokens_est} → {edited.n_tokens_est}")
    library.update(solo_mat.id, name="umbenannt.md")
    failures += not check("Umbenennen", library.get(solo_mat.id).name == "umbenannt.md")

    session_store.delete(solo.id)
    library.delete(solo_mat.id)
    library.delete(private.id)

    if "--chat" in sys.argv:
        print("\n9) Live-Call (streaming)")
        text, stats = "", {}
        for ev in ollama_client.chat_stream(model, payload, num_ctx=session.num_ctx, temperature=0.2):
            if ev["type"] == "token":
                text += ev["text"]
            elif ev["type"] == "done":
                stats = ev["stats"]
        failures += not check("Antwort erhalten", len(text) > 20, f"{len(text)} Zeichen")
        check("Server-Tokenzaehlung", bool(stats),
              f"prompt={stats.get('prompt_tokens')} eval={stats.get('completion_tokens')} "
              f"({stats.get('total_duration_s')}s)")
        if stats.get("prompt_tokens"):
            drift = abs(stats["prompt_tokens"] - report["prompt_tokens_est"]) / stats["prompt_tokens"]
            check("Schaetzung vs. Realitaet", drift < 0.5,
                  f"geschaetzt {report['prompt_tokens_est']}, real {stats['prompt_tokens']} "
                  f"({drift * 100:.0f}% Abweichung)")
        print(f"\n--- Antwort (Auszug) ---\n{text[:400]}\n")

        print("10) Lektions-Digest via Modell")
        digest = ctx.build_lesson_digest(lessons[0], model, force=True)
        failures += not check("Digest erzeugt", bool(digest), digest[:70].replace("\n", " "))
        failures += not check("Digest nicht mehr veraltet", not lessons[0].digest_stale())
        results = ctx.refresh_project_digests(project.id, model)
        failures += not check("Alle Projekt-Digests aktuell", all(ok for _, ok in results),
                              f"{sum(ok for _, ok in results)}/{len(results)}")

        print("11) Memory Compression")
        for i in range(6):
            session.add("user", f"Frage {i}: Erklaere Aspekt {i} des p-Werts ausfuehrlich.")
            session.add("assistant", f"Antwort {i}: " + "Ausfuehrliche Erklaerung. " * 40)
        session.save()
        summary = ctx.compress_history(session)
        failures += not check("Zusammenfassung erzeugt", bool(summary),
                              (summary or "")[:70].replace("\n", " "))
        failures += not check("Alte Turns archiviert",
                              any(m.archived for m in session.messages))
        after = ctx.build_payload(session)[1]
        check("Kontext danach kleiner", after["fill"] < 1.0, f"Auslastung {after['fill'] * 100:.0f}%")

    print("\n12) Aufraeumen")
    project_store.delete(project.id, delete_sessions=True)
    session_store.delete(session.id)
    for m in materials:
        library.delete(m.id)
    check("Testdaten entfernt",
          session_store.load(session.id) is None
          and project_store.get(project.id) is None
          and not project_store.sessions_of(project.id))

    print(f"\n{'🎉 Alle Checks bestanden.' if not failures else f'⚠️  {failures} Check(s) fehlgeschlagen.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        if "studyhub-selftest-" in TEST_DATA:
            shutil.rmtree(TEST_DATA, ignore_errors=True)
    raise SystemExit(code)

