"""Konkrete Hintergrund-Aufgaben (das, was `jobs.JobManager` ausfuehrt).

Jede Funktion hat die Signatur `(progress, should_cancel, **kwargs) -> dict`.
Sie laeuft in einem Worker-Thread – deshalb hier **kein** Streamlit-Aufruf und
kein Zugriff auf `st.session_state`. Ergebnisse gehen ueber die Platte
(`session.save()`, `library.add()`) zurueck in die UI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from . import context_manager as ctx
from . import library, ollama_client, project_store, session_store, study_tools
from .ingest import ingest_file


def _attach(session_id: str, material_id: str) -> None:
    """Material der Session zuordnen – frisch laden, da die UI sie zwischenzeitlich
    veraendert haben kann (der Job laeuft minutenlang parallel)."""
    session = session_store.load(session_id)
    if session is None:
        return
    if material_id not in session.material_ids:
        session.material_ids.append(material_id)
        session.save()


def ingest_task(
    progress: Callable[[float, str], None],
    should_cancel: Callable[[], bool],
    *,
    path: str,
    display_name: str,
    session_id: str,
    project_id: str = "",
    whisper_model: str | None = None,
    language: str | None = None,
    auto_summarize: bool = False,
    summary_model: str = "",
    num_ctx: int = 8192,
) -> dict[str, Any]:
    """Datei extrahieren/transkribieren, der Session zuordnen, optional zusammenfassen."""
    share = 0.85 if auto_summarize else 1.0   # Restanteil fuer die Zusammenfassung

    def forward(fraction: float, info: str) -> None:
        progress(fraction * share, info)

    material = ingest_file(
        Path(path),
        original_name=display_name,
        whisper_model=whisper_model,
        language=language,
        progress=forward,
        project_id=project_id,
        session_id=session_id,
    )
    _attach(session_id, material.id)

    result: dict[str, Any] = {
        "material_id": material.id,
        "name": material.name,
        "tokens": material.n_tokens_est,
        "message": f"{material.name} · ~{material.n_tokens_est:,} Tokens".replace(",", "'"),
    }
    if material.meta.get("speed_factor"):
        result["message"] += f" · {material.meta['speed_factor']}x Echtzeit"

    if auto_summarize and not should_cancel():
        progress(share, "erstelle Zusammenfassung…")
        summary = study_tools.summarize_material(
            material, summary_model or "", num_ctx,
            progress=lambda p, info: progress(share + (1 - share) * p, f"Zusammenfassung: {info}"),
        )
        if summary:
            note = library.add(
                name=f"Zusammenfassung – {material.name}", text=summary, kind="note",
                source=f"map-reduce({summary_model})",
                project_id=project_id, session_id=session_id,
            )
            _attach(session_id, note.id)
            session = session_store.load(session_id)
            if session is not None:
                session.add("assistant", f"**Zusammenfassung von {material.name}**\n\n{summary}")
                session.save()
            result["summary_id"] = note.id
            result["message"] += " · zusammengefasst"
    return result


def summarize_task(
    progress: Callable[[float, str], None],
    should_cancel: Callable[[], bool],
    *,
    material_id: str,
    session_id: str,
    model: str,
    num_ctx: int,
) -> dict[str, Any]:
    """Map-Reduce-Zusammenfassung eines vorhandenen Materials."""
    material = library.get(material_id)
    if material is None:
        raise ValueError("Material nicht mehr vorhanden.")
    summary = study_tools.summarize_material(material, model, num_ctx, progress=progress)
    if not summary:
        raise ValueError("Leere Zusammenfassung erhalten.")

    note = library.add(
        name=f"Zusammenfassung – {material.name}", text=summary, kind="note",
        source=f"map-reduce({model})",
        project_id=material.project_id, session_id=session_id,
    )
    _attach(session_id, note.id)
    session = session_store.load(session_id)
    if session is not None:
        session.add("assistant", f"**Zusammenfassung von {material.name}**\n\n{summary}")
        session.save()
    return {"note_id": note.id, "message": f"Zusammenfassung von {material.name} erstellt"}


def digests_task(
    progress: Callable[[float, str], None],
    should_cancel: Callable[[], bool],
    *,
    project_id: str,
    model: str,
    force: bool = False,
) -> dict[str, Any]:
    """Lektions-Digests eines Projekts aktualisieren (ein Modell-Call pro Lektion)."""
    lessons = project_store.lesson_sessions(project_id)
    ok = 0
    for i, lesson in enumerate(lessons):
        if should_cancel():
            break
        progress(i / max(len(lessons), 1), f"verdichte «{lesson.title}»")
        if not lesson.messages:
            continue
        if not force and not lesson.digest_stale():
            ok += 1
            continue
        try:
            if ctx.build_lesson_digest(lesson, model, force=force):
                ok += 1
        except ollama_client.OllamaError:
            pass                                  # einzelne Lektion scheitert -> Rest laeuft weiter
    progress(1.0, "fertig")
    return {"updated": ok, "total": len(lessons),
            "message": f"{ok}/{len(lessons)} Digests aktuell"}

