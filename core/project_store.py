"""Projekte: Klammer um mehrere Sessions (z. B. ein Semester-Modul).

Hierarchie:
    Projekt "Statistik HS26"
      ├── Session "Woche 1 – Deskriptive Statistik"   (kind="lesson")
      ├── Session "Woche 2 – Hypothesentests"          (kind="lesson")
      └── Session "🎓 Gesamtkontext"                   (kind="project")

Die Projekt-Session aggregiert Materialien UND verdichtete Lektions-Digests
aller Lektions-Sessions – damit laesst sich ueber das ganze Semester arbeiten,
ohne dass die Rohverlaeufe das Kontextfenster sprengen.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from . import config, session_store

INDEX_PATH = config.DATA_DIR / "projects.json"


@dataclass
class Project:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "Neues Projekt"
    description: str = ""          # z. B. "Semester HS26, Modul Statistik"
    model: str = ""                # Default fuer neue Sessions im Projekt
    persona: str = "Akademischer Tutor"
    system_prompt: str = ""        # Default-Prompt, wird an Sessions vererbt
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def save(self) -> None:
        self.updated = time.time()
        index = _load()
        index[self.id] = asdict(self)
        _save(index)


def _load() -> dict[str, dict[str, Any]]:
    if not INDEX_PATH.exists():
        return {}
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save(index: dict[str, dict[str, Any]]) -> None:
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- CRUD
def create(name: str, description: str = "", model: str = "", persona: str = "",
           system_prompt: str = "") -> Project:
    project = Project(
        name=name or "Neues Projekt",
        description=description,
        model=model,
        persona=persona or "Akademischer Tutor",
        system_prompt=system_prompt,
    )
    project.save()
    return project


def get(project_id: str) -> Project | None:
    data = _load().get(project_id)
    return Project(**data) if data else None


def list_all() -> list[Project]:
    return sorted((Project(**d) for d in _load().values()), key=lambda p: p.created)


def delete(project_id: str, delete_sessions: bool = False) -> None:
    """Projekt loeschen. Sessions werden entweder mitgeloescht oder entkoppelt."""
    for meta in sessions_of(project_id):
        session = session_store.load(meta["id"])
        if session is None:
            continue
        if delete_sessions:
            session_store.delete(session.id)
        else:
            session.project_id = ""
            if session.kind == "project":       # verwaiste Projekt-Session ergibt keinen Sinn
                session_store.delete(session.id)
            else:
                session.save()
    index = _load()
    index.pop(project_id, None)
    _save(index)


# ---------------------------------------------------------------- Zuordnung
def sessions_of(project_id: str, kind: str | None = None) -> list[dict[str, Any]]:
    """Alle Session-Metadaten eines Projekts, aelteste zuerst (= chronologisch nach Lektion)."""
    items = [
        s for s in session_store.list_sessions()
        if s.get("project_id") == project_id and (kind is None or s.get("kind") == kind)
    ]
    return sorted(items, key=lambda s: s.get("created", 0))


def lesson_sessions(project_id: str) -> list[session_store.Session]:
    """Geladene Lektions-Sessions (ohne die Projekt-Session selbst)."""
    out = []
    for meta in sessions_of(project_id, kind="lesson"):
        session = session_store.load(meta["id"])
        if session is not None:
            out.append(session)
    return out


def project_session(project_id: str) -> session_store.Session | None:
    metas = sessions_of(project_id, kind="project")
    return session_store.load(metas[0]["id"]) if metas else None


def get_or_create_project_session(project: Project) -> session_store.Session:
    """Die eine Session pro Projekt, die ueber den Gesamtkontext arbeitet."""
    existing = project_session(project.id)
    if existing is not None:
        return existing

    base = project.system_prompt or config.PERSONAS.get(project.persona, "")
    prompt = (
        base
        + f"\n\nDu arbeitest im Gesamtkontext des Projekts «{project.name}»"
        + (f" ({project.description})" if project.description else "")
        + ".\nDer Kontext enthaelt verdichtete Digests aller einzelnen Lektionen sowie das "
        "zugehoerige Studienmaterial. Stelle Querbezuege zwischen den Lektionen her, erkenne "
        "roten Faden und Widersprueche, und nenne bei Aussagen immer die Lektion bzw. "
        "[Quelle: <Datei>], auf die du dich beziehst."
    )
    session = session_store.create(
        model=project.model,
        persona=project.persona,
        system_prompt=prompt,
        title=f"🎓 Gesamtkontext – {project.name}",
        topic=project.description,
    )
    session.project_id = project.id
    session.kind = "project"
    session.save()
    return session


def assign(session: session_store.Session, project_id: str) -> dict[str, int]:
    """Session einem Projekt zuordnen oder herausloesen (project_id="").

    Material, das dieser Session gehoert, wandert mit: beim Hinzufuegen wird es auf das
    Projekt umgehaengt (fuer alle Lektionen sichtbar), beim Herausloesen wieder auf die
    Session beschraenkt. Fremdes Material, das dadurch unsichtbar wird, verliert seine
    Referenz – sonst waere es im Prompt, aber nicht in der UI.
    """
    from . import library  # lokal, vermeidet Import-Zyklus ueber context_manager

    old_project = session.project_id
    if old_project == project_id:
        return {"moved": 0, "unlinked": 0}

    session.project_id = project_id
    if not project_id and session.kind == "project":
        session.kind = "lesson"  # eine Projekt-Session ohne Projekt ergibt keinen Sinn

    moved = library.migrate_session_scope(session.id, old_project, project_id)

    # Referenzen bereinigen: nur behalten, was in der neuen Umgebung sichtbar ist.
    keep, unlinked = [], 0
    for mid in session.material_ids:
        mat = library.get(mid)
        if mat is not None and library.is_visible(mat, session):
            keep.append(mid)
        else:
            unlinked += 1
    session.material_ids = keep
    session.save()
    return {"moved": moved, "unlinked": unlinked}


def stats(project_id: str) -> dict[str, Any]:
    metas = sessions_of(project_id)
    lessons = [m for m in metas if m.get("kind") == "lesson"]
    return {
        "n_lessons": len(lessons),
        "n_messages": sum(m.get("n_messages", 0) for m in lessons),
        "n_materials": len({mid for m in metas for mid in m.get("material_ids", [])}),
        "has_project_session": any(m.get("kind") == "project" for m in metas),
    }

