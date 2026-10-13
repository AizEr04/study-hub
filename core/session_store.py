"""Persistenz fuer Sessions (JSON-Dateien, eine pro Session).

Designentscheidung: Nachrichten werden beim Komprimieren NICHT geloescht, sondern
mit `archived=True` markiert. Die UI zeigt weiterhin den vollen Verlauf,
an das Modell geht nur der aktive Teil + Zusammenfassung.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from . import config


def _now() -> float:
    return time.time()


def _slug(text: str, maxlen: int = 40) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip()
    return re.sub(r"[\s_-]+", "-", text)[:maxlen].lower() or "session"


@dataclass
class Message:
    role: str                      # system | user | assistant
    content: str
    ts: float = field(default_factory=_now)
    archived: bool = False         # von der Memory-Compression ersetzt
    meta: dict[str, Any] = field(default_factory=dict)  # z.B. {"stats": {...}}


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = "Neue Session"
    topic: str = ""                # Selective Context Loading: ein Thema pro Session
    project_id: str = ""           # Zugehoerigkeit zu einem Projekt (z. B. Semester-Modul)
    kind: str = "lesson"           # lesson = einzelne Lektion | project = Gesamtkontext
    # Verdichteter Stand dieser Lektion, den die Projekt-Session konsumiert:
    # {"content": str, "n_messages": int, "ts": float}
    digest: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    persona: str = "Akademischer Tutor"
    system_prompt: str = ""
    num_ctx: int = config.DEFAULT_NUM_CTX
    temperature: float = 0.7
    think: bool = False
    auto_compress: bool = True
    rag_mode: str = "auto"         # auto | full | off
    material_ids: list[str] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)  # {"content","ts","covered"}
    created: float = field(default_factory=_now)
    updated: float = field(default_factory=_now)

    # ---------- Verlauf ----------
    def add(self, role: str, content: str, **meta: Any) -> Message:
        msg = Message(role=role, content=content, meta=meta)
        self.messages.append(msg)
        self.updated = _now()
        return msg

    def active_messages(self) -> list[Message]:
        return [m for m in self.messages if not m.archived]

    def last_user_message(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return ""

    # ---------- Digest (Futter fuer die Projekt-Session) ----------
    def digest_stale(self) -> bool:
        """True, wenn seit dem letzten Digest neue Nachrichten dazugekommen sind."""
        if not self.digest:
            return bool(self.messages)
        return len(self.messages) > int(self.digest.get("n_messages", 0))

    def digest_text(self) -> str:
        return (self.digest or {}).get("content", "")

    def set_digest(self, content: str) -> None:
        self.digest = {"content": content, "n_messages": len(self.messages), "ts": _now()}

    def path(self) -> "config.Path":
        return config.SESSION_DIR / f"{self.id}.json"

    # ---------- IO ----------
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["messages"] = [asdict(m) if not isinstance(m, dict) else m for m in self.messages]
        return data

    def save(self) -> None:
        self.updated = _now()
        self.path().write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        msgs = [Message(**m) for m in data.pop("messages", [])]
        known = {f for f in cls.__dataclass_fields__}  # tolerant gegen alte Dateien
        session = cls(**{k: v for k, v in data.items() if k in known})
        session.messages = msgs
        return session


# ---------- Repository ----------
def list_sessions() -> list[dict[str, Any]]:
    """Kompakte Uebersicht (ohne Nachrichten zu parsen), neueste zuerst."""
    out: list[dict[str, Any]] = []
    for p in config.SESSION_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append(
            {
                "id": data.get("id", p.stem),
                "title": data.get("title", p.stem),
                "topic": data.get("topic", ""),
                "project_id": data.get("project_id", ""),
                "kind": data.get("kind", "lesson"),
                "model": data.get("model", ""),
                "created": data.get("created", 0),
                "updated": data.get("updated", 0),
                "n_messages": len(data.get("messages", [])),
                "n_materials": len(data.get("material_ids", [])),
                "material_ids": data.get("material_ids", []),
                "has_digest": bool((data.get("digest") or {}).get("content")),
                "digest_stale": len(data.get("messages", []))
                > int((data.get("digest") or {}).get("n_messages", 0)),
            }
        )
    return sorted(out, key=lambda s: s["updated"], reverse=True)


def load(session_id: str) -> Session | None:
    p = config.SESSION_DIR / f"{session_id}.json"
    if not p.exists():
        return None
    return Session.from_dict(json.loads(p.read_text(encoding="utf-8")))


def delete(session_id: str) -> None:
    (config.SESSION_DIR / f"{session_id}.json").unlink(missing_ok=True)


def create(model: str, persona: str, system_prompt: str, title: str = "", topic: str = "",
           project_id: str = "", kind: str = "lesson") -> Session:
    session = Session(
        title=title or f"Session {time.strftime('%d.%m. %H:%M')}",
        topic=topic,
        project_id=project_id,
        kind=kind,
        model=model,
        persona=persona,
        system_prompt=system_prompt,
    )
    session.save()
    return session


def export_markdown(session: Session) -> str:
    lines = [f"# {session.title}", "", f"*Modell:* `{session.model}` · *Persona:* {session.persona}"]
    if session.topic:
        lines.append(f"*Thema:* {session.topic}")
    lines += ["", "---", ""]
    for m in session.messages:
        if m.role == "system":
            continue
        who = "🧑 **Du**" if m.role == "user" else "🤖 **Modell**"
        stamp = time.strftime("%d.%m.%Y %H:%M", time.localtime(m.ts))
        lines += [f"### {who} · {stamp}", "", m.content, ""]
    if session.summaries:
        lines += ["---", "## Kontext-Zusammenfassungen", ""]
        lines += [s["content"] for s in session.summaries]
    return "\n".join(lines)


def export_filename(session: Session) -> str:
    return f"{_slug(session.title)}-{session.id}.md"

