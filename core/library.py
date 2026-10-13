"""Material-Bibliothek: extrahierte Texte aus PDFs, PPTX, Audio/Video usw.

Ein Material = 1 Textdatei in data/materials + 1 Eintrag in der Registry (index.json).

Sichtbarkeit (Scope) – Material wird NIE global geteilt:
* `project_id` gesetzt  -> sichtbar in allen Sessions genau dieses Projekts.
* nur `session_id`      -> sichtbar ausschliesslich in dieser einen Session.
* keins von beidem      -> "nicht zugeordnet" (Altbestand); nur sichtbar, wo es bereits
                           referenziert ist, und im UI explizit zuweisbar.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from . import config

INDEX_PATH = config.MATERIAL_DIR / "index.json"


@dataclass
class Material:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""              # Originaldateiname
    kind: str = "document"      # document | slides | transcript | note
    source: str = ""            # Extraktionsverfahren, z.B. "pypdf", "faster-whisper(small)"
    n_chars: int = 0
    n_tokens_est: int = 0
    project_id: str = ""        # Scope: ganzes Projekt
    session_id: str = ""        # Scope: nur diese Session (bzw. Herkunfts-Session)
    created: float = field(default_factory=lambda: time.time())
    edited: float = 0.0         # manuell bearbeitet am
    meta: dict[str, Any] = field(default_factory=dict)  # Seiten, Dauer, Sprache ...

    @property
    def text_path(self):
        return config.MATERIAL_DIR / f"{self.id}.md"

    def text(self) -> str:
        try:
            return self.text_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def label(self) -> str:
        icon = {"document": "📄", "slides": "📊", "transcript": "🎤", "note": "📝"}.get(self.kind, "📄")
        return f"{icon} {self.name} · ~{self.n_tokens_est:,} Tok".replace(",", "'")

    def scope_label(self) -> str:
        if self.project_id:
            return "📁 Projekt"
        if self.session_id:
            return "💬 nur diese Session"
        return "⚠️ nicht zugeordnet"


def _load_index() -> dict[str, dict[str, Any]]:
    if not INDEX_PATH.exists():
        return {}
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_index(index: dict[str, dict[str, Any]]) -> None:
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def add(name: str, text: str, kind: str, source: str, meta: dict[str, Any] | None = None,
        project_id: str = "", session_id: str = "") -> Material:
    from .context_manager import estimate_tokens

    mat = Material(
        name=name,
        kind=kind,
        source=source,
        n_chars=len(text),
        n_tokens_est=estimate_tokens(text),
        project_id=project_id,
        session_id=session_id,
        meta=meta or {},
    )
    mat.text_path.write_text(text, encoding="utf-8")
    index = _load_index()
    index[mat.id] = asdict(mat)
    _save_index(index)
    return mat


def get(material_id: str) -> Material | None:
    data = _load_index().get(material_id)
    if not data:
        return None
    known = {f for f in Material.__dataclass_fields__}
    return Material(**{k: v for k, v in data.items() if k in known})


def list_all() -> list[Material]:
    known = {f for f in Material.__dataclass_fields__}
    return sorted(
        (Material(**{k: v for k, v in d.items() if k in known}) for d in _load_index().values()),
        key=lambda m: m.created,
        reverse=True,
    )


def _put(mat: Material) -> Material:
    index = _load_index()
    index[mat.id] = asdict(mat)
    _save_index(index)
    return mat


def update(material_id: str, **fields: Any) -> Material | None:
    """Metadaten aendern (name, kind, project_id, session_id …)."""
    mat = get(material_id)
    if mat is None:
        return None
    for key, value in fields.items():
        if hasattr(mat, key):
            setattr(mat, key, value)
    mat.edited = time.time()
    return _put(mat)


def save_text(material_id: str, text: str) -> Material | None:
    """Extrahierten Text manuell korrigieren (OCR-Fehler, Whisper-Verhörer, Kürzungen)."""
    from .context_manager import estimate_tokens

    mat = get(material_id)
    if mat is None:
        return None
    mat.text_path.write_text(text, encoding="utf-8")
    mat.n_chars = len(text)
    mat.n_tokens_est = estimate_tokens(text)
    mat.edited = time.time()
    return _put(mat)


def delete(material_id: str) -> None:
    index = _load_index()
    mat = index.pop(material_id, None)
    if mat:
        (config.MATERIAL_DIR / f"{material_id}.md").unlink(missing_ok=True)
        _save_index(index)


# ---------------- Scope / Sichtbarkeit ----------------
def is_visible(mat: Material, session) -> bool:
    """Material gehoert zum Projekt der Session bzw. zur Session selbst."""
    if mat.project_id:
        return bool(session.project_id) and mat.project_id == session.project_id
    if mat.session_id:
        return mat.session_id == session.id
    return mat.id in session.material_ids  # Altbestand ohne Scope


def visible_for(session) -> list[Material]:
    return [m for m in list_all() if is_visible(m, session)]


def unassigned() -> list[Material]:
    """Materialien ohne jeden Scope (Altbestand oder Session wurde geloescht)."""
    return [m for m in list_all() if not m.project_id and not m.session_id]


def scope_for_session(session) -> dict[str, str]:
    """Default-Scope fuer neues Material: Projekt, wenn vorhanden, sonst die Session."""
    return {"project_id": session.project_id or "", "session_id": session.id}


def migrate_session_scope(session_id: str, old_project_id: str, new_project_id: str) -> int:
    """Haengt Material beim Projektwechsel einer Session mit um. -> Anzahl geaenderter Materialien."""
    changed = 0
    for mat in list_all():
        if mat.session_id != session_id:
            continue
        if mat.project_id == new_project_id:
            continue
        if mat.project_id in ("", old_project_id):
            update(mat.id, project_id=new_project_id)
            changed += 1
    return changed


# ---------------- Chunking (fuer selektives Nachladen / RAG) ----------------
@dataclass
class Chunk:
    material_id: str
    material_name: str
    ordinal: int
    text: str
    anchor: str = ""   # z.B. "S. 12" oder "00:14:20"


_ANCHOR_RE = re.compile(r"^(?:##+\s*)?(?:\[)?((?:Seite|Slide|S\.)\s*\d+|\d{2}:\d{2}:\d{2})", re.I)


def _split_oversized(para: str, target_chars: int) -> list[str]:
    """Zerlegt einen zu langen Absatz an Satz- bzw. Wortgrenzen.

    Noetig fuer Transkripte: eine 90-Min-Aufnahme kann als ein einziger Block ohne
    Absatzumbrueche ankommen. Ohne dieses Splitting entstuende ein 30k-Token-Chunk,
    der jedes Material-Budget sprengt und deshalb komplett verworfen wuerde.
    """
    if len(para) <= target_chars:
        return [para]
    # Zuerst an Satzenden trennen, dann uebergrosse Reststuecke hart an Wortgrenzen.
    units = re.split(r"(?<=[.!?])\s+", para)
    pieces: list[str] = []
    buf = ""
    for unit in units:
        while len(unit) > target_chars:              # ein einzelner "Satz" ist zu lang
            cut = unit.rfind(" ", 0, target_chars)
            cut = cut if cut > target_chars // 2 else target_chars
            pieces.append(unit[:cut].strip())
            unit = unit[cut:].strip()
        if len(buf) + len(unit) + 1 > target_chars and buf:
            pieces.append(buf.strip())
            buf = unit
        else:
            buf = f"{buf} {unit}".strip()
    if buf.strip():
        pieces.append(buf.strip())
    return [p for p in pieces if p]


def chunk_text(material: Material, target_chars: int = 3500, overlap: int = 250) -> list[Chunk]:
    """Absatzweises Chunking – respektiert Seiten-/Slide-/Zeitmarken aus der Extraktion.

    Uebergrosse Absaetze (typisch bei Transkripten) werden hart nachgeteilt, damit
    jeder Chunk sicher in ein Material-Budget passt.
    """
    text = material.text()
    if not text:
        return []
    # Ein Absatz kann selbst laenger als das Ziel sein -> vorab aufsplitten.
    raw_paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    paragraphs: list[str] = []
    for para in raw_paragraphs:
        paragraphs.extend(_split_oversized(para, target_chars))

    chunks: list[Chunk] = []
    buf: list[str] = []
    size = 0
    anchor = ""

    def flush() -> None:
        nonlocal buf, size
        if not buf:
            return
        chunks.append(
            Chunk(material.id, material.name, len(chunks), "\n\n".join(buf), anchor)
        )
        tail = buf[-1][-overlap:] if overlap and buf else ""
        buf = [tail] if tail else []
        size = len(tail)

    for para in paragraphs:
        m = _ANCHOR_RE.match(para)
        if m:
            anchor = m.group(1)
        if size + len(para) > target_chars:
            flush()
        buf.append(para)
        size += len(para)
    flush()
    return chunks

