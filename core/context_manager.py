"""Kontext-Management: Token-Budget, Memory-Compression und Material-Retrieval.

Drei Strategien (bewusst in dieser Reihenfolge angewandt):
1. Token Budgeter      – rechnet vor jedem Call, was ins Fenster passt.
2. Memory Compression  – alte Turns werden vom Modell selbst zusammengefasst.
3. Selective Loading   – nur relevante Material-Chunks statt ganzer Dokumente.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from . import config, library, ollama_client, project_store
from .session_store import Message, Session

# ---------------------------------------------------------------- Tokenizer
try:  # tiktoken ist nur eine Naeherung fuer Llama/Gemma, aber deutlich besser als len/4
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - optionale Abhaengigkeit
    _ENC = None


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    # Fallback-Heuristik: ~4 Zeichen pro Token (DE eher 3.5 -> leicht konservativ)
    return math.ceil(len(text) / 3.7)


def messages_tokens(messages: Iterable[dict[str, str] | Message]) -> int:
    total = 0
    for m in messages:
        content = m["content"] if isinstance(m, dict) else m.content
        total += estimate_tokens(content) + 4  # Rollen-Overhead
    return total


def _clip_to_tokens(text: str, max_tokens: int) -> str:
    """Kuerzt Text auf ~max_tokens – token-genau statt per Zeichenheuristik.

    Wird beim Abschneiden von Material genutzt; die alte Variante `text[:keep*4]`
    lag bei deutschem Text daneben und riskierte einen Kontext-Overflow.
    """
    if max_tokens <= 0 or not text:
        return ""
    if _ENC is not None:
        toks = _ENC.encode(text, disallowed_special=())
        if len(toks) <= max_tokens:
            return text
        return _ENC.decode(toks[:max_tokens])
    approx = max_tokens * 4
    return text if len(text) <= approx else text[:approx]


# ---------------------------------------------------------------- Budget
@dataclass
class Budget:
    num_ctx: int
    usable: int          # abzueglich Reserve fuer die Antwort
    material_max: int    # Obergrenze fuer injiziertes Material
    history_max: int     # Obergrenze fuer den Gespraechsverlauf
    digest_max: int = 0  # Obergrenze fuer Lektions-Digests (nur Projekt-Sessions)


def make_budget(session: Session) -> Budget:
    usable = int(session.num_ctx * (1 - config.RESPONSE_RESERVE))
    system_cost = estimate_tokens(session.system_prompt) + 200
    free = max(usable - system_cost, 512)
    if session.kind == "project":
        # Im Gesamtkontext sind die Lektions-Digests das Rueckgrat -> eigenes Budget.
        return Budget(session.num_ctx, usable, int(free * 0.40), int(free * 0.25), int(free * 0.35))
    return Budget(session.num_ctx, usable, int(free * 0.55), int(free * 0.45))


# ---------------------------------------------------------------- Projekt-Aggregation
def effective_material_ids(session: Session) -> list[str]:
    """Material der Session – bei Projekt-Sessions zusaetzlich das der Lektionen.

    Der Scope wird hier hart durchgesetzt: was in dieser Session nicht sichtbar ist,
    landet auch nicht im Prompt (Material bleibt im Projekt bzw. in der Einzelsession).
    """
    ids: list[str] = []
    for mid in session.material_ids:
        mat = library.get(mid)
        if mat is not None and library.is_visible(mat, session) and mid not in ids:
            ids.append(mid)

    if session.kind == "project" and session.project_id:
        for meta in project_store.sessions_of(session.project_id, kind="lesson"):
            for mid in meta.get("material_ids", []):
                if mid in ids:
                    continue
                mat = library.get(mid)
                # Nur projektweit freigegebenes Material wird geerbt,
                # nicht das auf eine einzelne Lektion beschraenkte.
                if mat is not None and mat.project_id == session.project_id:
                    ids.append(mid)
    return ids



# ---------------------------------------------------------------- Retrieval
_WORD_RE = re.compile(r"\w{3,}", re.UNICODE)
_STOP = {
    "der", "die", "das", "und", "oder", "ein", "eine", "einen", "ist", "sind", "war", "mit",
    "von", "fuer", "für", "auf", "den", "dem", "des", "nicht", "auch", "wie", "was", "wer",
    "the", "and", "for", "with", "that", "this", "are", "was", "from", "you", "your",
}


def _terms(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOP]


def rank_chunks(chunks: list[library.Chunk], query: str, top_k: int) -> list[library.Chunk]:
    """BM25-artiges Keyword-Ranking – deterministisch, offline, ohne Embedding-Modell.

    Bewusste Entscheidung: lokal ist oft kein Embedding-Modell gepullt
    (`ollama pull nomic-embed-text`), deshalb ist das der robuste Default.
    """
    q = set(_terms(query))
    if not q or not chunks:
        return chunks[:top_k]
    df: dict[str, int] = {}
    docs: list[list[str]] = []
    for c in chunks:
        terms = _terms(c.text)
        docs.append(terms)
        for t in set(terms) & q:
            df[t] = df.get(t, 0) + 1
    n = len(chunks)
    avg_len = sum(len(d) for d in docs) / max(n, 1)
    scored: list[tuple[float, int]] = []
    for i, terms in enumerate(docs):
        tf: dict[str, int] = {}
        for t in terms:
            if t in q:
                tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t, f in tf.items():
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * (f * 2.5) / (f + 1.5 * (0.25 + 0.75 * len(terms) / max(avg_len, 1)))
        scored.append((score, i))
    scored.sort(key=lambda s: (-s[0], s[1]))
    hits = [chunks[i] for score, i in scored[:top_k] if score > 0]
    # Kein Treffer (z. B. "Fasse die Vorlesung zusammen" ohne inhaltliche Stichworte):
    # dann breite Abdeckung aus dem Anfang statt nur eines einzelnen Chunks.
    return hits or chunks[:top_k]


def build_material_block(session: Session, query: str, budget: Budget) -> tuple[str, dict[str, Any]]:
    """Baut den Material-Kontext gemaess session.rag_mode (auto|full|off)."""
    report: dict[str, Any] = {"mode": session.rag_mode, "used": [], "tokens": 0, "truncated": False}
    material_ids = effective_material_ids(session)
    if session.rag_mode == "off" or not material_ids:
        return "", report

    materials = [m for m in (library.get(i) for i in material_ids) if m]
    if not materials:
        return "", report

    total = sum(m.n_tokens_est for m in materials)
    full = session.rag_mode == "full" or (session.rag_mode == "auto" and total <= budget.material_max)

    parts: list[str] = []
    used_tokens = 0
    if full:
        for m in materials:
            text = m.text()
            cost = estimate_tokens(text)
            if used_tokens + cost > budget.material_max:
                # Hartes Limit: lieber abschneiden als den Server in den Kontext-Overflow schicken
                keep = max(budget.material_max - used_tokens, 0)
                text = _clip_to_tokens(text, keep)
                report["truncated"] = True
                cost = estimate_tokens(text)
            if not text:
                continue
            parts.append(f"### Quelle: {m.name} ({m.kind})\n{text}")
            report["used"].append(m.name)
            used_tokens += cost
            if report["truncated"]:
                break
        report["mode"] = "full"
    else:
        chunks: list[library.Chunk] = []
        for m in materials:
            chunks.extend(library.chunk_text(m))
        top = rank_chunks(chunks, query, top_k=12)
        for c in top:
            if used_tokens >= budget.material_max:
                break
            text = c.text
            cost = estimate_tokens(text)
            if used_tokens + cost > budget.material_max:
                # Chunk passt nicht ganz -> kuerzen statt verwerfen. Frueher stand hier
                # ein `break`, wodurch ein einzelner uebergrosser Chunk (z. B. ein langes
                # Transkript-Segment) dazu fuehrte, dass GAR KEIN Material ankam.
                keep = max(budget.material_max - used_tokens, 0)
                if keep < 40:                    # Rest zu klein, um noch sinnvoll zu sein
                    break
                text = _clip_to_tokens(text, keep)
                cost = estimate_tokens(text)
                report["truncated"] = True
            anchor = f" – {c.anchor}" if c.anchor else ""
            parts.append(f"### Quelle: {c.material_name}{anchor} (Abschnitt {c.ordinal + 1})\n{text}")
            report["used"].append(f"{c.material_name}#{c.ordinal + 1}")
            used_tokens += cost
        report["mode"] = "retrieval"

    if not parts:
        return "", report
    report["tokens"] = used_tokens
    if report["mode"] == "full":
        header = (
            "Nachfolgend das VOLLSTAENDIGE Studienmaterial dieser Session. Beziehe dich darauf "
            "und kennzeichne Aussagen mit [Quelle: <Dateiname>].\n\n"
        )
    else:
        header = (
            "Nachfolgend die zur Frage passendsten AUSZUEGE aus dem Studienmaterial dieser Session "
            "(nicht der vollstaendige Text – es wurden relevante Abschnitte ausgewaehlt). "
            "Beziehe dich darauf und kennzeichne Aussagen mit [Quelle: <Dateiname>]. "
            "Fehlt eine Information, sage das, statt zu raten.\n\n"
        )
    return header + "\n\n".join(parts), report


# ---------------------------------------------------------------- Lektions-Digests
LESSON_DIGEST_PROMPT = (
    "Du erstellst einen Lektions-Digest fuer ein Semester-Gedaechtnis. "
    "Verdichte den folgenden Lektionsverlauf auf maximal 250 Woerter.\n"
    "Struktur:\n"
    "- Thema: <ein Satz>\n"
    "- Kernpunkte: <Stichpunkte, inkl. Definitionen, Formeln, Zahlen>\n"
    "- Bezuege/Beispiele: <konkrete Beispiele oder genannte Quellen>\n"
    "- Offene Fragen: <was ungeklaert blieb>\n"
    "Keine Einleitung, keine Meta-Kommentare, nur der Digest."
)


def _raw_excerpt(session: Session, limit: int = 900) -> str:
    """Notnagel, wenn noch kein vom Modell erzeugter Digest existiert."""
    parts: list[str] = []
    for m in session.messages:
        if m.role == "system":
            continue
        who = "Frage" if m.role == "user" else "Antwort"
        parts.append(f"{who}: {m.content.strip()[:300]}")
    if not parts:
        return ""
    text = "\n".join(parts[:2] + (["…"] + parts[-2:] if len(parts) > 4 else parts[2:]))
    return text[:limit]


def build_lesson_digest(session: Session, model: str | None = None, force: bool = False) -> str:
    """Erzeugt (und cached) den Digest einer Lektions-Session via Modell-Call."""
    if not session.messages:
        return ""
    if not force and not session.digest_stale():
        return session.digest_text()

    body = session.summaries[-1]["content"] + "\n\n" if session.summaries else ""
    body += "\n\n".join(
        f"{'BENUTZER' if m.role == 'user' else 'ASSISTENT'}: {m.content}"
        for m in session.messages
        if m.role != "system"
    )
    # Der Digest-Call selbst darf das Fenster nicht sprengen.
    max_chars = max(session.num_ctx * 2, 4000)
    if len(body) > max_chars:
        body = body[: max_chars // 2] + "\n\n[…]\n\n" + body[-max_chars // 2:]

    header = f"LEKTION: {session.title}" + (f" (Thema: {session.topic})" if session.topic else "")
    summary = ollama_client.chat_once(
        model or session.model,
        [
            {"role": "system", "content": LESSON_DIGEST_PROMPT},
            {"role": "user", "content": f"{header}\n\n{body}"},
        ],
        num_ctx=session.num_ctx,
        temperature=0.2,
    )
    if summary:
        session.set_digest(summary)
        session.save()
    return summary


def refresh_project_digests(project_id: str, model: str | None = None, force: bool = False,
                            progress: Any = None) -> list[tuple[str, bool]]:
    """Aktualisiert die Digests aller Lektionen eines Projekts. -> [(Titel, erfolgreich)]"""
    lessons = project_store.lesson_sessions(project_id)
    results: list[tuple[str, bool]] = []
    for i, lesson in enumerate(lessons):
        if progress:
            progress(i / max(len(lessons), 1), f"{lesson.title}")
        if not lesson.messages:
            results.append((lesson.title, False))
            continue
        if not force and not lesson.digest_stale():
            results.append((lesson.title, True))
            continue
        try:
            results.append((lesson.title, bool(build_lesson_digest(lesson, model, force))))
        except ollama_client.OllamaError:
            results.append((lesson.title, False))
    if progress:
        progress(1.0, "fertig")
    return results


def _relevance(query: str, text: str) -> float:
    q = set(_terms(query))
    if not q:
        return 0.0
    t = _terms(text)
    if not t:
        return 0.0
    return len(q & set(t)) / len(q)


def build_digest_block(session: Session, query: str, budget: Budget) -> tuple[str, dict[str, Any]]:
    """Semester-Gedaechtnis: Digests aller Lektionen des Projekts, chronologisch."""
    report: dict[str, Any] = {"lessons": [], "stale": [], "raw": [], "tokens": 0, "skipped": 0}
    if session.kind != "project" or not session.project_id:
        return "", report

    entries: list[tuple[int, str, str, float]] = []  # (order, titel, text, relevanz)
    for order, lesson in enumerate(project_store.lesson_sessions(session.project_id)):
        text = lesson.digest_text()
        if not text:
            text = _raw_excerpt(lesson)
            if text:
                report["raw"].append(lesson.title)
        elif lesson.digest_stale():
            report["stale"].append(lesson.title)
        if not text:
            continue
        entries.append((order, lesson.title, text, _relevance(query, text)))

    if not entries:
        return "", report

    # Passt alles? Sonst die relevantesten Lektionen behalten, Reihenfolge bleibt chronologisch.
    selected = entries
    total = sum(estimate_tokens(e[2]) for e in entries)
    if total > budget.digest_max:
        ranked = sorted(entries, key=lambda e: (-e[3], e[0]))
        selected, used = [], 0
        for entry in ranked:
            cost = estimate_tokens(entry[2])
            if used + cost > budget.digest_max:
                continue
            selected.append(entry)
            used += cost
        report["skipped"] = len(entries) - len(selected)
        selected.sort(key=lambda e: e[0])

    parts = [f"#### Lektion {order + 1}: {title}\n{text}" for order, title, text, _ in selected]
    report["lessons"] = [title for _, title, _, _ in selected]
    report["tokens"] = sum(estimate_tokens(p) for p in parts)
    header = (
        "SEMESTER-GEDAECHTNIS – verdichtete Digests der einzelnen Lektionen dieses Projekts. "
        "Nutze sie fuer Querbezuege und verweise auf die Lektion, aus der eine Aussage stammt.\n\n"
    )
    return header + "\n\n".join(parts), report


# ---------------------------------------------------------------- Payload
def build_payload(session: Session, query: str = "") -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Erzeugt das finale messages-Array fuer /api/chat + einen Kontext-Report fuer die UI."""
    budget = make_budget(session)
    query = query or session.last_user_message()

    payload: list[dict[str, str]] = []
    if session.system_prompt.strip():
        payload.append({"role": "system", "content": session.system_prompt.strip()})

    for s in session.summaries:
        payload.append(
            {
                "role": "system",
                "content": "ZUSAMMENFASSUNG DES BISHERIGEN GESPRAECHS (verdichtetes Gedaechtnis):\n"
                + s["content"],
            }
        )

    digest_block, digest_report = build_digest_block(session, query, budget)
    if digest_block:
        payload.append({"role": "system", "content": digest_block})

    material_block, material_report = build_material_block(session, query, budget)
    if material_block:
        payload.append({"role": "system", "content": material_block})

    # Verlauf von hinten nach vorne aufnehmen, bis das History-Budget voll ist.
    history: list[dict[str, str]] = []
    used = 0
    dropped = 0
    for m in reversed(session.active_messages()):
        if m.role == "system":
            continue
        cost = estimate_tokens(m.content) + 4
        if used + cost > budget.history_max and len(history) >= 2:
            dropped += 1
            continue
        history.append({"role": m.role, "content": m.content})
        used += cost
    history.reverse()
    payload.extend(history)

    report = {
        "budget": budget,
        "prompt_tokens_est": messages_tokens(payload),
        "history_tokens": used,
        "history_dropped": dropped,
        "material": material_report,
        "digests": digest_report,
        "summaries": len(session.summaries),
        "fill": messages_tokens(payload) / max(budget.usable, 1),
    }
    return payload, report


# ---------------------------------------------------------------- Compression
SUMMARY_PROMPT = (
    "Du bist ein Gedaechtnis-Kompressor. Fasse den folgenden Gespraechsverlauf so zusammen, "
    "dass ein anderes Modell die Konversation nahtlos fortsetzen kann.\n"
    "Regeln:\n"
    "- Stichpunkte, maximal 300 Woerter.\n"
    "- Erhalte: gestellte Aufgaben, Definitionen, Zahlen, Entscheidungen, offene Fragen, "
    "vereinbarte Formate und den Namen genannter Quellen.\n"
    "- Keine Floskeln, keine Einleitung, keine Meta-Kommentare. Gib NUR die Zusammenfassung aus."
)


def should_compress(session: Session, report: dict[str, Any]) -> bool:
    return bool(session.auto_compress) and report["fill"] >= config.COMPRESSION_TRIGGER


def compress_history(session: Session, model: str | None = None) -> str | None:
    """Fasst alte Turns zusammen, markiert sie als `archived` und speichert die Session.

    Gibt die erzeugte Zusammenfassung zurueck (oder None, wenn nichts zu tun war).
    """
    active = [m for m in session.active_messages() if m.role != "system"]
    to_compress = active[: -config.KEEP_RECENT_MESSAGES] if len(active) > config.KEEP_RECENT_MESSAGES else []
    if len(to_compress) < 2:
        return None

    transcript = "\n\n".join(
        f"{'BENUTZER' if m.role == 'user' else 'ASSISTENT'}: {m.content}" for m in to_compress
    )
    previous = session.summaries[-1]["content"] if session.summaries else ""
    user_block = (f"BISHERIGE ZUSAMMENFASSUNG:\n{previous}\n\n" if previous else "") + (
        f"NEUER VERLAUF:\n{transcript}"
    )

    summary = ollama_client.chat_once(
        model or session.model,
        [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": user_block},
        ],
        num_ctx=session.num_ctx,
        temperature=0.2,
    )
    if not summary:
        return None

    for m in to_compress:
        m.archived = True
    session.summaries = [  # kumulativ: eine einzige, fortgeschriebene Zusammenfassung
        {"content": summary, "ts": to_compress[-1].ts, "covered": len(to_compress)}
    ]
    session.save()
    return summary


TITLE_PROMPT = (
    "Gib einen praegnanten Titel (max. 6 Woerter, keine Anfuehrungszeichen, kein Punkt) "
    "fuer folgende Anfrage aus. Antworte NUR mit dem Titel."
)


def suggest_title(session: Session, first_message: str) -> str:
    try:
        title = ollama_client.chat_once(
            session.model,
            [{"role": "system", "content": TITLE_PROMPT}, {"role": "user", "content": first_message[:1500]}],
            num_ctx=2048,
            temperature=0.2,
        )
    except ollama_client.OllamaError:
        return ""
    return title.strip().strip('"').splitlines()[0][:60] if title else ""

