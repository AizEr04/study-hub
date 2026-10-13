"""Lokaler Study-Hub für Ollama – Chat, Session-/Kontext-Handling und Material-Ingest.

Start:  streamlit run app.py
"""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from core import (
    config,
    context_manager as ctx,
    jobs as jobmod,
    library,
    ollama_client,
    project_store,
    session_store,
    study_tools,
    tasks,
)
from core.ingest import IngestError, ingest_file, resolve_local_path, save_upload
from core.ingest.transcribe import ffmpeg_available, model_cached

st.set_page_config(page_title="Study Hub · Ollama", page_icon="🎓", layout="wide")

WHISPER_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
MAX_UPLOAD_MB = int(st.get_option("server.maxUploadSize") or 200)


@st.cache_resource(show_spinner=False)
def job_manager() -> jobmod.JobManager:
    """Ein Manager pro Prozess – ueberlebt Streamlit-Reruns dank cache_resource."""
    return jobmod.get_manager()


# --------------------------------------------------------------------- State
@st.cache_data(ttl=30, show_spinner=False)
def cached_models() -> list[dict]:
    try:
        return ollama_client.list_models()
    except ollama_client.OllamaError:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def cached_ctx_len(model: str) -> int:
    return ollama_client.model_context_length(model) or 8192


@st.cache_data(ttl=3600, show_spinner=False)
def cached_caps(model: str) -> list[str]:
    return ollama_client.model_capabilities(model)


def current_session() -> session_store.Session | None:
    sid = st.session_state.get("sid")
    if not sid:
        return None
    session = session_store.load(sid)
    if session is None:
        st.session_state.pop("sid", None)
    return session


def open_session(sid: str) -> None:
    st.session_state.sid = sid
    st.session_state.pop("queued_prompt", None)
    st.rerun()


# --------------------------------------------------------------------- Sidebar
def _digest_icon(meta: dict) -> str:
    """✅ Digest aktuell · 🔄 veraltet/fehlt · ○ leere Lektion"""
    if not meta.get("n_messages"):
        return "○"
    if not meta.get("has_digest"):
        return "🔄"
    return "🔄" if meta.get("digest_stale") else "✅"


def _session_button(meta: dict, container, show_digest: bool = False) -> None:
    active = meta["id"] == st.session_state.get("sid")
    col_a, col_b = container.columns([6, 1])
    prefix = "▶ " if active else ""
    icon = f"{_digest_icon(meta)} " if show_digest else ""
    if col_a.button(f"{prefix}{icon}{meta['title']}", key=f"open_{meta['id']}", use_container_width=True):
        open_session(meta["id"])
    if col_b.button("🗑", key=f"del_{meta['id']}", help="Session löschen"):
        session_store.delete(meta["id"])
        if active:
            st.session_state.pop("sid", None)
        st.rerun()
    col_a.caption(f"{meta['n_messages']} Nachrichten · {meta['n_materials']} Materialien")


def render_jobs(container) -> bool:
    """Job-Panel. -> True, wenn etwas laeuft (dann pollt die Seite automatisch)."""
    manager = job_manager()
    all_jobs = manager.list()
    active = [j for j in all_jobs if j.active]

    label = f"⚙️ Hintergrund-Jobs ({len(active)})" if active else "⚙️ Hintergrund-Jobs"
    with container.expander(label, expanded=bool(active)):
        if not all_jobs:
            st.caption(
                "Keine Jobs. Transkriptionen und Zusammenfassungen laufen hier im Hintergrund – "
                "du kannst währenddessen in anderen Sessions weiterarbeiten."
            )
            return False

        st.caption(jobmod.summary_line(all_jobs))
        for job in all_jobs[:8]:
            with st.container(border=True):
                st.markdown(f"{job.icon()} **{job.label}**")
                if job.session_title:
                    st.caption(f"→ {job.session_title}")
                if job.status == jobmod.RUNNING:
                    st.progress(job.progress)
                    eta = job.eta()
                    st.caption(
                        f"{job.progress * 100:.0f} % · {job.message}"
                        + (f" · noch ~{jobmod._fmt(eta)}" if eta else "")
                    )
                    if st.button("🚫 Abbrechen", key=f"cancel_{job.id}", use_container_width=True):
                        manager.cancel(job.id)
                        st.rerun()
                elif job.status == jobmod.PENDING:
                    st.caption("wartet auf freien Worker…")
                    if st.button("🚫 Abbrechen", key=f"cancel_{job.id}", use_container_width=True):
                        manager.cancel(job.id)
                        st.rerun()
                elif job.status == jobmod.DONE:
                    st.caption(f"✅ {job.message} · {jobmod._fmt(job.elapsed())}")
                    if job.session_id and st.button("↗ Zur Session", key=f"goto_job_{job.id}",
                                                    use_container_width=True):
                        open_session(job.session_id)
                elif job.status == jobmod.FAILED:
                    st.error(job.error or job.message, icon="❌")
                else:
                    st.caption(f"🚫 {job.message}")

        if len(all_jobs) > len(active) and st.button("🧹 Erledigte entfernen",
                                                     use_container_width=True, key="clr_jobs"):
            manager.clear_finished()
            st.rerun()
    return bool(active)


def render_sidebar(models: list[dict], session: session_store.Session | None) -> bool:
    st.sidebar.title("🎓 Study Hub")
    online = bool(models) or ollama_client.is_online()
    if online:
        st.sidebar.success(f"Ollama verbunden · {config.OLLAMA_HOST}", icon="🟢")
    else:
        st.sidebar.error("Ollama nicht erreichbar. Start: `ollama serve`", icon="🔴")

    busy = render_jobs(st.sidebar)

    # Einstellungen der offenen Session ganz oben: unabhaengig von der Laenge
    # des Chatverlaufs jederzeit erreichbar, ohne im Hauptbereich zu scrollen.
    if session is not None:
        render_settings(session, models, st.sidebar)

    names = [m["name"] for m in models]
    projects = project_store.list_all()
    current_sid = session.id if session else None
    current_pid = session.project_id if session else ""

    # ---- Neues Projekt ----
    with st.sidebar.expander("➕ Neues Projekt", expanded=not projects):
        with st.form("new_project", clear_on_submit=True):
            p_name = st.text_input("Projektname", placeholder="z. B. Statistik HS26")
            p_desc = st.text_input("Beschreibung", placeholder="z. B. Semester 3, Modul W.BA")
            p_model = st.selectbox("Standard-Modell", names or ["<kein Modell>"], key="p_model")
            p_persona = st.selectbox("Standard-Persona", list(config.PERSONAS), key="p_persona")
            if st.form_submit_button("Projekt anlegen", type="primary", use_container_width=True):
                if not names:
                    st.error("Kein Modell verfügbar – zuerst `ollama pull <modell>`.")
                else:
                    project_store.create(
                        p_name, p_desc, p_model, p_persona, config.PERSONAS[p_persona]
                    )
                    st.rerun()

    # ---- Projekte mit ihren Sessions ----
    for project in projects:
        metas = project_store.sessions_of(project.id)
        lessons = [m for m in metas if m["kind"] == "lesson"]
        is_current = project.id == current_pid
        with st.sidebar.expander(f"📁 {project.name} ({len(lessons)})", expanded=is_current):
            if project.description:
                st.caption(project.description)

            proj_meta = next((m for m in metas if m["kind"] == "project"), None)
            active = proj_meta and proj_meta["id"] == current_sid
            if st.button(
                ("▶ " if active else "") + "🎓 Gesamtkontext",
                key=f"psess_{project.id}",
                use_container_width=True,
                type="secondary" if active else "primary",
                help="Session über den Kontext aller Lektionen dieses Projekts",
            ):
                if not project.model and names:
                    project.model = names[0]
                    project.save()
                open_session(project_store.get_or_create_project_session(project).id)

            st.markdown("**Lektionen**")
            if not lessons:
                st.caption("Noch keine Lektion.")
            for meta in lessons:
                _session_button(meta, st.container(), show_digest=True)

            with st.form(f"new_lesson_{project.id}", clear_on_submit=True):
                l_title = st.text_input("Titel", placeholder="z. B. Woche 3 – Regression",
                                        key=f"lt_{project.id}", label_visibility="collapsed")
                if st.form_submit_button("➕ Lektion hinzufügen", use_container_width=True):
                    model = project.model or (names[0] if names else "")
                    prompt = project.system_prompt or config.PERSONAS.get(project.persona, "")
                    if l_title:
                        prompt += f"\n\nThema dieser Lektion: {l_title}."
                    session = session_store.create(
                        model=model, persona=project.persona, system_prompt=prompt,
                        title=l_title or f"Lektion {len(lessons) + 1}",
                        topic=l_title, project_id=project.id, kind="lesson",
                    )
                    session.num_ctx = min(config.DEFAULT_NUM_CTX, cached_ctx_len(model)) if model else config.DEFAULT_NUM_CTX
                    session.save()
                    open_session(session.id)

            with st.popover("🗑 Projekt löschen", use_container_width=True):
                st.caption(f"«{project.name}» wirklich löschen?")
                also = st.checkbox("Sessions mitlöschen", key=f"delall_{project.id}")
                if st.button("Endgültig löschen", key=f"delp_{project.id}", type="primary"):
                    project_store.delete(project.id, delete_sessions=also)
                    if current_pid == project.id:
                        st.session_state.pop("sid", None)
                    st.rerun()

    # ---- Sessions ohne Projekt ----
    loose = [s for s in session_store.list_sessions() if not s.get("project_id")]
    with st.sidebar.expander(f"💬 Ohne Projekt ({len(loose)})", expanded=not projects):
        for meta in loose:
            _session_button(meta, st.container())
        with st.form("new_session", clear_on_submit=True):
            title = st.text_input("Titel", placeholder="z. B. Schnelle Frage")
            topic = st.text_input("Thema", placeholder="z. B. Hypothesentests")
            model = st.selectbox("Modell", names or ["<kein Modell>"], key="new_model")
            persona = st.selectbox("Persona", list(config.PERSONAS), key="new_persona")
            if st.form_submit_button("➕ Session anlegen", use_container_width=True):
                if not names:
                    st.error("Kein Modell verfügbar – zuerst `ollama pull <modell>`.")
                else:
                    prompt = config.PERSONAS[persona]
                    if topic:
                        prompt += f"\n\nThema dieser Session: {topic}. Bleibe strikt bei diesem Thema."
                    session = session_store.create(model, persona, prompt, title, topic)
                    session.num_ctx = min(config.DEFAULT_NUM_CTX, cached_ctx_len(model))
                    session.save()
                    open_session(session.id)

    return busy



# --------------------------------------------------------------------- Settings
def render_settings(session: session_store.Session, models: list[dict], container) -> None:
    """Einstellungen der AKTUELLEN Session.

    Alle Widget-Keys enthalten die Session-ID. Ohne das teilen sich verschiedene
    Sessions denselben Widget-State und Aenderungen wuerden auf die falsche Session wirken.
    """
    sid = session.id
    names = [m["name"] for m in models] or [session.model]

    with container.expander("⚙️ Session-Einstellungen", expanded=False):
        st.caption(f"Gilt nur für «{session.title}»")

        idx = names.index(session.model) if session.model in names else 0
        model = st.selectbox("Modell", names, index=idx, key=f"cfg_model_{sid}")
        max_ctx = cached_ctx_len(model)
        num_ctx = st.select_slider(
            "Kontextfenster (num_ctx)",
            options=[v for v in (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144) if v <= max_ctx]
            or [max_ctx],
            value=min(session.num_ctx, max_ctx),
            key=f"cfg_ctx_{sid}",
            help=f"Modell-Maximum: {max_ctx:,} Tokens. Grösser = mehr RAM-Bedarf.".replace(",", "'"),
        )
        temperature = st.slider("Temperatur", 0.0, 1.5, float(session.temperature), 0.05,
                                key=f"cfg_temp_{sid}")
        rag_mode = st.radio(
            "Material-Modus", ["auto", "full", "off"],
            index=["auto", "full", "off"].index(session.rag_mode),
            horizontal=True, key=f"cfg_rag_{sid}",
            help="auto = ganzes Material solange es passt, sonst nur relevante Abschnitte.",
        )
        c1, c2 = st.columns(2)
        auto_compress = c1.toggle("Auto-Komprimierung", value=session.auto_compress,
                                  key=f"cfg_comp_{sid}")
        think = c2.toggle(
            "Reasoning", value=session.think, key=f"cfg_think_{sid}",
            disabled="thinking" not in cached_caps(model),
            help="Nur für Modelle mit 'thinking'-Capability.",
        )
        title = st.text_input("Titel", value=session.title, key=f"cfg_title_{sid}")
        system_prompt = st.text_area("System-Prompt", value=session.system_prompt, height=140,
                                     key=f"cfg_prompt_{sid}")

        if st.button("💾 Speichern", type="primary", use_container_width=True,
                     key=f"cfg_save_{sid}"):
            session.model = model
            session.num_ctx = int(num_ctx)
            session.temperature = float(temperature)
            session.rag_mode = rag_mode
            session.auto_compress = auto_compress
            session.think = think
            session.system_prompt = system_prompt
            session.title = title
            session.save()
            st.toast("Einstellungen gespeichert", icon="✅")
            st.rerun()

        # ---- Projektzuordnung (immer nur für diese eine Session) ----
        st.divider()
        st.markdown("**Projektzuordnung**")
        if session.kind == "project":
            project = project_store.get(session.project_id)
            st.caption(
                f"Gesamtkontext-Session von 📁 {project.name if project else '–'} – "
                "fest mit dem Projekt verbunden."
            )
        else:
            projects = project_store.list_all()
            options = [""] + [p.id for p in projects]
            labels = {"": "— ohne Projekt (eigenständig) —", **{p.id: f"📁 {p.name}" for p in projects}}
            current = session.project_id if session.project_id in options else ""
            target = st.selectbox(
                "Projekt", options, index=options.index(current),
                format_func=lambda i: labels[i], key=f"cfg_project_{sid}",
                help="Material dieser Session wandert mit: im Projekt sehen es alle Lektionen, "
                     "eigenständig nur diese Session.",
            )
            if target != session.project_id:
                if st.button(f"↪ Verschieben nach {labels[target]}", type="primary",
                             use_container_width=True, key=f"cfg_move_{sid}"):
                    result = project_store.assign(session, target)
                    msg = f"«{session.title}» verschoben · {result['moved']} Material(ien) umgehängt"
                    if result["unlinked"]:
                        msg += f" · {result['unlinked']} Referenz(en) entfernt"
                    st.toast(msg, icon="↪")
                    st.rerun()
            else:
                st.caption("Aktuell: " + labels[current])


# --------------------------------------------------------------------- Chat
def render_history(session: session_store.Session) -> None:
    for msg in session.messages:
        if msg.role == "system":
            continue
        with st.chat_message("user" if msg.role == "user" else "assistant"):
            if msg.archived:
                st.caption("🗜 komprimiert – nicht mehr im aktiven Kontext")
            st.markdown(msg.content)
            stats = msg.meta.get("stats") if msg.meta else None
            if stats:
                st.caption(
                    f"{stats.get('prompt_tokens', 0)} Prompt- / {stats.get('completion_tokens', 0)} "
                    f"Antwort-Tokens · {stats.get('total_duration_s', 0)} s"
                )


def run_turn(session: session_store.Session, user_text: str) -> None:
    session.add("user", user_text)
    session.save()
    with st.chat_message("user"):
        st.markdown(user_text)

    payload, report = ctx.build_payload(session, query=user_text)
    if ctx.should_compress(session, report):
        with st.status("🗜 Kontext wird komprimiert (Memory Compression)…", expanded=False) as status:
            try:
                summary = ctx.compress_history(session)
                status.update(
                    label="🗜 Kontext komprimiert" if summary else "Nichts zu komprimieren",
                    state="complete",
                )
                if summary:
                    payload, report = ctx.build_payload(session, query=user_text)
            except ollama_client.OllamaError as exc:
                status.update(label=f"Komprimierung fehlgeschlagen: {exc}", state="error")

    with st.chat_message("assistant"):
        placeholder = st.empty()
        think_box = st.empty()
        buffer, thinking = "", ""
        stats: dict = {}
        try:
            for event in ollama_client.chat_stream(
                session.model,
                payload,
                num_ctx=session.num_ctx,
                temperature=session.temperature,
                think=session.think if session.think else None,
            ):
                if event["type"] == "token":
                    buffer += event["text"]
                    placeholder.markdown(buffer + " ▌")
                elif event["type"] == "thinking":
                    thinking += event["text"]
                    think_box.caption("🧠 " + thinking[-300:])
                elif event["type"] == "done":
                    stats = event["stats"]
        except ollama_client.OllamaError as exc:
            placeholder.error(str(exc))
            session.messages.pop()  # fehlgeschlagenen Turn nicht im Verlauf verewigen
            session.save()
            return
        placeholder.markdown(buffer)
        think_box.empty()

    session.add("assistant", buffer, stats=stats)
    if len([m for m in session.messages if m.role == "user"]) == 1 and session.title.startswith("Session "):
        title = ctx.suggest_title(session, user_text)
        if title:
            session.title = title
    session.save()
    st.rerun()


def render_chat(session: session_store.Session) -> None:
    render_history(session)

    # Aktives Material sichtbar machen – so ist klar, worauf das Modell Zugriff hat.
    eff_ids = ctx.effective_material_ids(session)
    if eff_ids and session.rag_mode != "off":
        names = [(library.get(i) or type("x", (), {"name": "?"})).name for i in eff_ids]
        preview = ", ".join(names[:4]) + (f" +{len(names) - 4}" if len(names) > 4 else "")
        st.caption(f"📎 Material im Kontext ({len(eff_ids)}): {preview}")
    elif session.rag_mode == "off":
        st.caption("📎 Material-Modus **off** – dem Modell wird kein Material übergeben "
                   "(in den Einstellungen änderbar).")

    if not session.messages:
        if session.kind == "project":
            st.info(
                "Gesamtkontext-Session. Sie sieht die Digests **aller Lektionen** dieses Projekts "
                "plus deren Material. Aktualisiere die Digests im Tab **Projekt**, bevor du startest.",
                icon="🎓",
            )
        else:
            st.info(
                "Session ist leer. Lade Material hoch (Tab **Material**) oder stelle direkt eine Frage.",
                icon="👋",
            )

    prompts = study_tools.PROJECT_PROMPTS if session.kind == "project" else study_tools.QUICK_PROMPTS
    label = "⚡ Semester-Werkzeuge" if session.kind == "project" else "⚡ Lern-Werkzeuge (fertige Prompts)"
    with st.expander(label, expanded=not session.messages):
        cols = st.columns(3)
        for i, (name, prompt) in enumerate(prompts.items()):
            if cols[i % 3].button(name, key=f"qp_{session.id}_{i}", use_container_width=True):
                st.session_state.queued_prompt = prompt
                st.rerun()

    queued = st.session_state.pop("queued_prompt", None)
    user_text = st.chat_input("Frage an das Modell…") or queued
    if user_text:
        run_turn(session, user_text)


# --------------------------------------------------------------------- Projekt
def render_project(session: session_store.Session) -> None:
    """Tab für Projekt-Sessions: Überblick und Pflege des Semester-Gedächtnisses."""
    project = project_store.get(session.project_id)
    if project is None:
        st.warning("Projekt nicht gefunden.")
        return

    lessons = project_store.lesson_sessions(project.id)
    stale = [s for s in lessons if s.messages and s.digest_stale()]
    material_ids = ctx.effective_material_ids(session)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lektionen", len(lessons))
    c2.metric("Nachrichten gesamt", sum(len(s.messages) for s in lessons))
    c3.metric("Materialien geerbt", len(material_ids))
    c4.metric("Digests veraltet", len(stale))

    if stale:
        st.warning(
            f"{len(stale)} Lektion(en) ohne aktuellen Digest – der Gesamtkontext nutzt solange "
            "einen groben Rohauszug. Jetzt aktualisieren für maximale Qualität.",
            icon="🔄",
        )

    c1, c2 = st.columns(2)
    if c1.button("🔄 Digests aktualisieren (nur veraltete)", type="primary", use_container_width=True,
                 disabled=not lessons):
        _refresh_digests(project, session.model, force=False)
    if c2.button("♻️ Alle Digests neu erzeugen", use_container_width=True, disabled=not lessons):
        _refresh_digests(project, session.model, force=True)

    st.divider()
    st.markdown("#### 📚 Lektionen im Gesamtkontext")
    if not lessons:
        st.caption("Noch keine Lektionen – lege sie in der Sidebar unter diesem Projekt an.")
    for i, lesson in enumerate(lessons, start=1):
        icon = "✅" if lesson.digest_text() and not lesson.digest_stale() else (
            "🔄" if lesson.messages else "○"
        )
        with st.expander(f"{icon} Lektion {i}: {lesson.title} · {len(lesson.messages)} Nachrichten"):
            st.caption(
                f"Material: {len(lesson.material_ids)} · "
                + (f"Digest vom {time.strftime('%d.%m.%Y %H:%M', time.localtime(lesson.digest['ts']))}"
                   if lesson.digest else "kein Digest")
            )
            if lesson.messages:
                digest_text = st.text_area(
                    "Digest (bearbeitbar – genau dieser Text landet im Semester-Gedächtnis)",
                    value=lesson.digest_text(),
                    height=200,
                    key=f"digest_edit_{lesson.id}",
                    placeholder="Noch kein Digest – es wird ein Rohauszug verwendet. "
                                "Du kannst hier auch manuell einen schreiben.",
                )
            else:
                digest_text = ""
                st.caption("Leere Lektion, wird nicht in den Gesamtkontext aufgenommen.")

            b1, b2, b3 = st.columns(3)
            if b1.button("💾 Digest speichern", key=f"digsave_{lesson.id}", use_container_width=True,
                         disabled=not lesson.messages):
                if digest_text.strip():
                    lesson.set_digest(digest_text.strip())
                else:
                    lesson.digest = {}
                lesson.save()
                st.toast("Digest gespeichert", icon="💾")
                st.rerun()
            if b2.button("🔄 Neu erzeugen", key=f"dig_{lesson.id}", use_container_width=True,
                         disabled=not lesson.messages):
                with st.spinner(f"Verdichte «{lesson.title}»…"):
                    try:
                        ctx.build_lesson_digest(lesson, session.model, force=True)
                        st.toast("Digest aktualisiert", icon="✅")
                    except ollama_client.OllamaError as exc:
                        st.error(str(exc))
                st.rerun()
            if b3.button("↗ Lektion öffnen", key=f"goto_{lesson.id}", use_container_width=True):
                open_session(lesson.id)

    st.divider()
    with st.expander("⚙️ Projekt-Einstellungen"):
        name = st.text_input("Projektname", value=project.name)
        desc = st.text_input("Beschreibung", value=project.description)
        if st.button("💾 Projekt speichern"):
            project.name, project.description = name, desc
            project.save()
            st.rerun()


def _refresh_digests(project: project_store.Project, model: str, force: bool) -> None:
    """Digest-Refresh als Hintergrund-Job (ein Modell-Call pro Lektion = mehrere Minuten)."""
    job_manager().submit(
        "digests", f"Digests: {project.name}", tasks.digests_task,
        session_title=project.name,
        project_id=project.id, model=model, force=force,
    )
    st.toast("Digest-Aktualisierung läuft im Hintergrund", icon="⚙️")
    st.rerun()


# --------------------------------------------------------------------- Material
def _scope_hint(session: session_store.Session, project: project_store.Project | None) -> str:
    if project:
        return (f"Neues Material wird dem Projekt **{project.name}** zugeordnet und ist in allen "
                "Lektionen dieses Projekts verfügbar.")
    return ("Diese Session gehört zu keinem Projekt – neues Material ist **ausschliesslich hier** "
            "sichtbar. Über die Session-Einstellungen kannst du sie einem Projekt zuordnen.")


def _queue_ingest(session, path: Path, display_name: str, whisper_model: str,
                  language: str, scope_project: str, auto_summarize: bool) -> None:
    """Datei als Hintergrund-Job einreihen – die UI bleibt sofort bedienbar."""
    job_manager().submit(
        "transcribe" if Path(display_name).suffix.lower() in config.AUDIO_SUFFIXES else "ingest",
        display_name,
        tasks.ingest_task,
        session_id=session.id,
        session_title=session.title,
        path=str(path),
        display_name=display_name,
        project_id=scope_project,
        whisper_model=whisper_model,
        language=None if language == "auto" else language,
        auto_summarize=auto_summarize,
        summary_model=session.model,
        num_ctx=session.num_ctx,
    )


def render_material(session: session_store.Session) -> None:
    project = project_store.get(session.project_id) if session.project_id else None

    if session.kind == "project":
        inherited = [i for i in ctx.effective_material_ids(session) if i not in session.material_ids]
        st.info(
            f"Gesamtkontext-Session: {len(inherited)} Material(ien) werden aus den Lektionen geerbt "
            "(nur projektweit freigegebenes Material). Hier hochgeladenes gehört direkt zum Projekt "
            "– z. B. Skript, Modulbeschreibung, alte Prüfungen.",
            icon="🎓",
        )
    else:
        st.caption(_scope_hint(session, project))

    st.markdown("#### 📥 Material → Text → Kontext")
    if not ffmpeg_available():
        st.warning("ffmpeg fehlt – Audio/Video kann nicht transkribiert werden (`brew install ffmpeg`).")

    c1, c2, c3 = st.columns(3)
    whisper_model = c1.selectbox(
        "Whisper-Modell", WHISPER_SIZES, index=WHISPER_SIZES.index(config.WHISPER_MODEL)
        if config.WHISPER_MODEL in WHISPER_SIZES else 2,
        key=f"wm_{session.id}",
        format_func=lambda s: f"{s} {'✅' if model_cached(s) else '⬇️'}",
        help="✅ = lokal bereit · ⬇️ = muss erst geladen werden (nur ausserhalb des VPN möglich). "
             "90-Min-Vorlesung mit Batching: small ≈ 6–10 min, medium ≈ 20 min.",
    )
    if not model_cached(whisper_model):
        st.warning(
            f"Whisper-Modell «{whisper_model}» ist noch nicht lokal vorhanden. "
            f"Einmalig **ausserhalb des VPN** laden:\n\n"
            f"```bash\ncd ~/Hslu/Ollama && ./.venv/bin/python scripts/fetch_whisper.py {whisper_model}\n```",
            icon="⬇️",
        )
    language = c2.selectbox("Sprache der Aufnahme", ["auto", "de", "en", "fr", "it"], index=0,
                            key=f"lang_{session.id}")
    scope_choice = c3.radio(
        "Sichtbarkeit",
        ["Projekt", "Nur diese Session"] if project else ["Nur diese Session"],
        horizontal=True, key=f"scope_new_{session.id}",
        help="Bestimmt, welche Sessions dieses Material sehen. Später jederzeit änderbar.",
    )
    scope_project = session.project_id if (project and scope_choice == "Projekt") else ""
    auto_sum = st.checkbox(
        "Nach der Transkription automatisch zusammenfassen", value=False,
        key=f"autosum_{session.id}",
        help="Hängt die Map-Reduce-Zusammenfassung direkt an den Job an – "
             "du bekommst Transkript **und** Zusammenfassung, ohne erneut zu warten.",
    )

    tab_path, tab_upload, tab_note = st.tabs(
        ["📂 Lokale Datei (empfohlen für Aufzeichnungen)", "⬆️ Upload", "✍️ Notiz"]
    )

    # ---- Variante 1: direkt vom Dateisystem (kein Upload, keine Grössengrenze) ----
    with tab_path:
        st.caption(
            "Für grosse Vorlesungsaufzeichnungen: Pfad der Datei angeben – sie wird direkt "
            "vom Datenträger gelesen. Kein Browser-Upload, keine Grössenbeschränkung. "
            "Tipp: Datei im Finder mit ⌘⌥C kopieren (Pfad kopieren)."
        )
        raw_path = st.text_input(
            "Dateipfad", key=f"path_{session.id}",
            placeholder="/Users/du/Studium/Vorlesung_Woche1.mp4",
        )
        if raw_path and st.button("🚀 Im Hintergrund verarbeiten", type="primary",
                                  key=f"gopath_{session.id}"):
            try:
                src = resolve_local_path(raw_path)
                _queue_ingest(session, src, src.name, whisper_model, language,
                              scope_project, auto_sum)
                st.toast(f"Job gestartet: {src.name}", icon="⚙️")
                st.rerun()
            except IngestError as exc:
                st.error(str(exc))

    # ---- Variante 2: klassischer Upload ----
    with tab_upload:
        st.caption(
            f"Upload-Limit: {MAX_UPLOAD_MB} MB. Die Datei wird komplett durch den Browser "
            "geladen – bei Aufzeichnungen ist *Lokale Datei* deutlich schneller."
        )
        uploads = st.file_uploader(
            "PDF, PPTX, DOCX, TXT/MD, VTT/SRT, MP3/M4A/WAV, MP4/MOV",
            type=[s.lstrip(".") for s in sorted(config.DOC_SUFFIXES | config.AUDIO_SUFFIXES)],
            accept_multiple_files=True, key=f"up_{session.id}",
        )
        if uploads and st.button("🚀 Im Hintergrund verarbeiten", type="primary",
                                 key=f"goup_{session.id}"):
            for upload in uploads:
                path = save_upload(upload.name, upload.getbuffer())
                _queue_ingest(session, path, upload.name, whisper_model, language,
                              scope_project, auto_sum)
            st.toast(f"{len(uploads)} Job(s) gestartet", icon="⚙️")
            st.rerun()

    # ---- Variante 3: eigene Notiz ----
    with tab_note:
        with st.form(f"new_note_{session.id}", clear_on_submit=True):
            note_name = st.text_input("Titel", placeholder="z. B. Eigene Mitschrift Woche 3")
            note_text = st.text_area("Inhalt (Markdown)", height=150)
            if st.form_submit_button("Notiz speichern", use_container_width=True) and note_text.strip():
                note = library.add(
                    name=note_name or "Notiz", text=note_text, kind="note", source="manuell",
                    project_id=scope_project, session_id=session.id,
                )
                session.material_ids.append(note.id)
                session.save()
                st.rerun()

    st.divider()
    st.markdown("#### 📚 Material dieser " + ("Projekt-Umgebung" if project else "Session"))
    materials = library.visible_for(session)
    effective = ctx.effective_material_ids(session)
    if not materials:
        st.caption("Noch kein Material in diesem Bereich.")
    for mat in materials:
        attached = mat.id in session.material_ids
        inherited = mat.id in effective and not attached
        c1, c2, c3 = st.columns([6, 2, 2])
        c1.markdown(f"**{mat.label()}**" + (" · _geerbt_" if inherited else ""))
        c1.caption(
            f"{mat.scope_label()} · {mat.source} · "
            f"{time.strftime('%d.%m.%Y %H:%M', time.localtime(mat.created))}"
            + (f" · {mat.meta.get('pages')} Seiten" if mat.meta.get("pages") else "")
            + (f" · {mat.meta.get('slides')} Slides" if mat.meta.get("slides") else "")
            + (f" · {mat.meta.get('duration')} ({mat.meta.get('language')})" if mat.meta.get("duration") else "")
            + (" · ✍️ bearbeitet" if mat.edited else "")
        )
        if c2.toggle("im Kontext", value=attached, key=f"att_{session.id}_{mat.id}") != attached:
            if attached:
                session.material_ids.remove(mat.id)
            else:
                session.material_ids.append(mat.id)
            session.save()
            st.rerun()
        if c3.button("🗑", key=f"delmat_{session.id}_{mat.id}", help="Material endgültig löschen"):
            library.delete(mat.id)
            if attached:
                session.material_ids.remove(mat.id)
                session.save()
            st.rerun()

        with st.expander("✏️ Bearbeiten / Aktionen", expanded=False):
            e1, e2 = st.columns([3, 2])
            new_name = e1.text_input("Name", value=mat.name, key=f"name_{session.id}_{mat.id}")
            if project:
                current = "Projekt" if mat.project_id else "Nur diese Session"
                new_scope = e2.radio(
                    "Sichtbarkeit", ["Projekt", "Nur diese Session"],
                    index=0 if current == "Projekt" else 1,
                    horizontal=True, key=f"scope_{session.id}_{mat.id}",
                )
            else:
                new_scope = "Nur diese Session"
                e2.caption("Sichtbarkeit: nur diese Session (kein Projekt zugeordnet)")

            text = st.text_area(
                "Extrahierter Text (bearbeitbar – z. B. OCR-Fehler, Whisper-Verhörer, Kürzungen)",
                value=mat.text(), height=280, key=f"txt_{session.id}_{mat.id}",
            )
            s1, s2, s3 = st.columns(3)
            if s1.button("💾 Speichern", key=f"save_{session.id}_{mat.id}", type="primary",
                         use_container_width=True):
                if text != mat.text():
                    library.save_text(mat.id, text)
                target_pid = session.project_id if (project and new_scope == "Projekt") else ""
                library.update(mat.id, name=new_name, project_id=target_pid,
                               session_id=mat.session_id or session.id)
                st.toast("Material gespeichert", icon="💾")
                st.rerun()
            s2.download_button(
                "⬇️ Herunterladen", mat.text(), file_name=f"{Path(mat.name).stem}.md",
                key=f"dl_{session.id}_{mat.id}", use_container_width=True,
            )
            if s3.button("🧠 Zusammenfassen (Hintergrund)", key=f"sum_{session.id}_{mat.id}",
                         use_container_width=True):
                job_manager().submit(
                    "summarize", f"Zusammenfassung: {mat.name}", tasks.summarize_task,
                    session_id=session.id, session_title=session.title,
                    material_id=mat.id, model=session.model, num_ctx=session.num_ctx,
                )
                st.toast("Zusammenfassung läuft im Hintergrund", icon="⚙️")
                st.rerun()

    # ---- Altbestand / verwaistes Material ----
    orphans = [m for m in library.unassigned() if m.id not in session.material_ids]
    if orphans:
        with st.expander(f"⚠️ Nicht zugeordnetes Material ({len(orphans)})"):
            st.caption(
                "Material ohne Projekt- oder Session-Bindung (Altbestand oder gelöschte Session). "
                "Es landet in keinem Kontext, bis du es hier zuordnest."
            )
            for mat in orphans:
                o1, o2, o3 = st.columns([5, 3, 2])
                o1.markdown(f"**{mat.label()}**")
                o1.caption(mat.source)
                target = "dieses Projekt" if project else "diese Session"
                if o2.button(f"➕ {target} zuordnen", key=f"adopt_{session.id}_{mat.id}",
                             use_container_width=True):
                    library.update(
                        mat.id,
                        project_id=session.project_id if project else "",
                        session_id=session.id,
                    )
                    session.material_ids.append(mat.id)
                    session.save()
                    st.rerun()
                if o3.button("🗑", key=f"delorph_{session.id}_{mat.id}", use_container_width=True):
                    library.delete(mat.id)
                    st.rerun()


# --------------------------------------------------------------------- Kontext
def render_context(session: session_store.Session) -> None:
    payload, report = ctx.build_payload(session)
    budget: ctx.Budget = report["budget"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prompt (geschätzt)", f"{report['prompt_tokens_est']:,}".replace(",", "'"))
    c2.metric("Nutzbares Budget", f"{budget.usable:,}".replace(",", "'"), help=f"num_ctx {session.num_ctx}")
    c3.metric("Auslastung", f"{report['fill'] * 100:.0f} %")
    c4.metric("Zusammenfassungen", report["summaries"])
    st.progress(min(report["fill"], 1.0))
    if report["fill"] >= config.COMPRESSION_TRIGGER:
        st.warning("Budget-Schwelle erreicht – beim nächsten Turn wird komprimiert.", icon="🗜")

    mat = report["material"]
    st.caption(
        f"Material-Modus: **{mat['mode']}** · Quellen im Kontext: "
        f"{', '.join(mat['used']) if mat['used'] else '–'}"
        + (" · ⚠️ abgeschnitten" if mat.get("truncated") else "")
        + (f" · {report['history_dropped']} alte Nachrichten ausgelassen" if report["history_dropped"] else "")
    )

    # Materialabdeckung: wie viel des zugeordneten Materials passt tatsaechlich rein?
    eff_ids = ctx.effective_material_ids(session)
    total_mat = sum((library.get(i) or type("x", (), {"n_tokens_est": 0})).n_tokens_est for i in eff_ids)
    if session.rag_mode != "off" and total_mat:
        shown = mat["tokens"]
        coverage = min(shown / total_mat, 1.0) if total_mat else 1.0
        st.caption(
            f"📊 Materialabdeckung: **{coverage * 100:.0f} %** "
            f"({shown:,} von {total_mat:,} Tokens zugeordnet)".replace(",", "'")
        )
        if coverage < 0.95:
            max_ctx = cached_ctx_len(session.model)
            can_grow = session.num_ctx < max_ctx
            with st.container(border=True):
                st.markdown(
                    "⚠️ **Nicht das ganze Material passt ins Kontextfenster.** "
                    "Bei gezielten Fragen wählt das System automatisch die relevantesten "
                    "Abschnitte (Retrieval). Für einen **vollständigen** Überblick:"
                )
                b1, b2 = st.columns(2)
                if can_grow:
                    # Auf die naechste sinnvolle Stufe erhoehen, damit mehr/alles passt.
                    steps = [8192, 16384, 32768, 65536, 131072, 262144]
                    target = next((v for v in steps if v > session.num_ctx and v <= max_ctx),
                                  min(max_ctx, 32768))
                    if b1.button(f"🔼 Kontextfenster auf {target:,} erhöhen".replace(",", "'"),
                                 use_container_width=True, key=f"grow_ctx_{session.id}"):
                        session.num_ctx = target
                        session.save()
                        st.toast(f"num_ctx = {target}", icon="🔼")
                        st.rerun()
                else:
                    b1.caption(f"num_ctx bereits am Modell-Maximum ({max_ctx:,})".replace(",", "'"))
                b2.caption(
                    "Oder im Tab **Material** die Aufzeichnung **zusammenfassen** – "
                    "die kompakte Version passt vollständig in den Kontext."
                )

    dig = report.get("digests", {})
    if session.kind == "project":
        st.caption(
            f"🎓 Semester-Gedächtnis: {len(dig.get('lessons', []))} Lektion(en) im Prompt "
            f"(~{dig.get('tokens', 0)} Tokens, Budget {budget.digest_max})"
            + (f" · {dig['skipped']} wegen Budget ausgelassen" if dig.get("skipped") else "")
            + (f" · ⚠️ veraltete Digests: {', '.join(dig['stale'])}" if dig.get("stale") else "")
            + (f" · 📝 Rohauszug statt Digest: {', '.join(dig['raw'])}" if dig.get("raw") else "")
        )

    c1, c2, c3 = st.columns(3)
    if c1.button("🗜 Jetzt komprimieren", use_container_width=True):
        with st.spinner("Fasse Verlauf zusammen…"):
            try:
                st.toast("Komprimiert" if ctx.compress_history(session) else "Zu wenig Verlauf", icon="🗜")
            except ollama_client.OllamaError as exc:
                st.error(str(exc))
        st.rerun()
    c2.download_button(
        "⬇️ Session als Markdown", session_store.export_markdown(session),
        file_name=session_store.export_filename(session), use_container_width=True,
    )
    if c3.button("🧹 Verlauf leeren (Material behalten)", use_container_width=True):
        session.messages.clear()
        session.summaries.clear()
        session.save()
        st.rerun()

    if session.summaries:
        with st.expander("🗜 Verdichtetes Gedächtnis (bearbeitbar)", expanded=False):
            st.caption(
                "Diese Zusammenfassung ersetzt die archivierten Turns im Prompt. "
                "Du kannst sie korrigieren, ergänzen oder verwerfen."
            )
            for i, s in enumerate(session.summaries):
                edited = st.text_area(
                    f"Zusammenfassung {i + 1} · ~{ctx.estimate_tokens(s['content'])} Tokens",
                    value=s["content"], height=220, key=f"sum_edit_{session.id}_{i}",
                )
                b1, b2 = st.columns(2)
                if b1.button("💾 Speichern", key=f"sum_save_{session.id}_{i}", use_container_width=True):
                    session.summaries[i]["content"] = edited
                    session.save()
                    st.toast("Zusammenfassung gespeichert", icon="💾")
                    st.rerun()
                if b2.button("🗑 Verwerfen", key=f"sum_del_{session.id}_{i}", use_container_width=True,
                             help="Entfernt die Zusammenfassung; archivierte Nachrichten bleiben erhalten."):
                    session.summaries.pop(i)
                    session.save()
                    st.rerun()

    with st.expander("✏️ Nachrichten im Kontext verwalten", expanded=False):
        visible = [(i, m) for i, m in enumerate(session.messages) if m.role != "system"]
        if not visible:
            st.caption("Noch keine Nachrichten.")
        st.caption(
            "Archivierte Nachrichten bleiben im Verlauf sichtbar, werden aber nicht mehr an das "
            "Modell gesendet. So kürzt du den Kontext gezielt, ohne etwas zu verlieren."
        )
        for i, msg in visible:
            who = "🧑 Du" if msg.role == "user" else "🤖 Modell"
            state = "🗜 archiviert" if msg.archived else "aktiv"
            head = f"{who} · {state} · ~{ctx.estimate_tokens(msg.content)} Tokens · " \
                   f"{time.strftime('%d.%m. %H:%M', time.localtime(msg.ts))}"
            with st.container(border=True):
                st.caption(head)
                new_text = st.text_area(
                    "Inhalt", value=msg.content, height=120,
                    key=f"msg_edit_{session.id}_{i}", label_visibility="collapsed",
                )
                m1, m2 = st.columns(2)
                if m1.button("💾 Text speichern", key=f"msg_save_{session.id}_{i}",
                             use_container_width=True):
                    session.messages[i].content = new_text
                    session.save()
                    st.toast("Nachricht aktualisiert", icon="💾")
                    st.rerun()
                label = "♻️ Reaktivieren" if msg.archived else "🗜 Archivieren"
                if m2.button(label, key=f"msg_arch_{session.id}_{i}", use_container_width=True):
                    session.messages[i].archived = not msg.archived
                    session.save()
                    st.rerun()

    with st.expander("🔍 Exakter Payload an /api/chat", expanded=False):
        for msg in payload:
            st.markdown(f"**{msg['role']}** · ~{ctx.estimate_tokens(msg['content'])} Tokens")
            st.code(msg["content"][:2500] + ("…" if len(msg["content"]) > 2500 else ""))


# --------------------------------------------------------------------- Main
def main() -> None:
    models = cached_models()
    session = current_session()
    busy = render_sidebar(models, session)

    if session is None:
        st.title("🎓 Lokaler Study Hub")
        st.markdown(
            "Alles läuft offline über deine lokale Ollama-Instanz.\n\n"
            "1. **Projekt anlegen** (links) – z. B. ein Modul pro Semester.\n"
            "2. **Lektionen** darin anlegen – eine Session pro Woche hält den Kontext scharf.\n"
            "3. **Material hochladen** – PDFs, Slides und Aufzeichnungen werden extrahiert bzw. "
            "transkribiert und automatisch in den Kontext geladen.\n"
            "4. **Gesamtkontext** – die Projekt-Session sieht die verdichteten Digests *aller* "
            "Lektionen plus deren Material: ideal für Semester-Synthese und Prüfungsvorbereitung."
        )
        projects = project_store.list_all()
        if projects:
            st.markdown("#### 📁 Deine Projekte")
            cols = st.columns(min(len(projects), 3))
            for i, project in enumerate(projects):
                s = project_store.stats(project.id)
                with cols[i % len(cols)].container(border=True):
                    st.markdown(f"**{project.name}**")
                    st.caption(project.description or "–")
                    st.caption(
                        f"{s['n_lessons']} Lektionen · {s['n_messages']} Nachrichten · "
                        f"{s['n_materials']} Materialien"
                    )
                    if st.button("🎓 Gesamtkontext öffnen", key=f"home_{project.id}",
                                 use_container_width=True):
                        if not project.model and models:
                            project.model = models[0]["name"]
                            project.save()
                        open_session(project_store.get_or_create_project_session(project).id)
        if models:
            st.markdown("**Verfügbare Modelle:** " + ", ".join(f"`{m['name']}`" for m in models))
        return

    project = project_store.get(session.project_id) if session.project_id else None
    head, badge = st.columns([8, 2])
    head.title(session.title)
    head.caption(
        (f"📁 {project.name} · " if project else "💬 ohne Projekt · ")
        + f"`{session.model}` · {session.persona} · num_ctx {session.num_ctx:,}".replace(",", "'")
        + (f" · Thema: {session.topic}" if session.topic else "")
    )
    badge.caption("⚙️ Einstellungen und Projektzuordnung dieser Session findest du **in der Sidebar**.")

    if session.kind == "project":
        tabs = st.tabs(["💬 Chat", "🎓 Projekt", f"📚 Material ({len(ctx.effective_material_ids(session))})",
                        "🧠 Kontext"])
        with tabs[0]:
            render_chat(session)
        with tabs[1]:
            render_project(session)
        with tabs[2]:
            render_material(session)
        with tabs[3]:
            render_context(session)
    else:
        tabs = st.tabs(["💬 Chat", f"📚 Material ({len(session.material_ids)})", "🧠 Kontext"])
        with tabs[0]:
            render_chat(session)
        with tabs[1]:
            render_material(session)
        with tabs[2]:
            render_context(session)

    _autorefresh(busy)


def _autorefresh(busy: bool) -> None:
    """Solange Jobs laufen, die Seite alle 2 s neu zeichnen (Fortschrittsanzeige).

    Bewusst am Ende des Skripts: So ist die Seite bereits vollstaendig gerendert und
    bedienbar, bevor gewartet wird. Chat-Eingaben unterbrechen den Zyklus sofort,
    weil Streamlit bei Interaktion ohnehin neu ausfuehrt.
    """
    if not busy:
        return
    time.sleep(2)
    st.rerun()


if __name__ == "__main__":
    main()

