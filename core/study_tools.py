"""Lern-Werkzeuge: fertige Prompt-Vorlagen + Map-Reduce-Zusammenfassung.

Die Map-Reduce-Zusammenfassung umgeht die Kontextgrenze: lange Materialien werden
chunkweise verdichtet ("map") und die Teilzusammenfassungen anschliessend
zu einem Dokument verschmolzen ("reduce").
"""
from __future__ import annotations

from typing import Callable

from . import library, ollama_client
from .context_manager import estimate_tokens

QUICK_PROMPTS: dict[str, str] = {
    "📋 Strukturierte Zusammenfassung": (
        "Erstelle aus dem Material eine hierarchische Zusammenfassung: "
        "1) Kernaussage in einem Satz, 2) Gliederung mit Unterpunkten und je 1-2 Saetzen Erklaerung, "
        "3) Fachbegriffe mit Definition, 4) offene Fragen. Nichts weglassen, was pruefungsrelevant ist."
    ),
    "🎯 Drei Hauptthesen + Belege": (
        "Extrahiere die drei zentralen Thesen des Materials. Belege jede These mit einem woertlichen "
        "Zitat inkl. Quellenangabe [Quelle: <Datei>, <Seite/Slide/Zeit>] und erlaeutere sie in 3-5 Saetzen."
    ),
    "🃏 Flashcards": (
        "Erzeuge 15 Lernkarten aus dem Material. Format pro Karte exakt:\n"
        "F: <praezise Frage>\nA: <Antwort, max. 3 Saetze>\n---\n"
        "Decke unterschiedliche Schwierigkeitsgrade ab, keine Trivialfragen."
    ),
    "❓ Quiz (MC)": (
        "Erstelle 10 Multiple-Choice-Fragen (4 Optionen, genau eine korrekt) zum Material. "
        "Gib zuerst alle Fragen aus, danach unter '## Loesungen' die richtigen Antworten mit kurzer Begruendung."
    ),
    "🧪 Pruefungssimulation": (
        "Simuliere eine 60-minuetige Hochschulpruefung zum Material: 3 Verstaendnisfragen, "
        "2 Transferaufgaben, 1 kritische Diskussionsfrage. Gib Punkte und einen Erwartungshorizont an."
    ),
    "🔍 Verstaendnis-Check": (
        "Stelle mir nacheinander Fragen zum Material, um meinen Wissensstand zu pruefen. "
        "Eine Frage pro Turn, warte auf meine Antwort, bewerte sie kritisch und benenne Luecken."
    ),
    "🧭 Lernplan": (
        "Erstelle aus dem Material einen Lernplan fuer 5 Tage: pro Tag Themenblock, Lernziel, "
        "geschaetzte Zeit und eine Selbstkontrollfrage."
    ),
}

# Prompts fuer die Projekt-Session (Gesamtkontext ueber alle Lektionen eines Semesters)
PROJECT_PROMPTS: dict[str, str] = {
    "🧵 Roter Faden des Semesters": (
        "Zeichne den roten Faden durch alle Lektionen dieses Projekts nach: Wie bauen die Themen "
        "aufeinander auf? Gib eine chronologische Übersicht mit je 2-3 Sätzen pro Lektion und "
        "benenne explizit die Verbindungen zwischen den Lektionen."
    ),
    "🔗 Querbezüge & Widersprüche": (
        "Vergleiche die Lektionen dieses Projekts miteinander. Nenne (a) Konzepte, die in mehreren "
        "Lektionen vorkommen, (b) scheinbare Widersprüche oder unterschiedliche Definitionen, "
        "(c) Themen, die angekündigt, aber nie vertieft wurden. Verweise jeweils auf die Lektion."
    ),
    "🕳 Lücken-Analyse": (
        "Analysiere den Gesamtkontext: Welche Themen sind nur oberflächlich behandelt? "
        "Welche offenen Fragen aus früheren Lektionen wurden nie beantwortet? "
        "Priorisiere die Lücken nach vermuteter Prüfungsrelevanz."
    ),
    "📝 Semester-Zusammenfassung": (
        "Erstelle eine Gesamtzusammenfassung des Semesters: 1) Kernaussage des Moduls in 3 Sätzen, "
        "2) Gliederung nach Themenblöcken (nicht nach Lektionen), 3) Glossar aller Fachbegriffe, "
        "4) die 10 wichtigsten Merksätze."
    ),
    "🎓 Semester-Prüfung": (
        "Simuliere die Semesterabschlussprüfung über den gesamten Stoff: 6 Fragen, die bewusst "
        "mehrere Lektionen kombinieren (Transferaufgaben). Gib Punkte und Erwartungshorizont an."
    ),
    "🗺 Concept Map": (
        "Erstelle eine textuelle Concept Map des gesamten Stoffs: Hauptkonzepte als Knoten, "
        "Beziehungen als beschriftete Kanten (Format: A --[Beziehung]--> B), gruppiert nach Themenblock."
    ),
}

MAP_PROMPT = (
    "Verdichte den folgenden Materialabschnitt zu praezisen Stichpunkten. "
    "Erhalte Definitionen, Zahlen, Formeln, Beispiele und Eigennamen woertlich. "
    "Keine Einleitung, nur die Stichpunkte."
)

REDUCE_PROMPT = (
    "Fuege die folgenden Teilzusammenfassungen zu EINEM strukturierten Dokument zusammen: "
    "1) Kernaussage, 2) thematische Gliederung mit Unterpunkten, 3) Glossar der Fachbegriffe, "
    "4) offene Fragen. Entferne Redundanzen, erfinde nichts hinzu."
)


def summarize_material(
    material: library.Material,
    model: str,
    num_ctx: int = 8192,
    progress: Callable[[float, str], None] | None = None,
) -> str:
    """Map-Reduce-Zusammenfassung eines beliebig langen Materials."""
    chunks = library.chunk_text(material, target_chars=max(num_ctx * 2, 4000), overlap=0)
    if not chunks:
        return ""

    if len(chunks) == 1:
        if progress:
            progress(0.5, "fasse zusammen")
        return ollama_client.chat_once(
            model,
            [{"role": "system", "content": REDUCE_PROMPT}, {"role": "user", "content": chunks[0].text}],
            num_ctx=num_ctx,
            temperature=0.2,
        )

    partials: list[str] = []
    for i, chunk in enumerate(chunks):
        if progress:
            progress(i / (len(chunks) + 1), f"Abschnitt {i + 1}/{len(chunks)}")
        partials.append(
            ollama_client.chat_once(
                model,
                [{"role": "system", "content": MAP_PROMPT}, {"role": "user", "content": chunk.text}],
                num_ctx=num_ctx,
                temperature=0.2,
            )
        )

    joined = "\n\n".join(f"### Abschnitt {i + 1}\n{p}" for i, p in enumerate(partials))
    while estimate_tokens(joined) > num_ctx * 0.6 and len(partials) > 1:  # zweite Reduktionsstufe
        joined = joined[: int(len(joined) * 0.8)]
    if progress:
        progress(0.95, "fuege zusammen")
    return ollama_client.chat_once(
        model,
        [{"role": "system", "content": REDUCE_PROMPT}, {"role": "user", "content": joined}],
        num_ctx=num_ctx,
        temperature=0.3,
    )

