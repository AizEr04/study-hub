# 🎓 Study Hub – lokaler Ollama-Arbeitsplatz fürs Studium

Web-UI für deine lokalen Ollama-Modelle mit **Session-Verwaltung, explizitem Kontext-Management**
und **Ingest-Pipeline** für PDFs, PowerPoints und Vorlesungsaufzeichnungen.
Alles läuft offline auf deinem Rechner – keine Cloud, keine Uploads nach aussen.

## Start

```bash
cd ~/Hslu/Ollama
./setup.sh               # einmalig, NUR ausserhalb des Firmen-VPN (PyPI ist im VPN blockiert)
./run.sh                 # → http://localhost:8501
```

Voraussetzungen: `ollama serve` läuft, `ffmpeg` installiert (`brew install ffmpeg`).
Die Python-Abhängigkeiten (`requirements.txt`) müssen **ohne Firmen-VPN** installiert werden,
da PyPI im VPN blockiert ist.

Selbsttest ohne UI:

```bash
./.venv/bin/python scripts/selftest.py          # Ingest, Budget, Retrieval, Projekte
./.venv/bin/python scripts/selftest.py --chat   # zusätzlich echter Modell-Call + Kompression
./.venv/bin/python scripts/uitest.py            # rendert alle Ansichten headless
./.venv/bin/python scripts/jobtest.py           # Hintergrund-Jobs: Queue, Abbruch, Parallelität
./.venv/bin/python scripts/transcribetest.py    # echte Transkription (misst die Geschwindigkeit)
```

## Projekte, Lektionen, Gesamtkontext

```
📁 Projekt "Statistik HS26"
   ├── 💬 Woche 1 – Deskriptive Statistik   ← eine Session pro Lektion
   ├── 💬 Woche 2 – Hypothesentests
   ├── 💬 Woche 3 – Regression
   └── 🎓 Gesamtkontext                      ← sieht ALLE Lektionen zusammen
```

**Lektions-Sessions** funktionieren wie bisher: enges Thema, eigenes Material, scharfer Kontext.

Die **Projekt-Session** (`kind="project"`, eine pro Projekt) baut ihren Kontext aus drei Quellen:

1. **Semester-Gedächtnis** – pro Lektion ein **Digest** (max. 250 Wörter: Thema, Kernpunkte,
   Bezüge, offene Fragen), den das Modell selbst aus dem Lektionsverlauf erzeugt.
   Rohverläufe würden jedes Kontextfenster sprengen – Digests nicht.
2. **Geerbtes Material** – alle Materialien sämtlicher Lektionen, plus projekteigenes
   (Skript, Modulbeschreibung, alte Prüfungen).
3. Der eigene Gesprächsverlauf der Projekt-Session.

Budget-Aufteilung im Gesamtkontext: 40 % Material, 35 % Digests, 25 % Verlauf.
Passen nicht alle Digests hinein, werden die für die aktuelle Frage relevantesten
ausgewählt – die Reihenfolge bleibt chronologisch, damit der rote Faden erhalten bleibt.

**Digest-Status** in der Sidebar: ✅ aktuell · 🔄 veraltet (neue Nachrichten seit letztem Digest)
· ○ leere Lektion. Aktualisieren im Tab **Projekt** (einzeln oder alle auf einmal).
Solange kein Digest existiert, nutzt der Gesamtkontext automatisch einen groben Rohauszug –
er funktioniert also auch ohne Vorbereitung, nur schlechter.

Eigene Semester-Werkzeuge in der Projekt-Session: Roter Faden, Querbezüge & Widersprüche,
Lücken-Analyse, Semester-Zusammenfassung, Semester-Prüfung, Concept Map.

## Grosse Vorlesungsaufzeichnungen (500 MB – 2 GB)

Die 200-MB-Grenze war Streamlits Upload-Limit — sie hat **nichts** mit dem Kontextfenster zu tun:
ein 90-Minuten-Transkript sind ~150 KB Text. Das Limit steht jetzt auf 4 GB
(`.streamlit/config.toml`).

**Die Datei zerschneiden ist nicht nötig und schadet eher** — faster-whisper liest streamend,
Schnitte zerreissen Wörter an den Grenzen.

Drei Wege, empfohlene Reihenfolge:

1. **📂 Lokale Datei (empfohlen)** — Pfad angeben, die Datei wird direkt vom Datenträger
   gelesen. Kein Browser-Upload, keine Grössengrenze. Pfad im Finder mit `⌘⌥C` kopieren.
2. **⬆️ Upload** — bis 4 GB, läuft aber komplett durch Browser und Arbeitsspeicher.
3. Vorab selbst verkleinern (macht die App bei Video/>150 MB automatisch).

### ⚠️ macOS-Berechtigung für Downloads/Dokumente/Schreibtisch

macOS sperrt diese Ordner für Terminal-Programme. Symptom: *„Operation not permitted"*.
Zwei Lösungen:

* **Systemeinstellungen → Datenschutz & Sicherheit → Festplattenvollzugriff** → Terminal
  aktivieren, Terminal neu starten. Danach funktioniert jeder Pfad.
* Oder die Datei in einen unkritischen Ordner legen: `mv ~/Downloads/Vorlesung.mp4 ~/Studium/`

### Geschwindigkeit

Gemessen auf diesem Mac (M-Chip, `small`, int8): **~12× Echtzeit** → eine 90-Minuten-Vorlesung
in **ca. 8 Minuten**. Drei Stellschrauben, alle per Umgebungsvariable:

| Variable | Default | Wirkung |
|---|---|---|
| `WHISPER_BATCH_SIZE` | `8` | Batch-Verarbeitung (faster-whisper ≥ 1.1), bis 4× schneller. `1` = aus |
| `WHISPER_BEAM_SIZE` | `1` | `5` ist minimal genauer, aber deutlich langsamer |
| `WHISPER_THREADS` | 70 % der Kerne | ctranslate2-Default (4) verschenkt auf M-Chips Leistung |

**Whisper-Modell vorab laden** (huggingface.co, im VPN blockiert):

```bash
./.venv/bin/python scripts/fetch_whisper.py small     # einmalig, ausserhalb des VPN
./.venv/bin/python scripts/fetch_whisper.py --help    # zeigt, was lokal bereitliegt
```

Fehlt das Modell, bricht die Transkription nach 0,4 s mit Hinweis ab statt im Timeout zu hängen.
Im Dropdown steht ✅ (bereit) bzw. ⬇️ (fehlt).

## Hintergrund-Jobs & parallele Sessions

Transkriptionen, Zusammenfassungen und Digest-Refreshs laufen **im Hintergrund**
(`core/jobs.py`). Die Sidebar zeigt für jeden Job Fortschrittsbalken, Restzeit-Schätzung
und einen Abbrechen-Knopf; die Seite aktualisiert sich automatisch alle 2 Sekunden,
solange etwas läuft.

Währenddessen kannst du normal weiterarbeiten: andere Sessions öffnen, chatten, Material
bearbeiten. Zwei Jobs laufen gleichzeitig (`STUDYHUB_MAX_JOBS`), weitere warten in der
Schlange. Jobs überstehen Reruns; nach einem App-Neustart werden unterbrochene als solche
markiert.

Beim Upload lässt sich **„Nach der Transkription automatisch zusammenfassen"** aktivieren —
dann liefert ein einziger Job Transkript *und* Zusammenfassung.

**Mehrere Sessions gleichzeitig am Modell**: Ollama serialisiert Anfragen standardmässig.
Einmalig konfigurieren:

```bash
./scripts/ollama_parallel.sh 3      # 3 parallele Anfragen, Modell 30 min im RAM
```

Das setzt `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS` und `OLLAMA_KEEP_ALIVE`
(letzteres verhindert das 30–60 s dauernde Nachladen bei jedem Sessionwechsel).
Speicher-Faustregel bei `gemma4:26b`: 2–3 parallele Anfragen sind realistisch.

## Sichtbarkeit von Material (Scope)

Material wird **nie global geteilt**. Jedes Material hat genau einen Scope:

| Scope | sichtbar in | typischer Fall |
|---|---|---|
| 📁 **Projekt** | allen Lektionen *und* dem Gesamtkontext dieses Projekts | Vorlesungsfolien, Skript |
| 💬 **Nur diese Session** | ausschliesslich der Session, in der es hochgeladen wurde | private Mitschrift, Sonderfall |
| ⚠️ **nicht zugeordnet** | nirgends (bis du es zuweist) | Altbestand, gelöschte Session |

Der Scope wird beim Upload gewählt (Tab *Material* → *Sichtbarkeit*) und ist jederzeit
änderbar (*Bearbeiten / Aktionen* → *Sichtbarkeit*). Eine Session ohne Projekt kann nur
sessionprivates Material haben. Der Scope wird auch beim Payload-Bau durchgesetzt
(`context_manager.effective_material_ids`) – nicht nur in der Anzeige.

**Sessions nachträglich zuordnen**: *Session-Einstellungen → Projektzuordnung*.
Material dieser Session wandert automatisch mit: beim Hinzufügen wird es projektweit
freigegeben, beim Herauslösen wieder auf die Session beschränkt. Referenzen auf Material,
das dadurch unsichtbar würde, werden entfernt – so kann nie etwas im Prompt landen,
das du in der UI nicht siehst.

## Wie Material ins Kontextfenster kommt

`context_manager.build_material_block()` entscheidet je nach **Material-Modus** (Einstellungen):

* **`auto`** (Default): Passt das gesamte zugeordnete Material ins Budget, kommt es
  **vollständig** rein. Ist es grösser (typisch bei Vorlesungstranskripten, ~30 000 Tokens),
  wird auf **Retrieval** umgeschaltet – die zur Frage passendsten Abschnitte werden per
  Keyword-Ranking (BM25-artig, offline) ausgewählt.
* **`full`**: erzwingt das ganze Material, notfalls token-genau abgeschnitten.
* **`off`**: kein Material an das Modell.

Grosse Aufzeichnungen werden dafür in ~1 000-Token-Chunks entlang der Zeitanker
(`## 00:12:00`) zerlegt (`library.chunk_text`, splittet auch absatzlose Transkripte hart nach).

Der Chat zeigt oben, **welches Material aktiv** ist; der Tab **Kontext** zeigt die
**Materialabdeckung in Prozent**. Passt nicht alles, bietet er einen Knopf zum Erhöhen des
Kontextfensters (`num_ctx`) und weist auf die **Zusammenfassen**-Funktion hin – die kompakte
Version passt vollständig. Für einen echten Gesamt-Überblick eines Transkripts ist die
**Map-Reduce-Zusammenfassung** (Tab *Material*) der richtige Weg: Sie liest das Material in
Stücken und ist unabhängig vom Kontextfenster.

## Kontexte bearbeiten

Jeder Bestandteil des Kontexts ist editierbar und wird sofort persistiert:

* **Material-Text** (Tab *Material* → *Bearbeiten*) – OCR-Fehler, Whisper-Verhörer oder
  überflüssige Passagen direkt korrigieren; Tokenzahl wird neu berechnet. Auch Umbenennen
  und eigene Notizen als Material anlegen.
* **Lektions-Digest** (Tab *Projekt*) – der Text, der im Semester-Gedächtnis landet,
  lässt sich manuell überschreiben statt neu generieren.
* **Zusammenfassungen** der Memory Compression (Tab *Kontext*) – korrigieren oder verwerfen.
* **Einzelne Nachrichten** (Tab *Kontext* → *Nachrichten verwalten*) – Inhalt bearbeiten oder
  gezielt archivieren/reaktivieren. Archiviert = bleibt im Verlauf sichtbar, geht aber nicht
  mehr an das Modell. Gelöscht wird nie etwas.
* **System-Prompt** (Session-Einstellungen).

## Workflow

1. **Projekt anlegen** (Sidebar) – z. B. ein Modul pro Semester.
2. **Lektion hinzufügen** – eine Session pro Woche, direkt im Projekt.
   *Ein Thema pro Session* („Selective Context Loading") hält den Kontext scharf.
3. **Material hochladen** (Tab *Material*):
   | Typ | Verarbeitung |
   |---|---|
   | PDF | `pypdf`, Seitenanker `## Seite n` |
   | PPTX | `python-pptx`, inkl. **Dozentennotizen** |
   | DOCX / TXT / MD / HTML | direkte Extraktion |
   | VTT / SRT | Untertitel mit Zeitankern |
   | MP3/M4A/WAV/MP4/MOV | **Whisper-Transkription** mit Zeitankern `## 00:12:00` |
   Das Ergebnis landet als Material in der Bibliothek und wird automatisch in den Session-Kontext gelegt.
   Materialien sind sessionübergreifend wiederverwendbar (einmal transkribieren, mehrfach nutzen).
4. **Arbeiten** (Tab *Chat*): frei prompten oder ein Lern-Werkzeug klicken
   (Zusammenfassung, Thesen + Belege, Flashcards, MC-Quiz, Prüfungssimulation, Verständnis-Check, Lernplan).
5. **Semester-Synthese** (🎓 *Gesamtkontext*): Digests im Tab *Projekt* aktualisieren,
   dann über den ganzen Stoff arbeiten – roter Faden, Querbezüge, Lücken, Semester-Prüfung.
6. **Kontrollieren** (Tab *Kontext*): Token-Auslastung, welche Quellen und Lektionen gerade im
   Prompt sind, manuelle Komprimierung, Markdown-Export der Session.

## Kontext-Management (das Herzstück)

Ollama ist zustandslos – das Gedächtnis lebt ausschliesslich im `messages`-Array. Umgesetzt in `core/context_manager.py`:

1. **Token Budgeter** – vor jedem Call wird gerechnet: 20 % des `num_ctx` bleiben für die Antwort reserviert,
   der Rest wird auf Material (55 %) und Verlauf (45 %) aufgeteilt. Schätzung via `tiktoken`
   (im Test 3 % Abweichung zur echten Server-Zählung).
2. **Memory Compression** – ab 75 % Auslastung fasst das Modell seinen eigenen älteren Verlauf zusammen.
   Die alten Nachrichten werden `archived=True` markiert (in der UI weiterhin sichtbar, grau),
   gesendet wird nur noch die Zusammenfassung + die letzten 6 Turns. Nichts geht verloren.
3. **Selective Loading** (`rag_mode`):
   * `auto` – ganzes Material, solange es ins Budget passt, sonst automatisch nur die relevanten Abschnitte
     (BM25-artiges Keyword-Ranking, offline, kein Embedding-Modell nötig).
   * `full` – immer alles (hartes Truncating am Budgetlimit).
   * `off` – nur Chat, kein Material.

Für beliebig lange Dokumente gibt es zusätzlich die **Map-Reduce-Zusammenfassung**
(Tab *Material* → *Vorschau / Aktionen*): chunkweise verdichten, dann zu einem Dokument verschmelzen.
Das umgeht die Kontextgrenze vollständig.

## Struktur

```
app.py                     Streamlit-UI (Sidebar, Chat, Projekt, Material, Kontext)
run.sh / setup.sh          Start / einmalige Installation
core/config.py             Endpunkte, Pfade, Budget-Schwellen, Personas
core/ollama_client.py      /api/chat (streaming), /api/tags, /api/show, /api/embed
core/project_store.py      Projekte, Zuordnung von Sessions, Projekt-Session
core/session_store.py      Session & Message als Dataclass, JSON pro Session, Digest-Cache
core/library.py            Material-Bibliothek + Chunking mit Ankern
core/context_manager.py    Budget, Retrieval, Payload-Bau, Kompression, Lektions-Digests
core/study_tools.py        Prompt-Vorlagen (Lektion + Semester), Map-Reduce-Zusammenfassung
core/ingest/documents.py   PDF/PPTX/DOCX/HTML/VTT-Extraktion
core/ingest/transcribe.py  faster-whisper (CPU/int8, VAD-Filter)
scripts/selftest.py        End-to-End-Test ohne UI
scripts/uitest.py          Render-Test aller Ansichten
data/                      projects.json, Sessions, Materialien, Uploads (gitignored)
```

## Konfiguration (Umgebungsvariablen)

| Variable | Default | Zweck |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama-Adresse |
| `STUDYHUB_DATA` | `./data` | Ablageort für Sessions/Materialien |
| `WHISPER_MODEL` | `small` | `tiny`…`large-v3` |
| `WHISPER_LANGUAGE` | auto | z. B. `de` erzwingen |

## Bekannte Grenzen

* **Gescannte PDFs** enthalten keinen Text → vorher OCR: `ocrmypdf in.pdf out.pdf`.
* **Whisper auf CPU** braucht ca. 0.3–1× Echtzeit (`small`); `medium`/`large-v3` deutlich länger.
  Das Modell wird pro Prozess einmal geladen und gecached.
* **Grosses `num_ctx`** kostet RAM: `gemma4:26b` kann 262k Tokens, praktikabel sind 8k–32k.
* Transkription und Analyse sind bewusst getrennte Schritte – ein LLM kann kein Audio verarbeiten.
