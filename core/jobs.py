"""Hintergrund-Jobs (Transkription, Zusammenfassungen, Digest-Refresh).

Warum ueberhaupt: Eine 90-Minuten-Vorlesung zu transkribieren dauert 10-30 Minuten.
Streamlit fuehrt Callbacks synchron aus – ohne Jobs waere die gesamte UI so lange
blockiert. Hier laufen die Aufgaben in Worker-Threads, die UI pollt nur den Status.

Designentscheidungen:
* Ein einziger Job-Manager pro Prozess (`get_manager()`), gehalten via
  `st.cache_resource` in der UI – ueberlebt damit Streamlit-Reruns.
* Status liegt zusaetzlich als JSON auf Platte: so ist nach einem App-Neustart
  sichtbar, was lief (und was abgebrochen wurde).
* Threads statt Prozesse: Whisper/ctranslate2 und `requests` geben die GIL waehrend
  der Rechen- bzw. Wartezeit frei, echte Parallelitaet ist also gegeben – ohne dass
  das Whisper-Modell (bis 3 GB RAM) mehrfach geladen werden muss.
"""
from __future__ import annotations

import json
import inspect
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from queue import Empty, Queue
from typing import Any, Callable

from . import config

PENDING, RUNNING, DONE, FAILED, CANCELLED = "pending", "running", "done", "failed", "cancelled"
ACTIVE_STATES = {PENDING, RUNNING}


@dataclass
class Job:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    kind: str = "task"              # transcribe | summarize | digests
    label: str = ""                 # was die UI anzeigt
    session_id: str = ""            # Herkunft, fuer "zur Session springen"
    session_title: str = ""
    status: str = PENDING
    progress: float = 0.0
    message: str = "wartet…"
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0

    # ---------- Ableitungen fuer die UI ----------
    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATES

    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        return (self.finished or time.time()) - self.started

    def eta(self) -> float | None:
        if self.status != RUNNING or self.progress <= 0.02:
            return None
        return self.elapsed() * (1 - self.progress) / self.progress

    def icon(self) -> str:
        return {PENDING: "⏳", RUNNING: "⚙️", DONE: "✅",
                FAILED: "❌", CANCELLED: "🚫"}.get(self.status, "•")


def _fmt(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


class CancelledError(RuntimeError):
    """Wird im Worker geworfen, wenn der Job abgebrochen wurde."""


class JobManager:
    """Schlanke Thread-Pool-Warteschlange mit Fortschritt und Abbruch."""

    def __init__(self, workers: int | None = None) -> None:
        self._queue: Queue[str] = Queue()
        self._jobs: dict[str, Job] = {}
        self._funcs: dict[str, tuple[Callable, dict]] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        self._n = workers or config.MAX_PARALLEL_JOBS
        self._load_persisted()

    # ---------------- oeffentliche API ----------------
    def submit(self, kind: str, label: str, func: Callable, *,
               session_id: str = "", session_title: str = "", **kwargs: Any) -> Job:
        """`func(progress=..., should_cancel=...)` wird im Hintergrund ausgefuehrt.

        `session_id` dient der UI als Herkunftsangabe und wird zusaetzlich an die
        Aufgabe durchgereicht, falls deren Signatur es vorsieht – so muss der
        Aufrufer es nicht doppelt angeben.
        """
        job = Job(kind=kind, label=label, session_id=session_id, session_title=session_title)
        try:
            params = inspect.signature(func).parameters
            if "session_id" in params and "session_id" not in kwargs:
                kwargs["session_id"] = session_id
        except (TypeError, ValueError):
            pass
        with self._lock:
            self._jobs[job.id] = job
            self._funcs[job.id] = (func, kwargs)
            self._cancel[job.id] = threading.Event()
        self._persist(job)
        self._queue.put(job.id)
        self._ensure_workers()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, session_id: str = "", include_done: bool = True) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if session_id:
            jobs = [j for j in jobs if j.session_id == session_id]
        if not include_done:
            jobs = [j for j in jobs if j.active]
        return sorted(jobs, key=lambda j: (not j.active, -j.created))

    def active(self) -> list[Job]:
        return [j for j in self.list() if j.active]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            event = self._cancel.get(job_id)
            if job is None or not job.active:
                return False
            if event:
                event.set()
            if job.status == PENDING:       # noch nicht gestartet -> sofort beenden
                job.status = CANCELLED
                job.message = "abgebrochen"
                job.finished = time.time()
        self._persist(job)
        return True

    def clear_finished(self) -> int:
        with self._lock:
            done = [j.id for j in self._jobs.values() if not j.active]
            for jid in done:
                self._jobs.pop(jid, None)
                self._funcs.pop(jid, None)
                self._cancel.pop(jid, None)
                (config.JOB_DIR / f"{jid}.json").unlink(missing_ok=True)
        return len(done)

    # ---------------- intern ----------------
    def _ensure_workers(self) -> None:
        with self._lock:
            self._workers = [w for w in self._workers if w.is_alive()]
            missing = self._n - len(self._workers)
        for _ in range(max(missing, 0)):
            worker = threading.Thread(target=self._loop, daemon=True, name="studyhub-job")
            worker.start()
            with self._lock:
                self._workers.append(worker)

    def _loop(self) -> None:
        while True:
            try:
                job_id = self._queue.get(timeout=300)
            except Empty:
                return                      # Leerlauf -> Thread beenden
            self._run(job_id)
            self._queue.task_done()

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            entry = self._funcs.get(job_id)
            cancel = self._cancel.get(job_id)
        if job is None or entry is None or job.status == CANCELLED:
            return

        func, kwargs = entry
        job.status = RUNNING
        job.started = time.time()
        job.message = "startet…"
        self._persist(job)

        def report(fraction: float, message: str = "") -> None:
            if cancel and cancel.is_set():
                raise CancelledError()
            job.progress = max(0.0, min(float(fraction), 1.0))
            if message:
                job.message = message
            self._persist(job, throttle=True)

        def should_cancel() -> bool:
            return bool(cancel and cancel.is_set())

        try:
            result = func(progress=report, should_cancel=should_cancel, **kwargs) or {}
            if cancel and cancel.is_set():
                # Die Aufgabe hat den Abbruch selbst behandelt und ist regulaer
                # zurueckgekehrt – trotzdem ist das Ergebnis unvollstaendig.
                job.status = CANCELLED
                job.message = "abgebrochen"
            else:
                job.result = result
                job.status = DONE
                job.progress = 1.0
                job.message = result.get("message", "fertig")
        except CancelledError:
            job.status = CANCELLED
            job.message = "abgebrochen"
        except Exception as exc:
            job.status = FAILED
            job.error = str(exc)
            job.message = str(exc)[:200]
            job.result = {"traceback": traceback.format_exc()[-1500:]}
        finally:
            job.finished = time.time()
            self._persist(job)

    # ---------------- Persistenz ----------------
    _last_write: dict[str, float] = {}

    def _persist(self, job: Job, throttle: bool = False) -> None:
        now = time.time()
        if throttle and now - self._last_write.get(job.id, 0) < 1.0:
            return                          # max. 1 Schreibvorgang pro Sekunde
        self._last_write[job.id] = now
        try:
            (config.JOB_DIR / f"{job.id}.json").write_text(
                json.dumps(asdict(job), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass                            # Persistenz ist Komfort, kein Muss

    def _load_persisted(self) -> None:
        """Alte Jobs einlesen; beim Neustart laufende gelten als abgebrochen."""
        cutoff = time.time() - config.JOB_RETENTION_H * 3600
        known = {f for f in Job.__dataclass_fields__}
        for path in config.JOB_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                path.unlink(missing_ok=True)
                continue
            if data.get("created", 0) < cutoff:
                path.unlink(missing_ok=True)
                continue
            job = Job(**{k: v for k, v in data.items() if k in known})
            if job.active:                  # der Thread von damals existiert nicht mehr
                job.status = CANCELLED
                job.message = "durch App-Neustart unterbrochen"
                job.finished = time.time()
            self._jobs[job.id] = job


_MANAGER: JobManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_manager() -> JobManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = JobManager()
        return _MANAGER


def summary_line(jobs: list[Job]) -> str:
    """Kurzfassung fuer die Sidebar, z. B. '⚙️ 1 laufend · ✅ 3 fertig'."""
    running = sum(1 for j in jobs if j.status == RUNNING)
    pending = sum(1 for j in jobs if j.status == PENDING)
    done = sum(1 for j in jobs if j.status == DONE)
    failed = sum(1 for j in jobs if j.status == FAILED)
    parts = []
    if running:
        parts.append(f"⚙️ {running} laufend")
    if pending:
        parts.append(f"⏳ {pending} wartend")
    if done:
        parts.append(f"✅ {done} fertig")
    if failed:
        parts.append(f"❌ {failed} fehlgeschlagen")
    return " · ".join(parts) or "keine Jobs"


__all__ = ["Job", "JobManager", "CancelledError", "get_manager", "summary_line",
           "PENDING", "RUNNING", "DONE", "FAILED", "CANCELLED", "_fmt"]




