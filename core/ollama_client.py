"""HTTP-Client fuer die lokale Ollama-API.

Verwendet ausschliesslich /api/chat (Rollen-basiert, streaming-faehig).
Alle Funktionen werfen `OllamaError` mit einer klartext-Fehlermeldung.
"""
from __future__ import annotations

import json
from typing import Any, Iterator

import requests

from . import config


class OllamaError(RuntimeError):
    """Fehler beim Sprechen mit dem lokalen Ollama-Server."""


def _post(url: str, payload: dict[str, Any], stream: bool = False) -> requests.Response:
    try:
        resp = requests.post(url, json=payload, stream=stream, timeout=config.REQUEST_TIMEOUT)
    except requests.exceptions.ConnectionError as exc:  # Ollama laeuft nicht
        raise OllamaError(
            f"Keine Verbindung zu Ollama unter {config.OLLAMA_HOST}. "
            "Laeuft der Dienst? Start: `ollama serve`"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError("Zeitueberschreitung. Modell evtl. zu gross fuer diese Hardware.") from exc
    if resp.status_code >= 400:
        raise OllamaError(f"Ollama HTTP {resp.status_code}: {resp.text[:400]}")
    return resp


def is_online() -> bool:
    try:
        requests.get(config.TAGS_ENDPOINT, timeout=3).raise_for_status()
        return True
    except Exception:
        return False


def list_models() -> list[dict[str, Any]]:
    """Alle lokal gepullten Modelle inkl. Metadaten (Groesse, Familie, Capabilities)."""
    try:
        resp = requests.get(config.TAGS_ENDPOINT, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        raise OllamaError(f"Modelliste nicht abrufbar: {exc}") from exc
    return sorted(resp.json().get("models", []), key=lambda m: m.get("name", ""))


def model_context_length(model: str) -> int | None:
    """Maximales Kontextfenster laut Modell-Metadaten (z. B. 262144 bei gemma4:26b)."""
    try:
        data = _post(config.SHOW_ENDPOINT, {"model": model}).json()
    except OllamaError:
        return None
    info = data.get("model_info") or {}
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    details = data.get("details") or {}
    return details.get("context_length")


def model_capabilities(model: str) -> list[str]:
    try:
        return _post(config.SHOW_ENDPOINT, {"model": model}).json().get("capabilities", [])
    except OllamaError:
        return []


def _options(num_ctx: int, temperature: float, **extra: Any) -> dict[str, Any]:
    opts = {"num_ctx": int(num_ctx), "temperature": float(temperature)}
    opts.update({k: v for k, v in extra.items() if v is not None})
    return opts


def chat_stream(
    model: str,
    messages: list[dict[str, str]],
    num_ctx: int = config.DEFAULT_NUM_CTX,
    temperature: float = 0.7,
    think: bool | None = None,
) -> Iterator[dict[str, Any]]:
    """Streamt Chat-Tokens.

    Yields dicts: {"type": "token", "text": ...} bzw. {"type": "done", "stats": {...}}.
    `messages` ist die vollstaendige Historie – Ollama ist zustandslos, das Gedaechtnis
    lebt ausschliesslich in diesem Array.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": _options(num_ctx, temperature),
    }
    if think is not None:
        payload["think"] = think

    resp = _post(config.CHAT_ENDPOINT, payload, stream=True)
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if "error" in chunk:
            raise OllamaError(chunk["error"])
        msg = chunk.get("message") or {}
        if msg.get("thinking"):
            yield {"type": "thinking", "text": msg["thinking"]}
        if msg.get("content"):
            yield {"type": "token", "text": msg["content"]}
        if chunk.get("done"):
            yield {
                "type": "done",
                "stats": {
                    "prompt_tokens": chunk.get("prompt_eval_count", 0),
                    "completion_tokens": chunk.get("eval_count", 0),
                    "total_duration_s": round(chunk.get("total_duration", 0) / 1e9, 2),
                    "done_reason": chunk.get("done_reason", ""),
                },
            }


def chat_once(
    model: str,
    messages: list[dict[str, str]],
    num_ctx: int = config.DEFAULT_NUM_CTX,
    temperature: float = 0.3,
) -> str:
    """Blockierender Einzel-Call – fuer Zusammenfassungen/Titel, nicht fuer die UI-Konversation."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": _options(num_ctx, temperature),
    }
    data = _post(config.CHAT_ENDPOINT, payload).json()
    return (data.get("message") or {}).get("content", "").strip()


def embed(model: str, inputs: list[str]) -> list[list[float]]:
    """Embeddings via /api/embed. Fehler werden nach oben gereicht (Fallback: Keyword-Suche)."""
    data = _post(config.EMBED_ENDPOINT, {"model": model, "input": inputs}).json()
    return data.get("embeddings", [])

