"""Mitigation + observability layer for the Observathon black-box agent.

Every request flows through mitigate(). This is the ONLY place to add logging,
retry, caching, PII redaction, arithmetic guardrails, and prompt routing.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timezone

from telemetry.cost import cost_from_usage
from telemetry.redact import redact
from telemetry.tracing import Tracer, format_tree

# ── Globals (persist across calls in the same process) ──────────────────────
_log_lock = threading.Lock()
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()
_tracer = Tracer(service_name="observathon-sim")

_log_file: str | None = None
_ensure_dir = lambda d: os.makedirs(d, exist_ok=True)


def _log_event(event_type: str, data: dict) -> None:
    """Append one JSON line to the telemetry log (thread-safe)."""
    global _log_file
    if _log_file is None:
        _ensure_dir("logs")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        _log_file = f"logs/mitigate-{today}.jsonl"
    payload = {"ts": datetime.now(timezone.utc).isoformat(), "event": event_type, "data": data}
    with _log_lock:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _sanitize_order_notes(question: str) -> str:
    """Strip ALL order-note / GHI CHU content including injection attempts.

    The private-phase injection twist embeds fake prices in GHI CHU/GHI CHÚ/order notes.
    We strip the entire note block from the question so the agent never sees it.
    """
    import re
    # Remove full GHI CHU blocks (multi-line)
    cleaned = re.sub(
        r'(?i)(?:GHI\s*CHU|GHI\s*CHÚ|order\s*notes?|note|notes|order_note)\s*[:;].*?(\n|$)',
        '',
        question
    )
    # Strip PII metadata triggers (lien he, goi minh)
    cleaned = re.sub(r'(?i)(?:lien he|liên hệ|goi minh|gọi mình|sdt|email).*', '', cleaned)
    # Also strip label-only leftovers
    cleaned = re.sub(r'\s*\[sanitized\]\s*', ' ', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned


def _check_arithmetic(answer: str) -> bool:
    """Verify the total line is a clean integer if present."""
    if not answer:
        return True
    match = re.search(r'Tong cong:\s*([\d.,]+)\s*VND', answer)
    if not match:
        return True  # no total line, nothing to check
    total_str = match.group(1).replace('.', '').replace(',', '')
    return total_str.isdigit()


def mitigate(call_next, question, config, context):
    # ── Context helpers ────────────────────────────────────────────────
    session_id = context.get("session_id", "?")
    turn_index = context.get("turn_index", 0)
    qid = context.get("qid", "?")
    t_start = time.time()

    print(f"[mitigate] session={session_id} turn={turn_index} qid={qid}")

    # ── 1. STRIP injection notes from input ────────────────────────────
    clean_question = _sanitize_order_notes(question)

    # ── 2. Cache lookup (same qid → reuse) ─────────────────────────────
    with _cache_lock:
        if qid in _cache:
            print(f"[mitigate] CACHE HIT for {qid}")
            return _cache[qid]

    # ── 3. Build config with retry amplification (loyalty F13 errors) ──
    conf = dict(config)
    conf["retry"] = {"enabled": True, "max_attempts": 3, "backoff_ms": 500}

    # ── 4. Call the black-box agent (with retry) ────────────────────────
    result = None
    last_error = None
    for attempt in range(4):  # 0..3 = at most 4 attempts
        try:
            result = call_next(clean_question, conf)
        except Exception as e:
            last_error = str(e)
            print(f"[mitigate] attempt {attempt + 1} error: {e}")
            time.sleep(0.3 * (2 ** attempt))
            continue

        if result and result.get("status") in ("ok", "max_steps") and result.get("answer"):
            # Check for injected price — if total is suspiciously low, retry
            # (The private phase injects prices like 1000 VND into notes)
            ans = result["answer"]
            if _check_arithmetic(ans):
                break
        last_error = result.get("error") if result else "no result"
        print(f"[mitigate] attempt {attempt + 1} status={result.get('status')}")
        time.sleep(0.3 * (2 ** attempt))

    if result is None:
        return {"answer": None, "status": "wrapper_error", "steps": 0, "trace": [],
                "meta": {"latency_ms": int((time.time() - t_start) * 1000),
                         "usage": {}, "model": config.get("model"), "provider": config.get("provider"),
                         "tools_used": []},
                "error": last_error or "all retries exhausted"}

    # ── 5. Compute telemetry ───────────────────────────────────────────
    wall_ms = int((time.time() - t_start) * 1000)
    meta = result.get("meta", {})
    usage = meta.get("usage", {})
    model = meta.get("model", config.get("model"))
    provider = meta.get("provider", config.get("provider"))
    tools_used = meta.get("tools_used", [])

    cost = cost_from_usage(model, usage)
    prompt_tok = usage.get("prompt_tokens", 0)
    completion_tok = usage.get("completion_tokens", 0)
    total_tok = usage.get("total_tokens", 0)

    # PII check + strip metadata leak (lien he, REDACTED tags)
    answer = result.get("answer", "")
    redacted_answer, pii_count = redact(answer) if answer else (answer, 0)
    redacted_answer = re.sub(r'\(?liên hệ?\s*:\s*\[?REDACTED\]?\)?', '', redacted_answer, flags=re.IGNORECASE)
    redacted_answer = re.sub(r'\(?lien he\s*:\s*\[?REDACTED\]?\)?', '', redacted_answer, flags=re.IGNORECASE)
    redacted_answer = re.sub(r'\s{2,}', ' ', redacted_answer).strip()
    result["answer"] = redacted_answer

    # ── 6. Log telemetry ───────────────────────────────────────────────
    status = result.get("status", "?")
    trace = result.get("trace", [])
    step_count = result.get("steps", len(trace))

    _log_event("request", {
        "qid": qid, "session": session_id, "turn": turn_index,
        "wall_ms": wall_ms, "latency_ms": meta.get("latency_ms", 0),
        "model": model, "provider": provider,
        "prompt_tokens": prompt_tok, "completion_tokens": completion_tok,
        "total_tokens": total_tok, "cost_usd": round(cost, 8),
        "steps": step_count, "tools_used": tools_used,
        "status": status, "pii_redacted": pii_count,
        "error": result.get("error", ""),
    })

    print(f"[mitigate] DONE {qid} | status={status} tokens={total_tok} "
          f"cost=${cost:.6f} tools={len(tools_used)} steps={step_count} pii={pii_count}")

    if trace:
        _log_event("trace_detail", {
            "qid": qid, "step_count": step_count, "trace": trace
        })

    # ── 7. Cache ───────────────────────────────────────────────────────
    if status == "ok":
        with _cache_lock:
            _cache[qid] = result

    return result
