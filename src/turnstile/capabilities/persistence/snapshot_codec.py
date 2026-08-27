"""SessionSnapshot <-> JSON — the durable wire shape of a conversation.

Explicit field-by-field (no pickle, no generic reflection): the snapshot is the
one thing that outlives a process, so its bytes must stay readable by a kernel
that has moved on. SNAPSHOT_VERSION rides inside; the reader (engine resume)
checks it BEFORE interpreting messages — this codec only transports.
"""

import json
from typing import Any

from turnstile.kernel.dtos import (
    ImageContent,
    Message,
    MessageMeta,
    ReasoningBlock,
    Role,
    SessionSnapshot,
    TokenUsage,
    ToolCall,
)


def dumps(snapshot: SessionSnapshot) -> str:
    return json.dumps(
        {
            "version": snapshot.version,
            "cache_epoch": snapshot.cache_epoch,
            "turn_counter": snapshot.turn_counter,
            "request_counter": snapshot.request_counter,
            "messages": [_message(m) for m in snapshot.messages],
        },
        ensure_ascii=False,
    )


def loads(raw: str | bytes) -> SessionSnapshot:
    d = json.loads(raw)
    return SessionSnapshot(
        version=int(d["version"]),
        messages=[_message_from(m) for m in d["messages"]],
        cache_epoch=int(d.get("cache_epoch", 0)),
        turn_counter=int(d.get("turn_counter", 0)),
        request_counter=int(d.get("request_counter", 0)),
    )


def _message(m: Message) -> dict[str, Any]:
    return {
        "role": m.role.value,
        "text": m.text,
        "tool_calls": [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in m.tool_calls
        ],
        "tool_call_id": m.tool_call_id,
        "is_error": m.is_error,
        "meta": _meta(m.meta) if m.meta is not None else None,
        "synthetic": m.synthetic,
        "internal_origin": m.internal_origin,
        "reasoning": m.reasoning,
        "reasoning_blocks": [
            {"text": b.text, "opaque": b.opaque, "provider": b.provider}
            for b in m.reasoning_blocks
        ],
        "images": [{"media_type": i.media_type, "data": i.data} for i in m.images],
    }


def _message_from(d: dict[str, Any]) -> Message:
    return Message(
        role=Role(d["role"]),
        text=d.get("text", ""),
        tool_calls=[ToolCall(c["id"], c["name"], c["arguments"]) for c in d.get("tool_calls", [])],
        tool_call_id=d.get("tool_call_id"),
        is_error=bool(d.get("is_error", False)),
        meta=_meta_from(d["meta"]) if d.get("meta") is not None else None,
        synthetic=bool(d.get("synthetic", False)),
        internal_origin=d.get("internal_origin"),
        reasoning=d.get("reasoning"),
        reasoning_blocks=[
            ReasoningBlock(b["text"], b.get("opaque"), b.get("provider"))
            for b in d.get("reasoning_blocks", [])
        ],
        images=[ImageContent(i["media_type"], i["data"]) for i in d.get("images", [])],
    )


def _meta(meta: MessageMeta) -> dict[str, Any]:
    return {
        "tokens": {
            "prompt": meta.tokens.prompt,
            "completion": meta.tokens.completion,
            "cached": meta.tokens.cached,
        },
        "elapsed_ms": meta.elapsed_ms,
        "reasoning_elapsed_ms": meta.reasoning_elapsed_ms,
        "ctx_window": meta.ctx_window,
        "used_tokens": meta.used_tokens,
        "utilization": meta.utilization,
        "round": meta.round,
        "turn_id": meta.turn_id,
        "request_id": meta.request_id,
        "provider_response_id": meta.provider_response_id,
        "provider_model": meta.provider_model,
        "session_id": meta.session_id,
        "finish_reason": meta.finish_reason,
    }


def _meta_from(d: dict[str, Any]) -> MessageMeta:
    t = d.get("tokens") or {}
    return MessageMeta(
        tokens=TokenUsage(
            prompt=int(t.get("prompt", 0)),
            completion=int(t.get("completion", 0)),
            cached=int(t.get("cached", 0)),
        ),
        elapsed_ms=int(d.get("elapsed_ms", 0)),
        reasoning_elapsed_ms=int(d.get("reasoning_elapsed_ms", 0)),
        ctx_window=int(d.get("ctx_window", 0)),
        used_tokens=int(d.get("used_tokens", 0)),
        utilization=float(d.get("utilization", 0.0)),
        round=int(d.get("round", 0)),
        turn_id=int(d.get("turn_id", 0)),
        request_id=int(d.get("request_id", 0)),
        provider_response_id=d.get("provider_response_id"),
        provider_model=d.get("provider_model"),
        session_id=d.get("session_id"),
        finish_reason=d.get("finish_reason", ""),
    )
