"""OpenAI-compatible streaming provider (vLLM / LiteLLM / gateways).

The L1 adapter's whole job is MARSHALING: L0 Messages/ToolDefs out to the wire,
SSE chunks back into L0 StreamEvents. The kernel never sees HTTP; policy about
retries/timeouts lives in the kernel's resilience tiers — this adapter only
classifies failures into the structured ProviderError the tiers branch on.

Commit 1 scope: happy-path streaming (text, reasoning_content, tool-call
fragments, usage, ids, truncation, malformed chunks) + message/tool rendering.
Error taxonomy, ChatOptions mapping, and session affinity land next commit.
"""

import json
from collections.abc import AsyncIterator

import httpx

from turnstile.kernel.dtos import (
    ChatOptions,
    Done,
    Malformed,
    Message,
    ProviderError,
    Reasoning,
    ResponseId,
    ResponseModel,
    Role,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolCallEvent,
    ToolDef,
    UsageEvent,
)
from turnstile.kernel.ports import LlmProvider

_ROLE_WIRE = {
    Role.SYSTEM: "system",
    Role.USER: "user",
    Role.ASSISTANT: "assistant",
    Role.TOOL: "tool",
}


def render_messages(messages: list[Message]) -> list[dict]:
    """L0 Messages -> OpenAI chat format. Deterministic: identical input yields
    byte-identical output (the prefix-cache contract crosses this boundary)."""
    wire: list[dict] = []
    for m in messages:
        entry: dict = {"role": _ROLE_WIRE[m.role], "content": m.text}
        if m.role is Role.ASSISTANT and m.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in m.tool_calls
            ]
        if m.role is Role.TOOL:
            entry["tool_call_id"] = m.tool_call_id
        wire.append(entry)
    return wire


def render_tools(tools: list[ToolDef]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


class _ToolCallAssembler:
    """vLLM streams tool calls as indexed fragments (id/name usually in the
    first, arguments split across many). Forward each fragment live as a
    ToolCallDelta AND assemble the complete ToolCall per index for execution."""

    def __init__(self) -> None:
        self._by_index: dict[int, dict] = {}

    def feed(self, fragment: dict) -> ToolCallDelta:
        index = fragment.get("index", 0)
        slot = self._by_index.setdefault(index, {"id": "", "name": "", "arguments": ""})
        function = fragment.get("function") or {}
        if fragment.get("id"):
            slot["id"] = fragment["id"]
        if function.get("name"):
            slot["name"] += function["name"]
        if function.get("arguments"):
            slot["arguments"] += function["arguments"]
        return ToolCallDelta(
            index=index,
            id=fragment.get("id"),
            name=function.get("name"),
            arguments=function.get("arguments") or "",
        )

    def assembled(self) -> list[ToolCall]:
        return [
            ToolCall(id=slot["id"], name=slot["name"], arguments=slot["arguments"])
            for _, slot in sorted(self._by_index.items())
        ]


class OpenAICompatProvider(LlmProvider):
    """Streams /v1/chat/completions against any OpenAI-compatible backend."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        context_window: int = 0,
        client: httpx.AsyncClient | None = None,
        request_timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._context_window = context_window
        self._timeout = request_timeout
        # Injectable client: tests pass one with a MockTransport replaying
        # recorded SSE fixtures; production shares a connection pool.
        self._client = client or httpx.AsyncClient(verify=False, timeout=request_timeout)

    def model_name(self) -> str:
        return self._model

    def context_window(self) -> int:
        return self._context_window

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        options: ChatOptions,
    ) -> AsyncIterator[StreamEvent]:
        payload: dict = {
            "model": self._model,
            "messages": render_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = render_tools(tools)
        # (full ChatOptions mapping lands next commit)

        assembler = _ToolCallAssembler()
        truncated = False
        sent_response_id = False
        sent_response_model = False
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise ProviderError(
                        message=f"HTTP {response.status_code}: {body.strip()}",
                        http_status=response.status_code,
                    )
                try:
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue  # comments / keep-alives / event names
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except ValueError:
                            yield Malformed()  # dropped garbage is a SIGNAL, never content
                            continue
                        if not sent_response_id and chunk.get("id"):
                            sent_response_id = True
                            yield ResponseId(chunk["id"])
                        if not sent_response_model and chunk.get("model"):
                            sent_response_model = True
                            yield ResponseModel(chunk["model"])
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                            details = usage.get("prompt_tokens_details") or {}
                            cached = details.get("cached_tokens", 0)
                            yield UsageEvent(
                                TokenUsage(
                                    prompt=usage.get("prompt_tokens", 0),
                                    completion=usage.get("completion_tokens", 0),
                                    cached=cached or 0,
                                )
                            )
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            if delta.get("reasoning_content"):
                                yield Reasoning(delta["reasoning_content"])
                            if delta.get("content"):
                                yield TextDelta(delta["content"])
                            for fragment in delta.get("tool_calls") or []:
                                yield assembler.feed(fragment)
                            if choice.get("finish_reason") == "length":
                                truncated = True
                finally:
                    pass
        finally:
            pass
        # Complete tool calls are emitted AFTER the stream: execution needs the
        # whole call; the live fragments above were display-only.
        for call in assembler.assembled():
            yield ToolCallEvent(call)
        yield Done(truncated=truncated)
