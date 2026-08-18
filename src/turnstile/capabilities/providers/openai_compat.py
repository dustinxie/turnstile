"""OpenAI-compatible streaming provider (vLLM / LiteLLM / gateways).

The L1 adapter's whole job is MARSHALING: L0 Messages/ToolDefs out to the wire,
SSE chunks back into L0 StreamEvents. The kernel never sees HTTP; policy about
retries/timeouts lives in the kernel's resilience tiers — this adapter only
classifies failures into the structured ProviderError the tiers branch on.

Failure classification is the taxonomy the kernel's resilience tiers branch
on: retryable (5xx / transport / 429 with Retry-After) vs terminal (auth/4xx),
structured codes for the overflow classifier, mid-stream transport failures as
ErrorEvents (never raw httpx exceptions across the port boundary).
"""

import json
from collections.abc import AsyncIterator

import httpx

from turnstile.kernel.dtos import (
    ChatOptions,
    Done,
    ErrorEvent,
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


# Statuses the KERNEL may retry (its tiers own policy; we only classify).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}


def _parse_error_body(body: str) -> tuple[str, str | None]:
    """Best-effort (message, code) from an error body: OpenAI's
    {"error": {"message", "code"|"type"}}, vLLM's flat {"message", "code"},
    or the raw text when it isn't JSON."""
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip(), None
    if not isinstance(data, dict):
        return body.strip(), None
    error = data.get("error")
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        return error.get("message") or body.strip(), str(code) if code else None
    if isinstance(error, str):
        return error, None
    code = data.get("code")
    return data.get("message") or body.strip(), str(code) if code is not None else None


def _open_error(status: int, headers: httpx.Headers, body: str) -> ProviderError:
    """A failed OPEN, classified: the kernel branches on retryable /
    http_status / code / retry_after_secs / is_context_overflow()."""
    message, code = _parse_error_body(body)
    retry_after: int | None = None
    raw_retry = headers.get("retry-after")
    if raw_retry and raw_retry.isdigit():
        retry_after = int(raw_retry)
    return ProviderError(
        message=f"HTTP {status}: {message}",
        retryable=status in _RETRYABLE_STATUS,
        http_status=status,
        code=code,
        retry_after_secs=retry_after,
    )


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
        self._session_id: str | None = None
        # Injectable client: tests pass one with a MockTransport replaying
        # recorded SSE fixtures; production shares a connection pool.
        self._client = client or httpx.AsyncClient(verify=False, timeout=request_timeout)

    def model_name(self) -> str:
        return self._model

    def context_window(self) -> int:
        return self._context_window

    def bind_session_id(self, session_id: str) -> None:
        """One-shot affinity binding: forwarded on every request so a gateway
        (LiteLLM) can pin the whole conversation to one upstream worker — the
        provider prefix cache is per worker, so sticky routing keeps it warm.
        A session's id never changes; a second bind is ignored."""
        if self._session_id is None:
            self._session_id = session_id

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._session_id:
            headers["x-turnstile-session-id"] = self._session_id
        return headers

    @staticmethod
    def _apply_options(payload: dict, options: ChatOptions) -> None:
        """Neutral knobs -> this vendor's wire params. None = no opinion (omit;
        the backend's defaults rule). rate_limit_retry_owner is a runtime
        sideband and NEVER rides the wire."""
        if options.max_tokens is not None:
            payload["max_tokens"] = options.max_tokens
        if options.temperature is not None:
            payload["temperature"] = options.temperature
        if options.reasoning_effort is not None:
            payload["reasoning_effort"] = options.reasoning_effort
        if options.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": options.enable_thinking}
        if options.tool_choice == "auto":
            return  # the neutral default: omit, let the model decide
        if options.tool_choice in ("required", "none"):
            payload["tool_choice"] = options.tool_choice
        else:  # a specific tool name
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": options.tool_choice},
            }

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
        self._apply_options(payload, options)

        assembler = _ToolCallAssembler()
        truncated = False
        sent_response_id = False
        sent_response_model = False
        # Failure-phase flags: with plain `async with`/`async for`, an httpx
        # error escaping to the outer except could be from OPEN (raise a
        # classified ProviderError), from the BODY (already reported as an
        # ErrorEvent below), or from CLOSE after the stream resolved (noise —
        # cleanup must never mask events already yielded). The flags say which.
        failed_mid_stream = False
        body_resolved = False
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise _open_error(response.status_code, response.headers, body)
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
                            yield Malformed()  # dropped garbage: SIGNAL, not content
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
                except httpx.HTTPError as read_error:
                    # Mid-stream transport death: an EVENT (the stream opened),
                    # never a raw httpx exception across the port boundary.
                    failed_mid_stream = True
                    yield ErrorEvent(
                        ProviderError(
                            message=f"stream failed mid-flight: {read_error}",
                            retryable=True,
                        )
                    )
                body_resolved = True
        except httpx.HTTPError as transport_error:
            if not (failed_mid_stream or body_resolved):
                # Connect refused / DNS / TLS / timeout before any response: a
                # failed OPEN the kernel's transient tier may retry.
                raise ProviderError(
                    message=f"transport error: {transport_error}",
                    retryable=True,
                ) from transport_error
            # else: close-time noise after the body already resolved/failed.
        if failed_mid_stream:
            return
        # Complete tool calls are emitted AFTER the stream: execution needs the
        # whole call; the live fragments above were display-only.
        for call in assembler.assembled():
            yield ToolCallEvent(call)
        yield Done(truncated=truncated)
