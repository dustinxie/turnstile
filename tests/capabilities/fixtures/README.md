# SSE fixtures

Real bytes recorded from ds4 (vLLM 0.25.0, OpenAI-compatible), via the
tokenless route `https://10.83.135.205/model-long/v1`, alias `model-fast`.
Recorded 2026-08-18; extra vLLM fields (`logprobs`, `token_ids`,
`system_fingerprint`, …) are kept verbatim — the adapter must ignore them.

| fixture | request that produced it |
|---|---|
| `text_stream.sse` | user: `Reply with exactly: hello turnstile`, temperature 0, max_tokens 20 |
| `tool_call_stream.sse` | user prompt asking for kb_search + web_search in parallel, both tools declared, temperature 0 |
| `truncated.sse` | user: `Write a long essay about firewalls.`, max_tokens 4 → `finish_reason: length` |

Re-capture command (adjust messages/tools per the table):

    curl -k -s https://10.83.135.205/model-long/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"model-fast","messages":[...],"temperature":0,"stream":true,
           "stream_options":{"include_usage":true}}' > <name>.sse

Tests pin exact ids/usage from these bytes — re-capturing means updating the
assertions in `test_openai_compat.py` to the new values.

NOT capturable from ds4 on demand, so covered by synthetic inline bodies in
`test_openai_compat.py` instead of fixture files: malformed garbage lines
(a gateway hiccup), `reasoning_content` (model-fast has no reasoning parser),
and `prompt_tokens_details.cached_tokens` (this vLLM build doesn't report it).
