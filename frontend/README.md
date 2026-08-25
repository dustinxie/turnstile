# turnstile frontend

The web UI (M6): a history panel + one conversation window over the
turnstile service's SSE API. Vite + React + TypeScript + Tailwind; tests with
vitest + Testing Library (+ msw for mocked SSE).

    npm install           # once
    npm run dev           # http://localhost:5173, proxies /v1 /health /sso -> 127.0.0.1:8000
    npm run check         # the gate: typecheck, lint, tests, build (= `make frontend-check`)

Talks to the service only through its public API (`/v1/conversations...`,
`/v1/files/{token}`, `/health`, `/sso`); the auth token, when auth is on, is a
Bearer JWT the app attaches itself (see architecture.md §2).
