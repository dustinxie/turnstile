"""L2 web driver — FastAPI + SSE transport over an assembled agent.

Dependency rule: imports `turnstile.root` and `turnstile.kernel` only — never
products or capabilities. Product objects are reached solely as fields of the
`AssembledAgent` bundle returned by `root.assemble()`.
"""
