"""L1 capabilities — concrete implementations of the L0 ports.

Marshal real I/O (LLM API, KB/DB, network) into and out of L0 DTOs.
Dependency rule: imports `turnstile.kernel` only — never products or service.
"""
