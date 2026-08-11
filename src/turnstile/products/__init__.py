"""L2 products — agent specs plus product discipline (hooks, middleware).

Dependency rule: imports `turnstile.kernel` and `turnstile.capabilities`.
Product specs are SIBLINGS: no product ever imports another product — they are
composed only by the application root (`turnstile.root`).
"""
