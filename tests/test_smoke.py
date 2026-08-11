"""Scaffold smoke test: the package and every layer import.

Exists so the pytest gate always collects at least one test — bare pytest exits
with code 5 on an empty suite, which would fake-fail the CI gate.
"""

import pytest


@pytest.mark.unit
def test_all_layers_import():
    import turnstile
    import turnstile.capabilities
    import turnstile.kernel
    import turnstile.products
    import turnstile.service

    assert turnstile.__doc__
