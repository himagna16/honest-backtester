"""Lookahead guard (audit check #2) — an impossibility statement, not a test.

The engine applies the t → t+1 shift centrally and the DSL has no
"execute at signal price" primitive; the golden test suite proves it
(future-perturbation + API-level tests in test_golden.py). This line appears
on every report so the guarantee is visible, not assumed.
"""
from __future__ import annotations

from audit.findings import Finding


def lookahead_statement() -> Finding:
    return Finding(
        check="lookahead", severity="pass",
        name="Lookahead bias: impossible by construction",
        message=(
            "Signals at day t use closing data through t only; positions earn "
            "returns from t+1. The shift lives inside the engine — no spec can "
            "request a same-day fill, and the test suite proves that altering "
            "future prices cannot change past weights."
        ),
        explainer="lookahead-bias",
    )
