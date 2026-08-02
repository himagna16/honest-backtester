"""Audit findings: severity + plain-English message + evidence.

Severity semantics (Section 3): FATAL = the headline number cannot be
trusted; WARNING = the number is real but weaker than it looks; NOTE =
disclosed context; PASS = the check ran and found nothing (shown collapsed —
the audit is never silent, even when clean).
"""
from __future__ import annotations

from dataclasses import dataclass, field

SEVERITY_ORDER = {"fatal": 0, "warning": 1, "note": 2, "pass": 3}


@dataclass
class Finding:
    check: str            # machine id, e.g. "selection_bias"
    severity: str         # fatal | warning | note | pass
    name: str             # short human title, e.g. "Selection bias detected"
    message: str          # what was found + why it matters, plain English
    evidence: dict = field(default_factory=dict)
    action: str | None = None   # what the tool did / offers (e.g. re-run)
    explainer: str | None = None  # glossary anchor id

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "name": self.name,
            "message": self.message,
            "evidence": self.evidence,
            "action": self.action,
            "explainer": self.explainer,
        }


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.check))
