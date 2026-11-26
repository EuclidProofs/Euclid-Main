"""Core objects: Check, Verdict, and the fail-safe aggregation rules.

The rules in `aggregate` are the heart of the trust argument:
  1. Any failed check makes the verdict failed.
  2. An error or an unverified check can never roll up to verified.
  3. The verdict's strength is the weakest strength among its verified checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNVERIFIED = "unverified"  # no applicable method, or missing input
    ERROR = "error"  # the check itself crashed or timed out


class VerdictStatus(str, Enum):
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    FAILED = "failed"


class Strength(str, Enum):
    """How strongly a verified result is known. Ordered strongest first."""

    PROVEN = "proven"
    SOLVER_CHECKED = "solver_checked"
    RECOMPUTED = "recomputed"
    TESTED = "tested"
    NONE = "none"


_STRENGTH_ORDER = [
    Strength.PROVEN,
    Strength.SOLVER_CHECKED,
    Strength.RECOMPUTED,
    Strength.TESTED,
    Strength.NONE,
]


def weakest(strengths: list[Strength]) -> Strength:
    if not strengths:
        return Strength.NONE
    return max(strengths, key=_STRENGTH_ORDER.index)


@dataclass
class Check:
    """One method applied to one claim."""

    name: str
    claim: str
    method: str
    status: Status
    strength: Strength
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["strength"] = self.strength.value
        return d


@dataclass
class Verdict:
    status: VerdictStatus
    strength: Strength
    scope: str
    summary: dict[str, int]
    checks: list[Check]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "strength": self.strength.value,
            "scope": self.scope,
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
        }


def aggregate(checks: list[Check], scope: str) -> Verdict:
    """Combine checks into a verdict under the fail-safe rules."""
    summary = {s.value: 0 for s in Status}
    for c in checks:
        summary[c.status.value] += 1

    if summary[Status.FAILED.value] > 0:
        status = VerdictStatus.FAILED
    elif not checks or summary[Status.ERROR.value] or summary[Status.UNVERIFIED.value]:
        # Nothing checked, or a gap somewhere: never report this as verified.
        status = VerdictStatus.PARTIALLY_VERIFIED
    else:
        status = VerdictStatus.VERIFIED

    verified = [c.strength for c in checks if c.status is Status.VERIFIED]
    return Verdict(
        status=status,
        strength=weakest(verified),
        scope=scope,
        summary=summary,
        checks=checks,
    )
