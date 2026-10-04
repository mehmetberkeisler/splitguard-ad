"""The audit taxonomy: what blocks a split, and what is merely worth knowing.

A split can be leakage-free and still be statistically undesirable, and
conflating the two makes an audit useless in both directions: a reader who sees
every finding as fatal cannot ship, and a reader who sees them all as advisory
ships a broken split. So findings carry one of four levels and only one of them
is a refusal.

    FAIL   the split is invalid. One condition reaches this level: a connected
           component with images on both sides of a partition boundary, which
           is leakage by construction.
    WARN   the split is valid and statistically awkward. Component-size or
           class-mix imbalance, which the largest-first allocation order can
           produce and which the manuscript reports as a known limitation.
    INFO   a descriptive diagnostic with no pass or fail reading.
    PASS   a requirement that was checked and satisfied.

The decision rule is deliberately blunt: GO unless something FAILs. Promoting a
WARN to a refusal would make the auditor reject splits that are correct for the
purpose it exists to serve.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["Level", "Finding", "AuditReport"]


class Level(str, Enum):
    FAIL = "FAIL"
    WARN = "WARN"
    INFO = "INFO"
    PASS = "PASS"


@dataclass(frozen=True)
class Finding:
    """One audit observation, at one level, with the evidence that produced it."""

    level: Level
    check: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.level.value:<5} {self.check}" + (f": {self.detail}" if self.detail else "")


@dataclass
class AuditReport:
    """A set of findings and the single decision that follows from them."""

    findings: list[Finding]

    def at(self, level: Level) -> list[Finding]:
        return [f for f in self.findings if f.level is level]

    @property
    def failures(self) -> list[Finding]:
        return self.at(Level.FAIL)

    @property
    def warnings(self) -> list[Finding]:
        return self.at(Level.WARN)

    @property
    def decision(self) -> str:
        """GO unless a FAIL is present. Warnings never block."""
        return "NO-GO" if self.failures else "GO"

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        counts = {lvl.value: len(self.at(lvl)) for lvl in Level}
        parts = ", ".join(f"{n} {name}" for name, n in counts.items() if n)
        return f"{self.decision} ({parts or 'no findings'})"

    def render(self) -> str:
        order = [Level.FAIL, Level.WARN, Level.INFO, Level.PASS]
        lines = [str(f) for lvl in order for f in self.at(lvl)]
        return "\n".join(lines + [f"\nDECISION: {self.summary()}"])
