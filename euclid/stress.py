"""Stress testing: look for bugs in a strategy by breaking its broker.

`euclid verify` audits one backtest. This module asks a different question:
what does the strategy do when something goes wrong? It re-runs the strategy
many times, each time injecting one broker fault, and checks three things:

  solvency   equity stays above zero
  limits     the declared portfolio limits still hold
  recovery   positions come back to the path the strategy intended

Every result is `tested`: it covers the faults that were tried, not all
possible ones. A pass does not show the strategy has no bugs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .loader import Backtest
from .lookahead import StrategyHandle
from .model import Check, Status, Strength

METHOD = "fault_injection"
DEFAULT_SCENARIOS = 12
MATCH_TOLERANCE = 0.10  # positions within 10% of each other count as the same
SETTLED_SHARE = 0.05  # recovered = at most 5% of the remaining bars still differ
MIN_LATER_ORDERS = 3  # need this many later orders to judge recovery
FAULT_TEXT = {"reject": "refusing", "partial": "part-filling"}


@dataclass
class Scenario:
    kind: str
    order: int
    symbol: str = ""
    date: str = ""
    took_effect: bool = False
    insolvent_on: str | None = None
    limits_broken: dict[str, str] = field(default_factory=dict)
    recovery: str = "not_assessed"  # recovered | not_recovered | too_late_to_judge | not_assessed
    bars_to_recover: int | None = None
    final_positions: dict[str, float] = field(default_factory=dict)
    intended_positions: dict[str, float] = field(default_factory=dict)

    def label(self) -> str:
        return f"{FAULT_TEXT[self.kind]} order #{self.order} ({self.symbol}, {self.date})"


def plan_faults(order_count: int, scenarios: int) -> list[dict[str, Any]]:
    """Spread the faults over the first 80% of orders; two refusals per part-fill."""
    usable = max(int(order_count * 0.8), min(order_count, 1))
    if usable == 0:
        return []
    picks = sorted(set(np.linspace(0, usable - 1, min(scenarios, usable)).astype(int).tolist()))
    return [{"kind": "partial" if i % 3 == 2 else "reject", "order": k} for i, k in enumerate(picks)]


def limit_breaches(constraints: dict, run) -> dict[str, int]:
    """First bar on which each declared limit is broken in a run."""
    values = run.positions * run.closes
    gross, net = np.abs(values).sum(axis=1), np.abs(values.sum(axis=1))
    slack = 1e-9 * np.maximum(np.abs(run.equity), 1.0)
    tests = {
        "max_gross_leverage": lambda v: gross > v * run.equity + slack,
        "max_net_exposure": lambda v: net > v * run.equity + slack,
        "max_position_weight": lambda v: (np.abs(values) > (v * run.equity + slack)[:, None]).any(axis=1),
        "min_cash": lambda v: run.cash < v - slack,
        "long_only": lambda v: (run.positions < 0).any(axis=1) if v is True else np.zeros(len(gross), bool),
    }
    out = {}
    for key, value in constraints.items():
        if key in tests:
            broken = np.flatnonzero(tests[key](value))
            if len(broken):
                out[key] = int(broken[0])
    return out


def _differs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per bar: do two position paths differ by more than the tolerance?"""
    scale = np.maximum(np.abs(a), np.abs(b))
    return (np.abs(a - b) > MATCH_TOLERANCE * scale + 1e-9).any(axis=1)


def assess(fault: dict, run, baseline, handle: StrategyHandle, constraints: dict,
           baseline_breaches: dict[str, int]) -> Scenario:
    n = fault["order"]
    bar, symbol, _ = baseline.orders[n] if n < len(baseline.orders) else (0, "", 0.0)
    s = Scenario(kind=fault["kind"], order=n, symbol=symbol, date=handle.dates[bar])
    if run.fault_bar is None:
        return s  # the fault never fired (for example the order was never reached)
    s.took_effect, s.date = True, handle.dates[run.fault_bar]

    wiped = np.flatnonzero(run.equity <= 0)
    if len(wiped):
        s.insolvent_on = handle.dates[wiped[0]]
    for key, first in limit_breaches(constraints, run).items():
        if key not in baseline_breaches:
            s.limits_broken[key] = handle.dates[first]

    start = run.fault_bar
    later_orders = int(np.count_nonzero(baseline.decisions[start + 1:].any(axis=1)))
    differs = _differs(run.positions, baseline.positions)[start:]
    if not differs.any():
        s.recovery, s.bars_to_recover = "recovered", 0
    elif later_orders < MIN_LATER_ORDERS:
        s.recovery = "too_late_to_judge"
    else:
        # Share of the remaining bars that still differ, from each bar to the end.
        tail_share = np.cumsum(differs[::-1])[::-1] / np.arange(len(differs), 0, -1)
        settled = np.flatnonzero((tail_share <= SETTLED_SHARE) & ~differs)
        if len(settled):
            s.recovery, s.bars_to_recover = "recovered", int(settled[0])
        else:
            s.recovery = "not_recovered"
            s.final_positions = dict(zip(handle.symbols, run.positions[-1].tolist()))
            s.intended_positions = dict(zip(handle.symbols, baseline.positions[-1].tolist()))
    return s


def _summarise(name: str, claim: str, scenarios: list[Scenario], broken: list[Scenario],
               describe, extra: dict[str, Any] | None = None) -> Check:
    live = [s for s in scenarios if s.took_effect]
    evidence = {"scenarios_run": len(live), "scenarios_broken": len(broken), **(extra or {})}
    if not live:
        return Check(name, claim, METHOD, Status.UNVERIFIED, Strength.NONE,
                     "No injected fault took effect, so nothing was tested", evidence)
    if not broken:
        return Check(name, claim, METHOD, Status.VERIFIED, Strength.TESTED,
                     f"Held in all {len(live)} fault scenarios tried", evidence)
    first = min(broken, key=lambda s: s.order)
    evidence["first_failure"] = asdict(first)
    return Check(name, claim, METHOD, Status.FAILED, Strength.NONE,
                 f"Broken in {len(broken)} of {len(live)} scenarios; first: {describe(first)}",
                 evidence)


def run_stress(bt: Backtest, handle: StrategyHandle | Check,
               scenarios: int = DEFAULT_SCENARIOS) -> tuple[list[Check], list[Scenario]]:
    gap_claim = "The strategy stays sound when its broker misbehaves"
    if isinstance(handle, Check) or handle.runner is None:
        return [Check("stress", gap_claim, METHOD, Status.UNVERIFIED, Strength.NONE,
                      "Stress testing needs a strategy that runs in a supported engine "
                      "(backtrader); none was supplied", {})], []
    try:
        baseline = handle.runner(handle.data)
        faults = plan_faults(baseline.order_count, scenarios)
        constraints = bt.config.get("constraints") or {}
        baseline_breaches = limit_breaches(constraints, baseline)
        results = [assess(f, handle.runner(handle.data, fault=f), baseline, handle,
                          constraints, baseline_breaches) for f in faults]
    except Exception as exc:  # noqa: BLE001 - fail safe: record, don't hide
        return [Check("stress", gap_claim, METHOD, Status.ERROR, Strength.NONE,
                      f"A stress run failed: {type(exc).__name__}: {exc}", {})], []

    if not faults:
        return [Check("stress", gap_claim, METHOD, Status.UNVERIFIED, Strength.NONE,
                      "The strategy placed no orders, so there was nothing to break", {})], []

    one_fault = "when one order is refused or part-filled"
    checks = []

    if (baseline.equity <= 0).any():
        checks.append(Check("stress_solvency", f"Equity stays above zero {one_fault}", METHOD,
                            Status.UNVERIFIED, Strength.NONE,
                            "The unaltered run already loses all its equity", {}))
    else:
        checks.append(_summarise(
            "stress_solvency", f"Equity stays above zero {one_fault}", results,
            [s for s in results if s.insolvent_on],
            lambda s: f"{s.label()} leaves equity at or below zero by {s.insolvent_on}"))

    tested_limits = [k for k in constraints if k not in baseline_breaches]
    if not tested_limits:
        checks.append(Check("stress_limits", f"Declared limits still hold {one_fault}", METHOD,
                            Status.UNVERIFIED, Strength.NONE,
                            "No declared limit holds in the unaltered run, so none was stress-tested",
                            {}))
    else:
        checks.append(_summarise(
            "stress_limits", f"Declared limits still hold {one_fault}", results,
            [s for s in results if s.limits_broken],
            lambda s: f"{s.label()} breaks " + ", ".join(
                f"{k} on {d}" for k, d in s.limits_broken.items()),
            {"limits_tested": tested_limits}))

    judged = [s for s in results if s.recovery in ("recovered", "not_recovered")]
    lost = [s for s in judged if s.recovery == "not_recovered"]

    def describe_lost(s: Scenario) -> str:
        def show(p: dict) -> str:
            return ", ".join(f"{k} {v:g}" for k, v in p.items())
        return (f"after {s.label()} the positions never return to the intended path; "
                f"final positions {show(s.final_positions)} against intended "
                f"{show(s.intended_positions)}")

    recovery = _summarise(
        "stress_recovery", "After one order is refused or part-filled, positions return to "
        "the path the strategy intended", judged, lost, describe_lost,
        {"too_late_to_judge": sum(s.recovery == "too_late_to_judge" for s in results)})
    checks.append(recovery)
    return checks, results
