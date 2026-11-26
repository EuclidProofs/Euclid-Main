"""Stress testing: look for bugs in a strategy by breaking its broker.

`euclid verify` audits one backtest. This module asks a different question:
what does the strategy do when something goes wrong? It re-runs the strategy
many times, each time injecting one fault, and checks four things:

  completes  the strategy runs to the end
  solvency   equity stays above zero
  limits     the declared portfolio limits still hold
  recovery   positions come back to the path the strategy intended

Faults: an order refused, two orders refused in a row, an order part-filled,
a price gap against an order, and a bar missing for one instrument.

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
DEFAULT_GAP = 0.15  # size of an injected price gap
# Order faults leave the market data alone, so the normal run is a fair reference
# for where positions should be. Data faults change what the strategy sees, so
# its positions may rightly differ; recovery is not judged for them.
ORDER_FAULTS = ("reject", "double", "partial")
KIND_CYCLE = ("reject", "reject", "partial", "double", "gap", "missing")


@dataclass
class Scenario:
    kind: str
    order: int
    symbol: str = ""
    date: str = ""
    description: str = ""
    took_effect: bool = False
    crashed: str | None = None
    insolvent_on: str | None = None
    limits_broken: dict[str, str] = field(default_factory=dict)
    recovery: str = "not_assessed"  # recovered | not_recovered | too_late_to_judge | not_assessed
    bars_to_recover: int | None = None
    final_positions: dict[str, float] = field(default_factory=dict)
    intended_positions: dict[str, float] = field(default_factory=dict)

    def label(self) -> str:
        return self.description


def plan_faults(orders: list, n_bars: int, scenarios: int,
                gap: float = DEFAULT_GAP) -> list[dict[str, Any]]:
    """Choose the faults to inject, spread over the first 80% of the strategy's orders.

    `orders` is the normal run's order list: (bar, symbol, size) per order.
    """
    usable = max(int(len(orders) * 0.8), min(len(orders), 1))
    if usable == 0:
        return []
    picks = sorted(set(np.linspace(0, usable - 1, min(scenarios, usable)).astype(int).tolist()))
    faults = []
    for i, k in enumerate(picks):
        kind = KIND_CYCLE[i % len(KIND_CYCLE)]
        bar, symbol, size = orders[k]
        fill_bar = bar + 1  # a market order fills on the bar after it is placed
        if kind in ("gap", "missing") and fill_bar >= n_bars:
            kind = "reject"
        if kind == "reject":
            faults.append({"kind": "reject", "orders": [k], "order": k, "label": "reject"})
        elif kind == "double":
            faults.append({"kind": "reject", "orders": [k, k + 1], "order": k, "label": "double"})
        elif kind == "partial":
            faults.append({"kind": "partial", "order": k, "label": "partial"})
        elif kind == "gap":
            # Against the order: prices jump up before a buy fills, down before a sell.
            faults.append({"kind": "gap", "bar": fill_bar, "symbol": symbol, "order": k,
                           "size": gap if size > 0 else -gap, "label": "gap"})
        else:
            faults.append({"kind": "missing", "bar": fill_bar, "symbol": symbol, "order": k,
                           "label": "missing"})
    return faults


def _describe(fault: dict, symbol: str, date: str) -> str:
    kind, k = fault["label"], fault["order"]
    if kind == "reject":
        return f"refusing order #{k} ({symbol}, {date})"
    if kind == "double":
        return f"refusing orders #{k} and #{k + 1} ({date})"
    if kind == "partial":
        return f"part-filling order #{k} ({symbol}, {date})"
    if kind == "gap":
        return f"a {fault['size']:+.0%} price gap in {symbol} on {date}"
    return f"a missing {symbol} bar on {date}"


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
    bar, symbol, _ = baseline.orders[n]
    s = Scenario(kind=fault["label"], order=n, symbol=fault.get("symbol", symbol),
                 date=handle.dates[bar])
    if run.fault_bar is None:
        s.description = _describe(fault, s.symbol, s.date)
        return s  # the fault never fired (for example the order was never reached)
    s.took_effect, s.date = True, handle.dates[run.fault_bar]
    s.description = _describe(fault, s.symbol, s.date)

    wiped = np.flatnonzero(run.equity <= 0)
    if len(wiped):
        s.insolvent_on = handle.dates[wiped[0]]
    for key, first in limit_breaches(constraints, run).items():
        if key not in baseline_breaches:
            s.limits_broken[key] = handle.dates[first]

    if s.kind not in ORDER_FAULTS:
        return s
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


def _run_scenario(fault: dict, baseline, handle: StrategyHandle, constraints: dict,
                  baseline_breaches: dict[str, int]) -> Scenario:
    engine_fault = {k: v for k, v in fault.items() if k != "label"}
    try:
        run = handle.runner(handle.data, fault=engine_fault)
    except Exception as exc:  # noqa: BLE001 - a crash under a fault is itself a finding
        bar, symbol, _ = baseline.orders[fault["order"]]
        symbol = fault.get("symbol", symbol)
        date = handle.dates[min(fault.get("bar", bar), len(handle.dates) - 1)]
        return Scenario(kind=fault["label"], order=fault["order"], symbol=symbol, date=date,
                        description=_describe(fault, symbol, date), took_effect=True,
                        crashed=f"{type(exc).__name__}: {exc}"[:300])
    return assess(fault, run, baseline, handle, constraints, baseline_breaches)


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
               scenarios: int = DEFAULT_SCENARIOS,
               gap: float = DEFAULT_GAP) -> tuple[list[Check], list[Scenario]]:
    gap_claim = "The strategy stays sound when its broker misbehaves"
    if isinstance(handle, Check) or handle.runner is None:
        return [Check("stress", gap_claim, METHOD, Status.UNVERIFIED, Strength.NONE,
                      "Stress testing needs a strategy that runs in a supported engine "
                      "(backtrader); none was supplied", {})], []
    try:
        baseline = handle.runner(handle.data)
    except Exception as exc:  # noqa: BLE001 - fail safe: record, don't hide
        return [Check("stress", gap_claim, METHOD, Status.ERROR, Strength.NONE,
                      f"The unaltered run failed: {type(exc).__name__}: {exc}", {})], []
    faults = plan_faults(baseline.orders, len(handle.dates), scenarios, gap)
    constraints = bt.config.get("constraints") or {}
    baseline_breaches = limit_breaches(constraints, baseline)
    results = [_run_scenario(f, baseline, handle, constraints, baseline_breaches) for f in faults]

    if not faults:
        return [Check("stress", gap_claim, METHOD, Status.UNVERIFIED, Strength.NONE,
                      "The strategy placed no orders, so there was nothing to break", {})], []

    one_fault = "under every injected fault"
    finished = [s for s in results if not s.crashed]
    checks = [_summarise(
        "stress_completes", "The strategy runs to the end under every injected fault", results,
        [s for s in results if s.crashed],
        lambda s: f"with {s.label()} the run stopped: {s.crashed}")]

    if (baseline.equity <= 0).any():
        checks.append(Check("stress_solvency", f"Equity stays above zero {one_fault}", METHOD,
                            Status.UNVERIFIED, Strength.NONE,
                            "The unaltered run already loses all its equity", {}))
    else:
        checks.append(_summarise(
            "stress_solvency", f"Equity stays above zero {one_fault}", finished,
            [s for s in finished if s.insolvent_on],
            lambda s: f"{s.label()} leaves equity at or below zero by {s.insolvent_on}"))

    tested_limits = [k for k in constraints if k not in baseline_breaches]
    if not tested_limits:
        checks.append(Check("stress_limits", f"Declared limits still hold {one_fault}", METHOD,
                            Status.UNVERIFIED, Strength.NONE,
                            "No declared limit holds in the unaltered run, so none was stress-tested",
                            {}))
    else:
        checks.append(_summarise(
            "stress_limits", f"Declared limits still hold {one_fault}", finished,
            [s for s in finished if s.limits_broken],
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
        "stress_recovery", "After an order is refused or part-filled, positions return to "
        "the path the strategy intended", judged, lost, describe_lost,
        {"too_late_to_judge": sum(s.recovery == "too_late_to_judge" for s in results)})
    checks.append(recovery)
    return checks, results
