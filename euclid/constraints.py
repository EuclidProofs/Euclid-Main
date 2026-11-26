"""Constraint engine: portfolio limits checked over the whole backtest with Z3.

For each declared constraint we ask the solver one question: "is there any
date on which this limit is broken?" The portfolio state on every date is
given to Z3 as exact rational numbers (no floating-point tolerance).

  unsat -> no such date exists: the limit holds on every date (verified)
  sat   -> the model names a violating date (failed, with the counterexample)
  other -> the solver could not decide: reported as an error, never a pass

Supported keys in config.json "constraints":
  max_gross_leverage   sum(|position value|) <= L * equity
  max_net_exposure     |sum(position value)| <= X * equity
  max_position_weight  |position value|      <= W * equity, for every symbol
  min_cash             cash >= C
  long_only            every position >= 0 (when true)
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable

import z3

from .loader import Backtest
from .model import Check, Status, Strength

METHOD = "smt_z3"
SOLVER_TIMEOUT_MS = 60_000
BLOCK_SIZE = 50


def _frac(x: Any) -> Fraction:
    """Exact value of a number as it was written in the input file."""
    return Fraction(str(float(x)))


@dataclass
class DayState:
    date: str
    cash: Fraction
    equity: Fraction
    positions: dict[str, Fraction]  # units held
    values: dict[str, Fraction]  # signed market value

    @property
    def gross(self) -> Fraction:
        return sum((abs(v) for v in self.values.values()), Fraction(0))

    @property
    def net(self) -> Fraction:
        return sum(self.values.values(), Fraction(0))


def exact_states(bt: Backtest) -> list[DayState]:
    """Rebuild cash, positions and equity on every date in exact arithmetic."""
    symbols = list(bt.prices.columns)
    by_date: dict[Any, list[tuple[str, Fraction, Fraction, Fraction]]] = {}
    for row in bt.trades.itertuples(index=False):
        by_date.setdefault(row.date, []).append(
            (row.symbol, _frac(row.quantity), _frac(row.price), _frac(row.fee))
        )

    cash = _frac(bt.initial_capital)
    positions = {s: Fraction(0) for s in symbols}
    states: list[DayState] = []
    for date, closes in bt.prices.iterrows():
        for symbol, qty, price, fee in by_date.get(date, []):
            positions[symbol] += qty
            cash -= qty * price + fee
        values = {s: positions[s] * _frac(closes[s]) for s in symbols}
        states.append(DayState(
            date=str(date.date()), cash=cash,
            equity=cash + sum(values.values(), Fraction(0)),
            positions=dict(positions), values=values,
        ))
    return states


def _real(f: Fraction) -> z3.ArithRef:
    return z3.RealVal(f"{f.numerator}/{f.denominator}")


# A "case" is one place a limit could be broken: (label, lhs, rhs_scale).
# The limit holds in that case when lhs <= limit * rhs_scale.
Case = tuple[dict[str, str], Fraction, Fraction]


def _find_violations(cases: list[Case], limit: Fraction) -> tuple[str, list[int]]:
    """Ask Z3 for every case where lhs > limit * scale. Returns (result, indexes).

    Cases are given to the solver in blocks: solving time grows faster than
    linearly with the number of cases in one query, so many small queries are
    much quicker than one large one. Every block must come back unsat for the
    overall result to be unsat.
    """
    i, lhs, scale = z3.Int("case"), z3.Real("lhs"), z3.Real("scale")
    bound = _real(limit)
    found: list[int] = []
    for start in range(0, len(cases), BLOCK_SIZE):
        solver = z3.Solver()
        solver.set("timeout", SOLVER_TIMEOUT_MS)
        solver.add(z3.Or([
            z3.And(i == start + k, lhs == _real(case_lhs), scale == _real(case_scale))
            for k, (_, case_lhs, case_scale) in enumerate(cases[start:start + BLOCK_SIZE])
        ]))
        solver.add(lhs > bound * scale)
        while True:
            result = solver.check()
            if result == z3.unsat:
                break
            if result != z3.sat:
                return "unknown", sorted(found)
            k = solver.model()[i].as_long()
            found.append(k)
            solver.add(i != k)
    return "unsat", sorted(found)


def _ratio(lhs: Fraction, scale: Fraction) -> str:
    return f"{float(lhs / scale):.4f}" if scale != 0 else "undefined (zero equity)"


def _run(name: str, claim: str, cases: list[Case], limit: Fraction,
         describe: Callable[[Case], str]) -> Check:
    if not cases:
        return Check(name, claim, METHOD, Status.UNVERIFIED, Strength.NONE,
                     "No dates to check", {})
    result, violations = _find_violations(cases, limit)
    evidence: dict[str, Any] = {
        "solver": f"z3 {z3.get_version_string()}", "result": result,
        "cases_checked": len(cases), "limit": str(limit), "arithmetic": "exact rational",
    }
    if result == "unknown":
        return Check(name, claim, METHOD, Status.ERROR, Strength.NONE,
                     "Solver did not reach a decision within the time limit", evidence)
    if not violations:
        return Check(name, claim, METHOD, Status.VERIFIED, Strength.SOLVER_CHECKED,
                     f"No violating case exists among {len(cases)} checked (solver: unsat)",
                     evidence)
    first = cases[violations[0]]
    evidence["violations"] = len(violations)
    evidence["counterexample"] = {**first[0], "detail": describe(first)}
    return Check(name, claim, METHOD, Status.FAILED, Strength.NONE,
                 f"Broken in {len(violations)} case(s); first: {describe(first)}", evidence)


def _per_day(states: list[DayState], lhs: Callable[[DayState], Fraction],
             scale: Callable[[DayState], Fraction]) -> list[Case]:
    return [({"date": s.date}, lhs(s), scale(s)) for s in states]


def check_constraints(bt: Backtest) -> list[Check]:
    declared = bt.config.get("constraints")
    if not declared:
        return [Check("constraints", "Declared portfolio limits hold on every date", METHOD,
                      Status.UNVERIFIED, Strength.NONE,
                      "No constraints declared in config.json", {})]

    states = exact_states(bt)
    one = lambda s: Fraction(1)  # noqa: E731
    checks: list[Check] = []

    for key, raw in declared.items():
        name = f"constraint_{key}"
        if key == "max_gross_leverage":
            checks.append(_run(
                name, f"Gross leverage never exceeds {raw:g}",
                _per_day(states, lambda s: s.gross, lambda s: s.equity), _frac(raw),
                lambda c: f"{c[0]['date']}: gross leverage {_ratio(c[1], c[2])}",
            ))
        elif key == "max_net_exposure":
            checks.append(_run(
                name, f"Net exposure never exceeds {raw:g} of equity",
                _per_day(states, lambda s: abs(s.net), lambda s: s.equity), _frac(raw),
                lambda c: f"{c[0]['date']}: net exposure {_ratio(c[1], c[2])}",
            ))
        elif key == "max_position_weight":
            cases = [({"date": s.date, "symbol": sym}, abs(v), s.equity)
                     for s in states for sym, v in s.values.items()]
            checks.append(_run(
                name, f"No position exceeds {raw:g} of equity", cases, _frac(raw),
                lambda c: f"{c[0]['date']} {c[0]['symbol']}: weight {_ratio(c[1], c[2])}",
            ))
        elif key == "min_cash":
            # cash >= C  is  -cash <= -C
            checks.append(_run(
                name, f"Cash never falls below {raw:g}",
                _per_day(states, lambda s: -s.cash, one), -_frac(raw),
                lambda c: f"{c[0]['date']}: cash {float(-c[1]):.2f}",
            ))
        elif key == "long_only":
            if raw is not True:
                continue
            cases = [({"date": s.date, "symbol": sym}, -p, Fraction(1))
                     for s in states for sym, p in s.positions.items()]
            checks.append(_run(
                name, "No short positions are ever held", cases, Fraction(0),
                lambda c: f"{c[0]['date']} {c[0]['symbol']}: position {float(-c[1]):g}",
            ))
        else:
            # Fail safe: a limit we don't understand is a gap, not a pass.
            checks.append(Check(name, f"Constraint '{key}' holds", METHOD, Status.UNVERIFIED,
                                Strength.NONE, f"Unknown constraint '{key}'; not checked", {}))
    return checks


def run_all(bt: Backtest) -> list[Check]:
    try:
        return check_constraints(bt)
    except Exception as exc:  # noqa: BLE001 - fail safe: record, don't hide
        return [Check("constraints", "(check did not complete)", METHOD, Status.ERROR,
                      Strength.NONE, f"{type(exc).__name__}: {exc}")]
