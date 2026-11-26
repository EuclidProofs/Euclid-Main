"""Recompute engine: independent recalculation of a backtest's numbers.

Every check here has strength `recomputed`: the reported number matches (or
does not match) what follows from the supplied inputs, within a stated
tolerance. It says nothing about whether the inputs themselves are right.
"""

from __future__ import annotations

import math
from typing import Callable

import pandas as pd

from . import metrics
from .loader import Backtest
from .model import Check, Status, Strength

METHOD = "deterministic_recompute"
_COMMON_FACTORS = (252, 365, 260, 250, 52, 12)


def _check(name: str, claim: str, status: Status, detail: str, evidence: dict) -> Check:
    strength = Strength.RECOMPUTED if status is Status.VERIFIED else Strength.NONE
    return Check(name, claim, METHOD, status, strength, detail, evidence)


def check_returns_reconcile(bt: Backtest) -> Check:
    name = "returns_reconcile"
    claim = "Reported daily returns equal the returns rebuilt from trades and prices"
    tol = bt.tolerance("returns")

    equity = metrics.equity_from_trades(bt.prices, bt.trades, bt.initial_capital)
    rebuilt = metrics.returns_from_equity(equity, bt.initial_capital)
    reported = bt.reported_returns

    if not rebuilt.index.equals(reported.index):
        only_reported = len(reported.index.difference(rebuilt.index))
        only_prices = len(rebuilt.index.difference(reported.index))
        return _check(
            name, claim, Status.FAILED,
            f"Date mismatch: {only_reported} date(s) only in returns.csv, "
            f"{only_prices} only in prices.csv",
            {"dates_only_in_returns": only_reported, "dates_only_in_prices": only_prices},
        )

    diff = (reported - rebuilt).abs()
    bad = diff[diff > tol]
    evidence = {"tolerance": tol, "max_abs_diff": float(diff.max()), "mismatched_days": int(len(bad))}
    if bad.empty:
        return _check(name, claim, Status.VERIFIED,
                      f"All {len(reported)} daily returns match within {tol:g}", evidence)

    worst = bad.idxmax()
    evidence.update({
        "first_mismatch": str(bad.index[0].date()),
        "worst_mismatch": str(worst.date()),
        "reported_at_worst": float(reported[worst]),
        "rebuilt_at_worst": float(rebuilt[worst]),
    })
    return _check(
        name, claim, Status.FAILED,
        f"{len(bad)} of {len(reported)} days differ; worst on {worst.date()}: "
        f"reported {reported[worst]:.6f}, rebuilt {rebuilt[worst]:.6f}",
        evidence,
    )


def check_equity_positive(bt: Backtest) -> Check:
    name = "equity_positive"
    claim = "Account equity stays above zero throughout"
    equity = metrics.equity_from_trades(bt.prices, bt.trades, bt.initial_capital)
    low = equity.idxmin()
    evidence = {"minimum_equity": float(equity.min()), "minimum_on": str(low.date())}
    wiped = equity[equity <= 0]
    if wiped.empty:
        return _check(name, claim, Status.VERIFIED,
                      f"Lowest equity is {equity.min():,.2f} on {low.date()}", evidence)
    first = wiped.index[0]
    evidence.update({"first_non_positive": str(first.date()), "days_non_positive": int(len(wiped))})
    return _check(
        name, claim, Status.FAILED,
        f"Equity is zero or negative on {len(wiped)} of {len(equity)} dates, first on "
        f"{first.date()} ({equity[first]:,.2f}). Returns, Sharpe and drawdown computed past "
        "that point are arithmetic on a wiped-out account and carry no meaning", evidence)


def _metric_check(
    bt: Backtest, key: str, label: str, compute: Callable[[pd.Series], float],
    diagnose: Callable[[float], str | None] | None = None,
) -> Check:
    name = f"metric_{key}"
    reported = bt.config.get("reported", {}).get(key)
    if reported is None:
        return _check(name, f"Reported {label} is correct", Status.UNVERIFIED,
                      f"No reported {label} in config.json", {})

    reported = float(reported)
    claim = f"Reported {label} = {reported:.6g}"
    tol = bt.tolerance("metrics")
    value = compute(bt.reported_returns)
    evidence = {"reported": reported, "recomputed": value, "tolerance": tol}

    if math.isnan(value):
        return _check(name, claim, Status.UNVERIFIED,
                      f"{label} is undefined for these returns", evidence)
    if abs(value - reported) <= tol:
        return _check(name, claim, Status.VERIFIED,
                      f"Recomputed {value:.6g}; matches within {tol:g}", evidence)

    detail = f"Recomputed {value:.6g}, reported {reported:.6g}"
    hint = diagnose(reported) if diagnose else None
    if hint:
        detail += f"; {hint}"
        evidence["diagnosis"] = hint
    return _check(name, claim, Status.FAILED, detail, evidence)


def _annualisation_hint(bt: Backtest, compute_with: Callable[[int], float]) -> Callable[[float], str | None]:
    def diagnose(reported: float) -> str | None:
        for factor in _COMMON_FACTORS:
            if factor == bt.periods_per_year:
                continue
            if abs(compute_with(factor) - reported) <= bt.tolerance("metrics"):
                return (f"the reported value matches an annualisation factor of {factor}, "
                        f"but config.json declares {bt.periods_per_year}")
        return None
    return diagnose


def _declared_sharpe_convention(bt: Backtest) -> tuple[dict, bool]:
    """The Sharpe definition the user declared, or EUCLID's default."""
    raw = bt.config.get("sharpe_convention")
    base = {**metrics.SHARPE_DEFAULT, "risk_free_rate": bt.risk_free_rate}
    if raw is None:
        return base, False
    if isinstance(raw, str):
        if raw not in metrics.SHARPE_PRESETS:
            raise ValueError(f"unknown sharpe_convention preset {raw!r}; "
                             f"known: {sorted(metrics.SHARPE_PRESETS)}")
        raw = metrics.SHARPE_PRESETS[raw]
    unknown = set(raw) - set(metrics.SHARPE_DEFAULT)
    if unknown:
        raise ValueError(f"unknown sharpe_convention keys: {sorted(unknown)}")
    return {**base, **raw}, True


def _sharpe_alternatives(declared: dict, ppy: int):
    """Other well-known Sharpe definitions, closest to the declared one first."""
    options = []
    for returns in ("periodic", "annual"):
        for rf in dict.fromkeys((declared["risk_free_rate"], 0.0, 0.01)):
            for std in ("sample", "population"):
                if returns == "annual":
                    options.append(({"returns": returns, "risk_free_rate": rf, "std": std,
                                     "annualize": False, "rf_conversion": "simple"}, ppy))
                    continue
                for annualize in (True, False):
                    for conv in (("simple", "geometric") if rf else ("simple",)):
                        for factor in dict.fromkeys((ppy, *_COMMON_FACTORS)):
                            options.append(({"returns": returns, "risk_free_rate": rf, "std": std,
                                             "annualize": annualize, "rf_conversion": conv}, factor))

    def distance(option) -> int:
        conv, factor = option
        return sum(conv[k] != declared[k] for k in conv) + (factor != ppy)

    return sorted(options, key=distance)


def check_sharpe(bt: Backtest) -> Check:
    name = "metric_sharpe"
    reported = bt.config.get("reported", {}).get("sharpe")
    if reported is None:
        return _check(name, "Reported Sharpe ratio is correct", Status.UNVERIFIED,
                      "No reported Sharpe ratio in config.json", {})
    reported, ppy, tol = float(reported), bt.periods_per_year, bt.tolerance("metrics")
    r = bt.reported_returns
    declared, explicit = _declared_sharpe_convention(bt)
    definition = metrics.describe_sharpe(declared, ppy)
    claim = f"Reported Sharpe ratio = {reported:.6g} ({definition})"
    value = metrics.sharpe_with(r, ppy, declared)
    evidence = {"reported": reported, "recomputed": value, "tolerance": tol,
                "definition": definition, "definition_declared": explicit}

    if math.isnan(value):
        return _check(name, claim, Status.UNVERIFIED,
                      "Sharpe ratio is undefined for these returns", evidence)
    if abs(value - reported) <= tol:
        return _check(name, claim, Status.VERIFIED,
                      f"Recomputed {value:.6g}; matches within {tol:g}", evidence)

    for alt, factor in _sharpe_alternatives(declared, ppy):
        alt_value = metrics.sharpe_with(r, factor, alt)
        if math.isnan(alt_value) or abs(alt_value - reported) > tol:
            continue
        alt_text = metrics.describe_sharpe(alt, factor)
        evidence["matching_definition"] = alt_text
        contradicts = explicit or factor != ppy
        if contradicts:
            return _check(
                name, claim, Status.FAILED,
                f"Recomputed {value:.6g}, reported {reported:.6g}; the reported value instead "
                f"matches: {alt_text}", evidence)
        return _check(
            name, claim, Status.UNVERIFIED,
            f"Under EUCLID's default definition the value is {value:.6g}, but the reported "
            f"{reported:.6g} matches another definition: {alt_text}. Declare sharpe_convention "
            "in config.json to confirm which is intended", evidence)

    return _check(name, claim, Status.FAILED,
                  f"Recomputed {value:.6g}, reported {reported:.6g}; no recognised Sharpe "
                  "definition reproduces the reported value", evidence)


def check_metrics(bt: Backtest) -> list[Check]:
    r, ppy = bt.reported_returns, bt.periods_per_year
    return [
        _metric_check(bt, "total_return", "total return", metrics.total_return),
        _metric_check(
            bt, "volatility", "annualised volatility",
            lambda s: metrics.volatility(s, ppy),
            _annualisation_hint(bt, lambda f: metrics.volatility(r, f)),
        ),
        check_sharpe(bt),
        _metric_check(bt, "max_drawdown", "maximum drawdown", metrics.max_drawdown),
    ]


def check_costs_applied(bt: Backtest) -> Check:
    name = "costs_applied"
    cost_bps = bt.config.get("cost_bps")
    if cost_bps is None:
        return _check(name, "Declared transaction costs were applied to every trade",
                      Status.UNVERIFIED, "No cost_bps declared in config.json", {})

    cost_bps = float(cost_bps)
    claim = f"Transaction costs of {cost_bps:g} bps were applied to every trade"
    tol = bt.tolerance("fees")
    t = bt.trades
    if t.empty:
        return _check(name, claim, Status.UNVERIFIED, "trades.csv has no trades", {})

    expected = (t["quantity"] * t["price"]).abs() * cost_bps / 1e4
    short = t[t["fee"] < expected - tol]
    evidence = {
        "cost_bps": cost_bps, "trades": int(len(t)), "undercharged_trades": int(len(short)),
        "expected_fees": float(expected.sum()), "recorded_fees": float(t["fee"].sum()),
    }
    if short.empty:
        return _check(name, claim, Status.VERIFIED,
                      f"All {len(t)} trades carry at least the declared cost", evidence)

    first = short.iloc[0]
    evidence["first_undercharged"] = {"date": str(first["date"].date()), "symbol": first["symbol"]}
    return _check(
        name, claim, Status.FAILED,
        f"{len(short)} of {len(t)} trades carry less than the declared cost; "
        f"recorded fees {t['fee'].sum():.2f} vs expected {expected.sum():.2f}",
        evidence,
    )


def run_all(bt: Backtest) -> list[Check]:
    """Run every recompute check. A crash in one check becomes an ERROR, never a pass."""
    checks: list[Check] = []
    for fn in (check_returns_reconcile, check_equity_positive, check_metrics, check_costs_applied):
        try:
            result = fn(bt)
            checks.extend(result if isinstance(result, list) else [result])
        except Exception as exc:  # noqa: BLE001 - fail safe: record, don't hide
            checks.append(Check(
                name=fn.__name__.removeprefix("check_"), claim="(check did not complete)",
                method=METHOD, status=Status.ERROR, strength=Strength.NONE,
                detail=f"{type(exc).__name__}: {exc}",
            ))
    return checks
