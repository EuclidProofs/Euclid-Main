"""Execution checks: do the recorded trades follow from the strategy's signals?

The signal code and the trade record are otherwise verified separately. These
checks tie them together by re-running the strategy, applying a declared
execution rule, and comparing the result with trades.csv.

Execution rule (config.json "execution"):
  signal      "target_units"  signal x multiplier is the position to hold
              "order_units"   signal x multiplier is the quantity to trade
  multiplier  units per unit of signal (default 1)
  delay       bars between a signal and its trade (default 1)
  price       "close" or "open": the price each trade should have executed at
              (optional; omitted means prices are not checked)

A backtrader strategy gets that engine's rule by default: orders fill one bar
later at the open.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .loader import Backtest
from .lookahead import StrategyHandle
from .model import Check, Status, Strength

METHOD = "signal_replay"
FOLLOW = "trades_follow_signals"
PRICES = "execution_prices"
QTY_TOL = 1e-9


def _gap(name: str, claim: str, detail: str) -> Check:
    return Check(name, claim, METHOD, Status.UNVERIFIED, Strength.NONE, detail, {})


def _traded_units(bt: Backtest) -> np.ndarray:
    """Net units traded per [date, symbol], on the price calendar."""
    table = (
        bt.trades.pivot_table(index="date", columns="symbol", values="quantity", aggfunc="sum")
        .reindex(index=bt.prices.index, columns=bt.prices.columns)
        .fillna(0.0)
    )
    return table.to_numpy(dtype=float)


def _signals(handle: StrategyHandle, n_symbols: int) -> np.ndarray:
    out = np.asarray(handle.func(handle.data.copy()), dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2 or out.shape[1] != n_symbols:
        raise ValueError(f"signals have shape {out.shape}; expected one column per symbol "
                         f"({n_symbols})")
    return np.nan_to_num(out, nan=0.0)


def check_trades_follow_signals(bt: Backtest, handle: StrategyHandle, rule: dict[str, Any]) -> Check:
    kind = rule.get("signal")
    multiplier = float(rule.get("multiplier", 1))
    delay = int(rule.get("delay", 1))
    claim = (f"Recorded trades are the ones the signals call for "
             f"({kind}, x{multiplier:g}, {delay} bar delay)")
    if kind not in ("target_units", "order_units"):
        return _gap(FOLLOW, claim, f"Execution rule signal={kind!r} is not supported yet")
    if delay < 0:
        return _gap(FOLLOW, claim, "Execution delay must be zero or more bars")

    dates = [str(d.date()) for d in bt.prices.index]
    if handle.dates != dates:
        return Check(FOLLOW, claim, METHOD, Status.ERROR, Strength.NONE,
                     "The strategy's data and prices.csv cover different dates", {})

    signals = _signals(handle, len(bt.prices.columns)) * multiplier
    shifted = np.zeros_like(signals)
    if delay < len(signals):
        shifted[delay:] = signals[:len(signals) - delay]
    traded = _traded_units(bt)

    if kind == "order_units":
        expected, actual, what = shifted, traded, "traded"
    else:
        expected, actual, what = shifted, np.cumsum(traded, axis=0), "held"

    wrong = np.argwhere(np.abs(expected - actual) > QTY_TOL)
    evidence: dict[str, Any] = {"rule": {"signal": kind, "multiplier": multiplier, "delay": delay},
                                "cells_compared": int(expected.size),
                                "mismatches": int(len(wrong))}
    if delay and np.any(signals[len(signals) - delay:] != 0):
        evidence["note"] = f"signals in the last {delay} bar(s) have no later bar to trade on"
    if not len(wrong):
        return Check(FOLLOW, claim, METHOD, Status.VERIFIED, Strength.RECOMPUTED,
                     f"Units {what} match the replayed signals on all {len(dates)} dates", evidence)

    row, col = (int(x) for x in wrong[0])
    symbol = bt.prices.columns[col]
    evidence["first_mismatch"] = {"date": dates[row], "symbol": symbol,
                                  "expected": float(expected[row, col]),
                                  "recorded": float(actual[row, col])}
    days = len(set(wrong[:, 0].tolist()))
    return Check(
        FOLLOW, claim, METHOD, Status.FAILED, Strength.NONE,
        f"Units {what} differ from the signals on {days} date(s); first on {dates[row]} "
        f"{symbol}: signals call for {expected[row, col]:g}, record shows {actual[row, col]:g}",
        evidence)


def check_execution_prices(bt: Backtest, handle: StrategyHandle, rule: dict[str, Any]) -> Check:
    basis = rule.get("price")
    claim = f"Every trade executed at that day's {basis}"
    if basis == "close":
        reference = bt.prices
    elif basis == "open":
        if handle.open_prices is None or len(bt.prices.columns) != 1:
            return _gap(PRICES, claim, "Open prices are only available for single-instrument "
                                       "engine strategies")
        reference = bt.prices.copy()
        reference.iloc[:, 0] = handle.open_prices
    else:
        return _gap(PRICES, f"Trades executed at the declared price ({basis!r})",
                    f"Execution price basis {basis!r} is not supported")

    t = bt.trades
    if t.empty:
        return _gap(PRICES, claim, "trades.csv has no trades")
    expected = np.array([reference.at[d, s] for d, s in zip(t["date"], t["symbol"])])
    recorded = t["price"].to_numpy(dtype=float)
    off = ~np.isclose(recorded, expected, rtol=1e-9, atol=1e-6)
    evidence = {"basis": basis, "trades": int(len(t)), "mismatches": int(off.sum())}
    if not off.any():
        return Check(PRICES, claim, METHOD, Status.VERIFIED, Strength.RECOMPUTED,
                     f"All {len(t)} trade prices match the {basis}", evidence)
    first = t[off].iloc[0]
    evidence["first_mismatch"] = {"date": str(first["date"].date()), "symbol": first["symbol"],
                                  "recorded": float(first["price"]),
                                  "expected": float(expected[off][0])}
    return Check(
        PRICES, claim, METHOD, Status.FAILED, Strength.NONE,
        f"{int(off.sum())} of {len(t)} trades are not at the {basis}; first on "
        f"{first['date'].date()} {first['symbol']}: recorded {first['price']:.6g}, "
        f"{basis} {expected[off][0]:.6g}", evidence)


def run_all(bt: Backtest, handle: StrategyHandle | Check) -> list[Check]:
    claim = "Recorded trades are the ones the signals call for"
    if isinstance(handle, Check):
        return [_gap(FOLLOW, claim, "No strategy code available to replay")]
    rule = bt.config.get("execution") or handle.default_execution
    if not rule:
        return [_gap(FOLLOW, claim,
                     "No execution rule declared in config.json, so signals cannot be "
                     "compared with trades")]
    checks = []
    for fn in (check_trades_follow_signals, check_execution_prices):
        if fn is check_execution_prices and not rule.get("price"):
            continue
        try:
            checks.append(fn(bt, handle, rule))
        except Exception as exc:  # noqa: BLE001 - fail safe: record, don't hide
            name = FOLLOW if fn is check_trades_follow_signals else PRICES
            checks.append(Check(name, "(check did not complete)", METHOD, Status.ERROR,
                                Strength.NONE, f"{type(exc).__name__}: {exc}"))
    return checks
