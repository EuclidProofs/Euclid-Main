"""Reference definitions of performance metrics.

These are deliberately small and explicit: each function is the definition
EUCLID checks reported numbers against. Conventions:
  - returns are simple per-period returns
  - volatility and Sharpe use the sample standard deviation (ddof=1)
  - max drawdown is a negative fraction (e.g. -0.112 for an 11.2% drawdown)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def total_return(returns: pd.Series) -> float:
    return float(np.prod(1.0 + returns.to_numpy()) - 1.0)


def volatility(returns: pd.Series, periods_per_year: int) -> float:
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe(returns: pd.Series, periods_per_year: int, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or math.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * math.sqrt(periods_per_year))


SHARPE_DEFAULT = {
    "returns": "periodic",  # "periodic" (per row) or "annual" (calendar-year returns)
    "risk_free_rate": 0.0,  # annual rate
    "annualize": True,  # multiply by sqrt(periods_per_year); ignored for annual returns
    "std": "sample",  # "sample" (n-1) or "population" (n)
    "rf_conversion": "simple",  # "simple" (rate / periods) or "geometric"
}
SHARPE_PRESETS = {
    # backtrader's SharpeRatio analyzer with its default parameters
    "backtrader_default": {"returns": "annual", "risk_free_rate": 0.01, "annualize": False,
                           "std": "population"},
}


def sharpe_with(returns: pd.Series, periods_per_year: int, convention: dict) -> float:
    """Sharpe ratio under an explicit convention (see SHARPE_DEFAULT)."""
    c = {**SHARPE_DEFAULT, **convention}
    ddof = 1 if c["std"] == "sample" else 0
    rf = float(c["risk_free_rate"])
    if c["returns"] == "annual":
        annual = (1.0 + returns).groupby(returns.index.year).prod() - 1.0
        excess, scale = annual - rf, 1.0
    else:
        per_period = (rf / periods_per_year if c["rf_conversion"] == "simple"
                      else (1.0 + rf) ** (1.0 / periods_per_year) - 1.0)
        excess = returns - per_period
        scale = math.sqrt(periods_per_year) if c["annualize"] else 1.0
    if len(excess) < 2:
        return float("nan")
    sd = excess.std(ddof=ddof)
    if sd == 0 or math.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * scale)


def describe_sharpe(convention: dict, periods_per_year: int) -> str:
    c = {**SHARPE_DEFAULT, **convention}
    rf = f"{c['risk_free_rate'] * 100:g}% risk-free rate"
    std = f"{c['std']} standard deviation"
    if c["returns"] == "annual":
        return f"calendar-year returns, {rf}, {std}, not annualised"
    scale = f"annualised with {periods_per_year} periods" if c["annualize"] else "not annualised"
    conv = "" if c["risk_free_rate"] == 0 or c["rf_conversion"] == "simple" else " (geometric conversion)"
    return f"per-period returns, {rf}{conv}, {std}, {scale}"


def max_drawdown(returns: pd.Series) -> float:
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns.to_numpy())])
    peaks = np.maximum.accumulate(curve)
    return float((curve / peaks - 1.0).min())


def equity_from_trades(
    prices: pd.DataFrame, trades: pd.DataFrame, initial_capital: float
) -> pd.Series:
    """Rebuild the equity curve from trades, marked at each day's close."""
    qty = (
        trades.pivot_table(index="date", columns="symbol", values="quantity", aggfunc="sum")
        .reindex(index=prices.index, columns=prices.columns)
        .fillna(0.0)
    )
    positions = qty.cumsum()

    flows = -(trades["quantity"] * trades["price"]) - trades["fee"]
    cash_flow = flows.groupby(trades["date"]).sum().reindex(prices.index).fillna(0.0)
    cash = initial_capital + cash_flow.cumsum()

    return cash + (positions * prices).sum(axis=1)


def returns_from_equity(equity: pd.Series, initial_capital: float) -> pd.Series:
    previous = equity.shift(1)
    previous.iloc[0] = initial_capital
    return equity / previous - 1.0
