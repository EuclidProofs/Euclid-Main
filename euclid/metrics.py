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
