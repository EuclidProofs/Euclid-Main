"""Generate two sample backtests: one clean, one with three planted errors.

    python examples/make_sample.py

Writes examples/sample_clean and examples/sample_flawed. The data is synthetic
(a seeded random walk and a simple moving-average rule); it exists to exercise
the checks, not to represent a real strategy.

Planted errors in sample_flawed:
  1. One reported daily return is inflated by 0.5 percentage points.
  2. Fees are missing on every third trade.
  3. The reported Sharpe ratio uses a 365-day annualisation factor.
  4. An oversized position is held for one day, breaking the position limit.
  5. The signal code reads prices two days ahead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from euclid import metrics

SEED = 7
DAYS = 500
CAPITAL = 1_000_000.0
COST_BPS = 5.0
PPY = 252
SYMBOLS = {"AAA": 100.0, "BBB": 50.0}
CONSTRAINTS = {
    "max_gross_leverage": 1.5,
    "max_position_weight": 0.5,
    "min_cash": 0,
    "long_only": True,
}
OVERSIZE_DAY = 300

STRATEGY_CLEAN = '''"""Sample signal code: long when the 10-day average is above the 40-day."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def trailing_mean(close, window):
    out = np.full_like(close, np.nan)
    out[window - 1:] = sliding_window_view(close, window, axis=0).mean(axis=-1)
    return out


def signals(close):
    fast = trailing_mean(close, 10)
    slow = trailing_mean(close, 40)
    return np.where(fast > slow, 1.0, 0.0)
'''

STRATEGY_FLAWED = STRATEGY_CLEAN.replace(
    "    fast = trailing_mean(close, 10)\n",
    "    fast = np.roll(trailing_mean(close, 10), -2, axis=0)  # planted error: reads 2 days ahead\n",
)


def make_prices() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2023-01-02", periods=DAYS)
    data = {}
    for symbol, start in SYMBOLS.items():
        steps = rng.normal(0.0004, 0.012, DAYS)
        data[symbol] = np.round(start * np.cumprod(1 + steps), 2)
    return pd.DataFrame(data, index=dates).rename_axis("date")


def make_trades(prices: pd.DataFrame) -> pd.DataFrame:
    """Hold 2,000 shares when the 10-day average is above the 40-day, else flat."""
    rows = []
    for symbol in prices.columns:
        p = prices[symbol]
        target = ((p.rolling(10).mean() > p.rolling(40).mean()).shift(1, fill_value=False)) * 2000
        delta = target.diff().fillna(target)
        for date, qty in delta[delta != 0].items():
            price = float(p[date])
            rows.append({
                "date": date, "symbol": symbol, "quantity": float(qty), "price": price,
                "fee": round(abs(qty * price) * COST_BPS / 1e4, 2),
            })
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def returns_for(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    equity = metrics.equity_from_trades(prices, trades, CAPITAL)
    return metrics.returns_from_equity(equity, CAPITAL)


def write(folder: Path, prices: pd.DataFrame, trades: pd.DataFrame,
          returns: pd.Series, sharpe_factor: int, strategy: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "strategy.py").write_text(strategy)
    long = prices.reset_index().melt(id_vars="date", var_name="symbol", value_name="close")
    long.sort_values(["date", "symbol"]).to_csv(folder / "prices.csv", index=False)
    trades.to_csv(folder / "trades.csv", index=False)
    returns.rename("ret").rename_axis("date").to_csv(folder / "returns.csv")
    config = {
        "initial_capital": CAPITAL,
        "periods_per_year": PPY,
        "risk_free_rate": 0.0,
        "cost_bps": COST_BPS,
        "constraints": CONSTRAINTS,
        "strategy": "strategy.py:signals",
        "reported": {
            "total_return": metrics.total_return(returns),
            "volatility": metrics.volatility(returns, PPY),
            "sharpe": metrics.sharpe(returns, sharpe_factor),
            "max_drawdown": metrics.max_drawdown(returns),
        },
    }
    (folder / "config.json").write_text(json.dumps(config, indent=2))


def add_oversized_position(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Buy AAA up to about 65% of capital on one day and sell the excess the next."""
    day, next_day = prices.index[OVERSIZE_DAY], prices.index[OVERSIZE_DAY + 1]
    held = trades[(trades["symbol"] == "AAA") & (trades["date"] <= day)]["quantity"].sum()
    price = float(prices.loc[day, "AAA"])
    extra = round((0.65 * CAPITAL / price - held) / 100) * 100.0
    rows = []
    for date, qty in ((day, extra), (next_day, -extra)):
        p = float(prices.loc[date, "AAA"])
        rows.append({"date": date, "symbol": "AAA", "quantity": qty, "price": p,
                     "fee": round(abs(qty * p) * COST_BPS / 1e4, 2)})
    out = pd.concat([trades, pd.DataFrame(rows)], ignore_index=True)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def make_flawed(prices: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    flawed_trades = add_oversized_position(prices, trades)  # error 4
    flawed_trades.loc[flawed_trades.index % 3 == 0, "fee"] = 0.0  # error 2
    flawed_returns = returns_for(prices, flawed_trades)
    flawed_returns.iloc[250] += 0.005  # error 1
    return flawed_trades, flawed_returns


def main() -> None:
    here = Path(__file__).parent
    prices = make_prices()
    trades = make_trades(prices)

    write(here / "sample_clean", prices, trades, returns_for(prices, trades), PPY,
          STRATEGY_CLEAN)

    flawed_trades, flawed_returns = make_flawed(prices, trades)
    write(here / "sample_flawed", prices, flawed_trades, flawed_returns, 365,  # error 3
          STRATEGY_FLAWED)  # error 5

    print(f"Wrote {here / 'sample_clean'} and {here / 'sample_flawed'} ({len(trades)} trades)")


if __name__ == "__main__":
    main()
