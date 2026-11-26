"""Load a backtest directory and freeze its inputs by hash.

Expected files (see docs/backtest-format.md):
  prices.csv   date, symbol, close
  trades.csv   date, symbol, quantity, price, fee
  returns.csv  date, ret
  config.json  capital, conventions, reported metrics, tolerances
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_FILES = ("prices.csv", "trades.csv", "returns.csv", "config.json")


class BacktestFormatError(ValueError):
    pass


@dataclass
class Backtest:
    path: Path
    prices: pd.DataFrame  # index: date, columns: symbol, values: close
    trades: pd.DataFrame  # columns: date, symbol, quantity, price, fee
    reported_returns: pd.Series  # index: date
    config: dict[str, Any]
    input_hashes: dict[str, str]

    @property
    def initial_capital(self) -> float:
        return float(self.config["initial_capital"])

    @property
    def periods_per_year(self) -> int:
        return int(self.config.get("periods_per_year", 252))

    @property
    def risk_free_rate(self) -> float:
        return float(self.config.get("risk_free_rate", 0.0))

    def tolerance(self, kind: str) -> float:
        defaults = {"returns": 1e-8, "metrics": 1e-4, "fees": 1e-6}
        return float(self.config.get("tolerances", {}).get(kind, defaults[kind]))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise BacktestFormatError(f"{name} is missing columns: {', '.join(missing)}")


def load_backtest(path: str | Path) -> Backtest:
    root = Path(path)
    missing = [f for f in REQUIRED_FILES if not (root / f).is_file()]
    if missing:
        raise BacktestFormatError(f"{root} is missing files: {', '.join(missing)}")

    hashes = {f: _sha256(root / f) for f in REQUIRED_FILES}

    config = json.loads((root / "config.json").read_text())
    if "initial_capital" not in config:
        raise BacktestFormatError("config.json must set initial_capital")

    prices_long = pd.read_csv(root / "prices.csv", parse_dates=["date"])
    _require_columns(prices_long, ["date", "symbol", "close"], "prices.csv")
    if prices_long.duplicated(["date", "symbol"]).any():
        raise BacktestFormatError("prices.csv has duplicate (date, symbol) rows")
    prices = prices_long.pivot(index="date", columns="symbol", values="close").sort_index()
    if prices.isna().any().any():
        raise BacktestFormatError("prices.csv must have a close for every symbol on every date")

    trades = pd.read_csv(root / "trades.csv", parse_dates=["date"])
    _require_columns(trades, ["date", "symbol", "quantity", "price", "fee"], "trades.csv")
    unknown_dates = set(trades["date"]) - set(prices.index)
    if unknown_dates:
        raise BacktestFormatError(
            f"trades.csv has {len(unknown_dates)} trade date(s) with no prices, "
            f"e.g. {min(unknown_dates).date()}"
        )
    unknown_symbols = set(trades["symbol"]) - set(prices.columns)
    if unknown_symbols:
        raise BacktestFormatError(f"trades.csv has symbols with no prices: {sorted(unknown_symbols)}")

    returns = pd.read_csv(root / "returns.csv", parse_dates=["date"])
    _require_columns(returns, ["date", "ret"], "returns.csv")
    reported = returns.set_index("date")["ret"].sort_index()

    return Backtest(
        path=root,
        prices=prices,
        trades=trades,
        reported_returns=reported,
        config=config,
        input_hashes=hashes,
    )
