"""Adapter: check a backtrader Strategy class for lookahead.

backtrader strategies are event-driven classes, not functions of a price
array, so they cannot be traced. This adapter turns one into a signal function
for the perturbation layer:

    input   an array [bars, 5] of open, high, low, close, volume
    output  the strategy's order decisions: the net order size it submitted
            on each bar (0 when it did nothing)

Order decisions are used instead of positions on purpose. backtrader fills a
market order on the next bar, so the position series lags by a bar and would
hide a one-bar lookahead.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


def _backtrader():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import backtrader as bt
    except ImportError as exc:
        raise ImportError("the backtrader adapter needs backtrader: pip install backtrader") from exc
    return bt


def load_strategy_class(spec: str, base: Path) -> tuple[type, str]:
    """Load `file.py:Class` (relative to base) or `package.module:Class`."""
    target, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError("strategy class must be given as file.py:Class or module:Class")
    if target.endswith(".py"):
        path = Path(target) if Path(target).is_absolute() else base / target
        module_spec = importlib.util.spec_from_file_location(f"euclid_bt_{path.stem}", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        _backtrader()
        module = importlib.import_module(target)
    cls = getattr(module, class_name)
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        source = ""
    return cls, source


def make_signal_function(
    strategy_cls: type,
    dates: pd.DatetimeIndex,
    cash: float = 100_000.0,
    commission: float = 0.0,
    stake: int = 1,
    params: dict[str, Any] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    bt = _backtrader()

    class Decisions(bt.Analyzer):
        def start(self):
            self.bar_times: list[float] = []
            self.orders: list[tuple[float, float]] = []

        def next(self):
            self.bar_times.append(self.data.datetime[0])

        def notify_order(self, order):
            if order.status == order.Submitted:
                self.orders.append((order.created.dt, float(order.created.size)))

        def get_analysis(self):
            index = {t: i for i, t in enumerate(self.bar_times)}
            out = np.zeros(len(self.bar_times))
            for created, size in self.orders:
                if created in index:
                    out[index[created]] += size
            return out

    def signals(ohlcv: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(ohlcv, dtype=float), index=dates, columns=list(FIELDS))
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.adddata(bt.feeds.PandasData(dataname=frame))
        cerebro.addstrategy(strategy_cls, **(params or {}))
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission=commission)
        cerebro.addsizer(bt.sizers.FixedSize, stake=stake)
        cerebro.addanalyzer(Decisions, _name="decisions")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cerebro.run()[0]
        out = result.analyzers.decisions.get_analysis()
        if len(out) != len(dates):
            raise ValueError(f"strategy ran {len(out)} bars but the data has {len(dates)}")
        return out

    return signals


def load_ohlcv(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    missing = [f for f in FIELDS if f not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    return frame.index, frame[list(FIELDS)].to_numpy(dtype=float)
