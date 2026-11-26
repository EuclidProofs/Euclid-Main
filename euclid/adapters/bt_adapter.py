"""Adapter: check a backtrader Strategy class for lookahead.

backtrader strategies are event-driven classes, not functions of a price
array, so they cannot be traced. This adapter turns one into a signal function
for the perturbation layer:

    input   an array [bars, instruments, 5] of open, high, low, close, volume
    output  the strategy's order decisions, [bars, instruments]: the net order
            size it submitted for each instrument on each bar (0 for none)

Order decisions are used instead of positions on purpose. backtrader fills a
market order on the next bar, so the position series lags by a bar and would
hide a one-bar lookahead.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import inspect
import io
import sys
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
        module_spec = importlib.util.spec_from_file_location(f"euclid_bt_{path.stem.replace('-', '_')}", path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module  # backtrader looks strategy modules up by name
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
    symbols: list[str] | None = None,
    data_order: list[str] | None = None,
    cash: float = 100_000.0,
    commission: float = 0.0,
    stake: int | None = None,
    params: dict[str, Any] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build `signals(ohlcv) -> order decisions`.

    `ohlcv` has shape [bars, instruments, 5] (or [bars, 5] for one instrument),
    with instruments in `symbols` order. The result has shape [bars, instruments].
    `data_order` is the order the feeds are added to the engine (data0, data1,
    ...), for strategies that address instruments by position.
    """
    bt = _backtrader()
    symbols = list(symbols or ["asset"])
    feed_order = list(data_order or symbols)
    if sorted(feed_order) != sorted(symbols):
        raise ValueError(f"data_order {feed_order} does not match the instruments {symbols}")
    column = {name: i for i, name in enumerate(symbols)}

    class Decisions(bt.Analyzer):
        def start(self):
            self.bar_times: list[float] = []
            self.orders: list[tuple[float, str, float]] = []

        def next(self):
            self.bar_times.append(self.strategy.datetime[0])

        def notify_order(self, order):
            if order.status == order.Submitted:
                self.orders.append((order.created.dt, order.data._name, float(order.created.size)))

        def get_analysis(self):
            index = {t: i for i, t in enumerate(self.bar_times)}
            out = np.zeros((len(self.bar_times), len(symbols)))
            for created, name, size in self.orders:
                if created in index:
                    out[index[created], column[name]] += size
            return out

    def signals(ohlcv: np.ndarray) -> np.ndarray:
        data = np.asarray(ohlcv, dtype=float)
        if data.ndim == 2:
            data = data[:, None, :]
        if data.shape[1:] != (len(symbols), len(FIELDS)):
            raise ValueError(f"OHLCV data has shape {data.shape}; expected "
                             f"[bars, {len(symbols)}, {len(FIELDS)}]")
        cerebro = bt.Cerebro(stdstats=False)
        for name in feed_order:
            frame = pd.DataFrame(data[:, column[name], :], index=dates, columns=list(FIELDS))
            cerebro.adddata(bt.feeds.PandasData(dataname=frame), name=name)
        cerebro.addstrategy(strategy_cls, **(params or {}))
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission=commission)
        if stake is not None:
            cerebro.addsizer(bt.sizers.FixedSize, stake=int(stake))
        cerebro.addanalyzer(Decisions, _name="decisions")
        # Strategies often print on every bar; keep that out of the report.
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            result = cerebro.run()[0]
        out = result.analyzers.decisions.get_analysis()
        if len(out) != len(dates):
            raise ValueError(f"strategy ran {len(out)} bars but the data has {len(dates)}")
        return out

    return signals


def load_ohlcv(path: Path, symbols: list[str]) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Read ohlcv.csv into an array [bars, instruments, 5], instruments in `symbols` order.

    Columns: date, open, high, low, close, volume, and `symbol` when there is
    more than one instrument.
    """
    frame = pd.read_csv(path, parse_dates=["date"])
    missing = [f for f in FIELDS if f not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    if "symbol" not in frame.columns:
        if len(symbols) != 1:
            raise ValueError(f"{path.name} needs a symbol column for {len(symbols)} instruments")
        frame["symbol"] = symbols[0]
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError(f"{path.name} has duplicate (date, symbol) rows")
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    blocks = []
    for name in symbols:
        one = frame[frame["symbol"] == name].set_index("date").reindex(dates)
        if one[list(FIELDS)].isna().any().any():
            raise ValueError(f"{path.name} does not have a full bar for {name} on every date")
        blocks.append(one[list(FIELDS)].to_numpy(dtype=float))
    return dates, np.stack(blocks, axis=1)
