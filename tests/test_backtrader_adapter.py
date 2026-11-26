import warnings

import numpy as np
import pandas as pd
import pytest

from euclid.lookahead import check_lookahead
from euclid.model import Status, Strength

warnings.simplefilter("ignore")
bt = pytest.importorskip("backtrader")

from euclid.adapters import bt_adapter  # noqa: E402

BARS = 260


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(5)
    close = 50 * np.cumprod(1 + rng.normal(0.0004, 0.015, BARS))
    ohlcv = np.column_stack([close * 0.995, close * 1.01, close * 0.99, close,
                             np.full(BARS, 1e6)])
    return pd.bdate_range("2022-01-03", periods=BARS), ohlcv


class PeeksAtTomorrow(bt.Strategy):
    """Buys when tomorrow's close is higher: a deliberate lookahead."""

    def next(self):
        try:
            tomorrow = self.data.close[1]
        except IndexError:
            return
        if not self.position and tomorrow > self.data.close[0]:
            self.buy()
        elif self.position and tomorrow < self.data.close[0]:
            self.sell()


def check(strategy_cls, market):
    dates, ohlcv = market
    func = bt_adapter.make_signal_function(strategy_cls, dates, ["X"], cash=100_000, stake=10)
    return func, check_lookahead(func, ohlcv, [str(d.date()) for d in dates], trace=False, n_cuts=8)


def test_builtin_crossover_has_no_lookahead(market):
    func, result = check(bt.strategies.MA_CrossOver, market)
    assert np.count_nonzero(func(market[1])) > 0  # the strategy actually placed orders
    assert result.status is Status.VERIFIED and result.strength is Strength.TESTED


def test_strategy_that_peeks_at_tomorrow_is_caught(market):
    _, result = check(PeeksAtTomorrow, market)
    assert result.status is Status.FAILED
    assert result.evidence["counterexample"]["rows_ahead"] == 1


class TradesSecondInstrument(bt.Strategy):
    """Addresses instruments by position: trades data1 on data0's moves."""

    def next(self):
        if len(self) < 2:
            return
        if self.data0.close[0] > self.data0.close[-1] and not self.getposition(self.data1).size:
            self.buy(data=self.data1, size=5)
        elif self.data0.close[0] < self.data0.close[-1] and self.getposition(self.data1).size:
            self.sell(data=self.data1, size=5)


def test_two_instruments_and_feed_order(market):
    dates, one = market
    two = np.stack([one, one[::-1].copy()], axis=1)  # instruments "A" and "B"
    func = bt_adapter.make_signal_function(TradesSecondInstrument, dates, ["A", "B"],
                                           data_order=["B", "A"], cash=100_000)
    out = func(two)
    assert out.shape == (BARS, 2)
    # data0 is B, so orders go to data1, which is A (column 0).
    assert np.count_nonzero(out[:, 0]) > 0 and np.count_nonzero(out[:, 1]) == 0
    result = check_lookahead(func, two, [str(d.date()) for d in dates], trace=False, n_cuts=6)
    assert result.status is Status.VERIFIED
