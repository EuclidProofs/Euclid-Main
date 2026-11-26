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
    func = bt_adapter.make_signal_function(strategy_cls, dates, cash=100_000, stake=10)
    return func, check_lookahead(func, ohlcv, [str(d.date()) for d in dates], trace=False, n_cuts=8)


def test_builtin_crossover_has_no_lookahead(market):
    func, result = check(bt.strategies.MA_CrossOver, market)
    assert np.count_nonzero(func(market[1])) > 0  # the strategy actually placed orders
    assert result.status is Status.VERIFIED and result.strength is Strength.TESTED


def test_strategy_that_peeks_at_tomorrow_is_caught(market):
    _, result = check(PeeksAtTomorrow, market)
    assert result.status is Status.FAILED
    assert result.evidence["counterexample"]["rows_ahead"] == 1
