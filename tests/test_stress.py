import inspect
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from euclid import patterns
from euclid.lookahead import StrategyHandle
from euclid.model import Status, Strength
from euclid.stress import plan_faults, run_stress

warnings.simplefilter("ignore")
bt = pytest.importorskip("backtrader")

from euclid.adapters import bt_adapter  # noqa: E402

BARS = 400


class FlipsAndAssumesFills(bt.Strategy):
    """Flips between +10 and -10 units, tracking its own position in self.qty.

    The bug: it updates self.qty and self.state whether or not the order fills.
    """

    def __init__(self):
        self.sma = bt.indicators.SMA(period=10)
        self.state, self.qty = 0, 0

    def next(self):
        want = 1 if self.data.close[0] > self.sma[0] else -1
        if want != self.state:
            size = 10 + self.qty
            if want > 0:
                self.buy(size=size)
            else:
                self.sell(size=size)
            self.qty = 10
            self.state = want


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(8)
    close = 50 * np.cumprod(1 + rng.normal(0.0002, 0.015, BARS))
    ohlcv = np.column_stack([close * 0.999, close * 1.01, close * 0.99, close, np.full(BARS, 1e6)])
    return pd.bdate_range("2021-01-04", periods=BARS), ohlcv[:, None, :]


def handle_for(strategy_cls, market, **kwargs) -> StrategyHandle:
    dates, ohlcv = market
    runner = bt_adapter.make_runner(strategy_cls, dates, ["X"], cash=100_000, **kwargs)
    return StrategyHandle(
        func=lambda d: runner(d).decisions, runner=runner, data=ohlcv,
        dates=[str(d.date()) for d in dates], symbols=["X"],
        source=inspect.getsource(strategy_cls), hashed_file=Path("ohlcv.csv"), engine="backtrader")


def by_name(checks):
    return {c.name: c for c in checks}


def test_strategy_that_assumes_fills_does_not_recover(market):
    config = SimpleNamespace(config={"constraints": {"max_gross_leverage": 1.0}})
    checks, scenarios = run_stress(config, handle_for(FlipsAndAssumesFills, market), scenarios=6)
    checks = by_name(checks)
    assert checks["stress_recovery"].status is Status.FAILED
    assert checks["stress_solvency"].status is Status.VERIFIED
    refused = [s for s in scenarios if s.kind == "reject"]
    assert refused and all(s.recovery == "not_recovered" for s in refused)
    first = checks["stress_recovery"].evidence["first_failure"]
    assert first["final_positions"] != first["intended_positions"]


def test_strategy_that_reads_its_real_position_recovers(market):
    config = SimpleNamespace(config={"constraints": {"long_only": True}})
    checks, scenarios = run_stress(
        config, handle_for(bt.strategies.MA_CrossOver, market, stake=10), scenarios=6)
    assert all(c.status is Status.VERIFIED and c.strength is Strength.TESTED for c in checks)
    assert all(s.took_effect for s in scenarios)
    assert any(s.bars_to_recover for s in scenarios)  # the faults really moved positions


def test_injected_faults_do_what_they_say(market):
    dates, ohlcv = market
    run = bt_adapter.make_runner(FlipsAndAssumesFills, dates, ["X"], cash=100_000)
    base = run(ohlcv)
    first_bar = base.orders[0][0]
    refused = run(ohlcv, fault={"kind": "reject", "order": 0})
    assert refused.fault_bar == first_bar + 1
    assert refused.positions[first_bar + 1, 0] == 0 != base.positions[first_bar + 1, 0]
    partial = run(ohlcv, fault={"kind": "partial", "order": 0})
    assert partial.positions[first_bar + 1, 0] == base.positions[first_bar + 1, 0] / 2
    assert partial.positions[first_bar + 2, 0] == base.positions[first_bar + 2, 0]


def test_without_an_engine_strategy_stress_is_a_gap():
    plain = StrategyHandle(func=lambda d: d, data=np.zeros((3, 1)), dates=["a", "b", "c"],
                           symbols=["X"], source="", hashed_file=Path("s.py"))
    checks, _ = run_stress(SimpleNamespace(config={}), plain)
    assert [c.status for c in checks] == [Status.UNVERIFIED]


def test_fault_plan_covers_early_orders_only():
    faults = plan_faults(100, 12)
    assert len(faults) == 12 and max(f["order"] for f in faults) < 80
    assert {f["kind"] for f in faults} == {"reject", "partial"}
    assert plan_faults(0, 12) == []


def test_pattern_scan_flags_state_updated_before_the_fill():
    hints = patterns.scan(inspect.getsource(FlipsAndAssumesFills))
    flagged = {h["code"] for h in hints}
    assert flagged == {"self.qty = 10", "self.state = want"}
    assert patterns.scan(inspect.getsource(bt.strategies.MA_CrossOver)) == []


def test_pattern_scan_flags_future_bar_access_but_not_feed_selection():
    source = '''
class S:
    def next(self):
        a = self.data.close[1]
        b = self.datas[1].close[0]
        c = self.data0.close[-1]
        self.order = self.buy()
'''
    hints = patterns.scan(source)
    assert [h["line"] for h in hints] == [4]
    assert "has not happened yet" in hints[0]["pattern"]
