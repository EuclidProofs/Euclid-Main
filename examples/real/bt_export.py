"""Shared helper: run a backtrader Cerebro and export what the engine did, and
what it reported, in EUCLID's input format. Works for one or many instruments.
"""

from __future__ import annotations

import contextlib
import io
import json
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
import backtrader as bt  # noqa: E402
import pandas as pd  # noqa: E402


class Recorder(bt.Analyzer):
    """Record bars and executed orders exactly as the engine saw them."""

    def start(self):
        self.bars, self.trades, self.rejected = [], [], []

    def next(self):
        date = self.strategy.datetime.date(0).isoformat()
        for d in self.datas:
            self.bars.append((date, d._name, d.open[0], d.high[0], d.low[0], d.close[0],
                              d.volume[0]))

    def notify_order(self, order):
        if order.status == order.Completed:
            self.trades.append({
                "date": bt.num2date(order.executed.dt).date().isoformat(),
                "symbol": order.data._name,
                "quantity": float(order.executed.size),
                "price": float(order.executed.price),
                "fee": float(order.executed.comm),
            })
        elif order.status in (order.Margin, order.Rejected, order.Canceled, order.Expired):
            self.rejected.append({
                "created": bt.num2date(order.created.dt).date().isoformat(),
                "symbol": order.data._name, "size": float(order.created.size),
                "status": order.getstatusname(),
            })

    def get_analysis(self):
        return {"bars": self.bars, "trades": self.trades, "rejected": self.rejected}


def run_and_export(cerebro: bt.Cerebro, out_dir: str, *, cash: float, commission: float,
                   strategy_config: dict, sharpe_mode: str = "default",
                   constraints: dict | None = None, source: dict | None = None,
                   quiet: bool = True) -> dict:
    """Add analyzers, run, and write prices/trades/returns/ohlcv/config."""
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addanalyzer(Recorder, _name="rec")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="ret", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    if sharpe_mode == "daily":
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days,
                            annualize=True, riskfreerate=0.0)
        sharpe_convention = {"returns": "periodic", "risk_free_rate": 0.0, "annualize": True,
                             "std": "population"}
    else:
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        sharpe_convention = "backtrader_default"

    with contextlib.redirect_stdout(io.StringIO() if quiet else None):
        strat = cerebro.run()[0]
    final_value = cerebro.broker.getvalue()
    rec = strat.analyzers.rec.get_analysis()
    returns = strat.analyzers.ret.get_analysis()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bars = pd.DataFrame(rec["bars"], columns=["date", "symbol", "open", "high", "low", "close",
                                              "volume"])
    bars.to_csv(out / "ohlcv.csv", index=False)
    bars[["date", "symbol", "close"]].to_csv(out / "prices.csv", index=False)
    pd.DataFrame(rec["trades"], columns=["date", "symbol", "quantity", "price", "fee"]).to_csv(
        out / "trades.csv", index=False)
    pd.DataFrame({"date": [d.date().isoformat() for d in returns], "ret": list(returns.values())}
                 ).to_csv(out / "returns.csv", index=False)

    config = {
        "initial_capital": cash,
        "periods_per_year": 252,
        "risk_free_rate": 0.0,
        "cost_bps": commission * 1e4,
        "constraints": constraints or {},
        "reported": {
            "total_return": final_value / cash - 1.0,
            "sharpe": strat.analyzers.sharpe.get_analysis()["sharperatio"],
            "max_drawdown": -strat.analyzers.dd.get_analysis()["max"]["drawdown"] / 100.0,
        },
        "sharpe_convention": sharpe_convention,
        "strategy": strategy_config,
        "source": {"engine": f"backtrader {bt.__version__}", **(source or {})},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    summary = {"bars": bars["date"].nunique(), "trades": len(rec["trades"]),
               "orders_not_filled": len(rec["rejected"]), "final_value": round(final_value, 2),
               "reported": config["reported"]}
    print(json.dumps(summary))
    return summary
