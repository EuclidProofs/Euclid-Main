"""Run backtrader's built-in MA_CrossOver strategy, unmodified, on real prices,
and export what it did and what it reported in EUCLID's input format.

    python run_backtrader.py <yahoo-csv> <out-dir> [default|daily]

`default` exports the Sharpe ratio from backtrader's SharpeRatio analyzer with
its default settings; `daily` uses daily returns, annualised, zero risk-free.
"""

import json
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
import backtrader as bt  # noqa: E402
import pandas as pd  # noqa: E402

CASH = 100_000.0
COMMISSION = 0.001  # 10 bps of traded value
STAKE = 2000


class Recorder(bt.Analyzer):
    """Record closes and executed orders exactly as the engine saw them."""

    def start(self):
        self.closes, self.trades, self.bars = [], [], []

    def next(self):
        d = self.data
        self.closes.append((self.strategy.datetime.date(0).isoformat(), d.close[0]))
        self.bars.append((self.strategy.datetime.date(0).isoformat(), d.open[0], d.high[0],
                          d.low[0], d.close[0], d.volume[0]))

    def notify_order(self, order):
        if order.status == order.Completed:
            self.trades.append({
                "date": bt.num2date(order.executed.dt).date().isoformat(),
                "symbol": "ORCL",
                "quantity": float(order.executed.size),
                "price": float(order.executed.price),
                "fee": float(order.executed.comm),
            })

    def get_analysis(self):
        return {"closes": self.closes, "trades": self.trades, "bars": self.bars}


def main(csv_path: str, out_dir: str, sharpe_mode: str = "default") -> None:
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.YahooFinanceCSVData(dataname=csv_path))
    cerebro.addstrategy(bt.strategies.MA_CrossOver)
    cerebro.broker.setcash(CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.addsizer(bt.sizers.FixedSize, stake=STAKE)

    cerebro.addanalyzer(Recorder, _name="rec")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="ret", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    if sharpe_mode == "daily":
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days,
                            annualize=True, riskfreerate=0.0)
    else:
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")

    strat = cerebro.run()[0]
    final_value = cerebro.broker.getvalue()
    rec = strat.analyzers.rec.get_analysis()
    returns = strat.analyzers.ret.get_analysis()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    closes = pd.DataFrame(rec["closes"], columns=["date", "close"])
    closes.insert(1, "symbol", "ORCL")
    closes.to_csv(out / "prices.csv", index=False)
    pd.DataFrame(rec["trades"], columns=["date", "symbol", "quantity", "price", "fee"]).to_csv(
        out / "trades.csv", index=False)
    pd.DataFrame({"date": [d.date().isoformat() for d in returns], "ret": list(returns.values())}
                 ).to_csv(out / "returns.csv", index=False)

    pd.DataFrame(rec["bars"], columns=["date", "open", "high", "low", "close", "volume"]).to_csv(
        out / "ohlcv.csv", index=False)

    config = {
        "initial_capital": CASH,
        "periods_per_year": 252,
        "risk_free_rate": 0.0,
        "cost_bps": COMMISSION * 1e4,
        "constraints": {"long_only": True, "min_cash": 0, "max_position_weight": 1.0,
                        "max_gross_leverage": 1.0},
        "reported": {
            "total_return": final_value / CASH - 1.0,
            "sharpe": strat.analyzers.sharpe.get_analysis()["sharperatio"],
            "max_drawdown": -strat.analyzers.dd.get_analysis()["max"]["drawdown"] / 100.0,
        },
        "sharpe_convention": ("backtrader_default" if sharpe_mode == "default" else
                              {"returns": "periodic", "risk_free_rate": 0.0, "annualize": True,
                               "std": "population"}),
        "strategy": {"engine": "backtrader", "class": "backtrader.strategies:MA_CrossOver",
                     "ohlcv": "ohlcv.csv", "stake": STAKE},
        "source": {"engine": f"backtrader {bt.__version__}", "strategy": "MA_CrossOver (built-in)",
                   "data": Path(csv_path).name, "sharpe_settings": sharpe_mode},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{len(closes)} bars, {len(rec['trades'])} trades, final value {final_value:,.2f}")
    print("reported:", json.dumps(config["reported"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
