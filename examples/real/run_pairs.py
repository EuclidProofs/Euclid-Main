"""Run backtrader's contributed pairs-trading strategy, unmodified, on two real
stocks and export the result for EUCLID. Long/short, two instruments, sized by
notional value.

    python run_pairs.py <backtrader-repo> <out-dir> [portfolio_value] [fromdate] [todate]

portfolio_value is the notional the strategy sizes its two legs from. The
sample's own default is 10000 against 100000 of cash. Setting it equal to the
cash (100000) leaves no headroom, and the engine then rejects some orders.

<backtrader-repo> is a clone of https://github.com/mementum/backtrader, which
holds both the strategy (contrib/samples/pair-trading/pair-trading.py) and the
price files (datas/). Needs: pip install backtrader statsmodels
"""

import datetime
import importlib.util
import sys
from pathlib import Path

import backtrader as bt

from bt_export import run_and_export

CASH = 100_000.0
COMMISSION = 0.001
FILES = {"ORCL": "orcl-1995-2014.txt", "YHOO": "yhoo-1996-2014.txt"}  # data0, data1


def main(repo: str, out_dir: str, portfolio_value: str = "10000", fromdate: str = "2005-01-01",
         todate: str = "2014-12-31") -> None:
    repo_path = Path(repo).resolve()
    strategy_file = repo_path / "contrib" / "samples" / "pair-trading" / "pair-trading.py"
    spec = importlib.util.spec_from_file_location("pair_trading", strategy_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # backtrader looks strategy modules up by name
    spec.loader.exec_module(module)

    params = {"printout": False, "portfolio_value": float(portfolio_value)}
    window = {"fromdate": datetime.datetime.strptime(fromdate, "%Y-%m-%d"),
              "todate": datetime.datetime.strptime(todate, "%Y-%m-%d")}
    cerebro = bt.Cerebro()
    for symbol, filename in FILES.items():
        cerebro.adddata(bt.feeds.YahooFinanceCSVData(
            dataname=str(repo_path / "datas" / filename), **window), name=symbol)
    cerebro.addstrategy(module.PairTradingStrategy, **params)

    run_and_export(
        cerebro, out_dir, cash=CASH, commission=COMMISSION,
        strategy_config={"engine": "backtrader",
                         "class": f"{strategy_file}:PairTradingStrategy",
                         "ohlcv": "ohlcv.csv", "data_order": list(FILES), "params": params},
        constraints={"max_gross_leverage": 2.0, "max_position_weight": 1.0},
        source={"strategy": "PairTradingStrategy (backtrader contrib sample)",
                "data": list(FILES.values()), "window": [fromdate, todate]},
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
