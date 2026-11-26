# Stress testing a strategy

`euclid verify` audits one backtest: it tells you whether what happened was recorded and reported correctly. It cannot tell you what the strategy would do in conditions that backtest never met.

`euclid stress` looks for bugs in the strategy itself. It re-runs the strategy many times, breaking the broker in one way each time, and watches what the strategy does next.

```bash
euclid stress path/to/backtest                  # 12 fault scenarios
euclid stress path/to/backtest --scenarios 30   # more scenarios, more orders hit
euclid stress path/to/backtest --gap 0.25       # larger price gaps (default 0.15)
```

It needs a strategy that runs in a supported engine (backtrader for now), declared in `config.json` as described in [backtest-format.md](backtest-format.md).

## Faults injected

Each scenario injects one fault, so a failure points at one cause. Faults are placed at orders spread across the first 80% of the strategy's orders, cycling through the kinds below.

| Fault | What happens | Share of scenarios |
| --- | --- | --- |
| Refused order | One order is rejected by the broker | 2 in 6 |
| Part-filled order | One order fills half its size, and the rest one bar later | 1 in 6 |
| Two refusals in a row | An order and the next one are both rejected | 1 in 6 |
| Price gap | On the day an order would fill, that instrument's prices jump by 15% against the order (up before a buy, down before a sell) and stay at the new level | 1 in 6 |
| Missing bar | On the day an order would fill, that instrument has no bar at all | 1 in 6 |

## What is checked in every scenario

| Check | Holds when | Judged on |
| --- | --- | --- |
| `stress_completes` | The strategy runs to the end without raising an error | All faults |
| `stress_solvency` | Equity stays above zero | All faults |
| `stress_limits` | Every limit declared in `constraints` still holds (limits already broken in the normal run are skipped) | All faults |
| `stress_recovery` | Positions come back to the path the strategy intended | Order faults only |

"Back" in `stress_recovery` means within 10% of the normal run's positions on at least 95% of the remaining bars. Scenarios with fewer than three later orders are not judged.

Recovery is judged only for refused and part-filled orders. Those faults leave the market data untouched, so the normal run is a fair reference for where the strategy meant to be. A price gap or a missing bar changes what the strategy sees, so its positions may rightly differ afterwards.

A strategy that reads its real position from the broker recovers on its own: after a refused order it is out of step until its next signal, then back in line. A strategy that keeps its own record of what it holds, and updates that record without checking the fill, never recovers. That is the bug this check is built to find. Two refusals in a row can cancel out for a strategy that flips back and forth, so a pass on that fault alone means little.

## Source hints

The report ends with hints from a scan of the strategy source. They are not part of the verdict.

| Pattern | Why it matters |
| --- | --- |
| A `self.` attribute is updated right after an order is placed | The strategy may be recording a position it does not have |
| A positive index on a price line, such as `self.data.close[1]` | Reads a bar that has not happened yet |

## What a result means

Every stress result is labelled `tested`. A pass says the strategy survived the faults that were tried. It does not say the strategy has no bugs: only a sample of orders is hit, each scenario has a single fault, and other kinds (delayed fills, bad ticks that revert, exchange halts) are not injected yet. A limit broken by a price gap may be a market move rather than a bug; the report names the scenario so you can judge.

Each scenario is a full run of the strategy, so the time taken is roughly the number of scenarios plus one, times the time of one run.
