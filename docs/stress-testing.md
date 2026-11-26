# Stress testing a strategy

`euclid verify` audits one backtest: it tells you whether what happened was recorded and reported correctly. It cannot tell you what the strategy would do in conditions that backtest never met.

`euclid stress` looks for bugs in the strategy itself. It re-runs the strategy many times, breaking the broker in one way each time, and watches what the strategy does next.

```bash
euclid stress path/to/backtest            # 12 fault scenarios
euclid stress path/to/backtest --scenarios 30
```

It needs a strategy that runs in a supported engine (backtrader for now), declared in `config.json` as described in [backtest-format.md](backtest-format.md).

## Faults injected

Each scenario injects exactly one fault, so a failure points at one cause.

| Fault | What happens |
| --- | --- |
| Refused order | One of the strategy's orders is rejected by the broker |
| Part-filled order | One order fills half its size, and the rest one bar later |

Faults are spread across the first 80% of the strategy's orders, two refusals for every part-fill.

## What is checked in every scenario

| Check | Holds when |
| --- | --- |
| `stress_solvency` | Equity stays above zero |
| `stress_limits` | Every limit declared in `constraints` still holds (limits already broken in the normal run are skipped) |
| `stress_recovery` | Positions come back to the path the strategy intended. "Back" means within 10% of the normal run's positions on at least 95% of the remaining bars. Scenarios with fewer than three later orders are not judged |

A strategy that reads its real position from the broker recovers on its own: after a refused order it is out of step until its next signal, then back in line. A strategy that keeps its own record of what it holds, and updates that record without checking the fill, never recovers. That is the bug this check is built to find.

## Source hints

The report ends with hints from a scan of the strategy source. They are not part of the verdict.

| Pattern | Why it matters |
| --- | --- |
| A `self.` attribute is updated right after an order is placed | The strategy may be recording a position it does not have |
| A positive index on a price line, such as `self.data.close[1]` | Reads a bar that has not happened yet |

## What a result means

Every stress result is labelled `tested`. A pass says the strategy survived the faults that were tried. It does not say the strategy has no bugs: other faults (price gaps, missing bars, several failures at once) are not injected yet, and only a sample of orders is hit.

Each scenario is a full run of the strategy, so the time taken is roughly the number of scenarios plus one, times the time of one run.
