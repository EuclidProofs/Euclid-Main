# EUCLID

**AI generates. Verification proves.**

EUCLID is independent verification for quantitative and automated trading strategies. The aim is the most efficient way to know that a strategy's code, backtest and risk numbers are correct before capital depends on them.

## Why

A strategy can run without errors, produce a smooth equity curve, and still be wrong:

- The backtest used data that would not have existed at decision time (lookahead).
- Reported returns don't reconcile with the trades, or a metric uses the wrong convention.
- Costs were left out or applied inconsistently.
- A leverage or position limit held on most days and failed on one.
- A pricing or risk function is off by a sign or a term.

None of these crash. They show up later, with real money. AI code assistants now produce strategies far faster than anyone can check them, and the tool that wrote a strategy is poorly placed to judge it.

## Quick start

```bash
git clone https://github.com/EuclidProofs/Euclid-Main.git
cd Euclid-Main
pip install -e .

python examples/make_sample.py          # writes a clean and a flawed sample backtest
euclid verify examples/sample_clean     # every check passes
euclid verify examples/sample_flawed    # five planted errors, six failed checks
```

Output on the flawed sample (shortened):

```text
[FAIL] returns_reconcile
       result: 1 of 500 days differ; worst on 2023-12-18: reported 0.005000, rebuilt 0.000000
[FAIL] metric_sharpe
       result: Recomputed -0.419, reported -0.505; the reported value instead matches:
               per-period returns, 0% risk-free rate, sample standard deviation,
               annualised with 365 periods
[FAIL] costs_applied
       result: 9 of 27 trades carry less than the declared cost
[FAIL] constraint_max_position_weight
       result: Broken in 1 case(s); first: 2024-02-26 AAA: weight 0.6708
[FAIL] lookahead
       result: The code reads future data on 491 of 500 dates; e.g. the signal on 2023-03-23
               uses data from 2023-03-27 (2 row(s) ahead); source hint, line 14

Verdict: FAILED
```

Each run also writes an evidence file with the hashes of every input and the full result of every check. The exit code is 0 for verified, 1 for failed and 2 for partially verified, so it can gate a CI job.

To check your own backtest, see [docs/backtest-format.md](docs/backtest-format.md).

## What EUCLID checks

| Area | Checks | Status |
| --- | --- | --- |
| Backtest integrity | Rebuild returns from trades and prices; recompute total return, volatility, Sharpe and drawdown; confirm declared costs were applied | Available |
| Metric conventions | Declare how a Sharpe ratio was computed; when a number doesn't match, EUCLID names the definition that does | Available |
| Portfolio constraints | Gross leverage, net exposure, position weight, minimum cash and long-only, checked on every date with the failing date when they don't hold | Available |
| Lookahead | Trace every signal back to the latest data it depends on; fall back to altering future data when code can't be traced | Available |
| backtrader strategies | Lookahead check for event-driven `Strategy` classes | Available |
| Execution | Re-run the strategy and confirm the recorded trades are the ones the signals call for, at the declared price | Available |
| Automated trading rules | Order sizing, loss limits and kill-switch logic hold in every reachable state | Planned |
| Pricing and risk math | Reference implementations, identities such as put-call parity, sensitivities against finite differences | Planned |

## Tested against a real engine

EUCLID was run on backtrader's built-in moving-average crossover strategy over 20 years of Oracle daily prices (5,036 trading days, 195 trades). It rebuilt backtrader's daily returns from the trade record to 8 decimal places on every day, and reproduced its total return, maximum drawdown and Sharpe ratio. The script is in [examples/real/run_backtrader.py](examples/real/run_backtrader.py).

This is one long-only strategy on one instrument. Multi-asset and short-selling strategies have not been tested yet.

## How it works

1. **Submit** strategy code, backtest output (returns, trades), prices, and a file of constraints and conventions.
2. **Freeze** every input by hash, so each verdict refers to exact versions.
3. **Decompose** the work into small, specific claims.
4. **Route** each claim to the strongest method that can handle it.
5. **Check** each claim and record the method, the result and the evidence.
6. **Report** a verdict under fail-safe rules, with an evidence record tied to the input hashes.

## What a result means

Every result carries a status and a strength label. A test is never reported as a proof.

| Label | Method | Establishes | Does not establish |
| --- | --- | --- | --- |
| `proven` | Formal proof (Lean 4) | The property holds for all inputs, under stated definitions | That the definitions match your intent |
| `solver_checked` | Constraint solving (Z3) | The constraints hold, or here is a counterexample | Anything outside the encoded constraints |
| `recomputed` | Independent recalculation | The reported number matches the inputs | That the inputs are right |
| `traced` | Time-provenance tracing | The property holds for the code that ran, on this data | Other code paths or other data |
| `tested` | Numerical and perturbation tests | Behaviour is correct on the cases tried | Behaviour on untried cases |
| `unverified` | No applicable method, or the check did not finish | Nothing | Reported as a gap, never as a pass |

The overall strength of a verdict is the weakest strength among its parts.

## Principles

- **Independent.** The checker is separate from whatever wrote the strategy. No language model decides a verdict.
- **Evidence-first.** Every result ships with what is needed to re-check it.
- **Honest about strength.** Proofs, solver checks, recomputations and tests are labelled as what they are.
- **Fail safe.** A missing check, a timeout or an error is never a pass.
- **Efficient.** Cheapest check first, and fast enough to run on every commit.

## What EUCLID does not claim

- That a strategy is profitable, or will be.
- That a backtest is free of overfitting. That is a separate statistical question.
- That a whole strategy is verified when only some properties were checked. Every verdict states its scope.

## Roadmap

- [x] Core data model and fail-safe verdict rules
- [x] Recompute engine: return reconciliation, performance metrics, costs
- [x] Constraint engine (Z3): portfolio limits on every date
- [x] Lookahead detection: tracing, perturbation and static hints
- [x] CLI (`euclid verify`) with CI-friendly exit codes
- [x] First run against a real engine (backtrader)
- [x] Check that recorded trades follow from the signals
- [ ] Multi-asset and short-selling strategies
- [ ] Trading rules checked over every reachable state
- [ ] Pricing and risk checks
- [ ] Lean 4 library of reusable lemmas
- [ ] Signed evidence records
- [ ] Adapters for more backtesting frameworks

## Get involved

We are looking for quant researchers and trading teams who want to try EUCLID on their own strategies and tell us where it falls short. Open an issue to get in touch.

---

*Mission: make computational trust independently verifiable.*
