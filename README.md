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

## What EUCLID checks

| Area | Checks |
| --- | --- |
| Backtest integrity | Rebuild P&L from trades and compare with reported returns; recompute Sharpe, volatility and drawdown; confirm costs were applied; detect lookahead by perturbing future data |
| Portfolio constraints | Weights, leverage, exposure and position limits hold on every rebalance date, with the failing date when they don't |
| Automated trading rules | Order sizing, loss limits and kill-switch logic hold in every reachable state, with a counterexample when they don't |
| Pricing and risk math | Compare against independent reference implementations; check identities such as put-call parity; check sensitivities against finite differences |

## How it works

1. **Submit** strategy code, backtest output (returns, trades, positions), market data, and a file of constraints and conventions.
2. **Freeze** every input by hash, so each verdict refers to exact versions.
3. **Decompose** the work into small, specific claims.
4. **Route** each claim to the strongest method that can handle it.
5. **Check** in isolated, reproducible environments with pinned tool versions.
6. **Report** a signed verdict with an evidence bundle that can be re-checked without EUCLID.

## What a result means

Every result carries a status and a strength label. A test is never reported as a proof.

| Label | Method | Establishes | Does not establish |
| --- | --- | --- | --- |
| `proven` | Formal proof (Lean 4) | The property holds for all inputs, under stated definitions | That the definitions match your intent |
| `solver_checked` | Constraint solving (Z3) | The constraints hold, or here is a counterexample | Anything outside the encoded constraints |
| `recomputed` | Exact independent recalculation | The reported number matches the inputs | That the inputs are right |
| `traced` | Time-provenance tracing | The property holds for the code that ran, on this data | Other code paths or other data |
| `tested` | Numerical and property tests | Behaviour is correct on the cases tried | Behaviour on untried cases |
| `unverified` | No applicable method, or the check did not finish | Nothing | Reported as a gap, never as a pass |

The overall strength of a verdict is the weakest strength among its parts.

## Principles

- **Independent.** The checker is separate from whatever wrote the strategy. No language model decides a verdict.
- **Evidence-first.** Every result ships with what is needed to re-check it.
- **Honest about strength.** Proofs, solver checks, recomputations and tests are labelled as what they are.
- **Fail safe.** A missing check, a timeout or an error is never a pass.
- **Efficient.** Cheapest check first, cached results, and re-checking only what changed, so it can run on every commit.

## What EUCLID does not claim

- That a strategy is profitable, or will be.
- That a backtest is free of overfitting. That is a separate statistical question.
- That a whole strategy is verified when only some properties were checked. Every verdict states its scope.

## Roadmap

- [ ] Core data model: submission, claim, check, evidence, verdict
- [ ] Recompute engine: P&L reconciliation and performance metrics
- [ ] Constraint engine (Z3): portfolio limits and trading rules
- [ ] Lookahead detection
- [ ] Pricing and risk checks
- [ ] Lean 4 library of reusable lemmas
- [ ] CLI (`euclid verify`) and CI integration
- [ ] Proof of concept: an open-source strategy with planted errors, with detection, misses, false alarms and timings published

## Get involved

We are looking for quant researchers and trading teams who want to try EUCLID on their own strategies and tell us where it falls short. Open an issue to get in touch.

---

*Mission: make computational trust independently verifiable.*
