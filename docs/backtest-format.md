# Backtest input format (v0)

`euclid verify <dir>` reads four files from one directory.

## prices.csv

| Column | Meaning |
| --- | --- |
| `date` | Trading date, `YYYY-MM-DD` |
| `symbol` | Instrument identifier |
| `close` | Closing price used to mark positions |

Every symbol needs a close on every date. These dates define the backtest calendar.

## trades.csv

| Column | Meaning |
| --- | --- |
| `date` | Execution date; must be a date in `prices.csv` |
| `symbol` | Must appear in `prices.csv` |
| `quantity` | Signed: positive buys, negative sells |
| `price` | Execution price per unit |
| `fee` | Total cost charged on the trade, in currency |

## returns.csv

| Column | Meaning |
| --- | --- |
| `date` | One row per date in `prices.csv` |
| `ret` | The strategy's reported simple return for that period |

## config.json

```json
{
  "initial_capital": 1000000,
  "periods_per_year": 252,
  "risk_free_rate": 0.0,
  "cost_bps": 5,
  "reported": {
    "total_return": 0.1234,
    "volatility": 0.0876,
    "sharpe": 1.41,
    "max_drawdown": -0.0652
  },
  "tolerances": {"returns": 1e-8, "metrics": 1e-4, "fees": 1e-6}
}
```

Only `initial_capital` is required. A missing reported metric or `cost_bps` is reported as **unverified**, never as a pass.

## Constraints

Add a `constraints` object to `config.json` to have limits checked on every date:

```json
"constraints": {
  "max_gross_leverage": 1.5,
  "max_net_exposure": 1.0,
  "max_position_weight": 0.5,
  "min_cash": 0,
  "long_only": true
}
```

| Key | Holds when |
| --- | --- |
| `max_gross_leverage` | Sum of absolute position values is at most this multiple of equity |
| `max_net_exposure` | Absolute net position value is at most this multiple of equity |
| `max_position_weight` | Each symbol's absolute value is at most this fraction of equity |
| `min_cash` | Cash is at least this amount |
| `long_only` | No position is negative |

These are checked with the Z3 solver in exact rational arithmetic, using values exactly as written in the input files. A pass means the solver found no date that breaks the limit; a failure names the first violating date. An unrecognised key, or no `constraints` object, is reported as **unverified**.

## Strategy code (lookahead check)

To check for lookahead, give EUCLID the signal code. Add to `config.json`:

```json
"strategy": "strategy.py:signals"
```

or pass `--strategy path/to/file.py:function`. The file path is relative to the backtest directory.

The function receives closing prices as an array of shape `[dates, symbols]` (symbols in alphabetical order) and returns signals whose first axis is dates. The signal in row `t` may use rows up to and including `t`.

EUCLID checks it in layers:

| Layer | How | Result label |
| --- | --- | --- |
| Tracing | One run on data where every value carries the latest date it depends on | `traced` |
| Perturbation | Used when tracing is not possible: alter data after a cut-off and confirm earlier signals do not move | `tested` |
| Static scan | Reads the source for known leak patterns and reports line numbers | hints only |

Tracing works on NumPy code that operates on the array it is given. It stops, and EUCLID falls back to perturbation, when the code converts the input (pandas, `np.asarray`), branches in Python on a data value, or calls a NumPy function the tracer does not support. The report says which.

A traced leak is reported as a failure only after altering the future data actually changes a signal. If the code reads future rows but no sampled signal changes, the result is **unverified**.

Without strategy code, lookahead is reported as **unverified**. EUCLID runs the strategy file as ordinary Python, so only point it at code you trust. It does not check that the recorded trades follow from the signals.

## Conventions

- Equity on a date is cash plus positions marked at that date's close, after that date's trades.
- The first period's return is measured against `initial_capital`.
- Volatility and Sharpe use the sample standard deviation and `periods_per_year`.
- Maximum drawdown is a negative fraction (`-0.0652` means 6.52%).
- `cost_bps` is checked as a minimum: each trade's fee must be at least `|quantity x price| x cost_bps / 10000`.

## Current limits

- Long and short positions in cash instruments only: no margin interest, dividends, borrow costs, splits or multiple currencies.
- Arithmetic is floating point with stated tolerances, not exact decimals.
- Evidence records are hashed but not yet signed.
