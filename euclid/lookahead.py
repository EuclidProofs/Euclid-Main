"""Lookahead detection for signal functions.

A signal function takes a price array of shape [dates, symbols] and returns
signals whose first axis is dates. Convention: the signal in row t may use
data from rows up to and including t, and nothing later.

Three layers, strongest first:
  1. Tracing      one run on time-tagged data; exact for the code that ran.
  2. Perturbation used when the tracer cannot follow the code: alter all data
                  after a cut-off and check that earlier signals do not move.
  3. Static scan  reads the source for known leak patterns; produces hints
                  with line numbers. Hints never decide the result.
"""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .model import Check, Status, Strength
from .tracer import TimeTagged, Untraceable, tag_by_row

NAME = "lookahead"
CLAIM = "Every signal uses only data available on or before its own date"
N_CUTS = 24
SEEDS = (11, 23)
CONFIRM_ROWS = 20
ENGINE_CUTS = 12  # event-driven engines are slow to re-run, so sample fewer dates
ENGINE_SEEDS = (11,)  # and alter the data once per date


# ---- loading ----------------------------------------------------------------

def load_signal_function(spec: str, base: Path) -> tuple[Callable, Path]:
    """Load `file.py:function`, with the file resolved relative to `base`."""
    file_part, _, func_name = spec.partition(":")
    path = Path(file_part)
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise FileNotFoundError(f"strategy file not found: {path}")
    module_spec = importlib.util.spec_from_file_location(f"euclid_strategy_{path.stem}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    func_name = func_name or "signals"
    if not hasattr(module, func_name):
        raise AttributeError(f"{path.name} has no function named {func_name!r}")
    return getattr(module, func_name), path


# ---- static scan ------------------------------------------------------------

def _negative_number(node: ast.AST | None) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float))
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value < 0


def _kw(call: ast.Call, name: str) -> ast.AST | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def _const(node: ast.AST | None) -> Any:
    return node.value if isinstance(node, ast.Constant) else None


def static_scan(source: str) -> list[dict[str, Any]]:
    """Find well-known lookahead patterns. Heuristic: may miss leaks or flag safe code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    hints = []

    def add(node: ast.AST, pattern: str) -> None:
        hints.append({"line": node.lineno, "pattern": pattern,
                      "code": lines[node.lineno - 1].strip()})

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        attr, first = node.func.attr, (node.args[0] if node.args else None)
        second = node.args[1] if len(node.args) > 1 else None
        if attr == "shift" and _negative_number(first or _kw(node, "periods")):
            add(node, "negative shift pulls future rows back")
        elif attr == "roll" and _negative_number(second or _kw(node, "shift")):
            add(node, "negative roll pulls future rows back")
        elif attr == "rolling" and _const(_kw(node, "center")) is True:
            add(node, "centred rolling window includes future rows")
        elif attr in ("bfill", "backfill"):
            add(node, "backward fill copies future values into the past")
        elif attr == "fillna" and _const(_kw(node, "method")) in ("bfill", "backfill"):
            add(node, "backward fill copies future values into the past")
        elif attr == "convolve" and "same" in (_const(_kw(node, "mode")),
                                               _const(node.args[2] if len(node.args) > 2 else None)):
            add(node, "mode='same' convolution is a centred window")
    return sorted(hints, key=lambda h: h["line"])


# ---- helpers ----------------------------------------------------------------

def _run_plain(func: Callable, close: np.ndarray) -> np.ndarray:
    out = np.asarray(func(close.copy()), dtype=float)
    if out.ndim == 0 or out.shape[0] != close.shape[0]:
        raise ValueError(f"signal function returned shape {out.shape}; "
                         f"its first axis must have {close.shape[0]} rows")
    return out


def _changed_rows(a: np.ndarray, b: np.ndarray, upto: int) -> np.ndarray:
    """Row indexes <= upto where two signal arrays differ."""
    same = np.isclose(a[:upto + 1], b[:upto + 1], rtol=1e-9, atol=1e-12, equal_nan=True)
    return np.flatnonzero(~same.reshape(upto + 1, -1).all(axis=1))


def _perturb_after(close: np.ndarray, cut: int, seed: int) -> np.ndarray:
    """Copy of the prices with every row after `cut` multiplied by random noise."""
    out = close.copy()
    tail = out[cut + 1:]
    rng = np.random.default_rng(seed * 100_003 + cut)
    out[cut + 1:] = tail * np.exp(rng.normal(0.0, 0.2, tail.shape))
    return out


def _row_moves(func: Callable, close: np.ndarray, baseline: np.ndarray, row: int, cut: int,
               seeds: tuple = SEEDS) -> bool:
    """Does the signal in `row` change when data after `cut` is altered?"""
    for seed in seeds:
        out = _run_plain(func, _perturb_after(close, cut, seed))
        if row in _changed_rows(baseline, out, row):
            return True
    return False


def _furthest_dependency(func: Callable, close: np.ndarray, baseline: np.ndarray,
                         row: int, known_cut: int, seeds: tuple = SEEDS) -> int:
    """Largest row index the signal in `row` is seen to depend on (binary search)."""
    lo, hi = known_cut, close.shape[0] - 2  # invariant: altering rows > lo moves the signal
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _row_moves(func, close, baseline, row, mid, seeds):
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def _check(status: Status, strength: Strength, method: str, detail: str,
           evidence: dict[str, Any]) -> Check:
    return Check(NAME, CLAIM, method, status, strength, detail, evidence)


def _hint_text(hints: list[dict[str, Any]]) -> str:
    if not hints:
        return ""
    h = hints[0]
    more = f" (+{len(hints) - 1} more)" if len(hints) > 1 else ""
    return f"; source hint, line {h['line']}: {h['pattern']}{more}"


# ---- the check --------------------------------------------------------------

def check_lookahead(func: Callable, close: np.ndarray, dates: list[str],
                    source: str = "", trace: bool = True, n_cuts: int = N_CUTS,
                    seeds: tuple = SEEDS, cache: dict | None = None) -> Check:
    close = np.asarray(close, dtype=float)
    n_rows = close.shape[0]
    hints = static_scan(source)
    evidence: dict[str, Any] = {"static_hints": hints, "rows": n_rows}

    try:
        baseline = _run_plain(func, close)
        repeat = _run_plain(func, close)
        if cache is not None:
            cache["baseline"] = baseline
    except Exception as exc:  # noqa: BLE001
        return _check(Status.ERROR, Strength.NONE, "none",
                      f"The signal function failed to run: {type(exc).__name__}: {exc}", evidence)
    if len(_changed_rows(baseline, repeat, n_rows - 1)):
        return _check(Status.UNVERIFIED, Strength.NONE, "none",
                      "The signal function gives different results on identical input, "
                      "so lookahead cannot be assessed", evidence)

    # Layer 1: tracing.
    untraceable_reason = None
    try:
        if not trace:
            raise Untraceable("this kind of strategy is event-driven and cannot be traced")
        traced = func(tag_by_row(close))
        if not isinstance(traced, TimeTagged):
            untraceable_reason = "the function's result was not derived from the traced input"
        elif traced.shape != baseline.shape or len(
                _changed_rows(baseline, traced.values.astype(float), n_rows - 1)):
            untraceable_reason = "traced run did not reproduce the normal run"
    except Exception as exc:  # noqa: BLE001 - includes Untraceable
        untraceable_reason = str(exc) or type(exc).__name__

    if untraceable_reason is None:
        row_index = np.arange(n_rows).reshape((-1,) + (1,) * (traced.tags.ndim - 1))
        ahead = (traced.tags - row_index).reshape(n_rows, -1).max(axis=1)  # per row
        evidence.update({"method": "trace",
                         "max_lookahead_rows": int(max(ahead.max(), 0))})
        leaking = np.flatnonzero(ahead > 0)
        if not len(leaking):
            return _check(
                Status.VERIFIED, Strength.TRACED, "time_provenance_trace",
                f"All {baseline.size} signal values traced in one run; none depends on data "
                f"after its own date", evidence)

        # Tags show structural dependence. Confirm that the value really moves.
        sample = leaking[np.linspace(0, len(leaking) - 1, min(CONFIRM_ROWS, len(leaking))).astype(int)]
        for row in sample:
            if row < n_rows - 1 and _row_moves(func, close, baseline, int(row), int(row), seeds):
                source_row = int(row + ahead[row])
                evidence["counterexample"] = {
                    "signal_date": dates[row], "uses_data_from": dates[source_row],
                    "rows_ahead": int(ahead[row]), "leaking_rows": int(len(leaking))}
                return _check(
                    Status.FAILED, Strength.NONE, "time_provenance_trace",
                    f"The code reads future data on {len(leaking)} of {n_rows} dates; e.g. the signal on "
                    f"{dates[row]} uses data from {dates[source_row]} "
                    f"({int(ahead[row])} row(s) ahead), confirmed by altering that data"
                    + _hint_text(hints), evidence)
        return _check(
            Status.UNVERIFIED, Strength.NONE, "time_provenance_trace",
            f"The code reads future rows on {len(leaking)} dates (up to {int(ahead.max())} "
            "row(s) ahead), but altering that data did not change any sampled signal. "
            "Review the code" + _hint_text(hints), evidence)

    # Layer 2: perturbation.
    cuts = sorted(set(np.linspace(0, n_rows - 2, min(n_cuts, n_rows - 1)).astype(int).tolist()))
    evidence.update({"method": "perturbation", "tracing_skipped_because": untraceable_reason,
                     "cut_offs_tested": len(cuts), "trials_per_cut_off": len(seeds)})
    try:
        for cut in cuts:
            for seed in seeds:
                moved = _changed_rows(baseline, _run_plain(func, _perturb_after(close, cut, seed)), cut)
                if len(moved):
                    row = int(moved[0])
                    source_row = _furthest_dependency(func, close, baseline, row, cut, seeds)
                    evidence["counterexample"] = {
                        "signal_date": dates[row], "uses_data_from": dates[source_row],
                        "rows_ahead": source_row - row, "altered_after": dates[cut]}
                    return _check(
                        Status.FAILED, Strength.NONE, "perturbation",
                        f"The signal on {dates[row]} changes when later data is altered; it "
                        f"depends on data as late as {dates[source_row]} "
                        f"({source_row - row} row(s) ahead)" + _hint_text(hints), evidence)
    except Exception as exc:  # noqa: BLE001
        return _check(Status.ERROR, Strength.NONE, "perturbation",
                      f"The signal function failed on altered data: {type(exc).__name__}: {exc}",
                      evidence)

    return _check(
        Status.VERIFIED, Strength.TESTED, "perturbation",
        f"No signal moved when later data was altered, at {len(cuts)} cut-off dates x "
        f"{len(seeds)} trial(s). Tracing was not possible: {untraceable_reason}" + _hint_text(hints),
        evidence)


@dataclass
class StrategyHandle:
    """A loaded strategy, ready to be run on data by any check."""

    func: Callable  # data array -> signals, first axis = dates
    data: np.ndarray
    dates: list[str]
    symbols: list[str]
    source: str
    hashed_file: Path
    engine: str | None = None  # None for a plain signal function
    traceable: bool = True
    n_cuts: int = N_CUTS
    seeds: tuple = SEEDS
    baseline: np.ndarray | None = None  # signals on the unaltered data, once computed

    def signals(self) -> np.ndarray:
        """Signals on the unaltered data (computed once, then reused)."""
        if self.baseline is None:
            self.baseline = np.asarray(self.func(self.data.copy()), dtype=float)
        return self.baseline
    default_execution: dict | None = None  # how this engine turns signals into trades
    open_prices: np.ndarray | None = None  # per date, when the engine fills at the open


def resolve(bt, strategy_spec: str | None) -> StrategyHandle | Check:
    """Load the strategy named in config.json or on the command line.

    Returns a handle, or a Check explaining why there is nothing to run.
    """
    spec = strategy_spec or bt.config.get("strategy")
    if not spec:
        return _check(Status.UNVERIFIED, Strength.NONE, "none",
                      "No strategy code supplied (set \"strategy\" in config.json or pass "
                      "--strategy); lookahead was not assessed", {})
    try:
        if isinstance(spec, dict):
            return _resolve_engine(bt, spec)
        func, path = load_signal_function(spec, bt.path)
    except _NoAdapter as exc:
        return _check(Status.UNVERIFIED, Strength.NONE, "none", str(exc), {})
    except Exception as exc:  # noqa: BLE001
        return _check(Status.ERROR, Strength.NONE, "none",
                      f"Could not load the strategy: {type(exc).__name__}: {exc}", {})
    return StrategyHandle(
        func=func, data=bt.prices.to_numpy(dtype=float),
        dates=[str(d.date()) for d in bt.prices.index], symbols=list(bt.prices.columns),
        source=path.read_text(), hashed_file=path)


class _NoAdapter(Exception):
    pass


def _resolve_engine(bt, spec: dict) -> StrategyHandle:
    engine = spec.get("engine")
    if engine != "backtrader":
        raise _NoAdapter(f"No adapter for strategy engine {engine!r}; lookahead was not assessed")
    from .adapters import bt_adapter

    symbols = list(bt.prices.columns)
    ohlcv_path = bt.path / spec.get("ohlcv", "ohlcv.csv")
    dates, ohlcv = bt_adapter.load_ohlcv(ohlcv_path, symbols)
    cls, source = bt_adapter.load_strategy_class(spec["class"], bt.path)
    func = bt_adapter.make_signal_function(
        cls, dates, symbols,
        data_order=spec.get("data_order"),
        cash=float(spec.get("cash", bt.initial_capital)),
        commission=float(spec.get("commission", float(bt.config.get("cost_bps", 0)) / 1e4)),
        stake=spec.get("stake"),
        params=spec.get("params"),
    )
    return StrategyHandle(
        func=func, data=ohlcv, dates=[str(d.date()) for d in dates], symbols=symbols,
        source=source, hashed_file=ohlcv_path, engine="backtrader", traceable=False,
        n_cuts=int(spec.get("cut_offs", ENGINE_CUTS)), seeds=ENGINE_SEEDS,
        # backtrader fills a market order at the next bar's open.
        default_execution={"signal": "order_units", "multiplier": 1, "delay": 1, "price": "open"},
        open_prices=ohlcv[:, :, 0],
    )


def run(handle: StrategyHandle | Check) -> Check:
    """Run the lookahead check for a resolved strategy."""
    if isinstance(handle, Check):
        return handle
    cache: dict = {}
    check = check_lookahead(handle.func, handle.data, handle.dates, handle.source,
                            trace=handle.traceable, n_cuts=handle.n_cuts, seeds=handle.seeds,
                            cache=cache)
    handle.baseline = cache.get("baseline")
    check.evidence["symbols"] = handle.symbols
    if handle.engine:
        check.evidence.update({"engine": handle.engine, "signal": "order decisions per bar"})
    return check
