import textwrap

import numpy as np
import pytest

from euclid.lookahead import check_lookahead, static_scan
from euclid.model import Status, Strength
from euclid.tracer import TimeTagged, Untraceable, tag_by_row

ROWS = 300


@pytest.fixture(scope="module")
def close():
    rng = np.random.default_rng(3)
    return 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, (ROWS, 2)), axis=0)


DATES = [f"d{i:03d}" for i in range(ROWS)]


def compile_signal(source: str):
    namespace: dict = {}
    exec(textwrap.dedent(source), namespace)
    return namespace["signals"], textwrap.dedent(source)


NUMPY_CLEAN = """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    def trailing_mean(close, window):
        out = np.full_like(close, np.nan)
        out[window - 1:] = sliding_window_view(close, window, axis=0).mean(axis=-1)
        return out

    def signals(close):
        return np.where(trailing_mean(close, 10) > trailing_mean(close, 40), 1.0, 0.0)
"""

NUMPY_LEAK = NUMPY_CLEAN.replace(
    "return np.where(trailing_mean(close, 10) > trailing_mean(close, 40), 1.0, 0.0)",
    "fast = np.roll(trailing_mean(close, 10), -2, axis=0)\n"
    "        return np.where(fast > trailing_mean(close, 40), 1.0, 0.0)",
)

PANDAS_CLEAN = """
    import pandas as pd

    def signals(close):
        df = pd.DataFrame(close)
        return (df.rolling(10).mean() > df.rolling(40).mean()).astype(float).to_numpy()
"""

PANDAS_LEAK = """
    import pandas as pd

    def signals(close):
        df = pd.DataFrame(close)
        tomorrow = df.shift(-1)
        return (tomorrow > df).astype(float).to_numpy()
"""

STRUCTURAL_ONLY = """
    import numpy as np

    def signals(close):
        return close + 0.0 * np.roll(close, -1, axis=0)
"""

PYTHON_BRANCH = """
    import numpy as np

    def signals(close):
        out = np.zeros(close.shape)
        for t in range(1, close.shape[0]):
            if close[t, 0] > close[t - 1, 0]:
                out[t] = 1.0
        return out
"""


def run(source, close):
    func, text = compile_signal(source)
    return check_lookahead(func, close, DATES, text)


def test_clean_numpy_strategy_is_verified_by_tracing(close):
    c = run(NUMPY_CLEAN, close)
    assert c.status is Status.VERIFIED and c.strength is Strength.TRACED
    assert c.evidence["method"] == "trace" and c.evidence["max_lookahead_rows"] == 0


def test_numpy_leak_is_found_with_exact_horizon_and_line_hint(close):
    c = run(NUMPY_LEAK, close)
    assert c.status is Status.FAILED
    assert c.evidence["counterexample"]["rows_ahead"] == 2
    assert c.evidence["static_hints"][0]["pattern"].startswith("negative roll")


def test_pandas_strategy_falls_back_to_perturbation(close):
    c = run(PANDAS_CLEAN, close)
    assert c.status is Status.VERIFIED and c.strength is Strength.TESTED
    assert c.evidence["method"] == "perturbation"
    assert "Tracing was not possible" in c.detail


def test_pandas_leak_is_caught_by_perturbation(close):
    c = run(PANDAS_LEAK, close)
    assert c.status is Status.FAILED and c.evidence["method"] == "perturbation"
    assert c.evidence["counterexample"]["rows_ahead"] == 1
    assert c.evidence["static_hints"][0]["pattern"].startswith("negative shift")


def test_future_read_that_never_changes_a_value_is_unverified_not_failed(close):
    c = run(STRUCTURAL_ONLY, close)
    assert c.status is Status.UNVERIFIED


def test_python_branch_on_data_is_not_traced_but_still_checked(close):
    c = run(PYTHON_BRANCH, close)
    assert c.status is Status.VERIFIED and c.strength is Strength.TESTED


def test_failing_function_is_an_error_not_a_pass(close):
    c = check_lookahead(lambda x: 1 / 0, close, DATES)
    assert c.status is Status.ERROR


def test_tags_follow_common_operations(close):
    x = tag_by_row(close[:, 0])
    rows = np.arange(ROWS)
    assert np.array_equal((x * 2 + 1).tags, rows)
    assert np.array_equal(np.cumsum(x).tags, rows)
    assert np.array_equal(np.diff(x).tags, rows[1:])
    assert np.array_equal(np.roll(x, -3).tags, np.roll(rows, -3))
    assert np.array_equal(np.convolve(x, np.ones(5) / 5, mode="valid").tags, rows[4:])
    centred = np.convolve(x, np.ones(5) / 5, mode="same").tags
    assert centred[10] == 12  # a centred window looks two rows ahead
    assert np.mean(x).tags == ROWS - 1  # a whole-series mean depends on the last row
    assert np.array_equal((x - np.mean(x)).tags, np.full(ROWS, ROWS - 1))
    assert np.array_equal(x[5:10].tags, rows[5:10])
    assert np.allclose((x * 2).values, close[:, 0] * 2)


def test_tracer_refuses_what_it_cannot_follow(close):
    x = tag_by_row(close)
    with pytest.raises(Untraceable):
        np.asarray(x)
    with pytest.raises(Untraceable):
        bool(x[0, 0] > 1)
    with pytest.raises(Untraceable):
        np.fft.fft(x)
    assert isinstance(x @ np.ones(2), TimeTagged)


def test_static_scan_patterns():
    hints = static_scan(textwrap.dedent("""
        a = df.shift(-1)
        b = df.rolling(5, center=True).mean()
        c = df.bfill()
        d = df.shift(1)
    """))
    assert [h["line"] for h in hints] == [2, 3, 4]
