import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from euclid.cli import app
from euclid.loader import BacktestFormatError, load_backtest
from euclid.model import Check, Status, Strength, VerdictStatus, aggregate
from euclid import constraints, recompute


def run_all(bt):
    return recompute.run_all(bt) + constraints.run_all(bt)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("make_sample", ROOT / "examples" / "make_sample.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path_factory.mktemp("samples")
    prices = mod.make_prices()
    trades = mod.make_trades(prices)
    mod.write(out / "clean", prices, trades, mod.returns_for(prices, trades), mod.PPY)
    flawed, r = mod.make_flawed(prices, trades)
    mod.write(out / "flawed", prices, flawed, r, 365)
    return out


def by_name(checks):
    return {c.name: c for c in checks}


def test_clean_backtest_is_verified(samples):
    checks = run_all(load_backtest(samples / "clean"))
    assert all(c.status is Status.VERIFIED for c in checks), [c.detail for c in checks]
    verdict = aggregate(checks, "test")
    assert verdict.status is VerdictStatus.VERIFIED
    assert verdict.strength is Strength.RECOMPUTED  # weakest of recomputed and solver_checked
    solver = [c for c in checks if c.name.startswith("constraint_")]
    assert len(solver) == 4 and all(c.strength is Strength.SOLVER_CHECKED for c in solver)


def test_flawed_backtest_flags_exactly_the_planted_errors(samples):
    checks = by_name(run_all(load_backtest(samples / "flawed")))
    failed = {n for n, c in checks.items() if c.status is Status.FAILED}
    assert failed == {"returns_reconcile", "costs_applied", "metric_sharpe",
                      "constraint_max_position_weight"}
    weight = checks["constraint_max_position_weight"]
    assert weight.evidence["violations"] == 1
    assert weight.evidence["counterexample"]["symbol"] == "AAA"
    assert checks["returns_reconcile"].evidence["mismatched_days"] == 1
    assert "365" in checks["metric_sharpe"].detail
    assert aggregate(list(checks.values()), "test").status is VerdictStatus.FAILED


def test_missing_reported_metric_is_a_gap_not_a_pass(samples, tmp_path):
    src = samples / "clean"
    for f in ("prices.csv", "trades.csv", "returns.csv"):
        (tmp_path / f).write_bytes((src / f).read_bytes())
    config = json.loads((src / "config.json").read_text())
    del config["reported"]["sharpe"]
    del config["cost_bps"]
    del config["constraints"]
    (tmp_path / "config.json").write_text(json.dumps(config))

    checks = by_name(run_all(load_backtest(tmp_path)))
    assert checks["metric_sharpe"].status is Status.UNVERIFIED
    assert checks["costs_applied"].status is Status.UNVERIFIED
    assert checks["constraints"].status is Status.UNVERIFIED
    assert aggregate(list(checks.values()), "t").status is VerdictStatus.PARTIALLY_VERIFIED


def _check(status, strength=Strength.RECOMPUTED):
    return Check("n", "c", "m", status, strength, "d")


def test_aggregate_fail_safe_rules():
    assert aggregate([], "s").status is VerdictStatus.PARTIALLY_VERIFIED
    assert aggregate([_check(Status.VERIFIED), _check(Status.ERROR)], "s").status \
        is VerdictStatus.PARTIALLY_VERIFIED
    assert aggregate([_check(Status.VERIFIED), _check(Status.FAILED)], "s").status \
        is VerdictStatus.FAILED
    mixed = aggregate([_check(Status.VERIFIED, Strength.PROVEN),
                       _check(Status.VERIFIED, Strength.TESTED)], "s")
    assert mixed.status is VerdictStatus.VERIFIED
    assert mixed.strength is Strength.TESTED  # weakest wins


def test_loader_rejects_incomplete_directory(tmp_path):
    with pytest.raises(BacktestFormatError):
        load_backtest(tmp_path)


def test_cli_exit_codes_and_evidence(samples, tmp_path):
    runner = CliRunner()
    ok = runner.invoke(app, ["verify", str(samples / "clean"), "--out", str(tmp_path / "ok.json")])
    assert ok.exit_code == 0, ok.output
    bad = runner.invoke(app, ["verify", str(samples / "flawed"), "--out", str(tmp_path / "bad.json")])
    assert bad.exit_code == 1, bad.output
    evidence = json.loads((tmp_path / "bad.json").read_text())
    assert evidence["verdict"]["status"] == "failed"
    assert set(evidence["inputs_sha256"]) == {"prices.csv", "trades.csv", "returns.csv", "config.json"}


def test_unknown_constraint_is_a_gap_and_short_position_is_caught(samples, tmp_path):
    src = samples / "clean"
    for f in ("prices.csv", "returns.csv"):
        (tmp_path / f).write_bytes((src / f).read_bytes())
    trades = (src / "trades.csv").read_text().splitlines()
    first_date = (src / "prices.csv").read_text().splitlines()[1].split(",")[0]
    trades.insert(1, f"{first_date},BBB,-10.0,50.0,0.0")
    (tmp_path / "trades.csv").write_text("\n".join(trades) + "\n")
    config = json.loads((src / "config.json").read_text())
    config["constraints"] = {"long_only": True, "max_turnover": 3}
    (tmp_path / "config.json").write_text(json.dumps(config))

    checks = by_name(constraints.run_all(load_backtest(tmp_path)))
    assert checks["constraint_max_turnover"].status is Status.UNVERIFIED
    assert checks["constraint_long_only"].status is Status.FAILED
    assert checks["constraint_long_only"].evidence["counterexample"]["symbol"] == "BBB"
