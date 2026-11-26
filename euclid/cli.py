"""Command-line interface: `euclid verify <backtest-dir>`."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .loader import BacktestFormatError, _sha256, load_backtest
from .model import Status, Verdict, VerdictStatus, aggregate
from . import constraints, execution, lookahead, recompute

app = typer.Typer(add_completion=False, help="Independent verification for trading strategies.")

_MARK = {
    Status.VERIFIED: "PASS",
    Status.FAILED: "FAIL",
    Status.UNVERIFIED: "GAP ",
    Status.ERROR: "ERR ",
}
_EXIT = {
    VerdictStatus.VERIFIED: 0,
    VerdictStatus.FAILED: 1,
    VerdictStatus.PARTIALLY_VERIFIED: 2,
}


def build_evidence(verdict: Verdict, input_hashes: dict[str, str]) -> dict:
    return {
        "euclid_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs_sha256": input_hashes,
        "verdict": verdict.to_dict(),
    }


def print_report(verdict: Verdict) -> None:
    typer.echo("")
    for c in verdict.checks:
        typer.echo(f"  [{_MARK[c.status]}] {c.name}")
        typer.echo(f"         claim:  {c.claim}")
        typer.echo(f"         result: {c.detail}")
    typer.echo("")
    s = verdict.summary
    line = f"  Verdict: {verdict.status.value.upper()}"
    if verdict.status is not VerdictStatus.FAILED:
        line += f"  (strength of verified checks: {verdict.strength.value})"
    typer.echo(line)
    typer.echo(
        f"  {s['verified']} verified, {s['failed']} failed, "
        f"{s['unverified']} unverified, {s['error']} errors"
    )
    typer.echo(f"  Scope: {verdict.scope}")
    typer.echo("")


@app.callback()
def main() -> None:
    """EUCLID command-line tool."""


@app.command()
def verify(
    backtest: Path = typer.Argument(..., help="Directory with prices, trades, returns and config."),
    out: Optional[Path] = typer.Option(None, "--out", help="Where to write the evidence JSON."),
    strategy: Optional[str] = typer.Option(
        None, "--strategy", help="Signal code to check for lookahead, as file.py:function."),
) -> None:
    """Verify a backtest and write an evidence record."""
    try:
        bt = load_backtest(backtest)
    except BacktestFormatError as exc:
        typer.echo(f"Cannot load backtest: {exc}", err=True)
        raise typer.Exit(code=3)

    handle = lookahead.resolve(bt, strategy)
    if isinstance(handle, lookahead.StrategyHandle):
        bt.input_hashes[handle.hashed_file.name] = _sha256(handle.hashed_file)
    checks = (recompute.run_all(bt) + constraints.run_all(bt)
              + [lookahead.run(handle)] + execution.run_all(bt, handle))
    scope = (
        f"{len(checks)} checks on {len(bt.reported_returns)} periods and "
        f"{len(bt.trades)} trades; annualisation {bt.periods_per_year}; "
        "does not assess profitability or overfitting"
    )
    verdict = aggregate(checks, scope)
    print_report(verdict)

    out_path = out or Path(f"{bt.path.name}.evidence.json")
    out_path.write_text(json.dumps(build_evidence(verdict, bt.input_hashes), indent=2))
    typer.echo(f"  Evidence written to {out_path}")
    raise typer.Exit(code=_EXIT[verdict.status])


if __name__ == "__main__":
    app()
