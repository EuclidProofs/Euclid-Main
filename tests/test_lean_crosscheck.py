"""Model versus code: the tracer agrees with the Lean model on random signals.

Runs only where a `lean` binary is installed; skipped otherwise.
"""
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(shutil.which("lean") is None, reason="needs a Lean 4 toolchain on PATH")
def test_tracer_matches_lean_model():
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "lean", "crosscheck.py"), "--signals", "300", "--seed", "11"],
        capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
    assert "tag disagreements:      0" in out.stdout
    assert "PASS" in out.stdout
