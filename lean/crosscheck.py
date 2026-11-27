"""Model versus code: does euclid/tracer.py follow the rule proved in Lookahead.lean?

Lookahead.lean proves that the tag rule cannot miss a lookahead. That proof is about a model.
This script tests the remaining link: it builds random signals, computes each one twice,

  * in Lean, with the model's `tag` and `eval`, and
  * in Python, with NumPy operations running under the real `TimeTagged` tracer,

and compares the two bar by bar, tags and values. It then plants bugs in the tracer and
checks that the comparison notices each one.

Usage:  python lean/crosscheck.py [--signals 3000] [--seed 1] [--lean lean]

Needs a `lean` binary (core Lean is enough). This is a test, not a proof: agreement on
random signals is evidence that the code follows the rule, for the operations exercised.
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from euclid import tracer  # noqa: E402
from euclid.tracer import TimeTagged, tag_by_row  # noqa: E402

N = 20            # bars compared
MARGIN = 12       # extra bars so that forward shifts never run off the end of the arrays
LONG = N + MARGIN
BIG = 10 ** 12    # stands in for "minus infinity" as the start of a running maximum
EXACT = 2 ** 50   # values below this are exact in float64, so they can be compared exactly

# ----------------------------------------------------------------------------- random signals

MAP1 = ['neg', 'abs', 'affine', 'square']
MAP2 = ['add', 'sub', 'mul', 'max', 'min', 'gt']
FOLD = ['sum', 'max']


def gen(rng: random.Random, depth: int, budget: list[int]):
    """A random signal as a nested tuple. `budget[0]` caps the total forward shift."""
    if depth == 0 or rng.random() < 0.12:
        return ('input', rng.randrange(2)) if rng.random() < 0.85 else ('const', rng.randrange(-3, 4))
    kind = rng.choices(
        ['lag', 'lead', 'map1', 'map2', 'select', 'rolling', 'cumulative', 'whole'],
        weights=[16, 12, 10, 22, 8, 14, 9, 9])[0]
    sub = lambda: gen(rng, depth - 1, budget)  # noqa: E731
    if kind == 'lag':
        return ('lag', rng.randrange(0, 5), rng.randrange(-2, 3), rng.randrange(3), sub())
    if kind == 'lead':
        k = rng.randrange(1, 4)
        if budget[0] < k:
            return sub()
        budget[0] -= k
        return ('lead', k, rng.randrange(2), sub())
    if kind == 'map1':
        return ('map1', rng.choice(MAP1), sub())
    if kind == 'map2':
        return ('map2', rng.choice(MAP2), sub(), sub())
    if kind == 'select':
        return ('select', sub(), sub(), sub())
    if kind == 'rolling':
        return ('rolling', rng.choice(FOLD), rng.randrange(1, 7), rng.randrange(2), sub())
    if kind == 'cumulative':
        return ('cumulative', rng.choice(FOLD), sub())
    return ('whole', rng.choice(FOLD), sub())


# ----------------------------------------------------------------------------- the Lean side

def lean_expr(e) -> str:
    k = e[0]
    if k == 'input':
        return f'(.input {e[1]})'
    if k == 'const':
        return f'(.const ({e[1]}))'
    if k == 'lag':
        return f'(.lag {e[1]} ({e[2]}) {lean_expr(e[4])})'
    if k == 'lead':
        return f'(.lead {e[1]} {lean_expr(e[3])})'
    if k == 'map1':
        f = {'neg': 'fun a => -a', 'abs': 'fun a => if a < 0 then -a else a',
             'affine': 'fun a => a * 2 + 1', 'square': 'fun a => a * a'}[e[1]]
        return f'(.map1 ({f}) {lean_expr(e[2])})'
    if k == 'map2':
        f = {'add': 'fun a b => a + b', 'sub': 'fun a b => a - b', 'mul': 'fun a b => a * b',
             'max': 'fun a b => max a b', 'min': 'fun a b => min a b',
             'gt': 'fun a b => if a > b then 1 else 0'}[e[1]]
        return f'(.map2 ({f}) {lean_expr(e[2])} {lean_expr(e[3])})'
    if k == 'select':
        return f'(.select (fun c => decide (c > 0)) {lean_expr(e[1])} {lean_expr(e[2])} {lean_expr(e[3])})'
    fold = {'sum': ('fun a b => a + b', '0'), 'max': ('fun a b => max a b', f'(-{BIG})')}
    if k == 'rolling':
        f, init = fold[e[1]]
        return f'(.rolling ({f}) {init} {e[2]} {lean_expr(e[4])})'
    if k == 'cumulative':
        f, init = fold[e[1]]
        return f'(.cumulative ({f}) {init} {lean_expr(e[2])})'
    if k == 'whole':
        f, init = fold[e[1]]
        return f'(.wholeSample ({f}) {init} {N} {lean_expr(e[2])})'
    raise ValueError(k)


def run_lean(lean: str, olean_dir: str, prices: np.ndarray, exprs: list) -> list[list[tuple[int, int]]]:
    def table(col):
        return '#[' + ', '.join(str(int(v)) for v in prices[:, col]) + ']'
    lines = ['import Lookahead', 'open Euclid.Lookahead',
             f'def tbl0 : Array Int := {table(0)}', f'def tbl1 : Array Int := {table(1)}',
             'def P : Nat → Nat → Int := fun i s => if i = 0 then tbl0.getD s 0 else tbl1.getD s 0',
             f'def row (e : Expr Int) : String := "R " ++ String.intercalate " " '
             f'((List.range {N}).map (fun t => s!"{{tag e t}}:{{eval P e t}}"))']
    for n, e in enumerate(exprs):
        lines.append(f'def e{n} : Expr Int := {lean_expr(e)}')
        lines.append(f'#eval IO.println (row e{n})')
    with tempfile.NamedTemporaryFile('w', suffix='.lean', dir=olean_dir, delete=False, encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
        path = fh.name
    env = dict(os.environ, LEAN_PATH=olean_dir)
    out = subprocess.run([lean, path], capture_output=True, text=True, env=env, timeout=1800)
    os.unlink(path)
    rows = [[tuple(int(x) for x in cell.split(':')) for cell in l[2:].split()]
            for l in out.stdout.splitlines() if l.startswith('R ')]
    if out.returncode != 0 or len(rows) != len(exprs):
        raise RuntimeError('Lean failed:\n' + (out.stdout + out.stderr)[-2000:])
    return rows


# ----------------------------------------------------------------------------- the Python side

def py_eval(e, X):
    """Evaluate a signal with ordinary NumPy calls. X is the traced price table."""
    k = e[0]
    if k == 'input':
        return X[:, e[1]]
    if k == 'const':
        return np.full(LONG, float(e[1]))
    if k == 'lag':
        shift, fill, how, v = e[1], float(e[2]), e[3], py_eval(e[4], X)
        if shift == 0:
            return v
        if how == 0:
            return np.concatenate([np.full(shift, fill), v[:LONG - shift]])
        if how == 1:
            r = np.roll(v, shift)
            r[:shift] = fill
            return r
        return np.pad(v[:LONG - shift], (shift, 0), mode='constant', constant_values=fill)
    if k == 'lead':
        shift, how, v = e[1], e[2], py_eval(e[3], X)
        if how == 0:
            return np.concatenate([v[shift:], np.zeros(shift)])
        r = np.roll(v, -shift)
        r[LONG - shift:] = 0.0
        return r
    if k == 'map1':
        v = py_eval(e[2], X)
        return {'neg': lambda a: -a, 'abs': np.abs, 'affine': lambda a: a * 2 + 1,
                'square': lambda a: a * a}[e[1]](v)
    if k == 'map2':
        a, b = py_eval(e[2], X), py_eval(e[3], X)
        if e[1] == 'gt':
            return np.where(a > b, 1.0, 0.0)
        return {'add': np.add, 'sub': np.subtract, 'mul': np.multiply,
                'max': np.maximum, 'min': np.minimum}[e[1]](a, b)
    if k == 'select':
        c, a, b = py_eval(e[1], X), py_eval(e[2], X), py_eval(e[3], X)
        return np.where(c > 0, a, b)
    if k == 'rolling':
        op, w, how, v = e[1], e[2], e[3], py_eval(e[4], X)
        if op == 'sum' and how == 0:
            return np.convolve(v, np.ones(w), mode='full')[:LONG]
        pad = 0.0 if op == 'sum' else -float(BIG)
        win = sliding_window_view(np.pad(v, (w - 1, 0), mode='constant', constant_values=pad), w)
        return win.sum(axis=1) if op == 'sum' else win.max(axis=1)
    if k == 'cumulative':
        v = py_eval(e[2], X)
        return np.cumsum(v) if e[1] == 'sum' else np.maximum.accumulate(v)
    if k == 'whole':
        v = py_eval(e[2], X)
        s = np.sum(v[:N]) if e[1] == 'sum' else np.max(v[:N])
        return np.zeros(LONG) + s
    raise ValueError(k)


def run_python(prices: np.ndarray, e) -> tuple[np.ndarray, np.ndarray]:
    out = py_eval(e, tag_by_row(prices))
    if isinstance(out, TimeTagged):
        return out.values[:N], out.tags[:N]
    return np.asarray(out)[:N], np.full(N, tracer.NO_DATA)


# ----------------------------------------------------------------------------- comparison

def compare(lean_rows, py_results):
    """Returns counts and the first few disagreements."""
    tag_bars = value_bars = skipped_values = 0
    tag_bad, value_bad, verdict_bad = [], [], []
    for idx, (rows, (vals, tags)) in enumerate(zip(lean_rows, py_results)):
        for t, (ltag, lval) in enumerate(rows):
            tag_bars += 1
            # the model tags a constant 0; the tracer tags it -1. Both mean "nothing after bar 0".
            if ltag != max(int(tags[t]), 0):
                tag_bad.append((idx, t, ltag, int(tags[t])))
            if (ltag <= t) != (int(tags[t]) <= t):
                verdict_bad.append((idx, t))
            if abs(lval) < EXACT and np.isfinite(vals[t]) and abs(vals[t]) < EXACT:
                value_bars += 1
                if int(round(float(vals[t]))) != lval:
                    value_bad.append((idx, t, lval, float(vals[t])))
            else:
                skipped_values += 1
    return dict(tag_bars=tag_bars, value_bars=value_bars, skipped_values=skipped_values,
                tag_bad=tag_bad, value_bad=value_bad, verdict_bad=verdict_bad)


# ----------------------------------------------------------------------------- planted bugs

def _bug_where_ignores_condition(func, condition, x=None, y=None):
    parts = [tracer._split(p) for p in (condition, x, y)]
    res = np.where(*[p[0] for p in parts])
    return TimeTagged(res, tracer._max_tags(res.shape, parts[1][1], parts[2][1]))


def _change_cumsum_keeps_current_tag(func, a, axis=None, **kwargs):
    v, t = tracer._split(a)
    return TimeTagged(func(v, axis=axis, **kwargs), t)


def _bug_roll_leaves_tags_behind(func, a, *args, **kwargs):
    v, t = tracer._split(a)
    return TimeTagged(func(v, *args, **kwargs), t if func is np.roll else func(t, *args, **kwargs))


def _bug_reduction_takes_earliest(func, a, *args, **kwargs):
    v, t = tracer._split(a)
    res = func(v, *args, **kwargs)
    tag = t.min(axis=kwargs.get('axis'), keepdims=kwargs.get('keepdims', False))
    return TimeTagged(res, np.broadcast_to(tag, np.shape(res)))


# Each of these breaks the rule and must be noticed.
BUGS = {
    'np.where drops the tag of its condition': {np.where: _bug_where_ignores_condition},
    'np.roll moves values but not tags': {np.roll: _bug_roll_leaves_tags_behind},
    'np.sum and np.max keep the earliest tag, not the latest':
        {np.sum: _bug_reduction_takes_earliest, np.max: _bug_reduction_takes_earliest},
}
# This one looks like a bug and is not, for the signals of this model: `tag_mono` in
# Lookahead.lean proves every tag here is non-decreasing in time, so the latest tag so far is
# always the current one. The comparison must stay silent.
HARMLESS = {
    'np.cumsum keeps only the current tag': {np.cumsum: _change_cumsum_keeps_current_tag},
}


def disagreements_with(patch, chunks) -> int:
    saved = {f: tracer._HANDLERS[f] for f in patch}
    tracer._HANDLERS.update(patch)
    try:
        bad = 0
        for prices, exprs, lean_rows in chunks:
            py_results = [run_python(prices, e) for e in exprs]
            bad += len(compare(lean_rows, py_results)['tag_bad'])
        return bad
    finally:
        tracer._HANDLERS.update(saved)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--signals', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--lean', default='lean')
    ap.add_argument('--chunk', type=int, default=250)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    t0 = time.time()
    work = tempfile.mkdtemp(prefix='euclid-crosscheck-')
    subprocess.run([args.lean, '-o', os.path.join(work, 'Lookahead.olean'),
                    os.path.join(HERE, 'Lookahead.lean')], check=True, capture_output=True, timeout=600)

    chunks, total = [], dict(tag_bars=0, value_bars=0, skipped_values=0, tag_bad=[], value_bad=[], verdict_bad=[])
    kinds: dict[str, int] = {}
    flagged = 0
    done = 0
    while done < args.signals:
        m = min(args.chunk, args.signals - done)
        prices = np.array([[rng.randrange(-9, 10), rng.randrange(-9, 10)] for _ in range(LONG)], dtype=float)
        exprs = [gen(rng, rng.randrange(1, 5), [MARGIN]) for _ in range(m)]
        lean_rows = run_lean(args.lean, work, prices, exprs)
        py_results = [run_python(prices, e) for e in exprs]
        r = compare(lean_rows, py_results)
        for key in ('tag_bars', 'value_bars', 'skipped_values'):
            total[key] += r[key]
        for key in ('tag_bad', 'value_bad', 'verdict_bad'):
            total[key] += [(done + x[0],) + tuple(x[1:]) for x in r[key]]
        for e, rows in zip(exprs, lean_rows):
            flagged += any(ltag > t for t, (ltag, _) in enumerate(rows))
            stack = [e]
            while stack:
                x = stack.pop()
                kinds[x[0]] = kinds.get(x[0], 0) + 1
                stack.extend(y for y in x[1:] if isinstance(y, tuple))
        chunks.append((prices, exprs, lean_rows))
        done += m
        print(f'  {done}/{args.signals} signals compared', flush=True)

    print(f'\nsignals: {args.signals}   bars compared: {total["tag_bars"]}')
    print(f'signals with at least one flagged bar (a real lookahead in the model): {flagged}')
    print('operations exercised:', ', '.join(f'{k} {v}' for k, v in sorted(kinds.items())))
    print(f'tag disagreements:      {len(total["tag_bad"])}')
    print(f'verdict disagreements:  {len(total["verdict_bad"])}')
    print(f'value disagreements:    {len(total["value_bad"])} of {total["value_bars"]} compared '
          f'({total["skipped_values"]} too large to compare exactly)')
    for name in ('tag_bad', 'value_bad'):
        for row in total[name][:5]:
            print('   ', name, row)

    print('\nplanted bugs (each must be noticed):')
    caught_all = True
    for label, patch in BUGS.items():
        bad = disagreements_with(patch, chunks)
        caught_all &= bad > 0
        print(f'  {"noticed" if bad else "MISSED "}  {label}: {bad} disagreeing bars')
    print('harmless change (must not raise an alarm):')
    quiet = True
    for label, patch in HARMLESS.items():
        bad = disagreements_with(patch, chunks)
        quiet &= bad == 0
        print(f'  {"silent " if bad == 0 else "ALARM  "}  {label}: {bad} disagreeing bars')
    caught_all &= quiet

    ok = not total['tag_bad'] and not total['value_bad'] and not total['verdict_bad'] and caught_all
    print(f'\n{"PASS" if ok else "FAIL"}  ({time.time() - t0:.0f}s)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
