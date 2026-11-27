# The lookahead rule, proved

`lean/Lookahead.lean` is EUCLID's first formal proof. It shows that the rule the lookahead
tracer uses cannot miss a lookahead, for the operations it models.

## What the tracer does

The tracer gives every number a tag: the latest bar it was computed from. A price read at
bar `t` is tagged `t`. Every operation passes on the largest tag among the values it used.
A signal for bar `t` whose tag is at most `t` is reported clean.

## What is proved

The file defines a small language of signals (prices, constants, backward and forward
shifts, any bar-by-bar function, a conditional, rolling windows, running totals and
whole-sample statistics), their values, and the tag rule. Then it proves:

| Theorem | In plain words |
| --- | --- |
| `tag_sound` | The value of a signal at a bar depends only on prices up to its tag. |
| `no_lookahead` | If the tracer says clean at bar `t`, no change to any later price, on any instrument, can change the signal at `t`. |
| `history_stable` | If a run is clean, new data can never rewrite the signals already produced. |
| `cleanUpTo_sound` | The check written as a program is correct: when it returns true, the run is clean. |
| `tag_mono` | Tags never go backwards: a later bar never depends on less data than an earlier one. |

And two results about money:

| Theorem | In plain words |
| --- | --- |
| `cheat_never_loses` | A strategy that peeks one bar ahead never loses, on any price path. |
| `cheat_wins_when_price_moves` | It makes money whenever the price moves at all. |
| `clean_can_lose` | For any signal the tracer reports clean, and any history, a one-unit move against the position is always possible. No clean signal is a sure thing. |

Together: lookahead is exactly what separates a sure thing from a bet, and the tracer's rule
catches every instance of it.

## Worked examples in the file

| Signal | Verdict | Proved |
| --- | --- | --- |
| Tomorrow's price | Flagged, and it really leaks | `peek_flagged`, `peek_leaks` |
| Price minus a whole-sample total | Flagged before the last bar | `demeanWhole_flagged` |
| Shift forward, then back | Clean | `roundTrip_clean` |
| Moving-total crossover, traded a bar later | Clean at every bar, for every window length | `crossover_clean` |
| Tomorrow's price minus itself | Flagged, though it is always zero | `harmless_flagged`, `harmless_constant` |

The last one is a false alarm, kept on purpose. The rule is sound, not exact: it can flag a
harmless computation, and it cannot pass a harmful one.

## How it was checked

- Compiles with Lean 4.19.0 and 4.36.0, with no errors and no warnings.
- Core Lean only. No dependencies.
- No `sorry`, no added axioms, no `native_decide`. The file prints its own axiom report when
  compiled: every theorem rests on Lean's standard axioms only.
- The definitions are run inside the file on a price series where the price at bar `s` is
  `s`, so a reader can see they compute what they claim.
- Three false statements were given to Lean as a control (a flagged signal claimed clean,
  the main theorem without its hypothesis, the cheater claimed clean). Lean rejected all
  three.

Check it yourself:

```
lean lean/Lookahead.lean
```

Every line of output should read `depends on axioms: [...]` with only `propext`,
`Classical.choice` and `Quot.sound` inside the brackets.

## Does the code follow the rule?

The proof is about a model. `lean/crosscheck.py` tests the link to the real tracer. It builds
random signals and computes each one twice: in Lean with the model, and in Python with
ordinary NumPy calls running under `euclid/tracer.py`. Then it compares tags and values bar
by bar.

| Run | Signals | Bars compared | Tag disagreements | Value disagreements |
| --- | --- | --- | --- | --- |
| seed 1 | 5,000 | 100,000 | 0 | 0 |
| seed 7 | 5,000 | 100,000 | 0 | 0 |

About a third of the signals contain a real lookahead, so both verdicts are exercised. The
Python side uses more than one NumPy route for the same operation (a backward shift by
`concatenate`, by `roll` and by `pad`; a rolling total by `convolve` and by a sliding window),
so several of the tracer's handlers are covered.

A comparison that never complains proves little, so the script also plants bugs in the
tracer and checks that it notices:

| Planted change | Outcome |
| --- | --- |
| `np.where` drops the tag of its condition | Noticed |
| `np.roll` moves values but not tags | Noticed |
| `np.sum` and `np.max` keep the earliest tag instead of the latest | Noticed |
| `np.cumsum` keeps only the current tag | Silent, correctly |

The last one looks like a bug and is not one here. `tag_mono` proves that for these signals
the latest tag so far is always the current tag, so the two rules give the same answer. It
would matter for operations that reorder time, which are outside the model.

Run it:

```
python lean/crosscheck.py --signals 5000 --seed 1
```

The saved output of both runs is in `lean/results/`.

## Limits

- **This is a proof about the rule, not about the Python.** The theorem covers the model in
  the Lean file. That `euclid/tracer.py` implements the same rule is established by the
  comparison above: 10,000 random signals with no disagreement. That is a test, not a proof,
  so a result that relies on both is labelled `traced`, not `proven`.
- **The comparison covers ten kinds of operation**, the ones in the model. The tracer handles
  many more NumPy functions than that; those are covered by the ordinary test suite only.
- **Only the modelled operations are covered.** Sorting, data-dependent indexing and
  anything the tracer cannot follow are outside the model. The tracer already refuses to
  give a verdict on code it cannot follow.
- **Prices are integers in the two money theorems**, and the model lets a price go to zero
  or below. The soundness theorems hold for values of any type.
- **One-bar profit only.** The money theorems ignore costs, sizing and compounding. They are
  there to show what a lookahead is worth, not to model a strategy.
