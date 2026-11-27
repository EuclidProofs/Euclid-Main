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

## Limits

- **This is a proof about the rule, not about the Python.** The theorem covers the model in
  the Lean file. That `euclid/tracer.py` implements the same rule is established by tests,
  not by this proof. A result that relies on both is labelled `traced`, not `proven`.
- **Only the modelled operations are covered.** Sorting, data-dependent indexing and
  anything the tracer cannot follow are outside the model. The tracer already refuses to
  give a verdict on code it cannot follow.
- **Prices are integers in the two money theorems**, and the model lets a price go to zero
  or below. The soundness theorems hold for values of any type.
- **One-bar profit only.** The money theorems ignore costs, sizing and compounding. They are
  there to show what a lookahead is worth, not to model a strategy.
