/-!
# No lookahead: the tracer's rule is sound

EUCLID's lookahead tracer gives every number a *tag*: the latest bar it was computed from.
A signal for bar `t` whose tag is at most `t` is reported clean.

This file states that rule for a small language of signal expressions and proves it sound:

  if the tag of a signal at bar `t` is at most `t`, then no change to any price after bar `t`
  can change the signal at bar `t`.

The file uses core Lean only. It has no dependencies, no `sorry` and no added axioms.

What is proved is a property of the *rule*, for the operations modelled here. It is not a
proof about the Python implementation line by line; that link is covered by tests.
-/

namespace Euclid.Lookahead

/-! ## Folds over the first `n` indices -/

/-- `foldTo f init g n` combines `g 0, g 1, …, g (n-1)` from the left, starting at `init`.
Sums, maxima, minima and counts over a window are all folds of this shape. -/
def foldTo {α : Type} (f : α → α → α) (init : α) (g : Nat → α) : Nat → α
  | 0 => init
  | n + 1 => f (foldTo f init g n) (g n)

/-- `maxTo g n` is the largest of `g 0, …, g (n-1)`, and `0` when `n = 0`. -/
def maxTo (g : Nat → Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => max (maxTo g n) (g n)

/-- A fold only depends on the values it actually reads. -/
theorem foldTo_congr {α : Type} {f : α → α → α} {init : α} {g g' : Nat → α} :
    ∀ n, (∀ j, j < n → g j = g' j) → foldTo f init g n = foldTo f init g' n
  | 0, _ => rfl
  | n + 1, h => by
      show f (foldTo f init g n) (g n) = f (foldTo f init g' n) (g' n)
      rw [foldTo_congr n (fun j hj => h j (Nat.lt_succ_of_lt hj)), h n (Nat.lt_succ_self n)]

/-- Every value read is at most the maximum. -/
theorem le_maxTo {g : Nat → Nat} : ∀ {n j : Nat}, j < n → g j ≤ maxTo g n
  | 0, j, h => absurd h (Nat.not_lt_zero j)
  | n + 1, j, h => by
      show g j ≤ max (maxTo g n) (g n)
      by_cases hj : j < n
      · exact Nat.le_trans (le_maxTo hj) (Nat.le_max_left _ _)
      · have hjn : j = n := by omega
        subst hjn
        exact Nat.le_max_right _ _

/-- If every value read is at most `b`, so is the maximum. -/
theorem maxTo_le {g : Nat → Nat} {b : Nat} : ∀ n, (∀ j, j < n → g j ≤ b) → maxTo g n ≤ b
  | 0, _ => Nat.zero_le b
  | n + 1, h => by
      show max (maxTo g n) (g n) ≤ b
      exact Nat.max_le.mpr ⟨maxTo_le n (fun j hj => h j (Nat.lt_succ_of_lt hj)), h n (Nat.lt_succ_self n)⟩

/-! ## Signals -/

/-- A signal expression over values of type `α`. Prices are read with `input`; everything
else builds a new series from existing ones. Bars are numbered `0, 1, 2, …`. -/
inductive Expr (α : Type) where
  /-- The price of instrument `i` at the current bar. -/
  | input (i : Nat) : Expr α
  /-- A constant. -/
  | const (c : α) : Expr α
  /-- The value `k` bars ago, or `fill` when there is no such bar. A backward shift. -/
  | lag (k : Nat) (fill : α) (e : Expr α) : Expr α
  /-- The value `k` bars ahead. A forward shift: this is what a lookahead bug looks like. -/
  | lead (k : Nat) (e : Expr α) : Expr α
  /-- Any function applied bar by bar. -/
  | map1 (f : α → α) (e : Expr α) : Expr α
  /-- Any two-argument function applied bar by bar: add, multiply, compare, and so on. -/
  | map2 (f : α → α → α) (a b : Expr α) : Expr α
  /-- Bar by bar: `a` where the test holds on `c`, otherwise `b`. -/
  | select (test : α → Bool) (c a b : Expr α) : Expr α
  /-- A rolling window: fold the last `w` bars, ending at the current bar. -/
  | rolling (f : α → α → α) (init : α) (w : Nat) (e : Expr α) : Expr α
  /-- A running total: fold every bar from the first up to the current bar. -/
  | cumulative (f : α → α → α) (init : α) (e : Expr α) : Expr α
  /-- A whole-sample statistic: fold all `n` bars of the sample, whatever the current bar. -/
  | wholeSample (f : α → α → α) (init : α) (n : Nat) (e : Expr α) : Expr α

/-- The value of a signal at bar `t`, given prices `p instrument bar`. -/
def eval {α : Type} (p : Nat → Nat → α) : Expr α → Nat → α
  | .input i, t => p i t
  | .const c, _ => c
  | .lag k fill e, t => if k ≤ t then eval p e (t - k) else fill
  | .lead k e, t => eval p e (t + k)
  | .map1 f e, t => f (eval p e t)
  | .map2 f a b, t => f (eval p a t) (eval p b t)
  | .select test c a b, t => if test (eval p c t) then eval p a t else eval p b t
  | .rolling f init w e, t => foldTo f init (fun j => eval p e (t - j)) (min w (t + 1))
  | .cumulative f init e, t => foldTo f init (fun j => eval p e j) (t + 1)
  | .wholeSample f init n e, _ => foldTo f init (fun j => eval p e j) n

/-- The tracer's rule. `tag e t` is the latest bar that the value of `e` at bar `t` was
computed from. It never looks at prices or at the functions applied, only at the shape of
the computation: a price read at bar `t` is tagged `t`, and every operation passes on the
largest tag among the values it used. -/
def tag {α : Type} : Expr α → Nat → Nat
  | .input _, t => t
  | .const _, _ => 0
  | .lag k _ e, t => if k ≤ t then tag e (t - k) else 0
  | .lead k e, t => tag e (t + k)
  | .map1 _ e, t => tag e t
  | .map2 _ a b, t => max (tag a t) (tag b t)
  | .select _ c a b, t => max (tag c t) (max (tag a t) (tag b t))
  | .rolling _ _ w e, t => maxTo (fun j => tag e (t - j)) (min w (t + 1))
  | .cumulative _ _ e, t => maxTo (fun j => tag e j) (t + 1)
  | .wholeSample _ _ n e, _ => maxTo (fun j => tag e j) n

/-- Two price histories agree on every instrument at every bar up to and including `T`.
After `T` they may differ in any way. -/
abbrev AgreeUpTo {α : Type} (T : Nat) (p q : Nat → Nat → α) : Prop :=
  ∀ i s, s ≤ T → p i s = q i s

/-! ## The theorem -/

/-- **Soundness of the tag.** The value of a signal at bar `t` depends only on prices at bars
up to its tag. If two price histories agree up to `tag e t`, the signal has the same value
at `t` under both. -/
theorem tag_sound {α : Type} (p q : Nat → Nat → α) :
    ∀ (e : Expr α) (t : Nat), AgreeUpTo (tag e t) p q → eval p e t = eval q e t := by
  intro e
  induction e with
  | input i =>
      intro t h
      exact h i t (Nat.le_refl t)
  | const c =>
      intro t _
      rfl
  | lag k fill e ih =>
      intro t h
      by_cases hk : k ≤ t
      · have h' : AgreeUpTo (tag e (t - k)) p q := by
          simpa [tag, hk] using h
        simpa [eval, hk] using ih (t - k) h'
      · simp [eval, hk]
  | lead k e ih =>
      intro t h
      exact ih (t + k) h
  | map1 f e ih =>
      intro t h
      show f (eval p e t) = f (eval q e t)
      rw [ih t h]
  | map2 f a b iha ihb =>
      intro t h
      show f (eval p a t) (eval p b t) = f (eval q a t) (eval q b t)
      rw [iha t (fun i s hs => h i s (Nat.le_trans hs (Nat.le_max_left _ _))),
          ihb t (fun i s hs => h i s (Nat.le_trans hs (Nat.le_max_right _ _)))]
  | select test c a b ihc iha ihb =>
      intro t h
      show (if test (eval p c t) then eval p a t else eval p b t)
         = (if test (eval q c t) then eval q a t else eval q b t)
      rw [ihc t (fun i s hs => h i s (Nat.le_trans hs (Nat.le_max_left _ _))),
          iha t (fun i s hs => h i s
            (Nat.le_trans hs (Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_right _ _)))),
          ihb t (fun i s hs => h i s
            (Nat.le_trans hs (Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_right _ _))))]
  | rolling f init w e ih =>
      intro t h
      show foldTo f init (fun j => eval p e (t - j)) (min w (t + 1))
         = foldTo f init (fun j => eval q e (t - j)) (min w (t + 1))
      apply foldTo_congr
      intro j hj
      exact ih (t - j) (fun i s hs => h i s
        (Nat.le_trans hs (le_maxTo (g := fun j => tag e (t - j)) hj)))
  | cumulative f init e ih =>
      intro t h
      show foldTo f init (fun j => eval p e j) (t + 1) = foldTo f init (fun j => eval q e j) (t + 1)
      apply foldTo_congr
      intro j hj
      exact ih j (fun i s hs => h i s (Nat.le_trans hs (le_maxTo (g := fun j => tag e j) hj)))
  | wholeSample f init n e ih =>
      intro t h
      show foldTo f init (fun j => eval p e j) n = foldTo f init (fun j => eval q e j) n
      apply foldTo_congr
      intro j hj
      exact ih j (fun i s hs => h i s (Nat.le_trans hs (le_maxTo (g := fun j => tag e j) hj)))

/-- The tracer's verdict for one bar: the signal at `t` carries no tag later than `t`. -/
def CleanAt {α : Type} (e : Expr α) (t : Nat) : Prop := tag e t ≤ t

/-- **No lookahead.** If the tracer reports the signal clean at bar `t`, then changing prices
after bar `t`, in any way, on any instrument, cannot change the signal at bar `t`. -/
theorem no_lookahead {α : Type} {e : Expr α} {t : Nat} (hclean : CleanAt e t)
    {p q : Nat → Nat → α} (hpq : AgreeUpTo t p q) : eval p e t = eval q e t :=
  tag_sound p q e t (fun i s hs => hpq i s (Nat.le_trans hs hclean))

/-- The tracer's verdict for a run of `n` bars: clean at every bar. -/
def Clean {α : Type} (e : Expr α) (n : Nat) : Prop := ∀ t, t < n → CleanAt e t

/-- **History is stable.** If the signal is clean over a run, then everything it produced up
to bar `T` is fixed by the prices up to bar `T`. New data can never rewrite old signals. -/
theorem history_stable {α : Type} {e : Expr α} {n : Nat} (hclean : Clean e n)
    {p q : Nat → Nat → α} {T : Nat} (hpq : AgreeUpTo T p q) :
    ∀ t, t < n → t ≤ T → eval p e t = eval q e t :=
  fun t htn htT => no_lookahead (hclean t htn) (fun i s hs => hpq i s (Nat.le_trans hs htT))

/-- The check as a program: `true` exactly when the signal is clean at each of the first `n`
bars. It can be run, because the tag never looks at prices. -/
def cleanUpTo {α : Type} (e : Expr α) : Nat → Bool
  | 0 => true
  | n + 1 => cleanUpTo e n && decide (tag e n ≤ n)

/-- If the program says clean, the run is clean. -/
theorem cleanUpTo_sound {α : Type} (e : Expr α) : ∀ n, cleanUpTo e n = true → Clean e n
  | 0, _ => fun t ht => absurd ht (Nat.not_lt_zero t)
  | n + 1, h => by
      have h' : cleanUpTo e n = true ∧ tag e n ≤ n := by
        simpa [cleanUpTo] using h
      intro t ht
      by_cases htn : t < n
      · exact cleanUpTo_sound e n h'.1 t htn
      · have : t = n := by omega
        subst this
        exact h'.2

/-! ## Tags never go backwards

For every signal in this language the tag is non-decreasing in time: a later bar never
depends on less data than an earlier one. This is why a running total needs no special
treatment here, and it marks the edge of the model: operations that reorder time, such as
reversing or sorting a series, break it and are not covered. -/

/-- Folding more values can only raise the maximum. -/
theorem maxTo_mono_len {g : Nat → Nat} {n m : Nat} (h : n ≤ m) : maxTo g n ≤ maxTo g m :=
  maxTo_le n (fun _ hj => le_maxTo (Nat.lt_of_lt_of_le hj h))

/-- **Tags are monotone in time.** -/
theorem tag_mono {α : Type} : ∀ (e : Expr α) {t t' : Nat}, t ≤ t' → tag e t ≤ tag e t' := by
  intro e
  induction e with
  | input i =>
      intro t t' h
      exact h
  | const c =>
      intro t t' _
      exact Nat.le_refl 0
  | lag k fill e ih =>
      intro t t' h
      by_cases hk : k ≤ t
      · have hk' : k ≤ t' := Nat.le_trans hk h
        have hsub : t - k ≤ t' - k := by omega
        simpa [tag, hk, hk'] using ih hsub
      · have h0 : tag (Expr.lag k fill e) t = 0 := by simp [tag, hk]
        rw [h0]
        exact Nat.zero_le _
  | lead k e ih =>
      intro t t' h
      have hadd : t + k ≤ t' + k := by omega
      exact ih hadd
  | map1 f e ih =>
      intro t t' h
      exact ih h
  | map2 f a b iha ihb =>
      intro t t' h
      show max (tag a t) (tag b t) ≤ max (tag a t') (tag b t')
      exact Nat.max_le.mpr ⟨Nat.le_trans (iha h) (Nat.le_max_left _ _),
                            Nat.le_trans (ihb h) (Nat.le_max_right _ _)⟩
  | select test c a b ihc iha ihb =>
      intro t t' h
      show max (tag c t) (max (tag a t) (tag b t)) ≤ max (tag c t') (max (tag a t') (tag b t'))
      exact Nat.max_le.mpr
        ⟨Nat.le_trans (ihc h) (Nat.le_max_left _ _),
         Nat.max_le.mpr
          ⟨Nat.le_trans (iha h) (Nat.le_trans (Nat.le_max_left _ _) (Nat.le_max_right _ _)),
           Nat.le_trans (ihb h) (Nat.le_trans (Nat.le_max_right _ _) (Nat.le_max_right _ _))⟩⟩
  | rolling f init w e ih =>
      intro t t' h
      show maxTo (fun j => tag e (t - j)) (min w (t + 1))
         ≤ maxTo (fun j => tag e (t' - j)) (min w (t' + 1))
      apply maxTo_le
      intro j hj
      have hj' : j < min w (t' + 1) := by omega
      have hsub : t - j ≤ t' - j := by omega
      exact Nat.le_trans (ih hsub) (le_maxTo (g := fun j => tag e (t' - j)) hj')
  | cumulative f init e ih =>
      intro t t' h
      show maxTo (fun j => tag e j) (t + 1) ≤ maxTo (fun j => tag e j) (t' + 1)
      exact maxTo_mono_len (by omega)
  | wholeSample f init n e ih =>
      intro t t' _
      exact Nat.le_refl _

/-! ## Worked examples

Five signals over integer prices. Each verdict below is a theorem, not a test. -/

/-- The price of the first instrument. -/
def price : Expr Int := .input 0

/-! ### 1. Tomorrow's price used today: flagged, and it really leaks -/

/-- Tomorrow's price. -/
def peek : Expr Int := .lead 1 price

/-- The tracer flags it at every bar. -/
theorem peek_flagged (t : Nat) : ¬ CleanAt peek t := by
  show ¬ (t + 1 ≤ t)
  omega

/-- And the flag is right: two histories identical up to bar `t` give different signals. -/
theorem peek_leaks (t : Nat) :
    ∃ p q : Nat → Nat → Int, AgreeUpTo t p q ∧ eval p peek t ≠ eval q peek t := by
  refine ⟨fun _ _ => 0, fun _ s => if s = t + 1 then 1 else 0, ?_, ?_⟩
  · intro i s hs
    have hne : ¬ (s = t + 1) := by omega
    simp [hne]
  · show (0 : Int) ≠ (if t + 1 = t + 1 then 1 else 0)
    simp

/-! ### 2. Subtracting a whole-sample total: flagged before the last bar -/

/-- Price minus the total over all `n` bars of the sample. Normalising by a statistic of the
whole sample is the most common lookahead bug in backtests. -/
def demeanWhole (n : Nat) : Expr Int :=
  .map2 (fun a b => a - b) price (.wholeSample (fun a b => a + b) 0 n price)

/-- Flagged at every bar except the last one of the sample. -/
theorem demeanWhole_flagged {n t : Nat} (h : t + 1 < n) : ¬ CleanAt (demeanWhole n) t := by
  intro hclean
  have h1 : tag (Expr.wholeSample (fun a b => a + b) (0 : Int) n price) t ≤ t :=
    Nat.le_trans (Nat.le_max_right _ _) hclean
  have h2 : t + 1 ≤ maxTo (fun j => tag price j) n :=
    le_maxTo (g := fun j => tag price j) h
  have h3 : maxTo (fun j => tag price j) n ≤ t := h1
  omega

/-! ### 3. A forward shift undone by a backward shift: clean -/

/-- Shift forward one bar, then back one bar. A text search for forward shifts would raise
an alarm here. The tracer follows the data and does not. -/
def roundTrip : Expr Int := .lag 1 0 (.lead 1 price)

theorem roundTrip_clean (t : Nat) : CleanAt roundTrip t := by
  by_cases h : 1 ≤ t
  · have : tag roundTrip t = t - 1 + 1 := by simp [roundTrip, price, tag, h]
    show tag roundTrip t ≤ t
    omega
  · have : tag roundTrip t = 0 := by simp [roundTrip, price, tag, h]
    show tag roundTrip t ≤ t
    omega

/-! ### 4. A moving-total crossover traded one bar later: clean at every bar -/

/-- The total of the last `w` prices. -/
def rollingTotal (w : Nat) : Expr Int := .rolling (fun a b => a + b) 0 w price

/-- A window ending at the current bar never reads past it. -/
theorem rollingTotal_clean (w t : Nat) : CleanAt (rollingTotal w) t := by
  show maxTo (fun j => tag price (t - j)) (min w (t + 1)) ≤ t
  apply maxTo_le
  intro j _
  show t - j ≤ t
  omega

/-- Long when the fast average is above the slow average, decided on the previous bar. -/
def crossover (fast slow : Nat) : Expr Int :=
  .lag 1 0 (.map2 (fun a b => if a * slow > b * fast then 1 else 0)
    (rollingTotal fast) (rollingTotal slow))

/-- Clean at every bar, for every pair of window lengths. -/
theorem crossover_clean (fast slow t : Nat) : CleanAt (crossover fast slow) t := by
  by_cases h : 1 ≤ t
  · have hf := rollingTotal_clean fast (t - 1)
    have hs := rollingTotal_clean slow (t - 1)
    have : tag (crossover fast slow) t
        = max (tag (rollingTotal fast) (t - 1)) (tag (rollingTotal slow) (t - 1)) := by
      simp [crossover, tag, h]
    show tag (crossover fast slow) t ≤ t
    rw [this]
    have hm := Nat.max_le.mpr ⟨hf, hs⟩
    omega
  · have : tag (crossover fast slow) t = 0 := by simp [crossover, tag, h]
    show tag (crossover fast slow) t ≤ t
    omega

/-- So no future price can change today's crossover signal. -/
theorem crossover_no_lookahead (fast slow t : Nat) {p q : Nat → Nat → Int}
    (hpq : AgreeUpTo t p q) : eval p (crossover fast slow) t = eval q (crossover fast slow) t :=
  no_lookahead (crossover_clean fast slow t) hpq

/-! ### 5. A false alarm: sound is not the same as exact -/

/-- Tomorrow's price minus tomorrow's price. Always zero, so it leaks nothing. -/
def harmless : Expr Int := .map2 (fun a b => a - b) peek peek

/-- The tracer still flags it, because it follows where numbers came from, not what the
arithmetic does with them. -/
theorem harmless_flagged (t : Nat) : ¬ CleanAt harmless t := by
  show ¬ (max (t + 1) (t + 1) ≤ t)
  have := Nat.le_max_left (t + 1) (t + 1)
  omega

/-- Yet its value never depends on any price. The rule can raise a false alarm. It cannot
miss a real one: that is `no_lookahead`. -/
theorem harmless_constant (p : Nat → Nat → Int) (t : Nat) : eval p harmless t = 0 := by
  show p 0 (t + 1) - p 0 (t + 1) = 0
  exact Int.sub_self _

/-! ## Why it matters: lookahead is the difference between a sure thing and a bet

Two more theorems, about money this time. Hold `x` units of the first instrument from bar
`t` to bar `t + 1`; the profit is `x` times the price change. -/

theorem int_mul_self_nonneg (x : Int) : 0 ≤ x * x := by
  rcases Int.le_total 0 x with h | h
  · exact Int.mul_nonneg h h
  · exact Int.mul_nonneg_of_nonpos_of_nonpos h h

theorem int_mul_self_pos {x : Int} (hx : x ≠ 0) : 0 < x * x := by
  have hsign : 0 < x ∨ x < 0 := by omega
  rcases hsign with h | h
  · exact Int.mul_pos h h
  · exact Int.mul_pos_of_neg_of_neg h h

/-- Profit over the next bar from holding `x` units from bar `t`. -/
def nextBarPnl (p : Nat → Nat → Int) (x : Int) (t : Nat) : Int :=
  x * (p 0 (t + 1) - p 0 t)

/-- The cheater: long one unit if tomorrow's price is higher, short one unit if it is lower. -/
def cheat : Expr Int :=
  .map2 (fun next cur => if next > cur then 1 else if next < cur then -1 else 0) peek price

/-- The tracer flags the cheater at every bar. -/
theorem cheat_flagged (t : Nat) : ¬ CleanAt cheat t := by
  show ¬ (max (t + 1) t ≤ t)
  have := Nat.le_max_left (t + 1) t
  omega

/-- **The cheater never loses.** On every price path, at every bar, its profit over the next
bar is at least zero. -/
theorem cheat_never_loses (p : Nat → Nat → Int) (t : Nat) :
    0 ≤ nextBarPnl p (eval p cheat t) t := by
  show 0 ≤ (if p 0 (t + 1) > p 0 t then 1 else if p 0 (t + 1) < p 0 t then -1 else 0)
            * (p 0 (t + 1) - p 0 t)
  by_cases h1 : p 0 (t + 1) > p 0 t
  · simp only [h1, ite_true]
    omega
  · by_cases h2 : p 0 (t + 1) < p 0 t
    · simp only [h1, h2, ite_true, ite_false]
      omega
    · simp only [h1, h2, ite_false]
      omega

/-- And it makes money whenever the price moves at all. -/
theorem cheat_wins_when_price_moves (p : Nat → Nat → Int) (t : Nat)
    (hmove : p 0 (t + 1) ≠ p 0 t) : 0 < nextBarPnl p (eval p cheat t) t := by
  show 0 < (if p 0 (t + 1) > p 0 t then 1 else if p 0 (t + 1) < p 0 t then -1 else 0)
            * (p 0 (t + 1) - p 0 t)
  by_cases h1 : p 0 (t + 1) > p 0 t
  · simp only [h1, ite_true]
    omega
  · have h2 : p 0 (t + 1) < p 0 t := by omega
    simp only [h1, h2, ite_true, ite_false]
    omega

/-- **An honest strategy can always lose.** Take any signal the tracer reports clean at bar
`t`, and any price history. There is a history identical up to bar `t`, whose next price
differs from the current one by at most one unit, on which the position taken at `t` does
not make money over the next bar, and loses money if the position is not zero. No clean
signal is a sure thing: a one-unit move against it is always possible. -/
theorem clean_can_lose {e : Expr Int} {t : Nat} (hclean : CleanAt e t) (p : Nat → Nat → Int) :
    ∃ q : Nat → Nat → Int, AgreeUpTo t p q ∧
      (-1 ≤ q 0 (t + 1) - q 0 t ∧ q 0 (t + 1) - q 0 t ≤ 1) ∧
      nextBarPnl q (eval q e t) t ≤ 0 ∧
      (eval p e t ≠ 0 → nextBarPnl q (eval q e t) t < 0) := by
  -- after bar `t`, move the price one unit against the position
  let x := eval p e t
  let d : Int := if x > 0 then -1 else if x < 0 then 1 else 0
  let q : Nat → Nat → Int := fun i s => if s ≤ t then p i s else p i t + d
  have hagree : AgreeUpTo t p q := by
    intro i s hs
    show p i s = if s ≤ t then p i s else p i t + d
    simp [hs]
  have hsame : eval q e t = x := (no_lookahead hclean hagree).symm
  have hq1 : q 0 (t + 1) = p 0 t + d := by
    show (if t + 1 ≤ t then p 0 (t + 1) else p 0 t + d) = p 0 t + d
    have : ¬ (t + 1 ≤ t) := by omega
    simp [this]
  have hq0 : q 0 t = p 0 t := by
    show (if t ≤ t then p 0 t else p 0 t + d) = p 0 t
    simp
  have hmove : q 0 (t + 1) - q 0 t = d := by
    rw [hq1, hq0]
    omega
  have hpnl : nextBarPnl q (eval q e t) t = x * d := by
    show eval q e t * (q 0 (t + 1) - q 0 t) = x * d
    rw [hsame, hmove]
  have hd : (x > 0 ∧ d = -1) ∨ (x < 0 ∧ d = 1) ∨ (x = 0 ∧ d = 0) := by
    show (x > 0 ∧ (if x > 0 then -1 else if x < 0 then 1 else 0) = -1)
       ∨ (x < 0 ∧ (if x > 0 then -1 else if x < 0 then 1 else 0) = 1)
       ∨ (x = 0 ∧ (if x > 0 then -1 else if x < 0 then 1 else 0) = 0)
    by_cases h1 : x > 0
    · exact Or.inl ⟨h1, by simp [h1]⟩
    · by_cases h2 : x < 0
      · exact Or.inr (Or.inl ⟨h2, by simp [h1, h2]⟩)
      · exact Or.inr (Or.inr ⟨by omega, by simp [h1, h2]⟩)
  refine ⟨q, hagree, ?_, ?_, ?_⟩
  · rw [hmove]
    rcases hd with ⟨_, hd⟩ | ⟨_, hd⟩ | ⟨_, hd⟩ <;> rw [hd] <;> omega
  · rw [hpnl]
    rcases hd with ⟨hx, hd⟩ | ⟨hx, hd⟩ | ⟨hx, hd⟩ <;> rw [hd] <;> omega
  · intro hne
    rw [hpnl]
    rcases hd with ⟨hx, hd⟩ | ⟨hx, hd⟩ | ⟨hx, hd⟩
    · rw [hd]; omega
    · rw [hd]; omega
    · exact absurd hx hne

/-! ## The definitions compute what they say

A proof about the wrong definitions proves nothing, so here the definitions are run on a
price series where the price at bar `s` is simply `s`. -/

/-- The price at bar `s` is `s`, for every instrument. -/
def ramp : Nat → Nat → Int := fun _ s => s

example : eval ramp price 7 = 7 := by decide
example : eval ramp (.lag 2 0 price) 7 = 5 := by decide
example : eval ramp (.lag 2 0 price) 1 = 0 := by decide
example : eval ramp peek 7 = 8 := by decide
example : eval ramp (rollingTotal 3) 5 = 5 + 4 + 3 := by decide
example : eval ramp (rollingTotal 3) 1 = 1 + 0 := by decide
example : eval ramp (.cumulative (fun a b => a + b) 0 price) 4 = 0 + 1 + 2 + 3 + 4 := by decide
example : eval ramp (.wholeSample (fun a b => a + b) 0 6 price) 2 = 15 := by decide
example : eval ramp (.select (fun c => decide (c > 3)) price (.const 1) (.const 0)) 5 = 1 := by decide
example : tag (rollingTotal 3) 5 = 5 := by decide
example : tag peek 5 = 6 := by decide
example : tag (demeanWhole 10) 5 = 9 := by decide
example : cleanUpTo (crossover 3 10) 50 = true := by decide
example : cleanUpTo peek 50 = false := by decide

/-! ## Axiom report

Printed when the file is compiled. Every line should list only Lean's standard axioms. -/

#print axioms tag_sound
#print axioms no_lookahead
#print axioms history_stable
#print axioms cleanUpTo_sound
#print axioms tag_mono
#print axioms peek_leaks
#print axioms demeanWhole_flagged
#print axioms roundTrip_clean
#print axioms crossover_no_lookahead
#print axioms harmless_constant
#print axioms cheat_never_loses
#print axioms cheat_wins_when_price_moves
#print axioms clean_can_lose

end Euclid.Lookahead
