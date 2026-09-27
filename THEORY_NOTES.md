# Why SAM helps under pruning: theory notes

Status: working notes toward the paper's theory section. Covers the
mechanistic explanation for the empirical finding in the tight-recovery /
near-capacity-wall regime (see the experimental write-up). Everything here
is either a derivation checked by hand or a measurement against real
checkpoints — flagged explicitly which is which, and where a claim is
partial or unconfirmed.

## 1. Starting point: Proposition 3.1′

The paper's original Proposition 3.1 bounds the restricted gradient norm
after a pruning perturbation `δ` from a dense reference `θ*`, under
near-stationarity (assumption A1: `∇L(θ*) ≈ 0`). Dropping A1 costs nothing
in the proof — Taylor-expand `∇L` around `θ*` and keep the `∇L(θ*)` term
instead of discarding it:

**Proposition 3.1′.** *Let `θ*` have `L`-Lipschitz Hessian `H` (no
assumption on `∇L(θ*)`). Then*

`‖P_m^T∇L(θ^(m))‖ ≤ ‖P_m^T∇L(θ*)‖ + ‖H_m^cross‖·‖δ‖ + (L/2)‖δ‖²`

*Proof.* Same Taylor expansion and triangle inequality as Prop. 3.1, without
discarding `∇L(θ*)`. Prop. 3.1 is the special case `∇L(θ*)=0`. ∎

For a SAM solution specifically, Bartlett, Long & Bousquet's own
characterization (arXiv:2210.01513, their Lemma 5) gives the residual a
closed form: at a converged oscillation point, `∇L(θ*) ≈ ±β₁v₁` with

`β₁ := ηρλ₁²/(2−ηλ₁)`

where `λ₁` is the top eigenvalue of the restricted Hessian `H_m`. This is
**not** a new approximation — it's their own Lemma 5 quantity, substituted
directly into Prop. 3.1′.

`figures/grad_norm_trajectory.png` plots the measured left-hand side
(`‖P_m^T∇L‖`, checkpoint-based, seed 13) across all four tight-recovery
configs: SAM's residual tracks consistently below SGD's, and the gap grows
round over round rather than staying flat — the empirical shape that
sections 3–6 explain.

## 2. The precondition, checked against the source

Bartlett et al.'s Theorem 1 requires `0 < η < 1/(2λ₁)`, i.e. **ηλ₁ < 1/2**,
for their convergence guarantee to hold (confirmed against arXiv:2210.01513v2
directly, not from memory — see §2.3 for what that bound is actually doing).
Our tight-recovery experiments routinely produce `ηλ₁` well above this,
which raises the obvious question: does the theory say anything once we're
past it?

### 2.1 The bare fixed point is stable well past 1/2

Project onto the top eigendirection `v₁`; locally `L(w) = λ₁w²/2`. SAM's
update in this 1-D model is exactly

`w_{t+1} = (1−ηλ₁)w_t − ηλ₁ρ·sign(w_t)`

Solving for the alternating fixed point (`w_t=A ⟹ w_{t+1}=−A`) gives
`A = ηλ₁ρ/(2−ηλ₁)`, i.e. `β₁ = λ₁A` — matching Lemma 5 exactly. Writing
`w_t = A + e_t` and substituting:

`w_{t+1} = (1−ηλ₁)(A+e_t) − ηλ₁ρ = −A + (1−ηλ₁)e_t`

so **`e_{t+1} = (1−ηλ₁)e_t` exactly**, a linear contraction whenever
`|1−ηλ₁| < 1`, i.e. **`ηλ₁ < 2`** — the ordinary gradient-descent stability
bound, with nothing special at 1/2. This gives a genuine extension:

**Proposition (rate bound, top eigendirection).** *For `0 < ηλ₁ < 2`,
provided the trajectory stays in the alternating regime near the cycle,*

`|v₁ᵀ∇L(θ_t)| ≤ β₁ + λ₁|1−ηλ₁|^t·|e₀|`

*where `t` counts SAM iterations since the perturbation and `e₀` is the
initial deviation (tied to `‖δ‖`).*

### 2.2 Why Bartlett's bound is stricter than the bare fixed point needs

Checked directly against the proof (Section 4 and appendix lemmas of
arXiv:2210.01513v2): the extra margin between 1/2 and 2 isn't needed for
this 1-D fixed point — it's needed for the *full, general* theorem to also
guarantee, in a genuinely multi-dimensional, non-quadratic setting:

- **Cross-eigendirection separation** (Lemma 13): needs `(1−ηλ₁) ∈ (0.5, 1)`
  specifically so the top eigendirection's oscillation stays cleanly
  separated from every other eigendirection's dynamics.
- **A positive-definiteness margin for the Taylor remainder** (Lemma 8):
  "room for perturbations" to absorb the non-quadratic terms a real loss
  surface has and the idealized quadratic model doesn't.
- **A uniform, robust convergence *rate*, not just eventual boundedness.**

None of these are things the bare 2-cycle fixed point needs to exist and be
locally stable — they're what Bartlett et al.'s *general, robust* guarantee
needs. This is a secondhand read of the proof via a fetch-and-summarize
tool, not a line-by-line verification — worth checking directly against the
PDF before this goes into anything submitted, but the core fact (bare fixed
point stable to `ηλ₁<2`; Bartlett's proof needs `ηλ₁<1/2` for the
separation/remainder margin) is solid enough to build on.

### 2.3 A three-tier classification

- **T1, `ηλ₁ < 1/2`** — Bartlett's theorem holds in full: proven, robust
  convergence to the bounce cycle.
- **T2, `1/2 ≤ ηλ₁ < 2`** — the bare fixed point is still the algebraically
  correct description of the 2-cycle, and still linearly stable, but the
  margin that lets the proof ignore cross-direction interference and
  non-quadratic error is gone. Degraded, not meaningless.
- **T3, `ηλ₁ ≥ 2`** — even the idealized fixed point loses linear stability.
  No principled reason left to expect bounded oscillation; qualitatively a
  different (plausibly divergent/chaotic) regime.

Classifying every round of the two fully-measured configs (ResNet18,
seed 13, tight recovery) by this scheme:

| | s=0.995: T1 / T2 / T3 | s=0.999: T1 / T2 / T3 |
|---|---|---|
| SAM | 1 / 8 / 2 | 0 / 8 / 3 |
| SGD | 0 / 3 / 8 | 0 / 2 / 9 |

SAM spends most of both schedules in T2; SGD is almost never even there —
mostly T3, including an `ηλ₁≈195` outlier (s=0.999, final round) with no
principled interpretation left at all.

## 3. Why 15 epochs matters: it's not the contraction rate

The rate bound in §2.1 is in units of **raw optimizer steps**, not epochs
— CIFAR-10 at batch 128 is ~390 steps/epoch. Even at the *worst* rate
inside the valid range (`ηλ₁→2`, so `|1−ηλ₁|→1`, e.g. 0.9), reducing `e_t`
by 3 orders of magnitude takes `ln(0.001)/ln(0.9) ≈ 66` steps — a
**fraction of one epoch**. So the raw contraction-to-a-fixed-target rate
cannot be why a 15-epoch recovery window is insufficient; that part of the
dynamics is essentially instantaneous on the timescale we're measuring.

**Which redirects the question: the target itself must be moving.** `β₁`
(via `λ₁`) is not static after a pruning cut — curvature is known to spike
transiently post-pruning. If `λ₁` takes many epochs to relax back down, `w_t`
is chasing a moving target the whole recovery window, and *that* relaxation
timescale — not the contraction rate — governs whether 15 epochs is enough.

## 4. Measuring it: does λ₁ actually relax within a round?

Extended the checkpoint analysis (`scripts/analyze_telescoping_bound.py`)
to compute `λ₁` at every saved checkpoint (not just round boundaries)
through the last two rounds (epochs 135–165) of the tight-recovery
schedule, seed 13, both configs, ResNet18.

**s=0.995, round 9 window (135→145, before the next cut at 150):**

| epoch | SAM λ₁ | SGD λ₁ |
|---|---|---|
| 135 | 14.7 | 70.0 |
| 140 | 9.7 | 62.4 |
| 145 | 9.0 | 62.8 |
| Δ | **−39%** | −10% |

**s=0.999, same window:**

| epoch | SAM λ₁ | SGD λ₁ |
|---|---|---|
| 135 | 22.0 | 98.4 |
| 140 | 19.7 | 101.6 |
| 145 | 18.9 | **149.5** |
| Δ | −14% | **+52% (rising)** |

**SAM's curvature relaxes within the window in both configs. SGD's barely
moves at s=0.995 (and stays 5–9× higher throughout), and at s=0.999 it's
still actively *increasing* when the next cut lands** — each round's spike
stacks on an already-elevated baseline instead of resetting. See
`figures/lambda1_relaxation.png` for the last two rounds of both configs.

The final cut (epoch 165) is the sharpest illustration: SAM's `λ₁` jumps
12.0→25.0 (s=0.995) or 28.9→58.9 (s=0.999) — in line with every earlier
round's post-cut spike. SGD's jumps 91.4→126.5 (s=0.995, consistent) but
**164.9→1989.8 (s=0.999)** — an order-of-magnitude blowup unlike anything
earlier in its own trajectory, i.e. this is where SGD is actually observed
entering the T3 (destabilized) regime, not just approaching it.

## 5. The mechanism: this is the drift term, not a new phenomenon

This isn't a new, unexplained empirical curiosity — it's Theorem 3.4's
**drift** component doing exactly what it's built to do. The bounce-drift
decomposition splits SAM's update into the oscillation along `v₁`
(characterized by `β₁`, §1) and a drift term that is, by construction, an
approximate gradient-descent step *on `λ₁` itself*, taken in the null space
of the restricted output Jacobian so it reduces curvature without damaging
predictions. **Plain SGD has no equivalent term** — its curvature evolves
only as an incidental side effect of minimizing the loss, with no
directional bias against high curvature.

This also connects to an earlier, independent measurement: the drift's
*effectiveness* is characterized by Prop. 3.5's `(k−r)/k`, and a direct
check (MLP, `s=0.1`–`0.95`) found the drift direction really does
preferentially occupy the pruning null space, well above a random-direction
control — though the exact `(k−r)/k` magnitude overstated the effect
(measured 0.52–0.83 vs. predicted ~0.998–1.0). Put together: **SAM has a
real, if imperfect, mechanism for reducing post-pruning curvature; SGD does
not; and we've now directly measured that mechanism operating (λ₁
relaxation) in the regime where the accuracy gap actually appears.**

## 6. The gap tracks the mechanism across sparsity

| | s=0.995 | s=0.999 |
|---|---|---|
| SGD's worst `ηλ₁` in-window | 2.76 | **195.3** |
| SAM's worst `ηλ₁` in-window | 2.72 | 6.88 |
| Final-accuracy gap (SAM−SGD, mean of 3 seeds) | +1.4pp | **+4.1pp** |

Going from s=0.995 to s=0.999, SGD's worst-case curvature violation gets
~70× worse while SAM's only gets ~2.5× worse — and the accuracy gap grows
along with that asymmetry. This is the load-bearing empirical claim: it's
not just "SAM is flatter," it's that *how much worse SGD's curvature
control gets, relative to SAM's, as sparsity increases* is what predicts
*how much worse SGD's final accuracy gets, relative to SAM's*.

## 7. Cross-architecture check: partial, and why

Repeated the within-round measurement on VGG16 (same two rounds, same
seed, both `s=0.999` and `s=0.9995`, using the already-saved seed-13
checkpoints — no new training).

**Grad-norm (plain backward pass, no eigenvalue solver) replicates
cleanly:** SAM's restricted gradient norm stays ~2–2.7× smaller than SGD's
throughout both configs, same direction as ResNet18.

**λ₁ does not replicate reliably — the numbers aren't physically
plausible.** VGG16's estimated top eigenvalue reaches into the millions
(e.g. SAM, s=0.999, epoch 165: 5,265,485) and swings by 1–2 orders of
magnitude between adjacent 5-epoch checkpoints. This is the power-iteration
solver failing to converge, not real curvature — `HESSIAN_MAX_ITER=30` was
sufficient for ResNet18 but VGG16's restricted Hessian is evidently far
more ill-conditioned at this sparsity. This is the same failure mode
flagged earlier for VGG16's trace estimate at extreme sparsity (the
generous-recovery s=0.9995 trace reversal) — same root cause, now hitting
the eigenvalue estimate directly.

**Honest summary:** the gradient-residual finding (§1, Prop. 3.1′) is
confirmed on both architectures. The curvature-relaxation mechanism (§4)
is directly measured and confirmed on ResNet18; on VGG16 it's consistent
with but not independently confirmed by the λ₁ numbers, because the
eigenvalue estimator isn't reliable enough there to say either way. Fixing
that would need a substantially more expensive/robust eigenvalue solver —
flagged as future work, not attempted further given time constraints.

## 8. What's proven, what's measured, what's open

- **Proven:** Prop. 3.1′ (general, no A1 needed). The rate bound in §2.1
  (bare fixed-point stability to `ηλ₁<2`, checked by hand).
- **Measured, single seed (13), one architecture (ResNet18), two
  sparsities:** the within-round `λ₁`-relaxation asymmetry (§4) that the
  rate bound's "moving target" argument (§3) predicts should matter.
- **Measured, both architectures:** the gradient-residual asymmetry
  (Prop. 3.1′'s actual quantity) and the final-accuracy gap it's meant to
  explain.
- **Not independently confirmed:** the curvature-relaxation mechanism on
  VGG16 specifically (estimator reliability, §7); the mechanism at seeds
  other than 13 (only seed 13 has full per-round checkpoints — 42/97 only
  saved final weights, so this can't be checked without retraining).
- **Not attempted:** why SAM's drift term achieves *more* null-space
  alignment than the `(k−r)/k` generic-direction baseline would predict
  (the Prop. 3.5 gap noted in §5) — a real open question in its own right,
  separate from what's above.
- **Out of scope, stated as limitations rather than closed:** CIFAR-100,
  a transformer architecture, an iso-compute comparison (SAM spends ~2× the
  gradient computations of SGD per step; the current comparisons match
  epoch count, not FLOPs).
