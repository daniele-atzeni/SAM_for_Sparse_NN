# TODO / deferred items

Tracking what's intentionally out of scope for the current re-run, so it
doesn't get lost.

## Experiment grid

- [x] Add more sparsity levels beyond 0.7/0.9 — s=0.95/0.98/0.99 added in
      `configs/sparse/{ResNet18,VGG16}_CIFAR10_s{0.95,0.98,0.99}.json`,
      runnable via `scripts/run_sparse_grid_strong.sh`. First pass at
      0.7/0.9 showed little SAM-vs-SGD / dense-vs-sparse divergence, even
      in the training trajectories, not just final accuracy — plausible
      cause: pruning finishes at epoch 55 out of 180, leaving 125 epochs
      (69%) of recovery, which is the regime where trajectory differences
      are expected to wash out before final accuracy is measured (matches
      the "generous recovery" discussion in `archive/rebuttal`, Table 6).
      This sweep holds the schedule fixed and only pushes sparsity, to
      check whether that alone is enough.
- [x] Strong-pruning sweep (s=0.95/0.98/0.99) completed, 36/36 runs, no
      errors. Result: Hessian trace is consistently lower for SAM than SGD
      (3–10x, both architectures, all sparsities — the one fully robust
      finding), and SAM shows a visibly smaller/faster-recovering accuracy
      dip right after the later pruning rounds (epoch 45/55) than SGD. But
      **final accuracy at epoch 180 stays statistically indistinguishable
      between SAM/SGD at every sparsity level** — confirms the "generous
      recovery washes out trajectory differences" account rather than
      refuting it. Also checked training accuracy specifically: SGD still
      reaches ~99-100% at every sparsity up to 0.99 for both architectures,
      meaning the network is still comfortably over-parameterized (can
      still fully memorize the 50k-image training set) even at s=0.99 —
      SAM's lower/declining train accuracy there is its own implicit
      regularization, not a capacity ceiling (confirmed since SGD at the
      same sparsity doesn't show the same ceiling). See
      `figures/train_accuracy_curves.png`.
- [x] Recovery-budget variant (`s0.9_shortrecovery`, 6/6 runs) and
      capacity-wall sweep (`s{0.995,0.999,0.9995}`, 18/18 runs) both
      completed, no errors. Recovery-budget is the one condition in the
      whole campaign where final accuracy itself moved in SAM's favor,
      consistently: +0.9pp mean, SAM ahead in 5/6 seed×architecture cells
      (vs. a coin flip at the same s=0.9 under generous recovery). The
      capacity-wall sweep found both optimizers collapse together between
      s=0.995→0.999 (ResNet18) / s=0.999→0.9995 (VGG16) — walls align by
      *absolute* active-parameter count (~10-30k) rather than sparsity
      fraction or architecture, and it's a shared capacity limit, not a
      SAM-vs-SGD effect. One anomaly: at VGG16 s=0.9995 the otherwise
      universal SAM<SGD trace ordering reverses in 2/3 seeds — not yet
      explained (see full write-up artifact for everything, published
      2026-09-10).
- [ ] **CIFAR-100 replication** — declined (no time for another full
      training campaign). Configs/script were built and the real
      `src/data/cifar100.py` root-path bug was fixed along the way (it
      defaulted to `../data` instead of `./src/data/DATA` like every other
      loader — would have scattered the download outside the repo), so
      this is ready to run later if there's ever time; not run.
- [x] **Aggressive-steps** (3 rounds, ~54% cut each, `prune_every=75`) —
      6/6 runs, no errors. SAM ahead in 6/6 seeds (mean +0.47pp), but no
      visible per-round dip for either optimizer — 75 epochs between
      rounds 1-2 and 2-3 turned out to give most cuts a generous recovery
      window despite each cut being much bigger, so this doesn't isolate
      the "tight recovery" variable the way `s0.9_shortrecovery` does.
      Superseded by the wall x short-recovery sweep below.
- [x] **Wall x short-recovery sweep** — `configs/sparse/{ResNet18,VGG16}_CIFAR10_s{0.995,0.999,0.9995}_shortrecovery.json`
      + `scripts/run_sparse_wall_shortrecovery.sh`, 12/12 runs, no errors.
      Combines both levers (tight recovery at *every* round, not just the
      last one, per the aggressive-steps lesson above, at sparsities near
      each architecture's capacity wall). This is the strongest result in
      the whole campaign: SAM ahead in every seed tested, all 3 seeds now
      complete for all 4 configs (ResNet18 s=0.995: mean +1.4pp;
      **ResNet18 s=0.999: mean +4.1pp, up to +6.2pp in one seed**; VGG16
      s=0.999: mean +1.2pp; VGG16 s=0.9995: mean +2.7pp, up 0.6pp from the
      single-seed +2.1pp estimate now that all 3 have finished). The gap
      size tracks sparsity, which is what motivated the
      theory work in `THEORY_NOTES.md`. See `figures/test_accuracy_curves.png`
      for the full mean±std trajectories across all 4 configs — the final-round
      dip/recovery gap (epoch 165→180) is where most of the final-accuracy gap
      comes from, most visibly at ResNet18 s=0.999. `figures/final_accuracy_gap_summary.png`
      is the headline bar-chart version (mean + per-seed dots, all 4 configs).
      `figures/hessian_trace_curves.png` confirms the strong-sweep's trace
      finding (SAM 3-10x lower, see below) holds throughout training at
      these higher sparsities too, ResNet18 only — VGG16's eigenvalue
      estimator is unreliable here (see THEORY_NOTES.md Sec. 7), so its
      trace isn't plotted. `figures/test_loss_curves.png` shows the same
      story in loss: SGD drops faster early but SAM ends up lower and
      recovers better after each cut.
      Along the way: found and fixed a real path-collision bug —
      `main_training_{sparse,dense}.py` named `saved_models/`/`tensorboard/`
      output only by `prune_ratio`, so configs with the same sparsity but
      different schedules silently overwrote each other's checkpoints (the
      original baseline, short-recovery, and aggressive-steps s=0.9 runs
      all collided). Now tagged by the config file's own name instead.
- [ ] Add a transformer architecture (ViT is already in `src/models/`,
      wire up a `configs/sparse/ViT_CIFAR10_s*.json` once ResNet/VGG results
      are in).
- [x] **Compute accounting** — measured real forward+backward FLOPs
      directly against the model code (`torch.utils.flop_counter`, batch
      128): ResNet18 SGD 426.1 GFLOPs/step, SAM 852.2 (exactly 2x, confirmed
      against `src/train/training.py`'s two full `model(data)`+backward
      calls); vgg16_bn SGD 254.6, SAM 509.2. Pruning is unstructured
      masking (`prune.global_unstructured`/`custom_from_mask`) — a dense
      tensor gets zeroed entries but every matmul still runs full-size, so
      **sparse and dense cost identical training FLOPs at a fixed epoch
      count**; SAM's 2x-per-step is the *only* real compute asymmetry in
      the whole campaign. `figures/compute_vs_accuracy.png` puts final
      accuracy and total training PFLOPs side by side for
      {dense, sparse-at-wall} x {SGD, SAM}, both architectures. Notable:
      **dense is a wash** (ResNet18 dense: SGD actually *beats* SAM in all
      3 seeds, 94.6 vs 94.1 mean; VGG16 dense: SAM ahead by ~1pp) — SAM's
      edge is specific to the sparse/near-wall regime, not a general "SAM
      is worth 2x compute" effect.
- [ ] **Iso-compute follow-up (prepared, not run), scoped to ResNet18** —
      does SGD close the gap if given SAM's actual FLOP budget instead of
      matching epoch count? Checked first whether this is even an open
      question: at s=0.995 both optimizers' post-cut recovery curves are
      already flat by epoch 180 (mean slope 0.03-0.08pp/epoch over the last
      5 epochs of the uninterrupted 165->180 window) — more time wouldn't
      be expected to move that (small, 1.4pp) gap. At **s=0.999, SGD is
      still visibly climbing when training stops** (~0.26pp/epoch vs SAM's
      ~0.08pp/epoch, same window, mean of 3 seeds) — the 4.1pp gap there is
      measured before either curve has converged, so this is a genuinely
      open question, not a settled one. That's why the follow-up is
      ResNet18-only: VGG16 wasn't checked this way and s=0.995 is already
      answered by the slope data above.
      Two sparsities, to see whether the gap (and the still-climbing
      pattern) keeps growing, saturates, or reverses one step further past
      the capacity wall: **s=0.999** (existing wall config, ~11.2k active
      params) and **s=0.9995** (new, ~5.6k active params — confirmed
      non-degenerate first: 76-80% test accuracy under generous recovery,
      seeds 13/42/97, from the earlier capacity-wall sweep, so this isn't a
      shot in the dark). s=0.9999 (~1.1k active params) intentionally held
      off until s=0.9995 results are in — untested territory, real risk of
      hitting a shared capacity floor where the SAM/SGD distinction becomes
      noise.
      `configs/sparse/ResNet18_CIFAR10_s0.9995_shortrecovery.json` is the
      new "normal" (epoch-matched) wall point — run via
      `scripts/run_sparse_resnet_s0.9995_extra.sh` first, to get its SAM
      reference and see if the accuracy gap itself grows past 4.1pp.
      `configs/sparse/ResNet18_CIFAR10_s0.999{,5}_shortrecovery_isocompute_sgd.json`
      give SGD 360 epochs (2x) at both sparsities, same 11-round pruning
      schedule (prune_every/first_iter doubled to 30 to preserve the
      schedule shape) and LR milestones scaled proportionally (160/320 of
      360, same as 80/160 of 180) so the back half of training isn't
      wasted at a decayed LR — total SGD FLOPs then exactly equal the
      corresponding 180-epoch SAM runs' FLOPs. Run via
      `scripts/run_sparse_isocompute_sgd.sh` (uses `--use-sam False`, since
      the SAM reference numbers already exist or will from the point
      above). VGG16 iso-compute config still exists
      (`configs/sparse/VGG16_CIFAR10_s0.9995_shortrecovery_isocompute_sgd.json`)
      but is commented out of the run script — lower priority, add back in
      if full architecture coverage becomes worth it.
      If 360-epoch SGD still trails 180-epoch SAM, that's a much stronger
      result than the epoch-matched comparison — SAM wouldn't just be
      "spending extra compute," it'd be reaching a final accuracy SGD can't
      match at *any* compute budget along this schedule shape. If it
      catches up, that reframes the story as "SAM converts compute into
      accuracy more efficiently per-epoch," still useful but weaker. Either
      way, the discussion section can note this is deliberately narrow in
      scope — the paper's claim is about mechanism (why and when SAM
      helps), not a full efficiency-normalized recipe comparison (e.g.
      against structured/hardware-aware sparsity, where compute would
      actually shrink) — that's future work.
- [ ] Consider a stronger/adaptive-ρ SAM variant as an additional baseline
      once vanilla SAM-vs-SGD numbers are solid.

## Pipeline

- [ ] Confirm wall-clock time per run on the actual server hardware, then
      decide if `evaluate_flatness_every`/`eval_batches` in the configs need
      further tuning.
- [ ] `main_prune_finetune.py` (prune-then-finetune, as opposed to
      prune-during-training) is archived, not deleted — revisit if the
      iterative-pruning results need a finetuning-based comparison.

## Theory

Full write-up in `THEORY_NOTES.md` — supersedes
`archive/FUTURE_WORK_A1_RELAXATION.md` (the rebuttal-cycle draft of
Prop. 3.1′/the telescoping idea; that file's numbers predate the pruning-loop
and path-collision bug fixes and shouldn't be treated as ground truth).
Summary of where it landed:

- [x] Prop. 3.1′ (A1 relaxation) — proven, general, used directly.
- [x] Rate-bound extension showing the bare bounce fixed point is stable to
      `ηλ₁<2`, not just Bartlett et al.'s stricter `ηλ₁<1/2` (checked
      against arXiv:2210.01513v2 directly for what the stricter bound is
      actually needed for — cross-eigendirection separation and a
      non-quadratic-remainder margin, not the fixed point itself).
- [x] Identified that raw contraction rate is too fast (sub-epoch) to
      explain why 15-epoch recovery matters — redirected to, and then
      measured, `λ₁` itself relaxing within a round for SAM but not
      reliably for SGD (ResNet18, seed 13, both tight-recovery configs).
      Connected to the existing drift-term mechanism (Theorem 3.4/Prop.
      3.5) rather than treated as a new phenomenon.
- [ ] VGG16 λ₁-relaxation check attempted, inconclusive — the eigenvalue
      estimator (`HESSIAN_MAX_ITER=30`) doesn't converge on VGG16's more
      ill-conditioned restricted Hessian at these sparsities (values reach
      the millions, swing by orders of magnitude between adjacent
      checkpoints). Gradient-residual side (Prop. 3.1′ itself) does
      replicate on VGG16 cleanly; only the curvature-relaxation mechanism
      is unconfirmed there. Would need a substantially more robust
      eigenvalue solver to fix — not attempted further.
- [ ] Mechanism checked on one seed (13) only — 42/97 don't have per-round
      checkpoints (only seed 13 was saved every 5 epochs), so this can't be
      replicated without retraining.
- [ ] Open, not attempted: why SAM's drift achieves *more* null-space
      alignment than the `(k−r)/k` generic-direction baseline predicts
      (measured 0.52–0.83 vs. predicted ~0.998–1.0 in the original
      drift-effectiveness check) — a real gap in its own right.
