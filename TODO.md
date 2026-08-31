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
      same sparsity doesn't show the same ceiling).
- [ ] Two follow-ups now configured, not yet run:
      1. **Recovery-budget variant** — `configs/sparse/{ResNet18,VGG16}_CIFAR10_s0.9_shortrecovery.json`
         + `scripts/run_sparse_recovery_budget.sh`. Same s=0.9, but pruning
         spread over 11 rounds every 15 epochs (epochs 15-165), leaving
         only 15 epochs of recovery instead of 125. Isolates the recovery-
         budget variable while holding sparsity fixed.
      2. **Capacity-wall sweep** — `configs/sparse/{ResNet18,VGG16}_CIFAR10_s{0.995,0.999,0.9995}.json`
         + `scripts/run_sparse_grid_extreme.sh`. Same schedule as the
         original grid (125 epochs recovery), pushing sparsity to where
         active-parameter counts approach/fall below the 50k-image
         training set (down to ~5.6k active for ResNet18, ~16.8k for
         VGG16 at s=0.9995), looking for where SGD's own *training*
         accuracy finally drops — the real under-parameterized signature.
- [ ] Add a transformer architecture (ViT is already in `src/models/`,
      wire up a `configs/sparse/ViT_CIFAR10_s*.json` once ResNet/VGG results
      are in).
- [ ] Iso-compute comparison (SAM uses ~2x the gradient computations per
      step of SGD; current grid matches epoch count, not FLOPs).
- [ ] Consider a stronger/adaptive-ρ SAM variant as an additional baseline
      once vanilla SAM-vs-SGD numbers are solid.

## Pipeline

- [ ] Confirm wall-clock time per run on the actual server hardware, then
      decide if `evaluate_flatness_every`/`eval_batches` in the configs need
      further tuning.
- [ ] `main_prune_finetune.py` (prune-then-finetune, as opposed to
      prune-during-training) is archived, not deleted — revisit if the
      iterative-pruning results need a finetuning-based comparison.

## Theory (next phase, after the experimental re-run is running)

- Revisit the rebuttal-cycle findings in `archive/` once the new empirical
  numbers land — several of the earlier CNN measurements (`archive/cnn_logs`,
  `archive/tmp_analysis`) were retroactive analyses of runs affected by the
  pruning-loop bug fixed in this cleanup (see `src/train/training.py`,
  `train_prune_loop`), so they should not be treated as ground truth going
  forward.
- Proposition 3.1' (A1 relaxation) and the telescoping extension for
  iterative pruning are drafted in `archive/FUTURE_WORK_A1_RELAXATION.md` —
  worth revisiting once clean multi-seed CNN data exists to test against.
