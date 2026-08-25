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
- [ ] If the strong-pruning sweep still shows little divergence, the next
      lever to try is shortening the recovery budget itself (extend
      pruning closer to epoch 180 instead of finishing at epoch 55) rather
      than pushing sparsity further — not yet configured.
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
