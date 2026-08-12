# TODO / deferred items

Tracking what's intentionally out of scope for the current re-run, so it
doesn't get lost.

## Experiment grid

- [ ] Add more sparsity levels beyond 0.7/0.9 — e.g. 0.5, 0.95 — once the
      current grid's cost/runtime is known.
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
