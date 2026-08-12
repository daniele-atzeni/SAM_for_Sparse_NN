# SAM for Sparse Neural Networks

Code for studying how Sharpness-Aware Minimization (SAM) affects loss-landscape
flatness and pruning robustness, compared to SGD.

**Current focus (re-run in progress):** iterative pruning-during-training for
**ResNet18** and **VGG16** on **CIFAR-10**, SAM vs. SGD, at sparsities
**0.7** and **0.9**, 3 seeds each. See `TODO.md` for what's deferred (more
sparsity levels, a transformer architecture, iso-compute comparisons).

## Repository structure

```text
├── configs/
│   ├── dense/                 # dense (no pruning) training configs
│   └── sparse/                # pruning-during-training configs
├── src/
│   ├── data/                  # dataset loaders (MNIST, CIFAR, ImageNet)
│   ├── models/                # model definitions (ResNet, VGG, WideResNet, ViT, MLP)
│   ├── train/                 # SAM optimizer, training loops, LR schedulers
│   ├── eval/                  # evaluation + Hessian-based flatness metrics
│   ├── pyhessian/              # PyHessian library (Yao & Gholami)
│   └── registry.py            # shared model/dataset/optimizer registries
├── scripts/
│   ├── run_dense.sh           # optional dense baselines (ResNet18/VGG16, 3 seeds)
│   ├── run_sparse_grid.sh     # main grid: {ResNet18,VGG16} x {s=0.7,0.9} x 3 seeds
│   └── parquet_to_imagefolder.py   # convert HF ImageNet parquets
├── main_training_dense.py     # entry point: dense training
├── main_training_sparse.py    # entry point: sparse (prune-during-training)
├── TODO.md                    # deferred experiments / open items
├── archive/                   # old exploratory work (gitignored, not pushed)
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the main grid

```bash
# All 3 seeds, both architectures, both sparsities (seed 0 keeps a checkpoint
# at every pruning round; seeds 1-2 keep only the final model)
bash scripts/run_sparse_grid.sh

# Just specific seeds
bash scripts/run_sparse_grid.sh 0
bash scripts/run_sparse_grid.sh 1 2

# A single run directly
python main_training_sparse.py \
    --config configs/sparse/ResNet18_CIFAR10_s0.9.json --seed 0
```

Each invocation trains **both** SAM and SGD from the same random init for
that (architecture, sparsity, seed) cell, so the comparison is fair. Restrict
to one optimizer with `--use-sam True` or `--use-sam False`.

Optional dense baselines (not required for the SAM-vs-SGD comparison, since
pruning starts from a shared random init rather than a dense-trained model):

```bash
bash scripts/run_dense.sh
```

## Outputs

- `saved_models/{dense,sparse}/<model>_<dataset>_.../seed_<n>/` — checkpoints
  and final weights, one subtree per seed.
- `tensorboard/runs_{dense,sparse}/.../seed_<n>/` — training curves. View with
  `tensorboard --logdir tensorboard/`.

Both directories are gitignored — nothing here gets pushed.

## Notes on the pipeline

- Pruning during training is global-unstructured L1 magnitude pruning,
  applied in `n_iter` equal-ratio rounds starting at epoch `first_iter`,
  every `prune_every` epochs (see the `sparse/*.json` configs).
- Routine per-epoch evaluation is forward-pass-only (`light=True` in
  `src/eval/eval.py`); the expensive SAM-loss/random-perturbation/Hessian
  diagnostics only run every `evaluate_flatness_every` epochs — needed to
  keep CNN-scale runs tractable.
- Seeds control weight init, and are baked into the save/log paths so
  parallel seeds never collide.

## Acknowledgements

- ViT implementation adapted from [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)
- Hessian computation uses [PyHessian](https://github.com/amirgholami/PyHessian) (Yao & Gholami, GPL-3.0)
