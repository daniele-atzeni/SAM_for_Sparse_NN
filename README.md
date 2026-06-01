# SAM for Sparse Neural Networks

Code for the NeurIPS submission on **Sharpness-Aware Minimisation (SAM) and neural-network sparsity**.

This repository provides training pipelines for studying how SAM affects loss-landscape flatness and pruning performance across several architectures (MLP, ResNet, VGG, Wide ResNet, Vision Transformer) and datasets (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, ImageNet).

## Repository structure

```
├── configs/                  # JSON experiment configs
│   ├── dense/                #   dense (no pruning) training
│   ├── sparse/               #   pruning-during-training
│   └── finetune/             #   prune-then-finetune
├── src/
│   ├── data/                 # Dataset loaders (MNIST, CIFAR, ImageNet)
│   ├── models/               # Model definitions
│   ├── train/                # SAM optimizer, training loops, LR schedulers
│   ├── eval/                 # Evaluation, Hessian-based flatness metrics
│   ├── pyhessian/            # PyHessian library (Yao & Gholami)
│   └── registry.py           # Shared model/dataset/optimizer registries
├── scripts/
│   └── parquet_to_imagefolder.py   # Convert HF ImageNet parquets
├── notebooks/                # (*.ipynb at repo root) Analysis & visualisation
├── main_training_dense.py    # Entry point: dense training
├── main_training_sparse.py   # Entry point: sparse (prune-during-training)
├── main_prune_finetune.py    # Entry point: prune → finetune
└── requirements.txt
```

## Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Every entry point reads a JSON config file and accepts optional CLI flags.

### Dense training

```bash
python main_training_dense.py --config configs/dense/MLP_MNIST_config.json
```

By default this runs both with and without SAM. To run only one setting:

```bash
python main_training_dense.py --config configs/dense/ResNet_CIFAR10_config.json --use-sam True
```

### Sparse training (pruning during training)

```bash
python main_training_sparse.py --config configs/sparse/MLP_MNIST_config.json
```

### Prune → finetune

Requires a pre-trained dense model in `saved_models/dense/`:

```bash
python main_prune_finetune.py --config configs/finetune/MLP_MNIST_config.json
```

### ImageNet

1. Download the ImageNet dataset (or a subset like ImageNet-100) and convert to ImageFolder layout:
   ```bash
   python scripts/parquet_to_imagefolder.py \
       --parquet_root /path/to/hf_download \
       --output_root  ./src/data/DATA/ImageNet
   ```
2. Update the `"root"` field in the config JSON, then run:
   ```bash
   python main_training_dense.py --config configs/dense/ResNet50_ImageNet_config.json
   ```

### TensorBoard

Training metrics are logged to `tensorboard/`. Visualise with:

```bash
tensorboard --logdir tensorboard/
```

## Acknowledgements

- ViT implementation adapted from [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)
- Hessian computation uses [PyHessian](https://github.com/amirgholami/PyHessian) (Yao & Gholami, GPL-3.0)
