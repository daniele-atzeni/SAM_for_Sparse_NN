"""Prune-then-finetune: load a dense-trained model, prune, and finetune.

Revived from archive/main_prune_finetune.py (which was never touched by the
Aug 12 2026 bug-fix pass) with three fixes:
  - train_loop now gets pruned=True for the finetuning call, so the
    Hessian trace/eigenvalue/SAM-Loss diagnostics it logs are computed on
    the restricted (masked) Hessian instead of silently defaulting to the
    unmasked one -- the same class of bug train_prune_loop had, just never
    fixed here since this script wasn't part of the sparse-training redo.
  - --seed support, seed-scoped output paths, and seed-matched dense
    checkpoint loading (seed N's finetune starts from seed N's dense
    checkpoint, not a single shared one) -- consistent with every other
    entry point in this repo post-redo.
  - dense_tag in the config names the dense checkpoint directory directly
    (e.g. "ResNet18_cifar10"), since current dense checkpoints are keyed by
    config-filename-derived tags that don't necessarily match model_name.

Usage:
    python main_prune_finetune.py --config configs/finetune/ResNet18_CIFAR10.json --seed 13
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.tensorboard import SummaryWriter

from src.registry import (
    build_model,
    build_dataloaders,
    build_scheduler,
    build_criterion,
    build_optimizers,
)
from src.train.training import train_loop
from src.eval.eval import evaluate, post_pruning_metrics, weight_distribution_metrics


def main():
    parser = argparse.ArgumentParser(description="Prune -> finetune")
    parser.add_argument("--config", type=str, required=True, help="Path to a JSON config file")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--use-sam", nargs="+", default=["True", "False"],
        help="Which SAM settings to use for the ORIGINAL dense training, e.g. --use-sam True False",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    config = json.load(open(args.config))
    config_tag = os.path.splitext(os.path.basename(args.config))[0]

    # ---- Dataset ----
    dataset_name = config["dataset"]["name"]
    batch_size = config["dataset"]["batch_size"]
    train_loader, test_loader = build_dataloaders(dataset_name, batch_size)

    # ---- Shared training config ----
    model_name = config["model"]["name"]
    model_params = config["model"]["parameters"]
    learning_rate = config["training"]["learning_rate"]
    criterion = build_criterion(config["training"]["loss_function"])
    scheduler = build_scheduler(config, learning_rate)

    dense_tag = config["dense_tag"]
    dense_model_dir = os.path.join("saved_models", "dense", dense_tag, f"seed_{args.seed}")
    finetune_epochs = config["training"]["finetune_epochs"]
    pruning_ratios = config["pruning_ratios"]
    use_sam_train_options = [s.lower() == "true" for s in args.use_sam]
    use_sam_finetune_options = [True, False]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for pruning_ratio in pruning_ratios:
        finetune_dir = os.path.join(
            "saved_models", "prune_finetune", config_tag, f"seed_{args.seed}",
            f"prune_ratio_{pruning_ratio}",
        )
        tb_root = os.path.join(
            "tensorboard", "runs_prune_finetune", config_tag, f"seed_{args.seed}",
            f"prune_ratio_{pruning_ratio}",
        )

        for sam_train in use_sam_train_options:
            trained_path = os.path.join(
                dense_model_dir, f"{dense_tag}_sam_{sam_train}.pth",
            )
            if not os.path.exists(trained_path):
                raise FileNotFoundError(f"Missing dense checkpoint: {trained_path}")

            for sam_finetune in use_sam_finetune_options:
                tag = f"sam_train_{sam_train}_sam_finetune_{sam_finetune}"
                print(
                    f"\n{'='*60}\n"
                    f"Finetune {model_name}/{dataset_name} | seed {args.seed} | "
                    f"Train-SAM: {sam_train} | FT-SAM: {sam_finetune} | "
                    f"Prune: {pruning_ratio}\n"
                    f"{'='*60}"
                )

                model = build_model(model_name, model_params)
                model.load_state_dict(torch.load(trained_path, map_location="cpu"))
                model = model.to(device)

                # Pre-pruning evaluation
                eval_metrics = evaluate(model, device, test_loader, criterion)
                for k, v in eval_metrics.items():
                    if v is not None:
                        print(f"  Pre-prune test {k}: {v:.4f}")

                pre_prune_dist = weight_distribution_metrics(model)
                print(
                    "Pre-pruning weight distribution: "
                    + ", ".join(f"{k}: {v:.6f}" for k, v in pre_prune_dist.items())
                )

                # Prune
                params_to_prune = [
                    (m, "weight")
                    for _, m in model.named_modules()
                    if isinstance(m, (nn.Linear, nn.Conv2d))
                ]
                prune.global_unstructured(
                    params_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_ratio,
                )

                post_prune = post_pruning_metrics(model, device, train_loader, criterion)
                print(
                    "Post-pruning metrics: "
                    + ", ".join(f"{k}: {v:.6f}" for k, v in post_prune.items())
                )

                tb_log_dir = os.path.join(tb_root, tag)
                with SummaryWriter(log_dir=tb_log_dir) as writer:
                    for k, v in pre_prune_dist.items():
                        writer.add_scalar(f"{k}/pre_pruning", v, 0)
                    for k, v in post_prune.items():
                        writer.add_scalar(f"{k}/post_pruning", v, 0)

                base_opt, sam_opt = build_optimizers(model, config, learning_rate)

                ckpt_dir = os.path.join(finetune_dir, "checkpoint", tag)
                os.makedirs(ckpt_dir, exist_ok=True)

                train_loop(
                    epochs=finetune_epochs,
                    use_sam=sam_finetune,
                    model=model,
                    device=device,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    SGD_optimizer=base_opt,
                    SAM_optimizer=sam_opt,
                    criterion=criterion,
                    scheduler=scheduler,
                    tensorboard_log_dir=tb_log_dir,
                    first_epoch=0,
                    checkpoint_folder=ckpt_dir,
                    save_every=finetune_epochs,
                    evaluate_flatness_every=1,
                    pruned=True,
                )

                final_path = os.path.join(finetune_dir, f"{config_tag}_{tag}.pth")
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                torch.save(model.state_dict(), final_path)


if __name__ == "__main__":
    main()
