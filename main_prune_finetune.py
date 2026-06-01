"""Prune-then-finetune: load a dense-trained model, prune, and finetune.

Usage:
    python main_prune_finetune.py --config configs/finetune/MLP_MNIST_config.json
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
    parser = argparse.ArgumentParser(description="Prune → finetune")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to a JSON config file"
    )
    args = parser.parse_args()

    config = json.load(open(args.config))

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

    dense_model_dir = os.path.join("saved_models", "dense", f"{model_name}_{dataset_name}")
    dense_epochs = config["training"]["dense_epochs"]
    finetune_epochs = config["training"]["finetune_epochs"]
    pruning_ratios = config["pruning_ratios"]
    use_sam_options = [False, True]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for pruning_ratio in pruning_ratios:
        finetune_dir = os.path.join(
            "saved_models",
            "prune_finetune",
            f"{model_name}_{dataset_name}_prune_ratio_{pruning_ratio}",
        )
        tb_root = os.path.join(
            "tensorboard",
            "runs_prune_finetune",
            f"{model_name}_{dataset_name}_prune_ratio_{pruning_ratio}",
        )

        for sam_train in use_sam_options:
            trained_path = os.path.join(
                dense_model_dir,
                f"{model_name}_{dataset_name}_sam_{sam_train}.pth",
            )
            if not os.path.exists(trained_path):
                raise FileNotFoundError(f"Missing dense checkpoint: {trained_path}")

            for sam_finetune in use_sam_options:
                tag = f"sam_train_{sam_train}_sam_finetune_{sam_finetune}"
                print(
                    f"\n{'='*60}\n"
                    f"Finetune {model_name}/{dataset_name} | "
                    f"Train-SAM: {sam_train} | FT-SAM: {sam_finetune} | "
                    f"Prune: {pruning_ratio}\n"
                    f"{'='*60}"
                )

                model = build_model(model_name, model_params)
                model.load_state_dict(
                    torch.load(trained_path, map_location="cpu")
                )
                model = model.to(device)

                # Pre-pruning evaluation
                eval_metrics = evaluate(model, device, test_loader, criterion)
                print(f"Epoch {dense_epochs}:")
                for k, v in eval_metrics.items():
                    if v is not None:
                        print(f"  Test {k}: {v:.4f}")

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

                post_prune = post_pruning_metrics(
                    model, device, train_loader, criterion
                )
                print(
                    "Post-pruning metrics: "
                    + ", ".join(f"{k}: {v:.6f}" for k, v in post_prune.items())
                )

                # Log pre/post pruning metrics
                tb_log_dir = os.path.join(tb_root, tag)
                with SummaryWriter(log_dir=tb_log_dir) as writer:
                    for k, v in pre_prune_dist.items():
                        writer.add_scalar(f"{k}/pre_pruning", v, dense_epochs)
                    for k, v in post_prune.items():
                        writer.add_scalar(f"{k}/post_pruning", v, dense_epochs)

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
                    first_epoch=dense_epochs,
                    checkpoint_folder=ckpt_dir,
                    save_every=1,
                    evaluate_flatness_every=1,
                )

                final_path = os.path.join(
                    finetune_dir,
                    f"{model_name}_{dataset_name}_{tag}.pth",
                )
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                torch.save(model.state_dict(), final_path)


if __name__ == "__main__":
    main()
