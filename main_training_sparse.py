"""Sparse training: train with iterative pruning during training.

Usage:
    python main_training_sparse.py --config configs/sparse/ResNet18_CIFAR10_s0.9.json --seed 0
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from src.registry import (
    build_model,
    build_dataloaders,
    build_scheduler,
    build_criterion,
    build_optimizers,
)
from src.train.training import train_prune_loop


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Sparse (pruning-during-training)")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to a JSON config file"
    )
    parser.add_argument(
        "--use-sam",
        nargs="+",
        type=str,
        default=["True", "False"],
        help="Which SAM settings to run, e.g. --use-sam True False",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for init/data ordering"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Override config's save_every (checkpoint frequency in epochs). "
        "Pass a small value (e.g. equal to prune_every) to keep every "
        "pruning-round checkpoint for this run; leave unset for final-only.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    config = json.load(open(args.config))
    use_sam_list = [s.lower() == "true" for s in args.use_sam]

    # ---- Pruning schedule ----
    epochs = config["training"]["epochs"]
    first_iter = config.get("first_iter", 2)
    prune_every = config.get("prune_every", 2)
    prune_ratio = config.get("prune_ratio", 0.5)
    default_n_iter = (epochs - first_iter) // prune_every
    n_iter = config.get("n_iter", default_n_iter)

    # ---- Dataset ----
    dataset_name = config["dataset"]["name"]
    batch_size = config["dataset"]["batch_size"]
    train_loader, test_loader = build_dataloaders(dataset_name, batch_size)

    # ---- Model ----
    model_name = config["model"]["name"]
    model_params = config["model"]["parameters"]
    model = build_model(model_name, model_params)

    # ---- Save paths (seed-scoped so parallel seeds never collide) ----
    save_dir = os.path.join(
        "saved_models",
        "sparse",
        f"{model_name}_{dataset_name}_prune_ratio_{prune_ratio}",
        f"seed_{args.seed}",
    )
    checkpoint_dir = os.path.join(save_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    initial_path = os.path.join(
        save_dir, f"{model_name}_{dataset_name}_initial.pth"
    )
    torch.save(model.state_dict(), initial_path)

    # ---- Training config ----
    learning_rate = config["training"]["learning_rate"]
    criterion = build_criterion(config["training"]["loss_function"])
    scheduler = build_scheduler(config, learning_rate)
    evaluate_flatness_every = config.get("evaluate_flatness_every", 10)
    eval_batches = config.get("eval_batches")
    save_every = args.save_every if args.save_every is not None else config.get("save_every", epochs + 1)

    tb_root = os.path.join(
        "tensorboard",
        "runs_sparse",
        f"{model_name}_{dataset_name}_prune_ratio_{prune_ratio}",
        f"seed_{args.seed}",
    )

    for use_sam in use_sam_list:
        print(
            f"\n{'='*60}\n"
            f"Training {model_name} on {dataset_name} | SAM: {use_sam} | "
            f"Prune ratio: {prune_ratio} | Seed: {args.seed}\n"
            f"{'='*60}"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Re-create model from scratch to guarantee a clean state, loading
        # the SAME initial weights for both SAM and SGD (fair comparison).
        model = build_model(model_name, model_params)
        model.load_state_dict(torch.load(initial_path, map_location="cpu"))
        model = model.to(device)

        base_opt, sam_opt = build_optimizers(model, config, learning_rate)

        train_prune_loop(
            epochs=epochs,
            use_sam=use_sam,
            model=model,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            SGD_optimizer=base_opt,
            SAM_optimizer=sam_opt,
            criterion=criterion,
            scheduler=scheduler,
            tensorboard_log_dir=os.path.join(tb_root, f"SAM_{use_sam}"),
            checkpoint_folder=checkpoint_dir,
            save_every=save_every,
            first_iter=first_iter,
            prune_every=prune_every,
            prune_ratio=prune_ratio,
            n_iter=n_iter,
            evaluate_flatness_every=evaluate_flatness_every,
            eval_batches=eval_batches,
        )

        final_path = os.path.join(
            save_dir, f"{model_name}_{dataset_name}_sam_{use_sam}.pth"
        )
        torch.save(model.state_dict(), final_path)


if __name__ == "__main__":
    main()
