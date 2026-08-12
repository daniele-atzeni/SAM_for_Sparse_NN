"""Dense training: train a model from scratch (no pruning).

Usage:
    python main_training_dense.py --config configs/dense/ResNet18_CIFAR10.json --seed 0
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
from src.train.training import train_loop


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Dense training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to a JSON config file"
    )
    parser.add_argument(
        "--use-sam",
        nargs="+",
        type=str,
        default=["False", "True"],
        help="Which SAM settings to run, e.g. --use-sam False True",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for init/data ordering"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    config = json.load(open(args.config))
    use_sam_list = [s.lower() == "true" for s in args.use_sam]

    # ---- Dataset ----
    dataset_name = config["dataset"]["name"]
    batch_size = config["dataset"]["batch_size"]
    loader_kwargs = {}
    if "root" in config["dataset"]:
        loader_kwargs["root"] = config["dataset"]["root"]
    train_loader, test_loader = build_dataloaders(
        dataset_name, batch_size, **loader_kwargs
    )

    # ---- Model ----
    model_name = config["model"]["name"]
    model_params = config["model"]["parameters"]
    model = build_model(model_name, model_params)

    # ---- Save paths (seed-scoped so parallel seeds never collide) ----
    save_dir = os.path.join(
        "saved_models", "dense", f"{model_name}_{dataset_name}", f"seed_{args.seed}"
    )
    checkpoint_dir = os.path.join(save_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save initial weights so both SAM/non-SAM runs start identically
    initial_state = model.state_dict()
    initial_path = os.path.join(save_dir, f"{model_name}_{dataset_name}_initial.pth")
    torch.save(initial_state, initial_path)

    # ---- Training config ----
    learning_rate = config["training"]["learning_rate"]
    epochs = config["training"]["epochs"]
    criterion = build_criterion(config["training"]["loss_function"])
    scheduler = build_scheduler(config, learning_rate)

    tb_root = os.path.join(
        "tensorboard", "runs_dense", f"{model_name}_{dataset_name}", f"seed_{args.seed}"
    )

    for use_sam in use_sam_list:
        print(
            f"\n{'='*60}\n"
            f"Training {model_name} on {dataset_name} | SAM: {use_sam} | Seed: {args.seed}\n"
            f"{'='*60}"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reload initial weights for a fair comparison
        model.load_state_dict(torch.load(initial_path, map_location="cpu"))
        model = model.to(device)

        base_opt, sam_opt = build_optimizers(model, config, learning_rate)

        train_loop(
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
            save_every=config.get("save_every", epochs + 1),
            evaluate_flatness_every=config.get("evaluate_flatness_every", 10),
            eval_batches=config.get("eval_batches"),
        )

        # Save final model
        final_path = os.path.join(
            save_dir, f"{model_name}_{dataset_name}_sam_{use_sam}.pth"
        )
        torch.save(model.state_dict(), final_path)


if __name__ == "__main__":
    main()
