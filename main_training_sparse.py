"""Sparse training: train with iterative pruning during training.

Usage:
    python main_training_sparse.py --config configs/sparse/MLP_MNIST_config.json
"""

import argparse
import json
import os

import torch

from src.registry import (
    build_model,
    build_dataloaders,
    build_scheduler,
    build_criterion,
    build_optimizers,
)
from src.train.training import train_prune_loop


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
    args = parser.parse_args()

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

    # ---- Save paths ----
    save_dir = os.path.join(
        "saved_models",
        "sparse",
        f"{model_name}_{dataset_name}_prune_ratio_{prune_ratio}",
    )
    checkpoint_dir = os.path.join(save_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Try to load matching dense initialisation for a fair comparison
    dense_init_path = os.path.join(
        "saved_models",
        "dense",
        f"{model_name}_{dataset_name}",
        f"{model_name}_{dataset_name}_initial.pth",
    )
    if os.path.exists(dense_init_path):
        print(f"Loading initial weights from {dense_init_path}")
        model.load_state_dict(torch.load(dense_init_path, map_location="cpu"))
    else:
        print(f"No dense initialisation found at {dense_init_path}; using random init")

    initial_path = os.path.join(
        save_dir, f"{model_name}_{dataset_name}_initial.pth"
    )
    torch.save(model.state_dict(), initial_path)

    # ---- Training config ----
    learning_rate = config["training"]["learning_rate"]
    criterion = build_criterion(config["training"]["loss_function"])
    scheduler = build_scheduler(config, learning_rate)

    tb_root = os.path.join(
        "tensorboard",
        "runs_sparse",
        f"{model_name}_{dataset_name}_prune_ratio_{prune_ratio}",
    )

    for use_sam in use_sam_list:
        print(
            f"\n{'='*60}\n"
            f"Training {model_name} on {dataset_name} | SAM: {use_sam} | "
            f"Prune ratio: {prune_ratio}\n"
            f"{'='*60}"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Re-create model from scratch to guarantee a clean state
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
            save_every=config.get("save_every", epochs + 1),
            first_iter=first_iter,
            prune_every=prune_every,
            prune_ratio=prune_ratio,
            n_iter=n_iter,
        )

        final_path = os.path.join(
            save_dir, f"{model_name}_{dataset_name}_sam_{use_sam}.pth"
        )
        torch.save(model.state_dict(), final_path)


if __name__ == "__main__":
    main()
