import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from src.train.SAM import SAM
from src.train.training import train_loop, train_prune_loop
from torch.utils.tensorboard import SummaryWriter
from src.models import *
from src.data import get_mnist_loaders, get_fashion_mnist_loaders, get_cifar10_loaders, get_cifar100_loaders
from src.train.lr_scheduler import CosineAnnealingLR, MultiStepLR, WarmupCosineAnnealingLR

from src.eval.eval import evaluate, post_pruning_metrics, weight_distribution_metrics


MODEL_NAME_TO_CLASS = {
    # Simple NN
    "MLP": MLP,
    # ResNet Plus
    "ResNet20": ResNet20,
    "ResNet32": ResNet32,
    "ResNet44": ResNet44,
    "ResNet56": ResNet56,
    "ResNet110": ResNet110,
    # ResNet
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    # VGG Plus
    'vgg11_plus': vgg11_plus,
    'vgg11_bn_plus': vgg11_bn_plus,
    'vgg13_mingze': vgg13_mingze,
    'vgg16_mingze': vgg16_mingze,
    'vgg19_mingze': vgg19_mingze,
    # VGG
    'vgg11': vgg11,
    'vgg11_bn': vgg11_bn,
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19': vgg19,
    'vgg19_bn': vgg19_bn,
    # Wide ResNet madry
    "WideResNet34_10_madry": WideResNet34_10_madry,
    "WideResNet16_8_madry": WideResNet16_8_madry,
    # Wide ResNet
    "WideResNet16_8": WideResNet16_8,
    "WideResNet28_10": WideResNet28_10,
    # Vision Transformer
    "ViT": ViT,
}

DATASET_NAME_TO_LOADER = {
    "MNIST": get_mnist_loaders,
    "fashionMNIST": get_fashion_mnist_loaders,
    "cifar10": get_cifar10_loaders,
    "cifar100": get_cifar100_loaders
}

def build_scheduler(learning_rate: float, training_config: dict):
    scheduler_name = training_config.get("scheduler", {}).get("type")
    if scheduler_name == "MultiStepLR":
        step_size = training_config["scheduler"]["step_size"]
        gamma = training_config["scheduler"]["gamma"]
        return MultiStepLR(learning_rate, step_size, gamma=gamma)
    if scheduler_name == "CosineAnnealingLR":
        t_max = training_config["scheduler"]["T_max"]
        return CosineAnnealingLR(learning_rate, T_max=t_max)
    if scheduler_name == "WarmupCosineAnnealingLR":
        t_max = training_config["scheduler"]["T_max"]
        warmup_epochs = training_config["scheduler"]["warmup_epochs"]
        return WarmupCosineAnnealingLR(learning_rate, T_max=t_max, warmup_epochs=warmup_epochs)
    return None


if __name__ == "__main__":
    
    CONFIG_PATH = "/home/datzeni/SAM_for_Sparse_NN/configs/finetune/MLP_MNIST_config.json"
    config = json.load(open(CONFIG_PATH, "r"))

    dataset_name = config["dataset"]["name"]
    batch_size = config["dataset"]["batch_size"]
    train_loader, test_loader = DATASET_NAME_TO_LOADER[dataset_name](batch_size=batch_size)

    model_name = config["model"]["name"]
    model_params = config["model"]["parameters"]
    dense_model_dir = os.path.join(".", "saved_models", "dense", f"{model_name}_{dataset_name}")
    dense_epochs = config["training"]["dense_epochs"]
    finetune_epochs = config["training"]["finetune_epochs"]
    pruning_ratios = config["pruning_ratios"]
    use_sams = [False, True]

    learning_rate = config["training"]["learning_rate"]
    optimizer_name = config["training"]["optimizer"]
    scheduler = build_scheduler(learning_rate, config["training"])
    evaluate_flatness_every = 1

    criterion_name = config["training"]["loss_function"]
    if criterion_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {criterion_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for pruning_ratio in pruning_ratios:
        finetune_model_dir = os.path.join(
            ".",
            "saved_models",
            f"prune_finetune",
            f"{model_name}_{dataset_name}_prune_ratio_{pruning_ratio}",
        )
        tensorboard_root = os.path.join(
            ".",
            "tensorboard",
            f"runs_prune_finetune",
            f"{model_name}_{dataset_name}_prune_ratio_{pruning_ratio}",
        )
        
        for sam_train in use_sams:
            trained_model_path = os.path.join(dense_model_dir, f"{model_name}_{dataset_name}_sam_{sam_train}.pth")
            if not os.path.exists(trained_model_path):
                raise FileNotFoundError(f"Missing dense checkpoint: {trained_model_path}")
            
            for sam_finetune in use_sams:
                print(
                    f"\n\nFine-tuning {model_name} on {dataset_name} | "
                    f"Orig train SAM: {sam_train} | Fine-tune SAM: {sam_finetune} | "
                    f"Prune Ratio: {pruning_ratio}\n"
                )

                model = MODEL_NAME_TO_CLASS[model_name](**model_params)
                model.load_state_dict(torch.load(trained_model_path, map_location="cpu"))
                model = model.to(device)

                eval_metrics = evaluate(model, device, test_loader, criterion)
                print(f"Epoch {dense_epochs}:")
                for name, value in eval_metrics.items():
                    if value is not None:
                        print(f"  Test {name}: {value:.4f}")

                # Pre-pruning weight distribution (Proposition 6: L1/L2 ratio & Gini for dense model)
                pre_prune_dist = weight_distribution_metrics(model)
                print("Pre-pruning weight distribution: " + ", ".join(
                    f"{k}: {v:.6f}" for k, v in pre_prune_dist.items()
                ))

                parameters_to_prune = [
                    (module, "weight")
                    for _, module in model.named_modules()
                    if isinstance(module, (nn.Linear, nn.Conv2d))
                ]
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_ratio,
                )

                # Post-pruning metrics: masked gradient norm + weight distribution (Propositions 3 & 6)
                post_prune = post_pruning_metrics(model, device, train_loader, criterion)
                print("Post-pruning metrics: " + ", ".join(
                    f"{k}: {v:.6f}" for k, v in post_prune.items()
                ))

                # Log pre/post pruning metrics to TensorBoard at step=dense_epochs
                with SummaryWriter(log_dir=tensorboard_log_dir) as writer:
                    for k, v in pre_prune_dist.items():
                        writer.add_scalar(f"{k}/pre_pruning", v, dense_epochs)
                    for k, v in post_prune.items():
                        writer.add_scalar(f"{k}/post_pruning", v, dense_epochs)

                if optimizer_name == "sgd":
                    weight_decay = config["training"].get("weight_decay", 1e-4)
                    momentum = config["training"].get("momentum", 0.9)
                    rho = config["training"].get("rho", 0.5)
                    base_optimizer = optim.SGD

                    sgd_optimizer = base_optimizer(
                        model.parameters(),
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        momentum=momentum,
                    )
                    sam_optimizer = SAM(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        base_optimizer,
                        rho=rho,
                        adaptive=False,
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        momentum=momentum,
                    )
                elif optimizer_name == "adamw":
                    weight_decay = config["training"].get("weight_decay", 1e-4)
                    rho = config["training"].get("rho", 0.5)
                    base_optimizer = optim.AdamW

                    sgd_optimizer = base_optimizer(
                        model.parameters(),
                        lr=learning_rate,
                        weight_decay=weight_decay,
                    )
                    sam_optimizer = SAM(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        base_optimizer,
                        rho=rho,
                        adaptive=False,
                        lr=learning_rate,
                        weight_decay=weight_decay,
                    )
                else:
                    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

                tensorboard_log_dir = os.path.join(
                    tensorboard_root,
                    f"sam_train_{sam_train}_sam_finetune_{sam_finetune}",
                )
                checkpoint_dir = os.path.join(
                    finetune_model_dir,
                    "checkpoint",
                    f"sam_train_{sam_train}_sam_finetune_{sam_finetune}",
                )
                os.makedirs(checkpoint_dir, exist_ok=True)

                train_loop(
                    epochs=finetune_epochs,
                    use_sam=sam_finetune,
                    model=model,
                    device=device,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    SGD_optimizer=sgd_optimizer,
                    SAM_optimizer=sam_optimizer,
                    criterion=criterion,
                    scheduler=scheduler,
                    tensorboard_log_dir=tensorboard_log_dir,
                    first_epoch=dense_epochs,
                    checkpoint_folder=checkpoint_dir,
                    save_every=1,
                    evaluate_flatness_every=evaluate_flatness_every,
                )

                model_path = os.path.join(
                    finetune_model_dir,
                    f"{model_name}_{dataset_name}_sam_train_{sam_train}_sam_finetune_{sam_finetune}.pth",
                )
                torch.save(model.state_dict(), model_path)