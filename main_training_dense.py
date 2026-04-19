import os
import json

import torch
import torch.nn as nn
import torch.optim as optim


from src.train.SAM import SAM
from src.train.training import train_loop
from src.models import *
from src.data import get_mnist_loaders, get_fashion_mnist_loaders, get_cifar10_loaders, get_cifar100_loaders, get_imagenet_loaders
from src.train.lr_scheduler import MultiStepLR, CosineAnnealingLR, WarmupCosineAnnealingLR



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
    # ImageNet ResNets (standard 7x7 stem, AdaptiveAvgPool)
    "ResNet18_IN": ResNet18_IN,
    "ResNet34_IN": ResNet34_IN,
    "ResNet50_IN": ResNet50_IN,
    "ResNet101_IN": ResNet101_IN,
    "ResNet152_IN": ResNet152_IN,
}

DATASET_NAME_TO_LOADER = {
    "MNIST": get_mnist_loaders,
    "fashionMNIST": get_fashion_mnist_loaders,
    "cifar10": get_cifar10_loaders,
    "cifar100": get_cifar100_loaders,
    "ImageNet": get_imagenet_loaders,
}


if __name__ == "__main__":

    #CONFIG_PATH = "./configs/dense/ViT_CIFAR10_config.json"
    CONFIG_PATH = "/home/datzeni/SAM_for_Sparse_NN/configs/dense/ResNet50_ImageNet_config.json"
    #CONFIG_PATH = "./configs/dense/MLP_FashionMNIST_config.json"
    config = json.load(open(CONFIG_PATH, "r"))

    # DATASET
    dataset_name = config["dataset"]["name"]
    batch_size = config["dataset"]["batch_size"]
    loader_kwargs = {"batch_size": batch_size}
    if "root" in config["dataset"]:
        loader_kwargs["root"] = config["dataset"]["root"]
    train_loader, test_loader = DATASET_NAME_TO_LOADER[dataset_name](**loader_kwargs)

    # MODEL
    model_name = config["model"]["name"]
    model_params = config["model"]["parameters"]
    model = MODEL_NAME_TO_CLASS[model_name](**model_params)
    # save initial parameters
    #MODEL_SAVE_PATH = f"./saved_models/dense/{model_name}_{dataset_name}"
    MODEL_SAVE_PATH = f"/home/datzeni/SAM_for_Sparse_NN/saved_models/dense/{model_name}_{dataset_name}_9"
    CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, "checkpoint")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    initial_state_dict = model.state_dict()
    INITIAL_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, f"{model_name}_{dataset_name}_initial.pth")
    torch.save(initial_state_dict, INITIAL_MODEL_SAVE_PATH)

    # TRAINING
    learning_rate = config["training"]["learning_rate"]
    scheduler_name = config["training"].get("scheduler", {}).get("type", None)
    if scheduler_name == "MultiStepLR":
        step_size = config["training"]["scheduler"]["step_size"]
        gamma = config["training"]["scheduler"]["gamma"]
        scheduler = MultiStepLR(learning_rate, step_size, gamma=gamma)
    elif scheduler_name == "CosineAnnealingLR":
        T_max = config["training"]["scheduler"]["T_max"]
        scheduler = CosineAnnealingLR(learning_rate, T_max=T_max)
    elif scheduler_name == "WarmupCosineAnnealingLR":
        T_max = config["training"]["scheduler"]["T_max"]
        warmup_epochs = config["training"]["scheduler"]["warmup_epochs"]
        scheduler = WarmupCosineAnnealingLR(learning_rate, T_max=T_max, warmup_epochs=warmup_epochs)
    else:
        scheduler = None

    criterion_name = config["training"]["loss_function"]
    if criterion_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {criterion_name}")


    USE_SAMS = [False, True]
    USE_SAMS = [False]
    MODE = "dense"  # "dense" or "sparse"

    #tensorboard_main_log_dir = f"./tensorboard/runs_{MODE}/{model_name}_{dataset_name}"
    tensorboard_main_log_dir = f"/home/datzeni/SAM_for_Sparse_NN/tensorboard/runs_{MODE}_9/{model_name}_{dataset_name}"    

    for use_sam in USE_SAMS:

        print(f"\n\nTraining {model_name} on {dataset_name} | Using SAM: {use_sam}\n")
        print(f"\nModel config:\n{config}\n")
        tensorboard_log_dir = tensorboard_main_log_dir + f"/SAM_{use_sam}"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load initial parameters
        model.load_state_dict(torch.load(INITIAL_MODEL_SAVE_PATH))
        model = model.to(device)

        # OPTIMIZER
        optimizer_name = config["training"]["optimizer"]
        if optimizer_name == "sgd":
            base_optimizer = optim.SGD
            weight_decay = config["training"].get("weight_decay", 1e-4)
            momentum = config["training"].get("momentum", 0.9)
            rho = config["training"].get("rho", 0.5)
        
            SGD_optimizer = base_optimizer(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay,
                momentum=momentum
            )
            SAM_optimizer = SAM(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    base_optimizer,
                    rho=rho, 
                    adaptive=False, 
                    lr=learning_rate,
                    weight_decay=weight_decay, 
                    momentum=momentum
                )
        elif optimizer_name == "adamw":
            base_optimizer = optim.AdamW
            weight_decay = config["training"].get("weight_decay", 1e-4)
            rho = config["training"].get("rho", 0.5)

            SGD_optimizer = base_optimizer(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
            SAM_optimizer = SAM(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    base_optimizer,
                    rho=rho, 
                    adaptive=False, 
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        epochs = config["training"]["epochs"]
        train_loop(
            epochs=epochs,
            use_sam=use_sam,
            model=model,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            SGD_optimizer=SGD_optimizer,
            SAM_optimizer=SAM_optimizer,
            criterion=criterion,
            scheduler=scheduler,
            tensorboard_log_dir=tensorboard_log_dir,
            checkpoint_folder=CHECKPOINT_PATH,
            save_every=config.get("save_every", epochs + 1), # if not specified, only save at the end of training
            evaluate_flatness_every=config.get("evaluate_flatness_every", 1),
            eval_batches=config.get("eval_batches", None)  # if not specified, evaluate on all batches
        )

        # save the model
        model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}_{dataset_name}_sam_{use_sam}.pth")
        torch.save(model.state_dict(), model_path)