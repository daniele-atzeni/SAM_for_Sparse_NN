import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from src.train.SAM import SAM
from src.train.training import train_loop, train_prune_loop
from src.models import *
from src.data import get_mnist_loaders, get_fashion_mnist_loaders, get_cifar10_loaders, get_cifar100_loaders
from src.train.lr_scheduler import MultiStepLR

from src.eval.eval import evaluate


MODEL_NAME_TO_CLASS = {
    # Simple NN
    "SimpleNN": SimpleNN,
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
}


if __name__ == "__main__":
    # 1. Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.05
    EPOCHS = 5
    PRUNE_RATIOS = [0.5, 0.7, 0.9]
    STEP_SIZE = [80, 160]
    SCHEDULER = MultiStepLR(LEARNING_RATE, STEP_SIZE, gamma=0.1)
    CRITERION = nn.CrossEntropyLoss()
    DATASET_NAMES = ["cifar10"]
    # model
    MODEL_NAMES = ["vgg16_bn", "ResNet18", "WideResNet16_8"]
    MODEL_PARAMS = {}

    MODE = "dense"  # "dense"
    MODEL_SAVE_PATH = f"./src/saved_models"

    for model_name in MODEL_NAMES:
        TENSORBOARD_MAIN_LOG_DIR = f"./src/tensorboard/runs/{model_name}"

        for dataset_name in DATASET_NAMES:
            for prune_ratio in PRUNE_RATIOS:
                for sam_train in [True, False]:
                    for sam_finetune in [True, False]:
                        print(f"\n\nFine-tuning {model_name} on {dataset_name} | Orig train SAM: {sam_train} | Fine-tune SAM: {sam_finetune} | Prune Ratio: {prune_ratio}\n")
                
                        tensorboard_log_dir = TENSORBOARD_MAIN_LOG_DIR + f"/{dataset_name}_{sam_train}_{sam_finetune}_{prune_ratio}"

                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                        if dataset_name == "MNIST":
                            train_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE)
                        elif dataset_name == "fashionMNIST":
                            train_loader, test_loader = get_fashion_mnist_loaders(batch_size=BATCH_SIZE)
                        elif dataset_name == "cifar10":
                            train_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)
                        elif dataset_name == "cifar100":
                            train_loader, test_loader = get_cifar100_loaders(batch_size=BATCH_SIZE)
                        else:
                            raise ValueError(f"Unsupported dataset: {dataset_name}")
                
                        ###
                        if dataset_name in ["cifar10", "MNIST", "fashionMNIST"]:
                            num_classes = 10
                        elif dataset_name == "cifar100":
                            num_classes = 100
                        MODEL_PARAMS.update({"num_classes": num_classes})
                        ###
                        model = MODEL_NAME_TO_CLASS[model_name](**MODEL_PARAMS).to(device)
                        print("Loading trained model...")
                        trained_model_path = f"{MODEL_SAVE_PATH}/{model_name}_{dataset_name}_hiddenNone_sam{sam_train}_prune0.pth"
                        model.load_state_dict(torch.load(trained_model_path))

                        SGD_optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
                        SAM_optimizer = SAM(
                                filter(lambda p: p.requires_grad, model.parameters()), optim.SGD,
                                rho=0.5, adaptive=False, lr=LEARNING_RATE,
                                weight_decay=1e-4, momentum=0.9
                            )

                        # evaluate before training
                        eval_metrics = evaluate(model, device, test_loader, CRITERION)
                        # print metrics
                        print(f"Epoch {180-1}:")
                        for name, value in eval_metrics.items():
                            if value is not None:
                                    print(f"  Test {name}: {value:.4f}")
                        # prune the model
                        parameters_to_prune = [(module, 'weight') for _, module in model.named_modules() if isinstance(module, (nn.Linear, nn.Conv2d))]
                        prune.global_unstructured(
                            parameters_to_prune,
                            pruning_method=prune.L1Unstructured,
                            amount=prune_ratio,
                        )
                        

                        train_loop(
                            epochs=EPOCHS,
                            use_sam=sam_finetune,
                            model=model,
                            device=device,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            SGD_optimizer=SGD_optimizer,
                            SAM_optimizer=SAM_optimizer,
                            criterion=CRITERION,
                            scheduler=SCHEDULER,
                            tensorboard_log_dir=tensorboard_log_dir,
                            first_epoch=180
                        )   

                        # save the model
                        model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}_{dataset_name}_sam_train_{sam_train}_sam_finetune_{sam_finetune}_prune_{prune_ratio}.pth")
                        torch.save(model.state_dict(), model_path)