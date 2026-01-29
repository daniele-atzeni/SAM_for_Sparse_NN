import os

import torch
import torch.nn as nn
import torch.optim as optim


from src.train.SAM import SAM
from src.train.training import train_loop, train_prune_loop
from src.models import *
from src.data import get_mnist_loaders, get_fashion_mnist_loaders, get_cifar10_loaders, get_cifar100_loaders
from src.train.lr_scheduler import MultiStepLR



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
    #LEARNING_RATE = 0.01
    LEARNING_RATE = 0.05
    EPOCHS = 20
    #EPOCHS = 180
    ##
    FIRST_ITER = 2
    PRUNE_EVERY = 2
    N_ITER = 10
    #PRUNE_RATIOS = [0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 0.9999]
    PRUNE_RATIOS = [0.7, 0.9, 0.99]
    ##
    USE_SAMS = [False, True]
    #HIDDEN_SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    HIDDEN_SIZES = [128]
    STEP_SIZE = [80, 160]
    SCHEDULER = MultiStepLR(LEARNING_RATE, STEP_SIZE, gamma=0.1)
    CRITERION = nn.CrossEntropyLoss()
    #DATASET_NAMES = ["MNIST", "fashionMNIST", "cifar10", "cifar100"]
    DATASET_NAMES = ["fashionMNIST"]
    # model
    MODEL_NAME = "SimpleNN"
    #MODEL_NAME = "vgg16_bn"
    #MODEL_NAME = "ResNet18"
    #MODEL_NAME = "WideResNet16_8"
    MODEL_PARAMS = {}

    MODE = "dense"  # "dense" or "sparse"
    MODEL_SAVE_PATH = f"./src/saved_models_last"

    
    TENSORBOARD_MAIN_LOG_DIR = f"./src/tensorboard/runs/{MODEL_NAME}_test"

    # Instantiate the model
    for use_sam in USE_SAMS:
        for hidden_size in HIDDEN_SIZES:
            for dataset_name in DATASET_NAMES:
                #for prune_ratio in PRUNE_RATIOS:
                if MODE == "dense":
                    prune_ratio = 0

                print(f"\n\nTraining {MODEL_NAME} on {dataset_name} with hidden size {hidden_size} | Using SAM: {use_sam} | Prune Ratio: {prune_ratio}\n")
                tensorboard_log_dir = TENSORBOARD_MAIN_LOG_DIR + f"/{dataset_name}_{use_sam}_{hidden_size}_{prune_ratio}"

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
                if MODEL_NAME == "SimpleNN":
                    MODEL_PARAMS.update({"hidden_size": hidden_size})
                if dataset_name in ["cifar10", "MNIST", "fashionMNIST"]:
                    num_classes = 10
                elif dataset_name == "cifar100":
                    num_classes = 100
                MODEL_PARAMS.update({"num_classes": num_classes})
                ###
                model = MODEL_NAME_TO_CLASS[MODEL_NAME](**MODEL_PARAMS).to(device)
                # save model initial state
                initial_model_path = os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}_{dataset_name}_hidden_{hidden_size}_initial.pth")
                if not os.path.exists(initial_model_path):
                    torch.save(model.state_dict(), initial_model_path)
                else:
                    print("Loading initial model state...")
                    model.load_state_dict(torch.load(initial_model_path))

                SGD_optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
                SAM_optimizer = SAM(
                        filter(lambda p: p.requires_grad, model.parameters()), optim.SGD,
                        rho=0.5, adaptive=False, lr=LEARNING_RATE,
                        weight_decay=1e-4, momentum=0.9
                    )

                if MODE == "dense":
                    train_loop(
                        epochs=EPOCHS,
                        use_sam=use_sam,
                        model=model,
                        device=device,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        SGD_optimizer=SGD_optimizer,
                        SAM_optimizer=SAM_optimizer,
                        criterion=CRITERION,
                        scheduler=SCHEDULER,
                        tensorboard_log_dir=tensorboard_log_dir
                    )   
                elif MODE == "sparse":
                    train_prune_loop(
                        epochs=EPOCHS,
                        use_sam=use_sam,
                        model=model,
                        device=device,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        SGD_optimizer=SGD_optimizer,
                        SAM_optimizer=SAM_optimizer,
                        criterion=CRITERION,
                        prune_ratio=prune_ratio,
                        first_iter=FIRST_ITER,
                        prune_every=PRUNE_EVERY,
                        n_iter=N_ITER,
                        scheduler=SCHEDULER,
                        tensorboard_log_dir=tensorboard_log_dir
                    )
                else:
                    raise ValueError(f"Unsupported mode: {MODE}")

                # save the model
                model_path = os.path.join(MODEL_SAVE_PATH, f"{MODEL_NAME}_{dataset_name}_hidden_{hidden_size}_sam_{use_sam}_prune_{prune_ratio}_same_init.pth")
                torch.save(model.state_dict(), model_path)