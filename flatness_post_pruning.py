import torch
import torch.nn.utils.prune as prune

import pandas as pd


from src.data import get_cifar10_loaders, get_mnist_loaders
from src.models import WideResNet16_8, SimpleNN
from src.train.eval import evaluate


#dataset = "cifar10"
dataset = "MNIST"
batch_size = 128
if dataset == "MNIST":
    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
elif dataset == "cifar10":
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
else:
    raise ValueError(f"Unsupported dataset: {dataset}")

# model_sam = WideResNet16_8(num_classes=10)
# model_sgd = WideResNet16_8(num_classes=10)
model_sam = SimpleNN(hidden_size=128, num_classes=10)
model_sgd = SimpleNN(hidden_size=128, num_classes=10)

# sam_model_path = "src/saved_models/WideResNet16_8_cifar10_hiddenNone_samTrue_prune0.pth"
# sgd_model_path = "src/saved_models/WideResNet16_8_cifar10_hiddenNone_samFalse_prune0.pth"
sam_model_path = "src/saved_models/SimpleNN_MNIST_hidden_128_sam_True_prune_0.0.pth"
sgd_model_path = "src/saved_models/SimpleNN_MNIST_hidden_128_sam_False_prune_0.0.pth"

model_sam.load_state_dict(torch.load(sam_model_path))
model_sgd.load_state_dict(torch.load(sgd_model_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()
model_sam.to(device)
model_sgd.to(device)

RHOS = [0.05, 0.5, 1, 2, 5]
STDS = [1, 3, 5, 10, 15]

for std in STDS:
    for rho in RHOS:
        print("Evaluating unpruned models with rho = {}, std = {}...".format(rho, std))

        # eval_metrics_sam = evaluate(model_sam, device, test_loader, criterion, n_batch=10, rho=rho)
        # eval_metrics_sgd = evaluate(model_sgd, device, test_loader, criterion, n_batch=10, rho=rho)

        # print(eval_metrics_sam)
        # print(eval_metrics_sgd)

        # # Save every result to a dataframe
        res = []
        # eval_metrics_sam["prune_ratio"] = 0
        # eval_metrics_sam["is_sam"] = True
        # eval_metrics_sam["rho"] = rho
        # res.append(eval_metrics_sam)
        # eval_metrics_sgd["prune_ratio"] = 0
        # eval_metrics_sgd["is_sam"] = False
        # eval_metrics_sgd["rho"] = rho
        # res.append(eval_metrics_sgd)

        prune_percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        pruning_methods = ["l1_unstructured"]

        for prune_percent in prune_percents:
            for pruning_method in pruning_methods:
                # print(f"Pruning {prune_percent*100}% using {pruning_method}...")
                # parameters_to_prune = []
                # for name, module in model_sam.named_modules():
                #     if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                #         parameters_to_prune.append((module, 'weight'))
                # if pruning_method == "l1_unstructured":
                #     prune.global_unstructured(
                #         parameters_to_prune,
                #         pruning_method=prune.L1Unstructured,
                #         amount=prune_percent,
                #     )
                # eval_metrics_sam_pruned = evaluate(model_sam, device, test_loader, criterion, pruned=True, n_batch=10, rho=rho, std=std)
                # print(f"SAM model pruned {prune_percent*100}%: {eval_metrics_sam_pruned}")
                # # reset the model weights to original before pruning SGD model
                # model_sam = SimpleNN(hidden_size=128, num_classes=10)
                # model_sam.load_state_dict(torch.load(sam_model_path))
                # model_sam.to(device)
                # # add to res
                # eval_metrics_sam_pruned["prune_ratio"] = prune_percent
                # eval_metrics_sam_pruned["is_sam"] = True
                # eval_metrics_sam_pruned["pruning_method"] = pruning_method
                # eval_metrics_sam_pruned["rho"] = rho
                # res.append(eval_metrics_sam_pruned)

                # Repeat for SGD model
                parameters_to_prune = []
                for name, module in model_sgd.named_modules():
                    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                        parameters_to_prune.append((module, 'weight'))
                if pruning_method == "l1_unstructured":
                    prune.global_unstructured(
                        parameters_to_prune,
                        pruning_method=prune.L1Unstructured,
                        amount=prune_percent,
                    )
                eval_metrics_sgd_pruned = evaluate(model_sgd, device, test_loader, criterion, pruned=True, n_batch=10, rho=rho, std=std)
                print(f"SGD model pruned {prune_percent*100}%: {eval_metrics_sgd_pruned}")
                # reset the model weights to original before pruning SGD model
                model_sgd = SimpleNN(hidden_size=128, num_classes=10)
                model_sgd.load_state_dict(torch.load(sgd_model_path))
                model_sgd.to(device)
                # add to res
                eval_metrics_sgd_pruned["prune_ratio"] = prune_percent
                eval_metrics_sgd_pruned["is_sam"] = False
                eval_metrics_sgd_pruned["pruning_method"] = pruning_method
                eval_metrics_sgd_pruned["rho"] = rho
                eval_metrics_sgd_pruned["std"] = std
                res.append(eval_metrics_sgd_pruned)

        df = pd.DataFrame(res)
        df.to_csv(f"src/flatness_post_pruning_results_mnist_rho_{rho}_std_{std}.csv", index=False)