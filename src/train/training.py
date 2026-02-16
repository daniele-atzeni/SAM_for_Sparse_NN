import os
import tempfile

import torch
from torch import nn

import torch.nn.utils.prune as prune

from torch.utils.tensorboard import SummaryWriter

from src.models.mlp import MLP

from .SAM import SAM, disable_running_stats, enable_running_stats
from ..eval.eval import evaluate


def train_epoch(
        model: nn.Module, 
        device: str | torch.device, 
        train_loader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer, 
        epoch: int, 
        criterion: nn.Module,
        log_every: int = 100
        ) -> dict:
    
    model.train()

    batch_losses = []
    batch_sam_losses = []
    batch_corrects = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if isinstance(optimizer, SAM):
            enable_running_stats(model)
        output = model(data)
        loss = criterion(output, target).mean()

        if isinstance(optimizer, SAM):
            # backprop the loss
            loss.backward()
            optimizer.first_step(zero_grad=True)
            # set the second step
            disable_running_stats(model)
            #smooth_crossentropy(model(data), target, smoothing=0.1).mean().backward()
            sam_loss = criterion(model(data), target).mean()
            sam_loss.backward()
            optimizer.second_step(zero_grad=True)

            # reapply the mask for the pruned weights
            with torch.no_grad():
                for m in model.modules():
                    if hasattr(m, "weight_mask"):
                        m.weight_orig.data.mul_(m.weight_mask)  # keep zeros zero
        else:
            loss.backward()
            optimizer.step()

        if batch_idx % log_every == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tSAM Loss: {sam_loss.item() if isinstance(optimizer, SAM) else 'N/A'}   "
            )
        # append batch metrics
        batch_losses.append(loss.item() * data.size(0))
        if isinstance(optimizer, SAM):
            batch_sam_losses.append((sam_loss.item() - loss.item()) * data.size(0))
        batch_corrects.append(
            output.argmax(dim=1).eq(target).sum().item()
        )

    epoch_metrics = {
        "Loss": sum(batch_losses) / len(train_loader.dataset),
        "SAM Loss": sum(batch_sam_losses) / len(train_loader.dataset) if isinstance(optimizer, SAM) else None,
        "Accuracy": sum(batch_corrects) / len(train_loader.dataset)
        }

    return epoch_metrics


def train_loop(
        epochs: int,
        use_sam: bool,
        model: nn.Module,
        device: str | torch.device,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        SGD_optimizer: torch.optim.Optimizer,
        SAM_optimizer: SAM,
        criterion: nn.Module,
        scheduler = None,
        tensorboard_log_dir: str = "runs/exp",
        first_epoch: int = 0,
        log_every: int = 100,
        checkpoint_folder: str = "./checkpoint",
        save_every: int = 100
        ):
    
    assert use_sam and SAM_optimizer is not None or not use_sam and SGD_optimizer is not None, \
        "Optimizer configuration does not match use_sam flag."

    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # evaluate before training
    eval_metrics = evaluate(model, device, test_loader, criterion)
    # log metrics to TensorBoard
    for name, value in eval_metrics.items():
        if value is not None:
            writer.add_scalar(f"{name}/test", value, first_epoch)
    # print metrics
    print(f"Epoch {first_epoch}:")
    for name, value in eval_metrics.items():
        if value is not None:
                print(f"  Test {name}: {value:.4f}")

    for epoch in range(first_epoch + 1, first_epoch + epochs + 1):
        if not use_sam:
            train_metrics = train_epoch(model, device, train_loader, SGD_optimizer, epoch, criterion, log_every=log_every)
        else:
            train_metrics = train_epoch(model, device, train_loader, SAM_optimizer, epoch, criterion, log_every=log_every)
        #train_metrics = evaluate(model, device, train_loader, criterion)
        eval_metrics = evaluate(model, device, test_loader, criterion)

        # log metrics to TensorBoard
        for name, value in train_metrics.items():
            if value is not None:
                writer.add_scalar(f"{name}/train", value, epoch)
        for name, value in eval_metrics.items():
            if value is not None:
                writer.add_scalar(f"{name}/test", value, epoch)
        # print metrics
        print(f"Epoch {epoch}:")
        for name, value in train_metrics.items():
            if value is not None:
                print(f"  Train {name}: {value:.4f}")
        for name, value in eval_metrics.items():
            if value is not None:
                print(f"  Test {name}: {value:.4f}")

        # update the scheduler
        if scheduler:
            scheduler(SGD_optimizer, epoch)
            scheduler(SAM_optimizer, epoch)
        
        # save the model
        if epoch % save_every == 0:
            model_path = os.path.join(checkpoint_folder, f"sam_{use_sam}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_path)

    writer.close()


def train_prune_loop(
        epochs: int,
        use_sam: bool,
        model: nn.Module,
        device: str | torch.device,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        SGD_optimizer: torch.optim.Optimizer,
        SAM_optimizer: SAM,
        criterion: nn.Module,
        prune_ratio: float,
        first_iter: int,
        prune_every: int,
        n_iter: int,
        scheduler = None,
        tensorboard_log_dir: str = "runs/exp1",
        log_every: int = 100,
        checkpoint_folder: str = "./checkpoint",
        save_every: int = 100,
        first_epoch: int = 0
        ):
    
    assert use_sam and SAM_optimizer is not None or not use_sam and SGD_optimizer is not None, \
        "Optimizer configuration does not match use_sam flag."

    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # evaluate before training
    eval_metrics = evaluate(model, device, test_loader, criterion)
    # log metrics to TensorBoard
    for name, value in eval_metrics.items():
        if value is not None:
            writer.add_scalar(f"{name}/test", value, first_epoch)
    # print metrics
    print(f"Epoch {first_epoch}:")
    for name, value in eval_metrics.items():
        if value is not None:
                print(f"  Test {name}: {value:.4f}")

    iter_ratio = 1 - (1 - prune_ratio) ** (1 / n_iter)
    #iter_epochs = epochs // n_iter

    for epoch in range(1, epochs + 1):
        #if epoch == first_iter or ((epoch - 1) % iter_epochs == 0 and epoch != 1):
        if epoch >= first_iter and (epoch - first_iter) % prune_every == 0:
            print(f"Pruning iteration at epoch {epoch}: pruning additional {iter_ratio*100:.2f}% of weights.")
            parameters_to_prune = [(module, 'weight') for _, module in model.named_modules() if isinstance(module, (nn.Linear, nn.Conv2d))]
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=iter_ratio,
            )   # change model in-place
            # compute sparsity
            print(f"Global sparsity after pruning at epoch {epoch}:")
            total_zeros = 0
            total_params = 0
            for module, _ in parameters_to_prune:
                if hasattr(module, 'weight_mask'):
                    zeros = torch.sum(module.weight_mask == 0).item()
                    params = module.weight_mask.numel()
                    total_zeros += zeros
                    total_params += params
                    print(f"  Layer {module}: {zeros}/{params} ({100. * zeros / params:.2f}%)")
            print(f"  Total: {total_zeros}/{total_params} ({100. * total_zeros / total_params:.2f}%)")
            writer.add_scalar(f"sparsity", 100. * total_zeros / total_params, epoch)

        if not use_sam:
            train_metrics = train_epoch(model, device, train_loader, SGD_optimizer, epoch, criterion, log_every=log_every)
        else:
            train_metrics = train_epoch(model, device, train_loader, SAM_optimizer, epoch, criterion, log_every=log_every)
        #train_metrics = evaluate(model, device, train_loader, criterion)
        eval_metrics = evaluate(model, device, test_loader, criterion)

        # log metrics
        for name, value in train_metrics.items():
            if value is not None:
                writer.add_scalar(f"{name}/train", value, epoch)
        for name, value in eval_metrics.items():
            if value is not None:
                writer.add_scalar(f"{name}/test", value, epoch)

        # update the scheduler
        if scheduler:
            scheduler(SGD_optimizer, epoch)
            scheduler(SAM_optimizer, epoch)
        
        # save the model with a tmp file
        if epoch % save_every == 0:
            model_path = os.path.join(checkpoint_folder, f"sam_{use_sam}_epoch_{epoch}.pth")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                tmp_path = tmp.name
            torch.save(model, tmp_path)

            model_copy = torch.load(tmp_path, map_location="cpu", weights_only=False)
            for m in model_copy.modules():
                if hasattr(m, "weight_orig"):
                    prune.remove(m, "weight")

            torch.save(model_copy.state_dict(), model_path)

            # cleanup
            del model_copy
            os.remove(tmp_path)

    writer.close()