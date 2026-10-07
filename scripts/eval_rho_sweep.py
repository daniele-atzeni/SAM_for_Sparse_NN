import json, sys, torch, torch.nn as nn, torch.nn.utils.prune as prune
sys.path.insert(0, ".")
from src.registry import build_model, build_dataloaders
dev = torch.device("cuda")
_, tl = build_dataloaders("cifar10", 512)
out = {}
for rho in ["0.05", "0.1", "0.2"]:
    m = build_model("ResNet18", {"num_classes": 10}).to(dev)
    for mod in m.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            prune.identity(mod, "weight")
    m.load_state_dict(torch.load(f"saved_models/sparse/ResNet18_CIFAR10_s0.9995_shortrecovery_rho{rho}/seed_13/ResNet18_cifar10_sam_True.pth", map_location=dev))
    m.eval(); c = n = 0
    with torch.no_grad():
        for x, y in tl:
            c += m(x.to(dev)).argmax(1).eq(y.to(dev)).sum().item(); n += y.numel()
    out[rho] = c / n; print(rho, c / n, flush=True)
json.dump(out, open("results/rho_sweep_seed13_fulltest.json", "w"))
