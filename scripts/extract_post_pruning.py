"""Extract the metrics logged right after every cut (tag suffix /post_pruning:
restricted gradient norm ||P_m^T grad L(theta^(m))|| on one training batch,
BatchNorm in evaluation mode, plus weight-distribution statistics) from the
TensorBoard logs of the iterative CNN runs.

Usage (server):
    python scripts/extract_post_pruning.py
"""
import glob
import json

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

out = {}
for d in sorted(glob.glob("tensorboard/runs_sparse/*shortrecovery*/seed_*/SAM_*")):
    run, seed, sam = d.split("/")[-3:]
    if "fixedalloc" in run or "rho0" in run or "latedecay" in run:
        continue
    ea = EventAccumulator(d, size_guidance={"scalars": 0})
    ea.Reload()
    tags = [t for t in ea.Tags()["scalars"] if t.endswith("/post_pruning")]
    if not tags:
        continue
    out[f"{run}|{seed}|{sam}"] = {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in tags}
    print(run, seed, sam, len(tags), "tags", flush=True)
json.dump(out, open("results/post_pruning_metrics.json", "w"))
