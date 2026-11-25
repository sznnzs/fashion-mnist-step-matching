"""Compare SGD rate-transfer rules and a local step-matching proposal on fresh seeds.

The design uses previously frozen rates, five new seeds and no test-set evaluation.
Fit checkpoints, block counts and comparison rules are specified below.
"""
import argparse
import copy
import hashlib
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector

from study import (EPOCHS, OUT, TARGET, Samples, Update, configure, evaluate, make_model,
                   metadata, read_json, sync, train_step, write_json)

SEEDS = (131, 137, 149, 151, 157)
RULES = ("reference", "constant", "noise", "tuned", "linear", "once", "refresh")
ANCHORS = {"once": (0,), "refresh": (0, 1, 5)}
SMALL, LARGE = 64, 256
BLOCKS = 16
GRID = 2048


def block_moments(model, x, y, blocks, base_rate):
    """Return (||g||², <g,d>, ||d||²) for each disjoint block at frozen weights."""
    parameters = tuple(model.parameters())
    origin = parameters_to_vector(parameters).detach().double()
    shadow = copy.deepcopy(model)
    update = Update(shadow.parameters(), "SGD", base_rate)
    moments = []
    for indices in blocks:
        xx, yy = x.index_select(0, indices), y.index_select(0, indices)
        gradient = torch.autograd.grad(F.cross_entropy(model(xx), yy), parameters)
        g = parameters_to_vector(gradient).detach().double()
        with torch.no_grad():
            for dest, src in zip(shadow.parameters(), parameters):
                dest.copy_(src)
        for xb, yb in zip(xx.split(SMALL), yy.split(SMALL)):
            train_step(shadow, update, xb, yb)
        displacement = origin - parameters_to_vector(shadow.parameters()).detach().double()
        moments.append(torch.stack((g.dot(g), g.dot(displacement), displacement.dot(displacement))))
    return torch.stack(moments).cpu().tolist()


def calibrate(model, data, seed, completed_epochs, base_rate, previous_rate):
    x, y = data.train
    sync(data.device)
    start = time.perf_counter()
    # This separate generator leaves the formal training order untouched.
    rng = torch.Generator().manual_seed(710000 + 100 * seed + completed_epochs)
    indices = torch.randperm(len(y), generator=rng)[:2*BLOCKS*LARGE]
    blocks = indices.reshape(2*BLOCKS, LARGE).to(data.device)
    fit = block_moments(model, x, y, blocks[:BLOCKS], base_rate)
    a = sum(row[0] for row in fit)
    c = sum(row[1] for row in fit)
    rate = max(0.0, c/a) if a > 0 else previous_rate
    sync(data.device)
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    audit = block_moments(model, x, y, blocks[BLOCKS:], base_rate)
    sync(data.device)
    return {
        "after_epoch": completed_epochs, "rate": rate, "fit_s": fit_s,
        "audit_s": time.perf_counter() - start, "fit": fit, "audit": audit,
        "fit_gradient_images": 2*BLOCKS*LARGE,
        "fit_indices_sha256": hashlib.sha256(indices[:BLOCKS*LARGE].numpy().tobytes()).hexdigest(),
        "audit_indices_sha256": hashlib.sha256(indices[BLOCKS*LARGE:].numpy().tobytes()).hexdigest(),
    }


def run(data, env, seed, rule, rates):
    x, y = data.train
    base_rate = rates["SGD-b64"]
    batch = SMALL if rule == "reference" else LARGE
    noise_rate = base_rate * math.sqrt((1/SMALL - 1/len(y)) / (1/LARGE - 1/len(y)))
    fixed = {"reference": base_rate, "constant": base_rate, "noise": noise_rate,
             "tuned": rates["SGD-b256"], "linear": base_rate * LARGE/SMALL}
    rate = fixed.get(rule, base_rate)
    warm = make_model(seed, data.device)
    warm_update = Update(warm.parameters(), "SGD", rate)
    for _ in range(8):
        train_step(warm, warm_update, x[:batch], y[:batch])
    sync(data.device)
    del warm, warm_update
    model = make_model(seed, data.device)
    update = Update(model.parameters(), "SGD", rate)
    order_rng = torch.Generator().manual_seed(50000 + seed)
    history = []
    calibrations = []
    hit = None
    examples = updates = fit_images = 0
    train_s = fit_s = 0.0
    for epoch in range(EPOCHS + 1):
        if epoch:
            if epoch-1 in ANCHORS.get(rule, ()):
                fit = calibrate(model, data, seed, epoch-1, base_rate, update.rate)
                calibrations.append(fit)
                update.rate = fit["rate"]
                fit_s += fit["fit_s"]
                fit_images += fit["fit_gradient_images"]
            order = torch.randperm(len(y), generator=order_rng).to(data.device)
            for block in order.split(GRID):
                sync(data.device)
                start = time.perf_counter()
                for indices in block.split(batch):
                    train_step(model, update, x.index_select(0, indices), y.index_select(0, indices))
                sync(data.device)
                train_s += time.perf_counter() - start
                examples += len(block)
                updates += math.ceil(len(block)/batch)
                if hit is None:
                    loss = evaluate(model, data.train)["loss"]
                    if loss <= TARGET:
                        hit = {"examples": examples, "updates": updates, "loss": loss,
                               "gradient_images": examples + fit_images, "seconds": train_s + fit_s}
        train = evaluate(model, data.train)
        valid = evaluate(model, data.valid)
        history.append({"epoch": epoch, "examples": examples, "updates": updates,
                        "fit_gradient_images": fit_images, "train_s": train_s, "fit_s": fit_s,
                        "rate": update.rate if epoch else None, "train": train, "valid": valid})
    path = OUT / "step_matching" / f"{rule}-s{seed}.json"
    write_json(path, {"seed": seed, "rule": rule, "batch": batch, "base_rate": base_rate,
                      "tuned_rate": rates["SGD-b256"], "noise_rate": noise_rate,
                      "target": TARGET, "target_hit": hit, "split": data.info,
                      "environment": env, "calibrations": calibrations, "history": history})
    print(f"{rule:9} seed {seed}  validation {valid['loss']:.4f}  fit+train {train_s+fit_s:.2f}s",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    configure()
    device = torch.device(args.device)
    data = Samples(device, test=False)
    environment = metadata(device)
    environment["step_matching_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    selected = read_json(OUT / "selected.json")
    for seed in SEEDS:
        rules = list(RULES)
        random.Random(81000 + seed).shuffle(rules)
        for rule in rules:
            run(data, environment, seed, rule, selected)
