"""Replay the first epoch and locate early target crossings on a common image grid."""
import argparse
import hashlib
import math
from pathlib import Path
import random
import time

import torch

from study import (BATCHES, OUT, RATES, SEEDS, Samples, Update, choice_key, configure,
                   evaluate, make_model, metadata, read_json, sync, train_step, write_json)

GRID = 2048


def replay(data, env, seed, method, batch, rate, destination):
    original = read_json(OUT / "main" / f"{choice_key(method,batch)}-s{seed}.json")
    assert original["rate"] == rate and original["environment"]["code_sha256"] == env["code_sha256"]
    assert GRID % batch == 0
    x, y = data.train
    device = data.device
    warm = make_model(seed,device)
    warm_update = Update(warm.parameters(),method,rate)
    for _ in range(8):
        train_step(warm,warm_update,x[:batch],y[:batch])
    sync(device)
    del warm, warm_update
    model = make_model(seed,device)
    update = Update(model.parameters(),method,rate)
    rng = torch.Generator().manual_seed(50000+seed)
    order = torch.randperm(len(y),generator=rng).to(device)
    history = [{"examples":0,"updates":0,"train_s":0.0,"loss":evaluate(model,data.train)["loss"]}]
    elapsed = seen = 0
    for block in order.split(GRID):
        sync(device)
        start = time.perf_counter()
        for i in block.split(batch):
            train_step(model,update,x.index_select(0,i),y.index_select(0,i))
        sync(device)
        elapsed += time.perf_counter()-start
        seen += len(block)
        history.append({"examples":seen,"updates":math.ceil(seen/batch),"train_s":elapsed,
                        "loss":evaluate(model,data.train)["loss"]})
    difference = abs(history[-1]["loss"]-original["history"][1]["train"]["loss"])
    if difference > 1e-6:
        raise ValueError(f"Epoch-one replay differs by {difference}")
    result = {"seed":seed,"method":method,"rate":rate,"batch":batch,"grid":GRID,
              "history":history,"epoch_one_abs_diff":difference,"environment":env}
    write_json(destination / f"{choice_key(method,batch)}-s{seed}.json",result)
    print(seed,method,batch,"endpoint difference",difference,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    parser.add_argument("--rounds",type=int,default=5)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    configure()
    device = torch.device(args.device)
    samples = Samples(device)
    env = metadata(device)
    env["replay_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rates = read_json(OUT / "selected.json")
    for repeat in range(1,args.rounds+1):
        cells = [(s,m,b) for s in SEEDS for m in RATES for b in BATCHES]
        random.Random(19300+repeat).shuffle(cells)
        for s,m,b in cells:
            replay(samples,env,s,m,b,rates[choice_key(m,b)],OUT / "timing" / f"round{repeat}")
