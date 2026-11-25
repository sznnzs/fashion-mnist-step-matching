"""Paired Fashion-MNIST optimization experiment. Run --help for the stages."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data_io import ROOT, load_arrays, read_idx

OUT = Path(os.environ.get("STUDY_RESULTS", ROOT / "results")).resolve()
RATES = {"SGD": [0.01, 0.03, 0.1, 0.3],
         "Momentum": [0.003, 0.01, 0.03, 0.1],
         "Adam": [0.0003, 0.001, 0.003, 0.01]}
BATCHES = (64, 256)
SEEDS = (7, 19, 43)
TUNE_SEED = 101
TUNE_EPOCHS = 5
EPOCHS = 15
TARGET = 0.50
PROBE_EPOCHS = (0, 5, 15)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        json.dump(value, tmp, indent=2, allow_nan=False)
        tmp.write("\n")
    os.replace(tmp.name, path)


def read_json(path):
    return json.loads(path.read_text())


def source_digest():
    h = hashlib.sha256()
    for name in ("data_io.py", "study.py"):
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()


def configure():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def metadata(device):
    info = {"torch": str(torch.__version__), "numpy": np.__version__,
            "python": platform.python_version(), "cuda_build": torch.version.cuda,
            "device": str(device), "job_id": os.environ.get("SLURM_JOB_ID"),
            "cpu_threads": torch.get_num_threads(), "code_sha256": source_digest()}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        identity = str(getattr(p, "uuid", None) or os.environ.get("SLURM_JOB_GPUS", "unknown"))
        info.update(gpu=p.name, gpu_key=hashlib.sha256(identity.encode()).hexdigest()[:10],
                    vram_bytes=p.total_memory, compute_capability=[p.major,p.minor])
    return info


def make_model(seed, device):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)


class Update:
    """Four parameter tensors. Gradients are supplied by PyTorch autograd."""
    def __init__(self, params, method, rate):
        self.params = list(params)
        self.method = method
        self.rate = rate
        self.step_number = 0
        self.first = [torch.zeros_like(p) for p in self.params] if method != "SGD" else []
        self.second = [torch.zeros_like(p) for p in self.params] if method == "Adam" else []

    @torch.no_grad()
    def step(self):
        self.step_number += 1
        if self.method == "Adam":
            bias1 = 1.0 - 0.9 ** self.step_number
            root_bias2 = math.sqrt(1.0 - 0.999 ** self.step_number)
        for index, p in enumerate(self.params):
            g = p.grad
            if self.method == "SGD":
                p.add_(g, alpha=-self.rate)
            elif self.method == "Momentum":
                v = self.first[index]
                v.mul_(0.9).add_(g)
                p.add_(v, alpha=-self.rate)
            else:
                m, v = self.first[index], self.second[index]
                m.lerp_(g, 0.1)
                v.mul_(0.999).addcmul_(g, g, value=0.001)
                denominator = v.sqrt().div_(root_bias2).add_(1e-8)
                p.addcdiv_(m, denominator, value=-self.rate/bias1)


def native_update(params, method, rate, fused=False):
    if method == "Adam":
        return torch.optim.Adam(params, lr=rate, betas=(0.9,0.999), eps=1e-8, fused=fused)
    return torch.optim.SGD(params, lr=rate, momentum=0.9 if method == "Momentum" else 0,
                           fused=fused)


def check_updates():
    results = {}
    for method, rates in RATES.items():
        rng = torch.Generator().manual_seed(558)
        p = nn.Parameter(torch.randn(5, 7, generator=rng, dtype=torch.float64))
        q = nn.Parameter(p.detach().clone())
        custom = Update([p], method, rates[1])
        native = native_update([q], method, rates[1])
        maximum = 0.0
        for _ in range(23):
            p.grad = torch.randn(p.shape, generator=rng, dtype=p.dtype)
            q.grad = p.grad.clone()
            custom.step()
            native.step()
            maximum = max(maximum, (p-q).detach().abs().max().item())
        if maximum > 2e-12:
            raise ValueError((method, maximum))
        results[method] = maximum
    write_json(OUT / "optimizer_check.json", {"steps":23, "max_abs_error":results,
                                              "torch":str(torch.__version__)})
    print(results, flush=True)


class Samples:
    def __init__(self, device, test=False):
        arrays, self.info = load_arrays()
        self.device = device
        self.train = self.convert(*arrays[:2])
        self.valid = self.convert(*arrays[2:])
        if test:
            self.test = self.convert(read_idx("t10k-images-idx3-ubyte.gz"),
                                     read_idx("t10k-labels-idx1-ubyte.gz"))

    def convert(self, x, y):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x.sub_(self.info["train_pixel_mean"]).div_(self.info["train_pixel_std"])
        return x, torch.tensor(y, dtype=torch.long, device=self.device)


@torch.inference_mode()
def evaluate(model, data):
    x, y = data
    losses, hits = [], []
    for xx, yy in zip(x.split(2048), y.split(2048)):
        logits = model(xx)
        losses.append(F.cross_entropy(logits, yy, reduction="none").double().sum())
        hits.append((logits.argmax(1) == yy).sum())
    return {"loss":torch.stack(losses).sum().item()/len(y),
            "accuracy":torch.stack(hits).sum().item()/len(y)}


def train_step(model, update, x, y):
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    update.step()


def choice_key(method, batch):
    return f"{method}-b{batch}"


def tune_path(method, batch, rate):
    return OUT / "tune" / f"{choice_key(method,batch)}-lr{rate:g}.json"


def choose_rates():
    choices = {}
    for method, rates in RATES.items():
        for batch in BATCHES:
            rows = [read_json(tune_path(method,batch,r)) for r in rates]
            for row in rows:
                assert row["seed"] == TUNE_SEED and len(row["history"]) == TUNE_EPOCHS+1
                assert row["environment"]["code_sha256"] == source_digest()
            winner = min(rows,key=lambda row:(row["history"][-1]["valid"]["loss"],row["rate"]))
            choices[choice_key(method,batch)] = winner["rate"]
    write_json(OUT / "selected.json", choices)
    print("Frozen learning rates", choices, flush=True)


def run_one(data, environment, stage, method, batch, rate, seed):
    device = data.device
    tuning = stage == "tune"
    epochs = TUNE_EPOCHS if tuning else EPOCHS
    path = tune_path(method,batch,rate) if tuning else OUT / stage / f"{choice_key(method,batch)}-s{seed}.json"
    expects_probe = stage == "main" and (seed,method,batch) == (19,"SGD",64)
    snapshots = OUT / "snapshots"
    if path.exists():
        done = read_json(path)
        if (done["rate"] != rate or done["environment"]["code_sha256"] != source_digest()
                or done["environment"]["device"] != str(device)):
            raise ValueError(f"Different experiment at {path}. Use a new STUDY_RESULTS directory.")
        if not expects_probe or all((snapshots / f"epoch{k}.pt").exists() for k in PROBE_EPOCHS):
            print(f"Already complete: {path.name}", flush=True)
            return

    def build():
        model = make_model(seed,device)
        update = (native_update(model.parameters(),method,rate,fused=device.type=="cuda")
                  if stage == "native" else Update(model.parameters(),method,rate))
        return model, update

    x, y = data.train
    warm, optimizer = build()
    for _ in range(8):
        train_step(warm,optimizer,x[:batch],y[:batch])
    sync(device)
    del warm, optimizer
    model, update = build()
    history = []
    elapsed_train = elapsed_eval = 0.0
    order_rng = torch.Generator().manual_seed(50000+seed)
    for epoch in range(epochs+1):
        if epoch:
            order = torch.randperm(len(y),generator=order_rng).to(device)
            sync(device)
            start = time.perf_counter()
            for indices in order.split(batch):
                train_step(model,update,x.index_select(0,indices),y.index_select(0,indices))
            sync(device)
            elapsed_train += time.perf_counter()-start
        start = time.perf_counter()
        train = evaluate(model,data.train)
        valid = evaluate(model,data.valid)
        sync(device)
        elapsed_eval += time.perf_counter()-start
        history.append({"epoch":epoch,"examples":epoch*len(y),
                        "updates":epoch*math.ceil(len(y)/batch),"train_s":elapsed_train,
                        "eval_s":elapsed_eval,"train":train,"valid":valid})
        if expects_probe and epoch in PROBE_EPOCHS:
            snapshots.mkdir(parents=True,exist_ok=True)
            torch.save(model.state_dict(),snapshots / f"epoch{epoch}.pt")
    result = {"stage":stage,"method":method,"seed":seed,"batch":batch,"rate":rate,
              "history":history,"environment":environment,"split":data.info}
    if not tuning:
        result["test"] = evaluate(model,data.test)
    write_json(path,result)
    print(f"{stage} {method} B={batch} lr={rate:g} seed={seed} "
          f"val={valid['loss']:.4f} train={elapsed_train:.2f}s",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["check","tune","select","main","native","probe"])
    parser.add_argument("--device",default="cuda",choices=["cuda","cpu"])
    parser.add_argument("--shard",type=int,default=0,choices=[0,1])
    parser.add_argument("--seeds",type=int,nargs="+",choices=SEEDS,default=list(SEEDS))
    args = parser.parse_args()
    configure()
    if args.stage == "check":
        return check_updates()
    if args.stage == "select":
        return choose_rates()
    device = torch.device(args.device)
    data = Samples(device,test=args.stage in ("main","native"))
    env = metadata(device)
    print(json.dumps(env),flush=True)
    if args.stage == "probe":
        from probe import measure
        return measure(data,env)
    if args.stage == "tune":
        trials = [(m,b,r,TUNE_SEED) for m,rates in RATES.items() for b in BATCHES for r in rates]
        random.Random(761).shuffle(trials)
        trials = trials[args.shard::2]
    else:
        choices = read_json(OUT / "selected.json")
        trials = []
        for seed in args.seeds:
            paired = [(m,b,choices[choice_key(m,b)],seed) for m in RATES for b in BATCHES]
            random.Random(9000+seed).shuffle(paired)
            trials.extend(paired)
    for method,batch,rate,seed in trials:
        run_one(data,env,args.stage,method,batch,rate,seed)


if __name__ == "__main__":
    main()
