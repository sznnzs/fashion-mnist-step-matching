"""Fixed-weight gradient measurements for simple random sampling of images."""
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from study import BATCHES, OUT, PROBE_EPOCHS, make_model, write_json


def gradient(model, x, y):
    loss = F.cross_entropy(model(x), y)
    parts = torch.autograd.grad(loss, tuple(model.parameters()))
    return torch.cat([part.reshape(-1) for part in parts])


@torch.no_grad()
def individual_squared_norms(model, x, y):
    hidden = model[:2](x)
    delta = model[2](hidden).softmax(1)
    delta[torch.arange(len(y),device=y.device),y] -= 1
    back = (delta @ model[2].weight) * (hidden > 0)
    return ((hidden.square().sum(1)+1)*delta.square().sum(1)
            + (x.square().sum(1)+1)*back.square().sum(1))


def measure(data, env):
    x, y = data.train
    x = x.double()
    n = len(y)
    draws = 256
    rng = np.random.default_rng(31077)
    # The first 64 indices are nested inside the uniform 256-index draw.
    ix = torch.tensor(np.stack([rng.choice(n,256,replace=False) for _ in range(draws)]),
                      device=data.device)
    snapshots = []
    for epoch in PROBE_EPOCHS:
        path = OUT / "snapshots" / f"epoch{epoch}.pt"
        model = make_model(19,data.device)
        model.load_state_dict(torch.load(path,map_location=data.device,weights_only=True))
        model.double()
        ref = torch.zeros(sum(p.numel() for p in model.parameters()),
                          device=data.device,dtype=torch.float64)
        sum_squared = ref.new_zeros(())
        for xx, yy in zip(x.split(512),y.split(512)):
            ref.add_(gradient(model,xx,yy),alpha=len(yy)/n)
            sum_squared += individual_squared_norms(model,xx,yy).sum()
        check = individual_squared_norms(model,x[:8],y[:8])
        slow = torch.stack([gradient(model,x[i:i+1],y[i:i+1]).square().sum() for i in range(8)])
        torch.testing.assert_close(check,slow,rtol=1e-10,atol=1e-10)
        grad_norm2 = ref.square().sum().item()
        trace = (sum_squared.item()-n*grad_norm2)/(n-1)
        rows = []
        for batch in BATCHES:
            deviations = []
            for row in ix[:,:batch]:
                g = gradient(model,x.index_select(0,row),y.index_select(0,row))
                deviations.append((g-ref).square().sum())
            values = torch.stack(deviations).cpu().numpy()
            rows.append({"batch":batch,"prediction":(1/batch-1/n)*trace,
                         "empirical":float(values.mean()),
                         "mc_se":float(values.std(ddof=1)/np.sqrt(draws)),
                         "draws_squared_error":values.tolist()})
        snapshots.append({"epoch":epoch,"trace":trace,"full_gradient_norm2":grad_norm2,
                          "snapshot_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                          "norm_check_abs":(check-slow).abs().max().item(),"batches":rows})
        print(f"Probe epoch {epoch}: mean squared errors {[r['empirical'] for r in rows]}",flush=True)
    write_json(OUT / "probe.json",{"environment":env,"rows":snapshots,"draws":draws,
                                   "seed":31077,"trajectory":{"method":"SGD","batch":64,"seed":19}})
