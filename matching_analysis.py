"""Tables and figures for the independent SGD learning-rate transfer experiment."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent
SEEDS = (131, 137, 149, 151, 157)
FIXED = ("constant", "noise", "tuned", "linear")
RULES = ("reference", *FIXED, "once", "refresh")
LABEL = {"reference": "Small-batch reference", "constant": "Unchanged",
         "noise": "Noise matched", "tuned": "Grid selected", "linear": "Linear",
         "once": "Fit once", "refresh": "Refit twice"}


def relative_error(moments, rate):
    a, c, d = np.sum(moments, axis=0)
    return math.sqrt(max(0.0, (d - 2*rate*c + rate*rate*a)/d))


def save_csv(path, records):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def measurements(results):
    by_rule = {}
    checkpoints = []
    core = hashlib.sha256((ROOT/"data_io.py").read_bytes() + (ROOT/"study.py").read_bytes()).hexdigest()
    signature = hashlib.sha256((ROOT/"step_matching.py").read_bytes()).hexdigest()
    assert len(list((results/"step_matching").glob("*.json"))) == len(RULES)*len(SEEDS)
    expected_split = json.loads((results/"main"/"SGD-b64-s7.json").read_text())["split"]
    rates = json.loads((results/"selected.json").read_text())
    for rule in RULES:
        by_rule[rule] = []
        for seed in SEEDS:
            path = results/"step_matching"/f"{rule}-s{seed}.json"
            data = json.loads(path.read_text())
            h = data["history"]
            assert len(h) == 16 and data["seed"] == seed and data["rule"] == rule
            assert data["environment"]["code_sha256"] == core
            assert data["environment"]["step_matching_sha256"] == signature
            assert "test" not in data and expected_split == data["split"]
            assert data["base_rate"] == rates["SGD-b64"] and data["tuned_rate"] == rates["SGD-b256"]
            assert [p["examples"] for p in h] == [54000*k for k in range(16)]
            batch = data["batch"]
            assert batch == (64 if rule == "reference" else 256)
            assert h[-1]["updates"] == 15*math.ceil(54000/batch)
            fits = data["calibrations"]
            assert [f["after_epoch"] for f in fits] == {"once":[0], "refresh":[0,1,5]}.get(rule, [])
            assert h[-1]["fit_gradient_images"] == 8192*len(fits)
            for fit in fits:
                fm, am = np.array(fit["fit"]), np.array(fit["audit"])
                assert fm.shape == am.shape == (16,3)
                assert fit["fit_indices_sha256"] != fit["audit_indices_sha256"]
                for moments in (fm,am):
                    assert np.all(moments[:,0] > 0) and np.all(moments[:,2] > 0)
                    assert np.all(moments[:,1]**2 <= moments[:,0]*moments[:,2]*(1+1e-10))
                a, c, d = fm.sum(axis=0)
                au, cu, du = am.sum(axis=0)
                assert math.isclose(fit["rate"], max(0,c/a), rel_tol=1e-12)
                floor_rate = max(0, cu/au)
                # Each displacement component perpendicular to its block gradient
                # remains unchanged at every learning rate.
                grad_sq, grad_dot_step, step_sq = am.T
                perpendicular_sq = np.maximum(0.0, step_sq - grad_dot_step**2 / grad_sq)
                checkpoints.append({
                    "rule":rule, "seed":seed, "after_epoch":fit["after_epoch"], "rate":fit["rate"],
                    "fit_error":relative_error(fm,fit["rate"]), "audit_error":relative_error(am,fit["rate"]),
                    "linear_error":relative_error(am,4*data["base_rate"]),
                    "noise_error":relative_error(am,data["noise_rate"]),
                    "grid_error":relative_error(am,data["tuned_rate"]),
                    "audit_floor":relative_error(am,floor_rate),
                    "audit_direction_floor":math.sqrt(perpendicular_sq.sum() / du),
                    "fit_ms":1000*fit["fit_s"],
                })
            hit = data["target_hit"]
            if hit is not None:
                assert hit["loss"] <= data["target"]
                assert hit["gradient_images"] >= hit["examples"]
            by_rule[rule].append({
                "rule":rule,"seed":seed,"batch":batch,"lr_first":h[1]["rate"],"lr_last":h[-1]["rate"],
                "initial_ce":h[0]["train"]["loss"],"epoch1_ce":h[1]["train"]["loss"],
                "stream_images":hit["examples"] if hit else None,
                "charged_images":hit["gradient_images"] if hit else None,
                "val5":h[5]["valid"]["loss"],"val15":h[15]["valid"]["loss"],
                "train15":h[15]["train"]["loss"],"fit_s":h[-1]["fit_s"],
                "total_s":h[-1]["train_s"]+h[-1]["fit_s"],
                "device":data["environment"].get("gpu_key","cpu"),
            })
    for i, seed in enumerate(SEEDS):
        paired = [by_rule[r][i] for r in RULES]
        assert len({r["device"] for r in paired}) == 1
        assert np.ptp([r["initial_ce"] for r in paired]) < 1e-7
        for field in ("lr_first","epoch1_ce","stream_images","charged_images"):
            # Both policies have the same state until their first refit.
            if field.endswith("images") and (by_rule["once"][i]["stream_images"] or 54001)>54000:
                continue
            assert by_rule["once"][i][field] == by_rule["refresh"][i][field]
    return by_rule, checkpoints


def numbers(records, key):
    return np.array([r[key] for r in records],dtype=float)


def med(records, key):
    return float(np.median(numbers(records,key)))


def image_count(records, key):
    values = [r[key] for r in records if r[key] is not None]
    if len(values) == len(records):
        return f"{np.median(values):,.0f}"
    return f"{len(values)}/{len(records)} hit"


def style_axis(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",color="#dce3e6",lw=.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5,color="#738189")


def figures(by_rule, checkpoints, directory):
    ink, accent = "#205d7d", "#a44a24"
    fig, axes = plt.subplots(1,2,figsize=(5.55,2.0),sharex=True,sharey=True)
    rates = [by_rule[r][0]["lr_first"] for r in FIXED]
    for ax, field, title in zip(axes, ("val5","val15"), ("After 5 epochs","After 15 epochs")):
        style_axis(ax)
        matrix = np.array([numbers(by_rule[r],field) for r in FIXED])
        for ys in matrix.T:
            ax.plot(rates,ys,color="#9baeb7",lw=.9,alpha=.82,marker=".",ms=3.2)
        ax.plot(rates,np.median(matrix,axis=1),color=ink,lw=2,marker="o",ms=4)
        ax.set_title(title)
        ax.set_xticks(rates,["0.10","0.20","0.30","0.40"])
        ax.set_xlabel("SGD learning rate at batch 256")
        ax.margins(x=.10,y=.12)
    axes[0].set_ylabel("Validation cross-entropy")
    fig.subplots_adjust(left=.125,right=.985,top=.86,bottom=.255,wspace=.16)
    fig.savefig(directory/"rate_response.pdf", bbox_inches="tight", pad_inches=.04,
                metadata={"CreationDate": None})
    plt.close(fig)

    rows = [r for r in checkpoints if r["rule"] == "refresh"]
    after = [0,1,5]
    fig, axes = plt.subplots(1,2,figsize=(5.55,2.25))
    for ax in axes:
        style_axis(ax)
        ax.set_xticks([0,1,2],["Init.","Epoch 1","Epoch 5"])
        ax.set_xlim(-.20,2.22)
    for seed in SEEDS:
        seq = sorted([r for r in rows if r["seed"]==seed],key=lambda r:r["after_epoch"])
        axes[0].plot(range(3),numbers(seq,"rate"),color="#a0b2bb",lw=.8,marker=".",ms=3)
    axes[0].plot(range(3),[med([r for r in rows if r["after_epoch"]==e],"rate") for e in after],
                 color=ink,lw=2,marker="o",ms=4)
    axes[0].set(ylabel="Fitted learning rate",title="Rate fitted on 16 blocks")
    for field,label,color,mark,ls in [
        ("linear_error","Linear",accent,"s","--"), ("noise_error","Noise matched","#545a66","^",":"),
        ("audit_error","Fitted",ink,"o","-")]:
        ys = [med([r for r in rows if r["after_epoch"]==e],field) for e in after]
        axes[1].plot(range(3),ys,label=label,color=color,lw=1.65,marker=mark,ms=4,linestyle=ls)
    axes[1].set(ylabel="Relative displacement error",title="Error on 16 separate blocks")
    axes[1].legend(loc="upper center",bbox_to_anchor=(.43,-.22),frameon=False,ncol=3,
                   fontsize=7.8,handlelength=1.4,columnspacing=.85,handletextpad=.3)
    fig.subplots_adjust(left=.12,right=.98,top=.86,bottom=.23,wspace=.39)
    fig.savefig(directory/"calibration.pdf", bbox_inches="tight", pad_inches=.04,
                metadata={"CreationDate": None})
    fig.savefig(directory/"calibration.png", bbox_inches="tight", pad_inches=.04, dpi=200)
    plt.close(fig)


def matching_outputs(results, fig_dir, write_table):
    groups, fits = measurements(results)
    records = [row for rule in RULES for row in groups[rule]]
    save_csv(results/"step_summary.csv",records)
    save_csv(results/"calibration_summary.csv",fits)
    fixed_table, fitted_table = [], []
    base = groups["reference"][0]["lr_first"]
    for rule in ("reference",*FIXED):
        rows = groups[rule]
        batch, eta = rows[0]["batch"],rows[0]["lr_first"]
        relative = (eta/base)**2 * (1/batch-1/54000)/(1/64-1/54000)
        fixed_table.append([LABEL[rule],batch,f"{eta:.4f}",f"{relative:.2f}",image_count(rows,"stream_images"),
                            f"{med(rows,'val15'):.3f}"])
    write_table("fixed_rules","@{}lrrrrr@{}",["Rule","Batch",r"$\eta$","Step MSE ratio","Images to target","Val. CE"],fixed_table)
    for rule in ("noise","tuned","linear","once","refresh"):
        rows = groups[rule]
        fitted_table.append([LABEL[rule],image_count(rows,"stream_images"),image_count(rows,"charged_images"),
                             f"{med(rows,'total_s'):.2f}",f"{1000*med(rows,'fit_s'):.0f}",
                             f"{med(rows,'val15'):.3f}"])
    write_table("matched_rules","@{}lrrrrr@{}",["Rule","Stream","With fitting","Total (s)","Fit (ms)","Val. CE"],
                fitted_table,[r" & \multicolumn{2}{c}{Images to target} & \multicolumn{2}{c}{15-epoch time} & \\",
                              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"])
    figures(groups,fits,fig_dir)
