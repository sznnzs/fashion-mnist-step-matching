"""Validate the measurements and regenerate all numerical tables and vector figures."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
from matching_analysis import matching_outputs

ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("STUDY_RESULTS", ROOT / "results")).resolve()
ASSETS = ROOT if RESULTS == ROOT / "results" else RESULTS / "report_assets"
FIG = ASSETS / "figures"
TABLE = ASSETS / "tables"
METHODS = ("SGD", "Momentum", "Adam")
BATCHES = (64, 256)
SEEDS = (7, 19, 43)
NTRAIN = 54000
TARGET = 0.50


def load(p):
    return json.loads(p.read_text())


def put_json(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def table(name, layout, headings, rows, grouped_header=None):
    ending = r" \\"
    lines = [r"\begin{tabular}{"+layout+"}", r"\toprule"]
    if grouped_header:
        lines.extend(grouped_header)
    lines.extend([" & ".join(headings)+ending, r"\midrule"])
    for row in rows:
        if row is None:
            lines.append(r"\addlinespace[2pt]")
        else:
            lines.append(" & ".join(map(str,row))+ending)
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE / f"{name}.tex").write_text("\n".join(lines)+"\n")


def first(history, predicate):
    return next((row for row in history if predicate(row)), None)


def subset(rows, **conditions):
    return [r for r in rows if all(r[k] == v for k,v in conditions.items())]


def values(rows, key):
    return np.array([r[key] for r in rows],dtype=float)


def median(rows, key):
    return float(np.median(values(rows,key)))


def raw_data():
    choices = load(RESULTS / "selected.json")
    source = hashlib.sha256()
    for name in ("data_io.py","study.py"):
        source.update((ROOT / name).read_bytes())
    signature = source.hexdigest()
    rows, histories, tuning = [], {}, {}
    split_hashes, replay_cards = set(), set()
    for m in METHODS:
        for b in BATCHES:
            key = f"{m}-b{b}"
            trial = sorted([load(p) for p in (RESULTS/"tune").glob(f"{key}-*.json")],
                           key=lambda r:r["rate"])
            assert len(trial) == 4
            assert all(r["seed"]==101 and len(r["history"])==6 and "test" not in r for r in trial)
            winner = min(trial,key=lambda r:(r["history"][-1]["valid"]["loss"],r["rate"]))
            assert choices[key] == winner["rate"]
            tuning[m,b] = trial
            for s in SEEDS:
                name = f"{key}-s{s}.json"
                r = load(RESULTS/"main"/name)
                h = r["history"]
                assert r["environment"]["code_sha256"] == signature and r["rate"] == choices[key]
                assert len(h)==16 and [e["epoch"] for e in h]==list(range(16))
                assert all(e["examples"]==e["epoch"]*NTRAIN and
                           e["updates"]==e["epoch"]*math.ceil(NTRAIN/b) for e in h)
                assert all(e["train_s"] < f["train_s"] for e,f in zip(h,h[1:]))
                split_hashes.add(tuple(r["split"].values()))
                histories[m,b,s] = h
                threshold = first(h,lambda e:e["train"]["loss"]<=TARGET)
                strict = first(h,lambda e:e["train"]["loss"]<=.30)
                hits, end_diffs = [], []
                for rep in range(1,6):
                    replay = load(RESULTS/"timing"/f"round{rep}"/name)
                    assert replay["environment"]["code_sha256"] == signature
                    assert replay["environment"]["replay_code_sha256"] == hashlib.sha256((ROOT/"early.py").read_bytes()).hexdigest()
                    replay_cards.add(replay["environment"].get("gpu_key", "cpu"))
                    assert replay["grid"]==2048 and replay["rate"]==r["rate"]
                    trace = replay["history"]
                    assert [e["examples"] for e in trace] == [*range(0,NTRAIN,2048),NTRAIN]
                    assert all(e["updates"]==math.ceil(e["examples"]/b) for e in trace)
                    diff = abs(h[1]["train"]["loss"]-trace[-1]["loss"])
                    assert diff <= 1e-6
                    end_diffs.append(diff)
                    hit = first(trace,lambda e:e["loss"]<=TARGET)
                    if hit is not None:
                        hits.append(hit)
                if hits:
                    assert len(hits)==5 and len({e["examples"] for e in hits})==1
                    ms = [v["train_s"]*1000 for v in hits]
                else:
                    ms = [float("nan")]
                rows.append(dict(method=m,batch=b,seed=s,lr=r["rate"],epoch50=threshold["epoch"] if threshold else None,
                                 epoch30=strict["epoch"] if strict else None,sec30=strict["train_s"] if strict else None,
                                 early_examples=hits[0]["examples"] if hits else None,
                                 early_updates=hits[0]["updates"] if hits else None,early_ms=float(np.median(ms)),
                                 early_min_ms=min(ms),early_max_ms=max(ms),early_repeats=len(hits),
                                 replay_diff=max(end_diffs),train_s=h[-1]["train_s"],eval_s=h[-1]["eval_s"],
                                 ms_update=1000*h[-1]["train_s"]/h[-1]["updates"],loss=h[-1]["train"]["loss"],
                                 val_loss=h[-1]["valid"]["loss"],test_acc=100*r["test"]["accuracy"],
                                 test_ce=r["test"]["loss"],increases=sum(a["train"]["loss"]<z["train"]["loss"] for a,z in zip(h,h[1:])),
                                 device=r["environment"].get("gpu_key","cpu")))
    assert len(split_hashes)==1 and len(replay_cards)==1
    for s in SEEDS:
        initial = [histories[m,b,s][0]["train"]["loss"] for m in METHODS for b in BATCHES]
        assert np.ptp(initial)<1e-7 and len({r["device"] for r in subset(rows,seed=s)})==1
    return rows, histories, tuning


def result_tables(rows, tuning):
    tuning_rows, targets, endpoints, sensitivity = [], [], [], []
    macros, summaries = {}, {}
    for m in METHODS:
        if targets:
            for tab in (tuning_rows,targets,endpoints):
                tab.append(None)
        paired = {}
        for b in BATCHES:
            group = sorted(subset(rows,method=m,batch=b),key=lambda r:r["seed"])
            paired[b] = group
            trial = tuning[m,b]
            cells = [f"{v['history'][-1]['valid']['loss']:.3f}" for v in trial]
            idx = [v["rate"] for v in trial].index(group[0]["lr"])
            cells[idx] = r"\textbf{"+cells[idx]+"}"
            grid = ", ".join(f"{r['rate']:g}" for r in trial)
            tuning_rows.append([m if b==64 else "",b,grid if b==64 else "",*cells])
            stamps = values(group,"early_ms")
            counts = values(group,"early_examples")
            targets.append([m if b==64 else "",b,f"{np.median(counts):,.0f}",f"{median(group,'early_updates'):.0f}",
                            f"{np.median(stamps):.1f}",f"{min(stamps):.1f}--{max(stamps):.1f}"])
            acc = values(group,"test_acc")
            endpoints.append([m if b==64 else "",b,f"{median(group,'loss'):.3f}",f"{np.median(acc):.2f}",
                              f"{min(acc):.2f}--{max(acc):.2f}",f"{median(group,'train_s'):.2f}",
                              f"{median(group,'ms_update'):.3f}"])
        a, b = paired[64], paired[256]
        early_ratio = values(a,"early_ms")/values(b,"early_ms")
        budget_ratio = values(a,"train_s")/values(b,"train_s")
        second_ratio = values(a,"sec30")/values(b,"sec30")
        native = [load(RESULTS/"native"/f"{m}-b{k}-s19.json") for k in BATCHES]
        assert all(nr["stage"]=="native" and nr["rate"]==paired[k][0]["lr"] for k,nr in zip(BATCHES,native))
        native_ratio = native[0]["history"][-1]["train_s"]/native[1]["history"][-1]["train_s"]
        full = float(np.median(budget_ratio))
        speed = float(np.median(early_ratio))
        macros["Speed"+m] = f"{speed:.2f}"
        macros["Full"+m] = f"{full:.2f}"
        summaries[m] = {"paired_t50_ratio":early_ratio.tolist(),"paired_full_time_ratio":budget_ratio.tolist(),
                        "paired_t30_ratio":second_ratio.tolist(),"native_seed19_full_ratio":native_ratio,
                        "accuracy_difference_pp":(values(b,"test_acc")-values(a,"test_acc")).tolist()}
        sensitivity.append([m,f"{speed:.2f}",f"{min(early_ratio):.2f}--{max(early_ratio):.2f}",
                            f"{np.median(second_ratio):.2f}",f"{full:.2f}",f"{native_ratio:.2f}"])
    table("rates","@{}lr l rrrr@{}",["Method","Batch","Candidate rates (ascending)",
          r"$\eta_1$",r"$\eta_2$",r"$\eta_3$",r"$\eta_4$"],tuning_rows,
          [r" & & & \multicolumn{4}{c}{Validation cross-entropy} \\",r"\cmidrule(lr){4-7}"])
    table("target","@{}lrrrrr@{}",["Method","Batch","Images","Updates","Time (ms)","Seed range (ms)"],targets)
    table("endpoints","@{}lrrrrrr@{}",["Method","Batch","Train CE","Median","Seed range",
          "Total (s)","ms/update"],endpoints,
          [r" & & & \multicolumn{2}{c}{Test accuracy (\%)} & \multicolumn{2}{c}{Training cost} \\",
           r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}"])
    table("sensitivity","@{}lrrrrr@{}",["Method",r"$L\leq.50$","Seed range",r"$L\leq.30$","15 epochs","Fused$^{*}$"],sensitivity)
    probes = load(RESULTS/"probe.json")
    pr = []
    for p in probes["rows"]:
        small, large = p["batches"]
        for v in (small,large):
            draws = v["draws_squared_error"]
            assert len(draws)==256 and np.isclose(np.mean(draws),v["empirical"])
            assert np.isclose(np.std(draws,ddof=1)/16,v["mc_se"])
            assert np.isclose(v["prediction"],(1/v["batch"]-1/NTRAIN)*p["trace"])
        ratio = small["empirical"]/large["empirical"]
        pr.append([p["epoch"],f"{p['trace']:.2f}",f"{small['prediction']:.3f}",
                   f"{small['empirical']:.3f} $\\pm$ {small['mc_se']:.3f}",f"{large['prediction']:.3f}",
                   f"{large['empirical']:.3f} $\\pm$ {large['mc_se']:.3f}",f"{ratio:.2f}"])
    table("probe","@{}rrrrrrr@{}",["Epoch",r"$\mathrm{tr}(S)$","Exact",r"Estimate $\pm$ SE","Exact",r"Estimate $\pm$ SE","Ratio"],pr,
          [r" & & \multicolumn{2}{c}{Batch 64} & \multicolumn{2}{c}{Batch 256} & \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}"])
    (TABLE/"numbers.tex").write_text("\n".join(r"\newcommand{\%s}{%s}"%(k,v) for k,v in macros.items())+"\n")
    put_json(RESULTS/"paired_summary.json",summaries)


def clean_axis(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="both",color="#dfe4e7",linewidth=.55,zorder=0)
    ax.tick_params(length=2.5,color="#66717a")
    ax.set_axisbelow(True)


def figures(rows, histories):
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.size":8.5,
                         "axes.labelsize":8.5,"axes.titlesize":9,"legend.fontsize":8,
                         "axes.edgecolor":"#69747b","text.color":"#17262e",
                         "axes.labelcolor":"#17262e","xtick.color":"#394b56","ytick.color":"#394b56"})
    fig, ax = plt.subplots(figsize=(5.55,2.82))
    clean_axis(ax)
    ax.grid(False)
    ax.grid(axis="x", color="#dce3e6", linewidth=.65)
    ax.axvline(1, color="#485761", linewidth=1.05, linestyle=(0,(3,2)), zorder=1)
    times, images, positions = [], [], []
    for i, method in enumerate(METHODS):
        for j, seed in enumerate(SEEDS):
            a, b = [subset(rows, method=method, batch=k, seed=seed)[0] for k in BATCHES]
            y = 3.65*i + j
            positions.append(y)
            times.append(b["early_ms"] / a["early_ms"])
            images.append(b["early_examples"] / a["early_examples"])
            ax.plot([times[-1], images[-1]], [y,y], color="#aab6bc", lw=.9, zorder=1)
        ax.text(-.115, 3.65*i+1, method, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9, fontweight="bold")
    ax.scatter(times, positions, s=27, marker="s", facecolor="#275873", edgecolor="#275873",
               label="Training time", zorder=3)
    ax.scatter(images, positions, s=30, marker="o", facecolor="white", edgecolor="#a34b22",
               linewidth=1.45, label="Images seen", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xlim(min(.23, min(times+images)*.93), max(2.63, max(times+images)*1.06))
    ax.set_xticks([.25,.5,1,2], ["0.25×","0.5×","1× (equal)","2×"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(positions, [str(s) for _ in METHODS for s in SEEDS])
    ax.set_ylim(positions[-1]+.6, -.7)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Batch 256 / batch 64 at the first observed loss ≤ 0.50")
    ax.text(-.047,1.035,"Seed",transform=ax.transAxes,ha="center",va="bottom",fontsize=8)
    ax.legend(loc="lower center",bbox_to_anchor=(.54,1.01),ncol=2,frameon=False,
              handletextpad=.45,columnspacing=2)
    fig.subplots_adjust(left=.246,right=.982,bottom=.199,top=.847)
    fig.savefig(FIG/"target.pdf",bbox_inches="tight",pad_inches=.04)
    plt.close(fig)

    fig, axes = plt.subplots(3,2,figsize=(5.65,4.0),sharey=True,sharex="col")
    longest_run = max(h[-1]["train_s"] for h in histories.values())
    palette = {64:"#205e85",256:"#af4f22"}
    for i,m in enumerate(METHODS):
        for j,ax in enumerate(axes[i]):
            clean_axis(ax)
            ax.axhline(.30,color="#76838c",lw=.65,ls=":")
            for b in BATCHES:
                xs, ys = [], []
                for s in SEEDS:
                    h = histories[m,b,s][1:]
                    x = np.array([e["examples"]/1e6 if j==0 else e["train_s"] for e in h])
                    y = np.array([e["train"]["loss"] for e in h])
                    xs.append(x)
                    ys.append(y)
                    ax.plot(x,y,lw=.7,alpha=.31,color=palette[b],ls="-" if b==64 else "--")
                ax.plot(np.median(xs,axis=0),np.median(ys,axis=0),color=palette[b],
                        lw=1.65,ls="-" if b==64 else "--")
            ax.set_ylim(.135,.515)
            ax.set_yticks([.2,.3,.4,.5])
            if j==0:
                ax.text(.97,.94,m,ha="right",va="top",transform=ax.transAxes,
                        fontsize=8.5,color="#243a46",fontweight="bold")
                ax.set_xlim(0,.83)
                ax.set_xticks([0,.2,.4,.6,.8])
            else:
                ax.set_xlim(0,math.ceil(longest_run*1.02))
    axes[0,0].set_title("Equal image exposure")
    axes[0,1].set_title("Measured training time")
    axes[2,0].set_xlabel("Images seen (millions)")
    axes[2,1].set_xlabel("Cumulative training time (s)")
    handles=[Line2D([],[],color=palette[b],lw=1.8,ls=s,label=f"Batch {b}") for b,s in [(64,"-"),(256,"--")]]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.55,1),ncol=2,frameon=False)
    fig.supylabel("Full-training cross-entropy",x=.015,fontsize=9)
    fig.subplots_adjust(left=.119,right=.985,bottom=.115,top=.86,wspace=.14,hspace=.21)
    fig.savefig(FIG/"convergence.pdf",bbox_inches="tight",pad_inches=.03)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True,exist_ok=True)
    TABLE.mkdir(parents=True,exist_ok=True)
    records, histories, tuning = raw_data()
    with (RESULTS/"summary.csv").open("w",newline="") as f:
        output=csv.DictWriter(f,fieldnames=records[0].keys())
        output.writeheader()
        output.writerows(records)
    result_tables(records,tuning)
    figures(records,histories)
    matching_outputs(RESULTS, FIG, table)
    print("Verified 24 tuning runs, 18 main runs, 90 timing replays, 6 native runs and 35 rate-transfer runs.")
