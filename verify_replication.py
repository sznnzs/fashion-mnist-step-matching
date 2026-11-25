"""Compare a fresh replication with the archived numerical results."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def check(reference, fresh):
    result = {"selected_rates_match": read(reference/"selected.json") == read(fresh/"selected.json")}
    if not result["selected_rates_match"]:
        raise ValueError("Learning-rate selections differ")
    max_loss = max_accuracy = 0.0
    checkpoints = endpoints = 0
    jobs = set()
    for stage in ("tune", "main", "native"):
        original = sorted((reference/stage).glob("*.json"))
        assert [p.name for p in original] == sorted(p.name for p in (fresh/stage).glob("*.json"))
        for file in original:
            a, b = read(file), read(fresh/stage/file.name)
            for key in ("method", "batch", "seed", "rate", "split"):
                assert a[key] == b[key], (file.name, key)
            assert a["environment"]["code_sha256"] == b["environment"]["code_sha256"]
            jobs.add(str(b["environment"]["job_id"]))
            for aa, bb in zip(a["history"], b["history"], strict=True):
                assert (aa["epoch"], aa["updates"]) == (bb["epoch"], bb["updates"])
                for split in ("train", "valid"):
                    max_loss = max(max_loss, abs(aa[split]["loss"] - bb[split]["loss"]))
                    max_accuracy = max(max_accuracy, abs(aa[split]["accuracy"] - bb[split]["accuracy"]))
                checkpoints += 1
            if "test" in a:
                max_loss = max(max_loss, abs(a["test"]["loss"] - b["test"]["loss"]))
                max_accuracy = max(max_accuracy, abs(a["test"]["accuracy"] - b["test"]["accuracy"]))
                endpoints += 1

    dense_losses = []
    for path in sorted((reference/"timing").glob("round*/*.json")):
        a, b = read(path), read(fresh/path.relative_to(reference))
        for aa, bb in zip(a["history"], b["history"], strict=True):
            assert aa["examples"] == bb["examples"]
            dense_losses.append(abs(aa["loss"] - bb["loss"]))
    a_probe, b_probe = (read(folder/"probe.json")["rows"] for folder in (reference, fresh))
    errors = []
    for aa, bb in zip(a_probe, b_probe, strict=True):
        assert aa["epoch"] == bb["epoch"]
        for a, b in zip(aa["batches"], bb["batches"], strict=True):
            assert a["batch"] == b["batch"]
            errors.extend(abs(x-y) for x,y in zip(a["draws_squared_error"], b["draws_squared_error"], strict=True))

    timing = read(fresh/"paired_summary.json")
    ratios = [value for group in timing.values() for value in group["paired_t50_ratio"]]
    transfer_files = sorted((reference/"step_matching").glob("*.json"))
    assert len(transfer_files) == 35
    assert [p.name for p in transfer_files] == sorted(p.name for p in (fresh/"step_matching").glob("*.json"))
    transfer_checkpoints = fit_events = 0
    transfer_delta = moment_delta = 0.0
    for p in transfer_files:
        a, b = read(p), read(fresh/"step_matching"/p.name)
        for key in ("seed", "rule", "batch", "split", "base_rate", "noise_rate", "tuned_rate"):
            assert a[key] == b[key], (p.name, key)
        for key in ("code_sha256", "step_matching_sha256"):
            assert a["environment"][key] == b["environment"][key]
        jobs.add(str(b["environment"]["job_id"]))
        assert (a["target_hit"] is None) == (b["target_hit"] is None)
        if a["target_hit"] is not None:
            for key in ("examples", "gradient_images", "updates"):
                assert a["target_hit"][key] == b["target_hit"][key]
        for aa, bb in zip(a["history"], b["history"], strict=True):
            for split in ("train", "valid"):
                for key in ("loss", "accuracy"):
                    transfer_delta = max(transfer_delta, abs(aa[split][key] - bb[split][key]))
            if aa["rate"] is not None:
                transfer_delta = max(transfer_delta, abs(aa["rate"] - bb["rate"]))
            transfer_checkpoints += 1
        for aa, bb in zip(a["calibrations"], b["calibrations"], strict=True):
            for key in ("after_epoch", "fit_indices_sha256", "audit_indices_sha256"):
                assert aa[key] == bb[key]
            for key in ("fit", "audit"):
                for row_a, row_b in zip(aa[key],bb[key],strict=True):
                    moment_delta = max(moment_delta, *(abs(v-w) for v,w in zip(row_a,row_b,strict=True)))
            fit_events += 1
    result.update(reference_epochs_checked=checkpoints, test_endpoints_checked=endpoints,
                  max_abs_loss_diff=max_loss, max_abs_accuracy_diff=max_accuracy,
                  dense_losses_checked=len(dense_losses), max_abs_dense_loss_diff=max(dense_losses),
                  gradient_errors_checked=len(errors), max_abs_gradient_error_diff=max(errors),
                  fresh_slurm_jobs=sorted(jobs), fresh_pairs_faster_at_256=sum(r>1 for r in ratios),
                  fresh_pairs_total=len(ratios), fresh_paired_t50_ratio_range=[min(ratios), max(ratios)],
                  transfer_runs_checked=len(transfer_files), transfer_checkpoints_checked=transfer_checkpoints,
                  max_abs_transfer_metric_diff=transfer_delta, calibration_fits_checked=fit_events,
                  max_abs_calibration_moment_diff=moment_delta)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fresh", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(ROOT/"results", args.fresh)
    text = json.dumps(report, indent=2)+"\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text)
