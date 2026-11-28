# Can One Large-Batch Step Match Four Small Steps?

One SGD update on 256 images sees the same examples as four updates on batches of 64. The four small steps evaluate gradients at successively updated weights. This Fashion-MNIST study asks how closely one large step can match their total displacement, and whether fitting its learning rate improves training outcomes. A comparison of SGD, momentum SGD and Adam establishes how batch size affects image use and GPU time.

[Main findings](#main-findings) · [Experimental setup](#experimental-setup) · [Reproduce results](#reproduce-results) · [Repository guide](#repository-guide)

## Main findings

### Batch size: images and GPU time

All three optimizers reached full-training cross-entropy (CE) **0.50 with fewer images at batch 64 and in less GPU time at batch 256**.

| Optimizer | Images at batch 64 | Images at batch 256 | Speedup at batch 256 |
| --- | ---: | ---: | ---: |
| SGD | 14,336 | 26,624 | 2.24× |
| Momentum SGD | 14,336 | 20,480 | 3.08× |
| Adam | 10,240 | 20,480 | 2.29× |

Image counts are medians across three seeds. Speedup is the median of the paired time ratios, `T64 / T256`. GPU time covers training updates with data already on the device; it excludes warm-up, data loading and full-dataset evaluation. See [measurement details](#measurement-details), [paired comparisons](results/paired_summary.json) and [per-run measurements](results/summary.csv).

### Step matching: local agreement and training outcome

- **Closer local steps:** fitting reduced held-out relative displacement error at all **15 seed–checkpoint pairs**: five seeds, each measured at initialization and after epochs 1 and 5. The median reduction relative to linear scaling was **28.3%**.
- **A remaining directional mismatch:** a median **94.9% of the residual squared error** came from displacement components perpendicular to the corresponding large-batch gradients. Changing the learning rate alone cannot remove those components.
- **Similar final validation loss:** refitted and grid-selected SGD both ended at median validation CE **0.320**, rounded to three decimals, after 15 epochs. Refitting required extra gradient evaluations, so the better local match did not produce a clear final-loss advantage in these runs.

![Fitted rates for five seeds and median held-out relative displacement errors at initialization and after epochs 1 and 5.](figures/calibration.png)

[Training results](results/step_summary.csv) · [Calibration measurements](results/calibration_summary.csv) · [Training and fitting costs](tables/matched_rules.tex)

These findings cover one dataset, a small network and three or five paired seeds per comparison. The reported speedups depend on the recorded GPU and timing procedure.

## Experimental setup

| Setting | Value |
| --- | --- |
| Dataset | [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): 54,000 training, 6,000 validation and 10,000 official test images |
| Split and preprocessing | Split seed 6704; 600 validation images per class; standardization uses training pixel statistics |
| Network | 784–128–10 multilayer perceptron with a ReLU hidden layer; 101,770 parameters |
| Training budget | 15 epochs, or 810,000 training images processed per run |
| Main comparison | SGD, momentum SGD and Adam at batch sizes 64 and 256 |
| Learning-rate selection | Four candidates per optimizer and batch size, five epochs each on seed 101; select by validation CE |
| Main seeds | 7, 19 and 43 |
| SGD follow-up seeds | 131, 137, 149, 151 and 157 |

Within each seed, configurations share the same initial weights and shuffled image order. The main comparison evaluates the official test set after training; the SGD follow-up compares validation loss. The selected baseline rates are recorded in [selected.json](results/selected.json).

### Seven SGD learning-rate policies

The follow-up runs all seven policies on each of the five new seeds, giving **35 training runs**.

| Policy | Record key | Batch size | Learning rate or fitting schedule |
| --- | --- | ---: | --- |
| Small-batch reference | `reference` | 64 | 0.1 |
| Unchanged | `constant` | 256 | 0.1 |
| Noise matched | `noise` | 256 | Approximately 0.20036; preserves one-step sampling error at fixed weights |
| Grid selected | `tuned` | 256 | 0.3, selected on tuning seed 101 |
| Linear | `linear` | 256 | 0.4, four times the small-batch rate |
| Fit once | `once` | 256 | Fit at initialization and keep the rate fixed |
| Refit twice | `refresh` | 256 | Fit at initialization, then refit after epochs 1 and 5 |

Each fit chooses a common nonnegative learning rate by least squares, using 16 blocks of 256 images. Another 16 blocks measure the displacement error. Both sets come from the training split and are disjoint at that checkpoint. One fit costs **8,192 image-gradient evaluations**, counting each image once in the large-batch gradient and again along the four-step path.

### Measurement details

- **Images to target:** the first observed full-training CE at or below 0.50, checked every 2,048 images and at the end of the first epoch.
- **Paired speedup:** replay each main trajectory's first epoch five times, take its median time to target, then compute `T64 / T256` within each seed. The results table reports the median of the three ratios.
- **GPU timing:** synchronize before and after batch indexing, forward and backward passes, and optimizer updates. Full-dataset evaluations take place outside these intervals.
- **Fitting cost:** the follow-up reports both training-image counts and counts including calibration gradients. Its total time includes training and fitting; displacement-audit and full-dataset evaluation time are excluded.

<details>
<summary>Reading the calibration CSV</summary>

In [calibration_summary.csv](results/calibration_summary.csv), the 15 rows with `rule=refresh` provide the five-seed, three-checkpoint comparison above.

- `audit_error` is the pooled relative displacement error on blocks held out from fitting; `linear_error` uses linear scaling on the same blocks and weights.
- `1 - audit_error / linear_error` gives the relative error reduction for one seed–checkpoint pair. Its median is 28.3%.
- `(audit_direction_floor / audit_error) ** 2` gives the share of residual squared error from perpendicular displacement. Its median is 94.9%.

</details>

## Reproduce results

### 1. Rebuild the archived figures and tables on a CPU

```bash
git clone https://github.com/sznnzs/fashion-mnist-step-matching
cd fashion-mnist-step-matching
python -m pip install -r requirements.txt
python analyze.py
```

This checks the archived experiment records and regenerates **four vector figures, the PNG preview and seven tables** in [figures/](figures) and [tables/](tables). It needs NumPy and Matplotlib; once installed, the rebuild runs offline without downloading the dataset or using a GPU.

### 2. Repeat the complete training study on a GPU

Use the same clone and dependencies from step 1. The archived runs used this environment:

| Component | Recorded version or hardware |
| --- | --- |
| Python | 3.12.3 |
| NumPy | 2.5.5 |
| PyTorch / CUDA build | 2.9.1 / 12.8 |
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell |

Install a CUDA-compatible PyTorch build in the training environment. For the recorded PyTorch/CUDA combination:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

See the [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/) for other CUDA builds. From the repository root, run the study in an existing GPU allocation:

```bash
STUDY_PYTHON=python bash run.sh all
```

Alternatively, submit to the course cluster through Slurm:

```bash
mkdir -p logs
STUDY_PYTHON="$(command -v python)" sbatch --gres=gpu:pilot:1 -o 'logs/%j.out' run.sh all
```

The `gpu:pilot` resource name is specific to the course cluster. `STUDY_PYTHON` selects the training interpreter. The [launcher](run.sh) uses one GPU, downloads Fashion-MNIST to `data/`, verifies its published checksums and writes new measurements to `replication/` by default. Archived measurements remain in `results/`.

### 3. Inspect and compare a fresh run

After training finishes:

```bash
STUDY_RESULTS="$PWD/replication" python analyze.py
python verify_replication.py replication --output replication/reproduction_check.json
```

New figures and tables appear in `replication/report_assets/`. The [verifier](verify_replication.py) checks the run configuration and reports the largest numerical differences and measured timing ratios. Its report is printed and saved to `replication/reproduction_check.json`. An earlier [independent GPU replication](results/reproduction_check.json) is included with the archived results.

## Repository guide

| Experiment | Implementation | Records and figures |
| --- | --- | --- |
| Compare optimizers | [study.py](study.py) | [Tuning runs](results/tune) · [Main runs](results/main) · [Loss curves](figures/convergence.pdf) |
| Measure time and images to target | [early.py](early.py) | [Five replay rounds](results/timing) · [Paired comparisons](figures/target.pdf) |
| Measure gradient noise at identical weights | [probe.py](probe.py) | [Individual draws](results/probe.json) · [Estimates and exact values](tables/probe.tex) |
| Match one large SGD step to four small steps | [step_matching.py](step_matching.py) · [matching_analysis.py](matching_analysis.py) | [Per-run records](results/step_matching) · [Fixed-rate controls](figures/rate_response.pdf) · [Held-out fit](figures/calibration.pdf) |

[data_io.py](data_io.py) handles dataset download and splitting; [analyze.py](analyze.py) checks saved measurements and generates the report figures and tables.
