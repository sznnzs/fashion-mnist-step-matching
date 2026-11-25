# Can One Large-Batch Step Match Four Small Steps?

Experiments on Fashion-MNIST comparing SGD, momentum SGD and Adam at batch sizes 64 and 256. A second experiment fits the learning rate of one 256-image SGD update to the weight displacement of four consecutive 64-image updates on the same images. Each fit is evaluated on separate training blocks.

The repository includes the implementations, measured trajectories and data behind all four figures and seven tables. The two main datasets for analysis are [optimizer comparisons](results/summary.csv) and [SGD learning-rate comparisons](results/step_summary.csv).

## Rebuild the figures and tables

From the repository root, run

```bash
python -m pip install -r requirements.txt
python analyze.py
```

This checks the archived measurements and writes the vector figures to `figures/` and the LaTeX table fragments to `tables/`. Once the Python packages are installed, this step runs offline on a CPU.

## Repeat the experiments

Use Python 3.12 with NumPy and a CUDA-compatible PyTorch installation. The measured runs used PyTorch 2.9.1+cu128, NumPy 2.5.5 and NVIDIA RTX PRO 5000 72GB Blackwell GPUs. A single GPU can run the complete study. Set `STUDY_PYTHON` to the training interpreter if it differs from `python`. [PyTorch installation](https://pytorch.org/get-started/previous-versions/)

From the repository root in a GPU allocation, run

```bash
STUDY_PYTHON=python bash run.sh all
```

On the course cluster, submit the same script through Slurm instead.

```bash
mkdir -p logs
sbatch --gres=gpu:pilot:1 -o 'logs/%j.out' run.sh all
```

The script downloads the four Fashion-MNIST IDX files, checks their published MD5 digests and caches them in `data/`. It writes fresh results to `replication/` by default. To analyze those results and compare them with the archived measurements, run

```bash
STUDY_RESULTS="$PWD/replication" python analyze.py
python verify_replication.py replication
```

## Experimental design

- **Task.** A 784–128–10 ReLU network minimizes cross-entropy on 54,000 images. Split seed 6704 reserves 600 images per class for validation. The original 10,000-image test set is used for final endpoints in the optimizer comparison. [Dataset and checksums](https://github.com/zalandoresearch/fashion-mnist/tree/b2617bb6d3ffa2e429640350f613e3291e10b141)
- **Optimizer comparison.** Each optimizer and batch size gets four candidate learning rates and five tuning epochs on seed 101. Formal runs share initialization and image order within each of seeds 7, 19 and 43 and train for 15 epochs. The first epoch is replayed five times with the training loss checked every 2,048 images. Timing covers batch indexing and forward, backward and update operations on GPU-resident data.
- **Step matching.** Five new seeds (131, 137, 149, 151 and 157) compare seven SGD policies. A fit uses 16 training blocks of 256 images. A further 16 disjoint blocks measure the relative displacement error. The cost of fitting enters the training budget. The once-fitted rate and the policy that refits after epochs 1 and 5 share their initial fit.

## Code and records

| Component | Implementation | Records |
| --- | --- | --- |
| Data, custom updates, tuning and main runs | `data_io.py`, `study.py` | `results/tune/`, `results/selected.json`, `results/main/`, `results/native/` |
| Time to the first loss crossing | `early.py` | `results/timing/`, `results/paired_summary.json` |
| Fixed-weight gradient noise | `probe.py` | `results/probe.json`, `results/snapshots/` |
| Learning-rate fits and held-out errors | `step_matching.py` | `results/step_matching/`, `results/calibration_summary.csv` |
| Figures, tables and numerical checks | `analyze.py`, `matching_analysis.py`, `verify_replication.py` | `results/optimizer_check.json`, `results/reproduction_check.json` |
