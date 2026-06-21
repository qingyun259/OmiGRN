# OmiGRN

Self-contained training code for **OmiGRN**, a multi-omics deep-learning framework
that fuses SNP genotype and transcriptome data to predict complex crop traits.

## Architecture

OmiGRN is a dual-branch network: the genomic and transcriptomic branches are encoded
separately, then fused and mapped to the target trait.

- **Genomic branch** — Multi-scale Genomic Context Mixer (MGCM): a Cross-Stage-Partial
  (CSP) backbone with cascaded Poly-Kernel-Inception (PKI) units and Context-Anchor
  Attention (CAA).
  → [`omigrn/blocks.py`](omigrn/blocks.py) (`MGCM`), used by `ModalityEncoder`.
- **Transcriptomic branch** — GRN message-passing network (GRN-MPNN) with a gated
  attention readout.
  → [`omigrn/grn_encoder.py`](omigrn/grn_encoder.py) (`GRNModalityEncoder`).
- **Fusion + output** — concatenation feature-fusion followed by an MLP regression head.
  → [`omigrn/model.py`](omigrn/model.py) (`OmiGRN`).

```text
omigrn_main/
├── omigrn/                  # core package
│   ├── __init__.py          # public API
│   ├── blocks.py            # MGCM genomic branch (CSP / PKI / CAA)
│   ├── grn_encoder.py       # GRN-MPNN transcriptomic branch (needs DGL)
│   ├── model.py             # OmiGRN dual-branch model (fusion + MLP head)
│   ├── data.py              # genotype / transcriptome / phenotype loading
│   └── trainer.py           # k-fold cross-validation training
├── example_data/            # small anonymized runnable subset (100 samples)
│   ├── geno.txt             # genotype
│   ├── trans.txt            # transcriptome
│   ├── pheno.txt            # phenotype    
│   └── network_all.csv      # GRN edges
├── GRN_Script/              # build the GRN edge list (WGCNA + GRNBoost2)
│   ├── run_wgcna_modules.R                # step 1: WGCNA co-expression modules
│   └── infer_grn_grnboost2_by_module.py   # step 2: module-wise GRNBoost2
├── train.py                 # training entry point (k-fold CV)
├── predict.py               # inference on new samples (fold ensemble)
├── demo.py                  # synthetic smoke test (no data, no DGL)
└── requirements.txt
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch` is pinned to `>=2.0` so you can pick the
[CPU / CUDA build](https://pytorch.org/get-started/locally/) for your machine.

The **transcriptomic GRN-MPNN branch additionally requires
[DGL](https://www.dgl.ai/pages/start.html)** (install the wheel matching your
torch + CUDA/CPU build; the reported runs used DGL 2.4.0). The genome-only
scenario (S1) runs without DGL.

Verify the install with a tiny synthetic run:

```bash
python demo.py
```

## Data format

All inputs are tab-separated tables whose **first column is the sample id**:

- **Genotype** (`--geno`) — one column per SNP, values in `{0, 1, 2}`.
- **Transcriptome** (`--transcripts`) — one column per gene (expression values);
  column headers are gene names matched against the GRN edge list.
- **Phenotype** (`--pheno`) — one column per trait.
- **GRN edges** (`--grn-edges`, CSV) — columns `Source, Target, Importance`
  (a directed, weighted edge from transcription factor → target gene).

Only samples present in every provided table (with a non-missing target) are kept;
the kept / removed ids are written to the output directory.

## GRN construction (transcriptomic branch)

The transcriptomic branch needs a directed, weighted gene regulatory network — the
`--grn-edges` CSV. It is built in two steps under [`GRN_Script/`](GRN_Script/),
following the paper (WGCNA co-expression modules → module-wise GRNBoost2):

```bash
# Step 1 — WGCNA: assign genes to co-expression modules -> gene_modules.tsv
Rscript GRN_Script/run_wgcna_modules.R trans.txt gene_modules.tsv

# Step 2 — module-wise GRNBoost2 (TFs from PlantTFDB) -> network_all.csv
python GRN_Script/infer_grn_grnboost2_by_module.py \
    --expr trans.txt \
    --tf-list tf_list.txt \
    --gene-modules gene_modules.tsv \
    --outdir grn_out
```

Step 2 writes `network_<group>.csv` (`Source, Target, Importance`) — exactly the
format the training script reads via `--grn-edges`.

Building the GRN needs extra tools, used only for this step (not for training):
R + WGCNA for step 1, and `pip install arboreto dask distributed` for step 2.

## Train

Three input scenarios — genome only (S1), transcriptome only (S2), and multi-omics
fusion (S3):

```bash
# S1 — genome only
python train.py --geno example_data/geno.txt --pheno example_data/pheno.txt --target Yield \
  --folds 10 --epochs 300 --patience 100 --device cuda:0

# S2 — transcriptome only, with the GRN prior
python train.py --transcripts example_data/trans.txt --pheno example_data/pheno.txt \
  --grn-edges example_data/network_all.csv --target Yield \
  --folds 10 --epochs 300 --device cuda:0

# S3 — multi-omics fusion (genotype + transcriptome + GRN)
python train.py --geno example_data/geno.txt --transcripts example_data/trans.txt \
  --pheno example_data/pheno.txt --grn-edges example_data/network_all.csv --target Yield \
  --folds 10 --epochs 300 --device cuda:0

# Train every phenotype column at once
python train.py --geno example_data/geno.txt --pheno example_data/pheno.txt --train-all-targets \
  --folds 10 --epochs 300 --device cuda:0
```

Useful flags: `--mode {auto,geno,transcript,both}`, `--resume`. Run
`python train.py --help` for the full list.

## Outputs

Each run writes to `--output`:

- `metrics.csv` — per-fold train / val MSE and Pearson r.
- `run_config.json` — the exact configuration used.
- `kept_sample_ids.txt` / `removed_samples.txt`.
- `fold_<k>/model.pt` — best-epoch weights for that fold.
- `fold_<k>/predictions.csv` — validation predictions vs. observed values.

With `--train-all-targets`, `summary_metrics.csv` and `summary_by_target.csv`
aggregate across traits.

## Predict

Predict phenotypes for **new samples** (no phenotype file needed) directly from
saved model weights. You only provide the weight file(s) and the data —
`predict.py` infers the architecture from the weight shapes, so no
`run_config.json` or run directory is needed. Passing several weight files (e.g.
the CV folds) averages their predictions.

```bash
# S1 — genome-only model
python predict.py --weights outputs/run_Yield/fold_*/model.pt \
  --geno new/geno.txt \
  --output new/pred_Yield.csv

# S2 — transcriptome-only model, with the GRN prior
python predict.py --weights outputs/run_Yield/fold_*/model.pt \
  --transcripts new/trans.txt --grn-edges data/network_all.csv \
  --output new/pred_Yield.csv

# S3 — multi-omics model (genotype + transcriptome + GRN)
python predict.py --weights outputs/run_Yield/fold_1/model.pt \
  --geno new/geno.txt --transcripts new/trans.txt \
  --grn-edges data/network_all.csv \
  --output new/pred_Yield.csv
```

The output CSV has `sample_id, prediction` (mean over the weight files). Notes:

- New inputs must have the **same feature columns, in the same order**, as training.
- A GRN model (S2 / S3) needs `--grn-edges` and DGL; the genome-only model (S1)
  needs neither. Predictions are in the original phenotype units.

## Use as a library

```python
import numpy as np
from omigrn import TrainingConfig, run_cross_validation

modalities = {"genotype": geno_array}   # {name: array of shape (n_samples, n_features)}
config = TrainingConfig(folds=10, epochs=300)
results = run_cross_validation(modalities, targets, sample_ids, config)
print(np.mean([r.val_pearson for r in results]))
```
