# OmiGRN

OmiGRN is a dual-branch deep learning framework that integrates SNP genotypes, gene expression profiles, and gene regulatory networks (GRNs) for crop trait prediction.

- **Genomic branch**: Multi-scale Genomic Context Mixer (MGCM) for local SNP feature extraction across multiple receptive fields.
- **Transcriptomic branch**: Directed GRN message-passing network (GRN-MPNN) with attention readout and gated expression fusion.
- **Prediction head**: Per-branch feature normalization and concatenation followed by an MLP regressor.

```text
OmiGRN/
├── omigrn/
│   ├── __init__.py
│   ├── layers.py
│   ├── grn.py
│   ├── model.py
│   ├── data.py
│   └── engine.py
├── scripts/
│   └── build_grn.py
├── example_data/
│   ├── geno.txt
│   ├── trans.txt
│   ├── pheno.txt
│   └── grn.csv
├── train.py
├── predict.py
├── demo.py
├── environment.yml
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

Requires Python 3.11, PyTorch 2.5.1, and DGL 2.4.0.

```bash
conda env create -f environment.yml
conda activate omigrn
```

Or install with pip:

```bash
pip install -r requirements.txt -f https://data.dgl.ai/wheels/torch-2.4/repo.html
```

For GPU support, install the matching CUDA builds of [PyTorch](https://pytorch.org/get-started/locally/) and [DGL](https://www.dgl.ai/pages/start.html) (genome-only S1 runs without DGL). You can test the environment with `python demo.py`.

## Data Format

Input feature and phenotype files are tab-separated tables with sample IDs in the first column:

- `--geno`: SNP genotypes coded as 0, 1, or 2 (missing values as `NA`).
- `--transcripts`: Gene expression matrix with gene IDs as column headers.
- `--pheno`: Phenotype table with one column per trait.
- `--grn-edges`: Comma-separated directed regulatory edges (`Source,Target,Importance`).

To extract a GRN edge file matching an expression matrix from a PlantRegMap regulation table:

```bash
python scripts/build_grn.py --interactions example_data/grn.csv --transcripts example_data/trans.txt --output outputs/grn_mapped.csv
```

## Training

Use `train.py` to run 10-fold cross-validation under genome-only (S1), transcriptome-only (S2), or multi-omics (S3) inputs, or pass `--all-targets` to train all traits in `--pheno`:

```bash
python train.py --geno example_data/geno.txt --pheno example_data/pheno.txt --target DTS --output outputs/s1_dts
python train.py --transcripts example_data/trans.txt --grn-edges example_data/grn.csv --pheno example_data/pheno.txt --target DTS --output outputs/s2_dts
python train.py --geno example_data/geno.txt --transcripts example_data/trans.txt --grn-edges example_data/grn.csv --pheno example_data/pheno.txt --target DTS --output outputs/s3_dts
python train.py --geno example_data/geno.txt --transcripts example_data/trans.txt --grn-edges example_data/grn.csv --pheno example_data/pheno.txt --all-targets --output outputs/s3_all
```

Pass `--device cuda:0` to train on GPU, or run `python train.py --help` for full options. Training outputs are saved to `--output`:

- `metrics.csv`: per-fold training, validation, and test metrics.
- `oof_predictions.csv`: out-of-fold test predictions.
- `run_config.json`: run parameters and summary statistics.
- `fold_<k>/model.pt`: saved model checkpoint for each fold.

## Prediction

Use `predict.py` to predict phenotypes for new samples from saved fold weights (passing multiple checkpoints averages their predictions):

```bash
python predict.py --weights outputs/s3_dts/fold_*/model.pt --geno example_data/geno.txt --transcripts example_data/trans.txt --output outputs/pred_s3_dts.csv
python predict.py --weights outputs/s1_dts/fold_1/model.pt --geno example_data/geno.txt --output outputs/pred_s1_dts_fold1.csv
```

## Citation

Cao, Q., Zhao, X., Zhang, D., Guo, H., Yang, F., Zhang, Q., Xu, T., Cao, Y., Wang, K., & Liu, Z. OmiGRN: A multi-omics method integrating gene regulatory networks for crop phenotypic prediction.

## License

Released under the [MIT License](LICENSE).
