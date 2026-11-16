import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from omigrn import (
    TrainConfig,
    get_traits,
    load_data,
    run_cv,
)


def _sep(val):
    return "\t" if val == "\\t" else val


def get_parser():
    p = argparse.ArgumentParser(description="Nested cross-validation training for OmiGRN")
    p.add_argument("--geno", default=None, help="Genotype dosage matrix path")
    p.add_argument("--transcripts", nargs="+", default=None, help="Gene expression matrix path(s)")
    p.add_argument("--pheno", required=True, help="Phenotype table path")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--target", default=None, help="Target trait column in --pheno")
    grp.add_argument("--all-targets", action="store_true", help="Train every trait column in --pheno")
    p.add_argument("--grn-edges", default=None, help="Directed GRN edge CSV path")
    p.add_argument("--mode", choices=["auto", "geno", "transcript", "both", "S1", "S2", "S3"], default="auto", help="Input modality mode")
    p.add_argument("--folds", type=int, default=10, help="Number of outer CV folds")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Inner validation fraction of outer train")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--split-file", default=None, help="Custom split JSON path")
    p.add_argument("--refit-outer", action=argparse.BooleanOptionalAction, default=True, help="Refit selected model on full outer train fold")
    p.add_argument("--epochs", type=int, default=200, help="Maximum training epochs")
    p.add_argument("--patience", type=int, default=40, help="Early stopping patience")
    p.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient L2 norm clip")
    p.add_argument("--embed-dim", type=int, default=128, help="Branch embedding dimension")
    p.add_argument("--mlp-hidden", type=int, default=128, help="Output head hidden dimension")
    p.add_argument("--grn-dim", type=int, default=128, help="GRN hidden dimension")
    p.add_argument("--grn-layers", type=int, default=2, help="Number of GRN message-passing layers")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    p.add_argument("--stop-metric", choices=["mse", "pearson"], default="mse", help="Inner validation selection metric")
    p.add_argument("--tune-trials", type=int, default=0, help="Inner hyperparameter tuning trials per fold")
    p.add_argument("--tuner", choices=["auto", "optuna", "random"], default="auto", help="Hyperparameter search backend")
    p.add_argument("--tune-arch", action="store_true", help="Tune embed_dim and mlp_hidden during search")
    p.add_argument("--geno-sep", default="\t", help="Delimiter for --geno")
    p.add_argument("--transcript-sep", default="\t", help="Delimiter for --transcripts")
    p.add_argument("--pheno-sep", default="\t", help="Delimiter for --pheno")
    p.add_argument("--output", default="outputs/omigrn", help="Output directory")
    p.add_argument("--device", default="cpu", help="Device (e.g. cpu or cuda:0)")
    p.add_argument("--threads", type=int, default=1, help="CPU thread count")
    return p


def _pick_inputs(mode, geno, transcripts):
    tx = list(transcripts) if transcripts else None
    if mode == "auto":
        if geno is None and not tx:
            raise ValueError("Provide at least one of --geno or --transcripts.")
        return geno, tx
    if mode in ("geno", "S1"):
        if geno is None:
            raise ValueError(f"--mode {mode} requires --geno.")
        return geno, None
    if mode in ("transcript", "S2"):
        if not tx:
            raise ValueError(f"--mode {mode} requires --transcripts.")
        return None, tx
    if mode in ("both", "S3"):
        if geno is None or not tx:
            raise ValueError(f"--mode {mode} requires both --geno and --transcripts.")
        return geno, tx
    raise ValueError(f"Unsupported mode: {mode}.")


def main(argv=None):
    args = get_parser().parse_args(argv)
    geno_path, tx_paths = _pick_inputs(args.mode, args.geno, args.transcripts)
    geno_sep = _sep(args.geno_sep)
    tx_sep = _sep(args.transcript_sep)
    pheno_sep = _sep(args.pheno_sep)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        embed_dim=args.embed_dim,
        mlp_hidden=args.mlp_hidden,
        grn_dim=args.grn_dim,
        grn_layers=args.grn_layers,
        dropout=args.dropout,
        folds=args.folds,
        val_ratio=args.val_ratio,
        seed=args.seed,
        refit_outer=args.refit_outer,
        stop_metric=args.stop_metric,
        tune_trials=args.tune_trials,
        tuner=args.tuner,
        tune_arch=args.tune_arch,
        device=args.device,
        threads=args.threads,
    )

    out_base = Path(args.output)
    traits = get_traits(args.pheno, sep=pheno_sep) if args.all_targets else [args.target]
    rows = []

    for trait in traits:
        data = load_data(
            geno_path=geno_path,
            pheno_path=args.pheno,
            target=trait,
            geno_sep=geno_sep,
            pheno_sep=pheno_sep,
            transcript_paths=tx_paths,
            transcript_sep=tx_sep,
        )
        out_dir = out_base / data.target_name if args.all_targets else out_base
        res = run_cv(
            data,
            cfg,
            grn_edges=args.grn_edges,
            split_file=args.split_file,
            output_dir=out_dir,
            verbose=True,
        )
        valid_r = [r.test_pearson for r in res if math.isfinite(r.test_pearson)]
        mean_r = float(np.mean(valid_r)) if valid_r else float("nan")
        std_r = float(np.std(valid_r, ddof=1)) if len(valid_r) > 1 else 0.0
        mean_rmse = float(np.mean([r.test_rmse for r in res]))
        print(
            f"Summary [{data.target_name} | {res[0].scenario}]: "
            f"Pearson r = {mean_r:.4f} +/- {std_r:.4f}, RMSE = {mean_rmse:.4f} "
            f"({len(res)} folds, params={res[0].num_params:,}) -> {out_dir}"
        )
        rows.append(
            {
                "target": data.target_name,
                "scenario": res[0].scenario,
                "num_folds": len(res),
                "trainable_parameters": res[0].num_params,
                "mean_test_pearson": mean_r,
                "std_test_pearson": std_r,
                "mean_test_rmse": mean_rmse,
            }
        )

    if args.all_targets and rows:
        out_base.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_base / "summary_metrics.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
