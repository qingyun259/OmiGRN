"""
Train OmiGRN with k-fold cross-validation.

Three input scenarios (matching the paper):
    S1  genome only          --geno
    S2  transcriptome only   --transcripts --grn-edges
    S3  multi-omics fusion    --geno --transcripts --grn-edges

Examples
--------
# S1 — genome only
python train.py --geno data/geno.txt --pheno data/pheno.txt --target DTS \
    --folds 10 --epochs 300 --patience 100 --device cuda:0

# S3 — multi-omics with the GRN prior on the transcriptome branch
python train.py --geno data/geno.txt --pheno data/pheno.txt \
    --transcripts data/transcriptome.txt --grn-edges data/network_all.csv \
    --target DTS --folds 10 --epochs 300 --device cuda:0

# Train every phenotype column in the file at once
python train.py --geno data/geno.txt --pheno data/pheno.txt --train-all-targets \
    --folds 10 --epochs 300 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from omigrn import data as data_utils
from omigrn import trainer


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OmiGRN on genotype and/or transcriptome data with k-fold CV.",
    )
    parser.add_argument("--geno", help="Genotype txt file (first column = sample id).")
    parser.add_argument("--pheno", required=True, help="Phenotype txt file (first column = sample id).")
    parser.add_argument("--target", help="Phenotype column to predict.")
    parser.add_argument(
        "--transcripts", nargs="*", default=[],
        help="Optional transcriptome txt file(s) (first column = sample id).",
    )
    parser.add_argument(
        "--mode", choices=["auto", "geno", "transcript", "both"], default="auto",
        help="Which modalities to use. 'auto' uses whatever inputs you provide.",
    )
    parser.add_argument("--geno-sep", default="\t", help="Separator in the genotype file.")
    parser.add_argument("--pheno-sep", default="\t", help="Separator in the phenotype file.")
    parser.add_argument("--transcript-sep", default="\t", help="Separator in the transcriptome file(s).")

    # Training schedule
    parser.add_argument("--output", default="outputs/omigrn", help="Directory for run artifacts.")
    parser.add_argument("--epochs", type=int, default=300, help="Max epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=40, help="Early-stopping patience.")
    parser.add_argument(
        "--early-stop-metric", choices=["corr", "mse"], default="corr",
        help="Early-stopping criterion (corr maximises Pearson r; mse minimises val MSE).",
    )
    parser.add_argument("--folds", type=int, default=10, help="Number of CV folds.")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient-clip norm (0 disables).")
    parser.add_argument("--device", default="cpu", help="Device, e.g. cpu or cuda:0.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Model
    parser.add_argument("--embed-dim", type=int, default=256, help="Modality embedding dimension.")
    parser.add_argument("--mlp-hidden", type=int, default=256, help="Hidden size of the regression head.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")

    # GRN-MPNN (transcriptomic branch)
    parser.add_argument(
        "--grn-edges",
        help="GRN edge CSV (columns: Source, Target, Importance) enabling GRN-MPNN on transcripts.",
    )
    parser.add_argument("--grn-hidden-dim", type=int, default=None, help="GRN hidden dim (defaults to embed-dim).")
    parser.add_argument("--grn-layers", type=int, default=2, help="Number of GRN-MPNN layers.")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", help="Checkpoint directory (default: <output>/.checkpoints).")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save a checkpoint every N epochs.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints if present.")

    parser.add_argument(
        "--train-all-targets", action="store_true",
        help="Train a separate model for every phenotype column (excluding the sample id).",
    )

    args = parser.parse_args()

    mode = args.mode.lower()
    if mode == "auto":
        use_geno = bool(args.geno)
        use_transcripts = bool(args.transcripts)
    else:
        use_geno = mode in ("geno", "both")
        use_transcripts = mode in ("transcript", "both")

    if use_geno and not args.geno:
        parser.error("Mode requires genotype data; please provide --geno.")
    if use_transcripts and not args.transcripts:
        parser.error("Mode requires transcript data; please provide --transcripts.")
    if not use_geno and not use_transcripts:
        parser.error("No modalities selected. Provide --geno and/or --transcripts.")

    if not use_geno and args.geno:
        print("[Mode] Ignoring --geno because mode disables genotype.")
        args.geno = None
    if not use_transcripts and args.transcripts:
        print("[Mode] Ignoring --transcripts because mode disables transcripts.")
        args.transcripts = []

    args.use_geno = use_geno
    args.use_transcripts = use_transcripts

    if not args.train_all_targets and not args.target:
        parser.error("You must provide --target or enable --train-all-targets.")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be a positive integer.")
    return args


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_names(path: str, sep: str) -> List[str]:
    frame = pd.read_csv(path, sep=sep, nrows=0)
    columns = list(frame.columns)
    return columns[1:] if columns else []


def discover_targets(pheno_path: str, sep: str) -> List[str]:
    pheno_df = pd.read_csv(pheno_path, sep=sep)
    if pheno_df.shape[1] <= 1:
        raise ValueError("Phenotype file must contain at least one phenotype column besides the sample id.")
    id_column = pheno_df.columns[0]
    return [col for col in pheno_df.columns if col != id_column]


def sanitise_target_name(target: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in target.strip())
    return cleaned.strip("_") or "target"


def unique_output_subdir(name: str, used_names: set[str]) -> str:
    base = sanitise_target_name(name)
    candidate = base
    index = 1
    while candidate in used_names:
        candidate = f"{base}_{index}"
        index += 1
    used_names.add(candidate)
    return candidate


def save_run_artifacts(
    output_dir: Path,
    config: trainer.TrainingConfig,
    args: argparse.Namespace,
    bundle: data_utils.DatasetBundle,
    fold_results: List[trainer.FoldResult],
    target_name: str,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        payload = vars(config).copy()
        payload.pop("grn_gene_names", None)
        payload.update(
            {
                "geno": args.geno,
                "pheno": args.pheno,
                "target": target_name,
                "transcripts": args.transcripts,
            }
        )
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    with (output_dir / "kept_sample_ids.txt").open("w", encoding="utf-8") as handle:
        handle.write("sample_id\n")
        for sample_id in bundle.sample_ids:
            handle.write(f"{sample_id}\n")

    with (output_dir / "removed_samples.txt").open("w", encoding="utf-8") as handle:
        for reason, ids in bundle.removed_samples.items():
            handle.write(f"{reason}: {len(ids)} samples\n")
            for sample_id in ids:
                handle.write(f"{sample_id}\n")
            handle.write("\n")

    metrics_records = [
        {
            "target": target_name,
            "fold": result.fold_index,
            "train_mse": result.train_mse,
            "train_pearson": result.train_pearson,
            "val_mse": result.val_mse,
            "val_pearson": result.val_pearson,
        }
        for result in fold_results
    ]
    metrics_frame = pd.DataFrame(metrics_records)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)

    for result in fold_results:
        fold_dir = output_dir / f"fold_{result.fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        if result.model_state:
            torch.save(result.model_state, fold_dir / "model.pt")
        if result.predictions is not None:
            pd.DataFrame(
                {
                    "sample_id": result.sample_ids,
                    "true_value": result.targets,
                    "predicted_value": result.predictions,
                }
            ).to_csv(fold_dir / "predictions.csv", index=False)

    summary = metrics_frame.mean(numeric_only=True)
    print("-" * 70)
    print(f"Cross-validation summary for target '{target_name}':")
    print(summary.to_string())
    return metrics_frame


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.train_all_targets:
        target_list = discover_targets(args.pheno, args.pheno_sep)
        print("Discovered phenotype columns:")
        for col in target_list:
            print(f"  - {col}")
    else:
        target_list = [args.target]

    config = trainer.TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        early_stop_metric=args.early_stop_metric,
        gradient_clip=args.gradient_clip,
        folds=args.folds,
        device=args.device,
        embed_dim=args.embed_dim,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
        grn_edges_path=args.grn_edges,
        grn_hidden_dim=args.grn_hidden_dim,
        grn_layers=args.grn_layers,
    )

    transcript_paths = args.transcripts if args.use_transcripts else []

    # ---- wire up the GRN gene names for the transcriptomic branch ----------- #
    if args.grn_edges and args.use_transcripts:
        grn_gene_names = {
            f"transcript_{idx}": load_feature_names(path, args.transcript_sep)
            for idx, path in enumerate(transcript_paths, start=1)
        }
        config.grn_gene_names = grn_gene_names
        for idx, names in enumerate(grn_gene_names.values(), start=1):
            print(f"[GRN] transcript_{idx}: GRN-MPNN over {len(names)} genes.")
    elif args.grn_edges:
        print("[GRN] Warning: --grn-edges provided but no transcripts were specified; ignoring.")
        config.grn_edges_path = None
    elif args.grn_hidden_dim is not None or args.grn_layers != 2:
        print("[GRN] Warning: GRN hyperparameters provided but --grn-edges is not set; ignoring.")

    base_output = Path(args.output)
    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else base_output / ".checkpoints"
    metrics_frames: List[pd.DataFrame] = []
    used_names: set[str] = set()
    target_order: List[str] = []

    for target_name in target_list:
        print("=" * 70)
        print(f"Starting training for target: {target_name}")
        target_order.append(target_name)

        if args.train_all_targets:
            subdir_name = unique_output_subdir(target_name, used_names)
            output_dir = base_output / subdir_name
            target_checkpoint_dir = checkpoint_root / subdir_name
        else:
            subdir_name = sanitise_target_name(target_name)
            output_dir = base_output
            target_checkpoint_dir = checkpoint_root / subdir_name

        bundle = data_utils.prepare_dataset(
            geno_path=args.geno if args.use_geno else None,
            pheno_path=args.pheno,
            target=target_name,
            geno_sep=args.geno_sep,
            pheno_sep=args.pheno_sep,
            transcript_paths=transcript_paths,
            transcript_sep=args.transcript_sep,
        )

        sample_count = len(bundle.sample_ids)
        modality_summary = ", ".join(
            f"{name}: {array.shape[1]} features" for name, array in bundle.modalities.items()
        )
        print(
            f"Dataset prepared -> samples: {sample_count}; "
            f"modalities: {modality_summary if modality_summary else 'none'}"
        )

        fold_results = trainer.run_cross_validation(
            modalities=bundle.modalities,
            targets=bundle.targets,
            sample_ids=bundle.sample_ids,
            config=config,
            checkpoint=trainer.CheckpointConfig(
                directory=target_checkpoint_dir,
                resume=args.resume,
                save_every=args.checkpoint_every,
            ),
        )

        metrics_frame = save_run_artifacts(output_dir, config, args, bundle, fold_results, target_name)
        metrics_frames.append(metrics_frame)

    if args.train_all_targets and metrics_frames:
        aggregated = pd.concat(metrics_frames, ignore_index=True)
        base_output.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(base_output / "summary_metrics.csv", index=False)
        aggregated["target"] = pd.Categorical(aggregated["target"], categories=target_order, ordered=True)
        summary_by_target = aggregated.groupby("target", observed=False).mean(numeric_only=True)
        summary_by_target = summary_by_target.reindex(target_order)
        summary_by_target.to_csv(base_output / "summary_by_target.csv")
        print("=" * 70)
        print("Aggregated summary across all targets:")
        print(summary_by_target.to_string())


if __name__ == "__main__":
    main()
