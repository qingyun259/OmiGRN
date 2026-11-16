import argparse
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from omigrn import has_dgl
import predict as predict_cli
import train as train_cli


def _make_synthetic(root, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = 48
    n_snps = 96
    n_genes = 64

    sids = [f"L{i:03d}" for i in range(1, n_samples + 1)]
    snps = [f"SNP{i:03d}" for i in range(1, n_snps + 1)]
    genes = [f"GENE{i:03d}" for i in range(1, n_genes + 1)]

    geno = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps), p=[0.45, 0.10, 0.45])
    geno_na = geno.astype(object)
    for r, c in [(0, 3), (5, 11), (12, 27), (29, 44), (41, 80)]:
        geno_na[r, c] = "NA"

    tf_lat = rng.normal(0.0, 1.0, size=(n_samples, 6))
    expr = rng.normal(0.0, 0.6, size=(n_samples, n_genes))
    for i in range(6):
        expr[:, i] += tf_lat[:, i]
        for off in range(4):
            expr[:, 6 + i * 4 + off] += 0.75 * tf_lat[:, i]

    trait = (
        65.0
        + 1.8 * geno[:, 0]
        - 1.4 * geno[:, 5]
        + 1.2 * geno[:, 12]
        + 2.1 * expr[:, 0]
        - 1.6 * expr[:, 7]
        + rng.normal(0.0, 0.8, size=n_samples)
    )

    edges = []
    for i in range(6):
        for off in range(4):
            edges.append((genes[i], genes[6 + i * 4 + off], 1.0))

    geno_path = root / "geno.txt"
    trans_path = root / "trans.txt"
    pheno_path = root / "pheno.txt"
    grn_path = root / "grn.csv"

    geno_df = pd.DataFrame(geno_na, columns=snps)
    geno_df.insert(0, "ID", sids)
    geno_df.to_csv(geno_path, sep="\t", index=False)

    trans_df = pd.DataFrame(np.round(expr, 5), columns=genes)
    trans_df.insert(0, "ID", sids)
    trans_df.to_csv(trans_path, sep="\t", index=False)

    pd.DataFrame({"ID": sids, "DTS": np.round(trait, 4)}).to_csv(pheno_path, sep="\t", index=False)
    pd.DataFrame(edges, columns=["Source", "Target", "Importance"]).to_csv(grn_path, index=False)
    return {"geno": geno_path, "trans": trans_path, "pheno": pheno_path, "grn": grn_path}


def get_parser():
    p = argparse.ArgumentParser(description="Run a quick OmiGRN demo on generated data")
    p.add_argument("--device", default="cpu", help="Device (e.g. cpu or cuda:0)")
    return p


def main(argv=None):
    args = get_parser().parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="omigrn_demo_") as tmp:
        root = Path(tmp)
        paths = _make_synthetic(root, seed=42)

        s1_out = root / "s1_run"
        train_cli.main(
            [
                "--geno", str(paths["geno"]),
                "--pheno", str(paths["pheno"]),
                "--target", "DTS",
                "--folds", "3",
                "--epochs", "6",
                "--patience", "4",
                "--batch-size", "16",
                "--device", args.device,
                "--output", str(s1_out),
            ]
        )
        s1_csv = root / "s1_preds.csv"
        predict_cli.main(
            [
                "--weights", str(s1_out / "fold_1" / "model.pt"), str(s1_out / "fold_2" / "model.pt"),
                "--geno", str(paths["geno"]),
                "--output", str(s1_csv),
                "--device", args.device,
            ]
        )
        s1_df = pd.read_csv(s1_csv)
        if len(s1_df) != 48 or not np.isfinite(s1_df["prediction"]).all():
            raise RuntimeError("Invalid S1 prediction output.")

        if has_dgl():
            s2_out = root / "s2_run"
            train_cli.main(
                [
                    "--transcripts", str(paths["trans"]),
                    "--grn-edges", str(paths["grn"]),
                    "--pheno", str(paths["pheno"]),
                    "--target", "DTS",
                    "--folds", "3",
                    "--epochs", "6",
                    "--patience", "4",
                    "--batch-size", "16",
                    "--device", args.device,
                    "--output", str(s2_out),
                ]
            )

            s3_out = root / "s3_run"
            train_cli.main(
                [
                    "--geno", str(paths["geno"]),
                    "--transcripts", str(paths["trans"]),
                    "--grn-edges", str(paths["grn"]),
                    "--pheno", str(paths["pheno"]),
                    "--target", "DTS",
                    "--folds", "3",
                    "--epochs", "6",
                    "--patience", "4",
                    "--batch-size", "16",
                    "--device", args.device,
                    "--output", str(s3_out),
                ]
            )
            s3_csv = root / "s3_preds.csv"
            predict_cli.main(
                [
                    "--weights", str(s3_out),
                    "--geno", str(paths["geno"]),
                    "--transcripts", str(paths["trans"]),
                    "--output", str(s3_csv),
                    "--device", args.device,
                ]
            )
            s3_df = pd.read_csv(s3_csv)
            if len(s3_df) != 48 or not np.isfinite(s3_df["prediction"]).all():
                raise RuntimeError("Invalid S3 prediction output.")

        print("Demo finished.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
