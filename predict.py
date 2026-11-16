import argparse
import glob
from pathlib import Path
import re

import numpy as np
import pandas as pd

from omigrn import apply_scalers, load_ckpt, load_inputs, predict_batch, unscale_y


def _sep(val):
    return "\t" if val == "\\t" else val


def _fold_key(path):
    m = re.search(r"fold_(\d+)", str(path))
    return (int(m.group(1)) if m else 10**9, str(path))


def _find_weights(raw_paths):
    paths = []
    for item in raw_paths:
        s = str(item)
        p = Path(s)
        if p.is_dir():
            cands = sorted(p.glob("fold_*/model.pt"), key=_fold_key)
            if not cands and (p / "model.pt").is_file():
                cands = [p / "model.pt"]
            if not cands:
                raise FileNotFoundError(f"No fold_*/model.pt checkpoints found in '{p}'.")
            paths.extend(cands)
        elif p.is_file():
            paths.append(p)
        elif any(ch in s for ch in "*?["):
            cands = sorted((Path(m) for m in glob.glob(s) if Path(m).is_file()), key=_fold_key)
            if not cands:
                raise FileNotFoundError(f"No checkpoints matched pattern '{s}'.")
            paths.extend(cands)
        else:
            raise FileNotFoundError(f"Checkpoint path does not exist: '{p}'.")
    if not paths:
        raise ValueError("Provide at least one checkpoint via --weights.")
    return paths


def get_parser():
    p = argparse.ArgumentParser(description="Single-model or fold-ensemble OmiGRN inference")
    p.add_argument("--weights", nargs="+", required=True, help="Checkpoint model.pt path(s), glob pattern, or run directory")
    p.add_argument("--geno", default=None, help="Genotype dosage matrix path")
    p.add_argument("--transcripts", nargs="+", default=None, help="Gene expression matrix path(s)")
    p.add_argument("--grn-edges", default=None, help="Optional GRN edge CSV override")
    p.add_argument("--output", required=True, help="Output predictions CSV path")
    p.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    p.add_argument("--device", default="cpu", help="Device (e.g. cpu or cuda:0)")
    p.add_argument("--geno-sep", default="\t", help="Delimiter for --geno")
    p.add_argument("--transcript-sep", default="\t", help="Delimiter for --transcripts")
    return p


def main(argv=None):
    args = get_parser().parse_args(argv)
    ckpts = _find_weights(args.weights)
    geno_sep = _sep(args.geno_sep)
    tx_sep = _sep(args.transcript_sep)

    ref_ids = None
    fold_preds = []

    for ckpt_path in ckpts:
        model, ckpt = load_ckpt(ckpt_path, device=args.device, grn_edges=args.grn_edges)
        exp_feats = ckpt.get("feature_names", {})
        sids, _, raw_mods = load_inputs(
            geno_path=args.geno if "genotype" in exp_feats else None,
            transcript_paths=args.transcripts if any(k.startswith("transcript") for k in exp_feats) else None,
            geno_sep=geno_sep,
            transcript_sep=tx_sep,
            expected_features=exp_feats,
        )
        if ref_ids is None:
            ref_ids = sids
        elif sids != ref_ids:
            raise ValueError(f"Sample ID alignment differed across checkpoints at '{ckpt_path}'.")

        scaled = apply_scalers(raw_mods, ckpt["preprocessing"])
        pred_std = predict_batch(model, scaled, batch_size=args.batch_size, device=args.device)
        fold_preds.append(unscale_y(pred_std, ckpt["target_scaler"]))

    ens_pred = np.mean(np.stack(fold_preds, axis=0), axis=0)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": ref_ids, "prediction": ens_pred}).to_csv(out_path, index=False)
    if len(ckpts) == 1:
        print(f"Saved {len(ref_ids)} predictions to {out_path}")
    else:
        print(f"Saved {len(ref_ids)} ensemble predictions ({len(ckpts)} checkpoints) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
