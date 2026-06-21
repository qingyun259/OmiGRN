

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from omigrn.data import _clean_matrix, _load_feature_table
from omigrn.model import ModalityConfig, OmiGRN


def load_feature_names(path: str, sep: str) -> List[str]:
    frame = pd.read_csv(path, sep=sep, nrows=0)
    columns = list(frame.columns)
    return columns[1:] if columns else []


def load_modalities(
    geno: str | None,
    transcripts: List[str],
    *,
    geno_sep: str,
    transcript_sep: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load genotype / transcript features and inner-join them on sample id."""
    frames: List[Tuple[pd.DataFrame, str, str]] = []
    if geno:
        frames.append((_load_feature_table(geno, geno_sep, prefix="geno_"), "genotype", "geno_"))
    for index, path in enumerate(transcripts, start=1):
        prefix = f"tx{index}_"
        frames.append((_load_feature_table(path, transcript_sep, prefix=prefix), f"transcript_{index}", prefix))

    if not frames:
        raise ValueError("Provide --geno and/or --transcripts.")

    working = frames[0][0]
    for frame, _, _ in frames[1:]:
        working = working.merge(frame, on="sample_id", how="inner")

    if working.empty:
        raise ValueError("No samples shared across the provided feature tables.")

    sample_ids = working["sample_id"].to_numpy()
    modalities: Dict[str, np.ndarray] = {}
    for _, name, prefix in frames:
        cols = [c for c in working.columns if c.startswith(prefix)]
        modalities[name] = _clean_matrix(working[cols].to_numpy(dtype=np.float32))
    return sample_ids, modalities


_LEGACY_DROP_PREFIXES = ("transformer.", "reg_head.")
_LEGACY_DROP_EXACT = ("cls_token", "position_embeddings")


def _normalize_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map legacy keys onto the current layout: rename ``concat_head.*`` -> ``head.*``
    and drop the unused transformer / CLS-token parameters kept by older checkpoints."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key in _LEGACY_DROP_EXACT or key.startswith(_LEGACY_DROP_PREFIXES):
            continue
        if key.startswith("concat_head."):
            key = "head." + key[len("concat_head."):]
        out[key] = value
    return out


def load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        obj = obj["model_state_dict"]
    if not isinstance(obj, dict) or not all(isinstance(v, torch.Tensor) for v in obj.values()):
        raise RuntimeError(f"{path} does not contain an OmiGRN state_dict.")
    return _normalize_state_dict(obj)


def infer_architecture(state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """Recover the model hyper-parameters from the weight tensor shapes."""
    keys = list(state.keys())
    modal_names = sorted({k.split(".")[1] for k in keys if k.startswith("encoders.")})
    if not modal_names:
        raise RuntimeError("State dict has no 'encoders.*' parameters; not an OmiGRN checkpoint.")

    if "head.0.weight" not in state:
        raise RuntimeError(
            "Could not find the regression head ('head.0.weight') in the weights. These weights may come "
            "from a transformer-fusion model, which this concat-fusion OmiGRN does not support."
        )
    concat_dim = int(state["head.0.weight"].shape[0])
    mlp_hidden = int(state["head.1.weight"].shape[0])
    embed_dim = concat_dim // len(modal_names)

    grn: Dict[str, Dict[str, int]] = {}
    for name in modal_names:
        gate_key = f"encoders.{name}.grn.fuse_gate.0.weight"
        if gate_key in state:
            hidden_dim = int(state[gate_key].shape[0])
            layer_ids = {k.split(".")[4] for k in keys if k.startswith(f"encoders.{name}.grn.layers.")}
            grn[name] = {"hidden_dim": hidden_dim, "num_layers": len(layer_ids) or 2}

    return {
        "embed_dim": embed_dim,
        "mlp_hidden": mlp_hidden,
        "modal_names": modal_names,
        "grn": grn,
    }


def run_model(
    model: OmiGRN,
    modality_tensors: Dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    n = next(iter(modality_tensors.values())).shape[0]
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = start + batch_size
            inputs = {name: tensor[start:end].to(device) for name, tensor in modality_tensors.items()}
            preds.append(model(inputs).detach().cpu())
    return torch.cat(preds).numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict phenotypes from OmiGRN weights and new data.")
    parser.add_argument("--weights", nargs="+", required=True, help="One or more state_dict files (averaged if several).")
    parser.add_argument("--geno", help="New genotype file to predict on.")
    parser.add_argument("--transcripts", nargs="*", default=[], help="New transcriptome file(s) to predict on.")
    parser.add_argument("--output", required=True, help="Output CSV path for predictions.")

    parser.add_argument("--grn-edges", help="GRN edge CSV (required for a GRN model).")

    parser.add_argument("--geno-sep", default="\t")
    parser.add_argument("--transcript-sep", default="\t")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device.startswith("cuda"):
        print("[Predict] CUDA requested but unavailable; using CPU.")

    weight_paths = [Path(p) for p in args.weights]
    for path in weight_paths:
        if not path.is_file():
            raise SystemExit(f"Weights file not found: {path}")

    arch = infer_architecture(load_state_dict(weight_paths[0], device))
    modal_names: List[str] = arch["modal_names"]
    grn_arch: Dict[str, Dict[str, int]] = arch["grn"]


    expect_geno = "genotype" in modal_names
    expect_n_transcripts = sum(1 for n in modal_names if n.startswith("transcript_"))
    if expect_geno and not args.geno:
        raise SystemExit("These weights include a genotype branch; please provide --geno.")
    if not expect_geno and args.geno:
        raise SystemExit("These weights have no genotype branch; do not pass --geno.")
    if len(args.transcripts) != expect_n_transcripts:
        raise SystemExit(f"These weights expect {expect_n_transcripts} transcript file(s); you passed {len(args.transcripts)}.")

    sample_ids, modalities = load_modalities(
        args.geno, args.transcripts, geno_sep=args.geno_sep, transcript_sep=args.transcript_sep
    )
    summary = ", ".join(f"{name}: {arr.shape[1]} features" for name, arr in modalities.items())
    print(f"[Predict] {len(sample_ids)} samples | {summary}")

    grn_gene_names = None
    grn_hidden_dim = None
    grn_layers = 2
    if grn_arch:
        if not args.grn_edges:
            raise SystemExit("These weights include a GRN branch; please provide --grn-edges.")
        if not Path(args.grn_edges).is_file():
            raise SystemExit(f"GRN edge file not found: {args.grn_edges}")
        grn_gene_names = {}
        for name in grn_arch:
            idx = int(name.split("_")[1]) - 1
            grn_gene_names[name] = load_feature_names(args.transcripts[idx], args.transcript_sep)
        first = next(iter(grn_arch.values()))
        grn_hidden_dim = first["hidden_dim"]
        grn_layers = first["num_layers"]

    modality_configs = [ModalityConfig(name=name, feature_dim=arr.shape[1]) for name, arr in modalities.items()]
    model = OmiGRN(
        modality_configs,
        embed_dim=int(arch["embed_dim"]),
        mlp_hidden=int(arch["mlp_hidden"]),
        dropout=0.0,
        grn_edges_path=args.grn_edges if grn_gene_names else None,
        grn_gene_names=grn_gene_names,
        grn_hidden_dim=grn_hidden_dim,
        grn_layers=grn_layers,
    ).to(device)

    modality_tensors = {name: torch.from_numpy(arr).float() for name, arr in modalities.items()}

    fold_preds: List[np.ndarray] = []
    for path in weight_paths:
        state = load_state_dict(path, device)
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise SystemExit(
                f"Failed to load {path}: architecture mismatch (most likely the feature columns differ "
                f"from training).\n{exc}"
            )
        fold_preds.append(run_model(model, modality_tensors, device, args.batch_size))

    prediction = np.stack(fold_preds, axis=0).mean(axis=0)
    out = pd.DataFrame({"sample_id": sample_ids, "prediction": prediction})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"[Predict] Wrote {len(out)} predictions from {len(weight_paths)} weight file(s) -> {output_path}")


if __name__ == "__main__":
    main()
