"""
Training utilities for OmiGRN (k-fold cross-validation with checkpointing).
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import ModalityConfig, OmiGRN


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 30
    early_stop_metric: str = "corr"
    gradient_clip: float = 1.0
    folds: int = 5
    device: str = "cpu"
    embed_dim: int = 256
    mlp_hidden: int = 256
    dropout: float = 0.1
    grn_edges_path: str | None = None
    grn_gene_names: Dict[str, List[str]] | None = None
    grn_hidden_dim: int | None = None
    grn_layers: int = 2


@dataclass
class CheckpointConfig:
    directory: Path
    resume: bool = False
    save_every: int = 1

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        if self.save_every <= 0:
            raise ValueError("checkpoint.save_every must be a positive integer.")


@dataclass
class FoldResult:
    fold_index: int
    train_mse: float
    train_pearson: float
    val_mse: float
    val_pearson: float
    sample_ids: List[str] = field(default_factory=list)
    predictions: np.ndarray | None = None
    targets: np.ndarray | None = None
    model_state: Dict[str, torch.Tensor] | None = None


class _ModalTensorDataset(Dataset):
    """Internal dataset returning per-modality tensors and targets."""

    def __init__(
        self,
        modalities: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        sample_ids: Sequence[str],
    ):
        self.modalities = modalities
        self.targets = targets
        self.sample_ids = list(sample_ids)
        self.names = list(modalities.keys())

    def __len__(self) -> int:
        return self.targets.size(0)

    def __getitem__(self, index: int):
        item = {name: self.modalities[name][index] for name in self.names}
        target = self.targets[index]
        return item, target, self.sample_ids[index]


def _to_tensor_dict(arrays: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {name: torch.from_numpy(array).float() for name, array in arrays.items()}


def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _serialize_optimizer_state(optimizer: torch.optim.Optimizer) -> Dict[str, object]:
    state_dict = optimizer.state_dict()
    cpu_state = copy.deepcopy(state_dict)
    for value in cpu_state["state"].values():
        for key, tensor in value.items():
            if isinstance(tensor, torch.Tensor):
                value[key] = tensor.detach().cpu().clone()
    return cpu_state


def _load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state_dict: Dict[str, object],
    device: torch.device,
) -> None:
    optimizer.load_state_dict(state_dict)
    for param_state in optimizer.state.values():
        for key, tensor in param_state.items():
            if isinstance(tensor, torch.Tensor):
                param_state[key] = tensor.to(device)


def _compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    mse = torch.mean((predictions - targets) ** 2).item()
    pred_centered = predictions - predictions.mean()
    target_centered = targets - targets.mean()
    numerator = torch.sum(pred_centered * target_centered)
    denominator = torch.sqrt(
        torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2)
    )
    pearson = (numerator / denominator).item() if denominator.item() != 0 else 0.0
    return mse, pearson


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    predictions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    sample_ids: List[str] = []

    with torch.no_grad():
        for modalities, target, ids in loader:
            modal_inputs = {name: tensor.to(device) for name, tensor in modalities.items()}
            target = target.to(device)
            output = model(modal_inputs)
            loss = loss_fn(output, target)
            total_loss += loss.item() * target.size(0)
            total_count += target.size(0)
            predictions.append(output.detach().cpu())
            targets.append(target.detach().cpu())
            sample_ids.extend(ids)

    prediction_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    mse, pearson = _compute_metrics(prediction_tensor, target_tensor)
    avg_loss = total_loss / max(total_count, 1)
    return avg_loss, pearson, prediction_tensor.numpy(), target_tensor.numpy(), sample_ids


def _finalize_fold_result(
    fold_index: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    best_state: Dict[str, torch.Tensor],
) -> FoldResult:
    model.load_state_dict(best_state)
    train_loss, train_corr, _, _, _ = _evaluate(model, train_loader, loss_fn, device)
    val_loss, val_corr, val_predictions, val_targets, val_ids = _evaluate(
        model, val_loader, loss_fn, device
    )
    return FoldResult(
        fold_index=fold_index,
        train_mse=train_loss,
        train_pearson=train_corr,
        val_mse=val_loss,
        val_pearson=val_corr,
        sample_ids=val_ids,
        predictions=val_predictions,
        targets=val_targets,
        model_state=_clone_state_dict(best_state),
    )


def _save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_state: Dict[str, torch.Tensor] | None,
    best_val_loss: float,
    best_val_corr: float,
    early_stop_metric: str,
    best_epoch: int,
    patience_counter: int,
    completed: bool = False,
    extras: Dict[str, object] | None = None,
) -> None:
    payload = {
        "completed": completed,
        "epoch": epoch,
        "model_state_dict": _clone_state_dict(model.state_dict()),
        "optimizer_state_dict": _serialize_optimizer_state(optimizer),
        "best_state": best_state,
        "best_val_loss": best_val_loss,
        "best_val_corr": best_val_corr,
        "early_stop_metric": early_stop_metric,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
    }
    if extras:
        payload.update(extras)
    torch.save(payload, path)


def _write_completion_marker(path: Path, result: FoldResult, best_epoch: int, epochs_trained: int) -> None:
    payload = {
        "completed": True,
        "fold": result.fold_index,
        "train_mse": result.train_mse,
        "train_pearson": result.train_pearson,
        "val_mse": result.val_mse,
        "val_pearson": result.val_pearson,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_cross_validation(
    modalities: Dict[str, np.ndarray],
    targets: np.ndarray,
    sample_ids: Sequence[str],
    config: TrainingConfig,
    checkpoint: CheckpointConfig | None = None,
) -> List[FoldResult]:
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and config.device.startswith("cuda"):
        print("CUDA device requested but not available. Falling back to CPU.")

    checkpoint_dir: Path | None = None
    checkpoint_save_every = 1
    if checkpoint is not None:
        checkpoint_dir = checkpoint.directory
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_save_every = checkpoint.save_every

    modality_configs = [
        ModalityConfig(name=name, feature_dim=array.shape[1])
        for name, array in modalities.items()
    ]

    kfold = KFold(n_splits=config.folds, shuffle=True, random_state=3407)
    loss_fn = nn.MSELoss()
    early_stop_metric = str(config.early_stop_metric).lower()
    if early_stop_metric not in ("corr", "mse"):
        raise ValueError("early_stop_metric must be 'corr' or 'mse'.")
    fold_results: List[FoldResult] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(targets), start=1):
        print("=" * 70)
        print(f"Starting fold {fold_idx}/{config.folds}")
        train_modalities = {
            name: modalities[name][train_idx].copy() for name in modalities
        }
        val_modalities = {
            name: modalities[name][val_idx].copy() for name in modalities
        }
        train_targets = targets[train_idx].copy()
        val_targets = targets[val_idx].copy()

        train_dataset = _ModalTensorDataset(
            _to_tensor_dict(train_modalities),
            torch.from_numpy(train_targets).float(),
            [sample_ids[i] for i in train_idx],
        )
        val_dataset = _ModalTensorDataset(
            _to_tensor_dict(val_modalities),
            torch.from_numpy(val_targets).float(),
            [sample_ids[i] for i in val_idx],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )

        fold_dir: Path | None = None
        checkpoint_path: Path | None = None
        best_model_path: Path | None = None
        completion_marker: Path | None = None
        if checkpoint_dir is not None:
            fold_dir = checkpoint_dir / f"fold_{fold_idx:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = fold_dir / "checkpoint.pt"
            best_model_path = fold_dir / "best_model.pt"
            completion_marker = fold_dir / "complete.json"

        model = OmiGRN(
            modality_configs,
            embed_dim=config.embed_dim,
            mlp_hidden=config.mlp_hidden,
            dropout=config.dropout,
            grn_edges_path=config.grn_edges_path,
            grn_gene_names=config.grn_gene_names,
            grn_hidden_dim=config.grn_hidden_dim,
            grn_layers=config.grn_layers,
        ).to(device)

        if (
            checkpoint
            and checkpoint.resume
            and completion_marker is not None
            and completion_marker.is_file()
            and best_model_path is not None
            and best_model_path.is_file()
        ):
            print(f"[Checkpoint] Fold {fold_idx} already completed. Loading cached results.")
            best_state = torch.load(best_model_path, map_location="cpu")
            if not isinstance(best_state, dict):
                raise RuntimeError(f"Best model checkpoint for fold {fold_idx} is corrupted.")
            fold_result = _finalize_fold_result(
                fold_index=fold_idx,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                best_state=best_state,
            )
            fold_results.append(fold_result)
            continue

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        best_val_loss = math.inf
        best_val_corr = -math.inf
        best_state: Dict[str, torch.Tensor] | None = None
        best_epoch = 0
        patience_counter = 0
        start_epoch = 1

        if (
            checkpoint
            and checkpoint.resume
            and checkpoint_path is not None
            and checkpoint_path.is_file()
        ):
            try:
                payload = torch.load(checkpoint_path, map_location="cpu")
            except Exception as exc:
                print(f"[Checkpoint] Failed to load checkpoint for fold {fold_idx}: {exc}")
            else:
                start_epoch = max(int(payload.get("epoch", 0)) + 1, 1)
                model_state = payload.get("model_state_dict")
                if isinstance(model_state, dict):
                    model.load_state_dict(model_state)
                optimizer_state = payload.get("optimizer_state_dict")
                if isinstance(optimizer_state, dict):
                    _load_optimizer_state(optimizer, optimizer_state, device)
                stored_best_state = payload.get("best_state")
                if isinstance(stored_best_state, dict):
                    best_state = _clone_state_dict(stored_best_state)
                if "best_val_loss" in payload:
                    best_val_loss = float(payload.get("best_val_loss", math.inf))
                if "best_val_corr" in payload:
                    best_val_corr = float(payload.get("best_val_corr", -math.inf))

                stored_metric = str(payload.get("early_stop_metric", early_stop_metric)).lower()
                if stored_metric not in ("corr", "mse"):
                    stored_metric = early_stop_metric

                if best_state is not None and (
                    not math.isfinite(best_val_loss) or not math.isfinite(best_val_corr)
                ):
                    model.load_state_dict(best_state)
                    best_val_loss, best_val_corr, _, _, _ = _evaluate(
                        model, val_loader, loss_fn, device
                    )
                    if isinstance(model_state, dict):
                        model.load_state_dict(model_state)
                patience_counter = int(payload.get("patience_counter", 0))
                best_epoch = int(payload.get("best_epoch", 0))
                if stored_metric != early_stop_metric:
                    print(
                        f"[Checkpoint] early_stop_metric changed ({stored_metric} -> {early_stop_metric}); "
                        "resetting patience counter."
                    )
                    patience_counter = 0
                print(f"[Checkpoint] Resuming fold {fold_idx} from epoch {start_epoch}.")

        epochs_trained = start_epoch - 1

        if start_epoch > config.epochs:
            print(f"[Checkpoint] Fold {fold_idx} already trained for {config.epochs} epochs.")
            if best_state is None:
                best_state = _clone_state_dict(model.state_dict())
            if best_model_path is not None and not best_model_path.is_file():
                torch.save(best_state, best_model_path)
            fold_result = _finalize_fold_result(
                fold_index=fold_idx,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                best_state=best_state,
            )
            fold_results.append(fold_result)
            best_val_loss = fold_result.val_mse
            best_val_corr = fold_result.val_pearson
            if completion_marker is not None:
                _write_completion_marker(
                    completion_marker,
                    fold_result,
                    best_epoch or epochs_trained,
                    epochs_trained,
                )
            if checkpoint_path is not None:
                _save_training_checkpoint(
                    checkpoint_path,
                    epoch=epochs_trained,
                    model=model,
                    optimizer=optimizer,
                    best_state=best_state,
                    best_val_loss=best_val_loss,
                    best_val_corr=best_val_corr,
                    early_stop_metric=early_stop_metric,
                    best_epoch=best_epoch or epochs_trained,
                    patience_counter=patience_counter,
                    completed=True,
                )
            continue

        for epoch in range(start_epoch, config.epochs + 1):
            epochs_trained = epoch
            model.train()
            running_loss = 0.0
            sample_count = 0

            for batch_modalities, batch_targets, _ in train_loader:
                modal_inputs = {name: tensor.to(device) for name, tensor in batch_modalities.items()}
                batch_targets = batch_targets.to(device)

                optimizer.zero_grad()
                outputs = model(modal_inputs)
                loss = loss_fn(outputs, batch_targets)
                loss.backward()
                if config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()

                running_loss += loss.item() * batch_targets.size(0)
                sample_count += batch_targets.size(0)

            train_loss = running_loss / max(sample_count, 1)
            val_loss, val_corr, _, _, _ = _evaluate(model, val_loader, loss_fn, device)

            print(
                f"Fold {fold_idx} | Epoch {epoch:03d} "
                f"| train_mse={train_loss:.6f} | val_mse={val_loss:.6f} | val_corr={val_corr:.4f}"
            )

            if early_stop_metric == "corr":
                improved = val_corr > best_val_corr
                metric_name = "val_corr"
            else:
                improved = val_loss < best_val_loss
                metric_name = "val_mse"

            if improved:
                best_val_loss = val_loss
                best_val_corr = val_corr
                best_state = _clone_state_dict(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
                if best_model_path is not None:
                    torch.save(best_state, best_model_path)
            else:
                patience_counter += 1

            if checkpoint_path is not None and epoch % checkpoint_save_every == 0:
                _save_training_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    best_state=best_state,
                    best_val_loss=best_val_loss,
                    best_val_corr=best_val_corr,
                    early_stop_metric=early_stop_metric,
                    best_epoch=best_epoch,
                    patience_counter=patience_counter,
                    completed=False,
                )

            if patience_counter >= config.patience:
                print(
                    f"Early stopping triggered on fold {fold_idx} "
                    f"after {config.patience} epochs without improvement in {metric_name}."
                )
                break

        if best_state is None:
            best_state = _clone_state_dict(model.state_dict())
        if best_epoch == 0:
            best_epoch = epochs_trained
        if best_model_path is not None and not best_model_path.is_file():
            torch.save(best_state, best_model_path)
        if checkpoint_path is not None:
            _save_training_checkpoint(
                checkpoint_path,
                epoch=epochs_trained,
                model=model,
                optimizer=optimizer,
                best_state=best_state,
                best_val_loss=best_val_loss,
                best_val_corr=best_val_corr,
                early_stop_metric=early_stop_metric,
                best_epoch=best_epoch,
                patience_counter=patience_counter,
                completed=True,
            )

        fold_result = _finalize_fold_result(
            fold_index=fold_idx,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            best_state=best_state,
        )
        fold_results.append(fold_result)

        if completion_marker is not None:
            _write_completion_marker(
                completion_marker,
                fold_result,
                best_epoch,
                epochs_trained,
            )

    return fold_results
