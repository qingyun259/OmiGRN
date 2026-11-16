import json
import math
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .data import (
    OmicsData,
    apply_scalers,
    fit_scalers,
    fit_y_scaler,
    make_splits,
    parse_splits,
    read_splits,
    scale_y,
    unscale_y,
)
from .grn import read_edges
from .model import Modality, OmiGRN

METRIC_COLS = (
    "target",
    "scenario",
    "fold",
    "n_train",
    "n_inner_train",
    "n_val",
    "n_test",
    "best_epoch",
    "lr",
    "batch_size",
    "weight_decay",
    "dropout",
    "grad_clip",
    "trainable_parameters",
    "train_mse",
    "train_pearson",
    "val_mse",
    "val_pearson",
    "test_mse",
    "test_rmse",
    "test_pearson",
)

_CFG_FIELDS = (
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "grad_clip",
    "patience",
    "embed_dim",
    "mlp_hidden",
    "grn_dim",
    "grn_layers",
    "dropout",
    "folds",
    "val_ratio",
    "seed",
    "refit_outer",
    "stop_metric",
    "tune_trials",
    "tuner",
    "tune_arch",
    "device",
    "threads",
)


class TrainConfig:
    def __init__(
        self,
        epochs=200,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        patience=40,
        embed_dim=128,
        mlp_hidden=128,
        grn_dim=128,
        grn_layers=2,
        dropout=0.1,
        folds=10,
        val_ratio=0.1,
        seed=42,
        refit_outer=True,
        stop_metric="mse",
        tune_trials=0,
        tuner="auto",
        tune_arch=False,
        device="cpu",
        threads=1,
        grn_edges=None,
    ):
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.patience = int(patience)
        self.embed_dim = int(embed_dim)
        self.mlp_hidden = int(mlp_hidden)
        self.grn_dim = int(grn_dim)
        self.grn_layers = int(grn_layers)
        self.dropout = float(dropout)
        self.folds = int(folds)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.refit_outer = bool(refit_outer)
        self.stop_metric = str(stop_metric)
        self.tune_trials = int(tune_trials)
        self.tuner = str(tuner)
        self.tune_arch = bool(tune_arch)
        self.device = str(device)
        self.threads = int(threads)
        self.grn_edges = grn_edges

    def to_dict(self):
        return {k: getattr(self, k) for k in _CFG_FIELDS}

    def copy(self, **updates):
        kw = self.to_dict()
        kw["grn_edges"] = self.grn_edges
        kw.update(updates)
        return TrainConfig(**kw)


class FoldOutput:
    def __init__(
        self,
        target,
        scenario,
        fold,
        n_train,
        n_inner_train,
        n_val,
        n_test,
        best_epoch,
        num_params,
        train_mse,
        train_pearson,
        val_mse,
        val_pearson,
        test_mse,
        test_rmse,
        test_pearson,
        hparams,
        test_ids,
        test_targets,
        test_preds,
        val_ids,
        val_targets,
        val_preds,
        ckpt_path=None,
    ):
        self.target = str(target)
        self.scenario = str(scenario)
        self.fold = int(fold)
        self.n_train = int(n_train)
        self.n_inner_train = int(n_inner_train)
        self.n_val = int(n_val)
        self.n_test = int(n_test)
        self.best_epoch = int(best_epoch)
        self.num_params = int(num_params)
        self.train_mse = float(train_mse)
        self.train_pearson = float(train_pearson)
        self.val_mse = float(val_mse)
        self.val_pearson = float(val_pearson)
        self.test_mse = float(test_mse)
        self.test_rmse = float(test_rmse)
        self.test_pearson = float(test_pearson)
        self.hparams = dict(hparams)
        self.test_ids = list(test_ids)
        self.test_targets = np.asarray(test_targets, dtype=np.float64)
        self.test_preds = np.asarray(test_preds, dtype=np.float64)
        self.val_ids = list(val_ids)
        self.val_targets = np.asarray(val_targets, dtype=np.float64)
        self.val_preds = np.asarray(val_preds, dtype=np.float64)
        self.ckpt_path = ckpt_path


def get_scenario(modality_names):
    has_g = any(not n.startswith("transcript") for n in modality_names)
    has_t = any(n.startswith("transcript") for n in modality_names)
    if has_g and has_t:
        return "S3"
    if has_g:
        return "S1"
    if has_t:
        return "S2"
    raise ValueError(f"Cannot determine scenario from modalities: {list(modality_names)}")


def set_seed(seed, device=None):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_metrics(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape or yt.size == 0:
        raise ValueError(f"Mismatched or empty metric arrays: {yt.shape} vs {yp.shape}.")
    diff = yt - yp
    mse = float(np.mean(diff * diff))
    rmse = float(math.sqrt(max(0.0, mse)))
    if yt.size < 2:
        r = float("nan")
    else:
        dy = yt - yt.mean()
        dp = yp - yp.mean()
        denom = float(math.sqrt(float(np.dot(dy, dy)) * float(np.dot(dp, dp))))
        r = float(np.dot(dy, dp) / denom) if denom > 1e-15 else float("nan")
    return {"mse": mse, "rmse": rmse, "pearson": r}


class _TensorSet(Dataset):
    def __init__(self, modalities, targets):
        self.modalities = {
            k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
            for k, v in modalities.items()
        }
        self.targets = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))

    def __len__(self):
        return int(self.targets.numel())

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.modalities.items()}, self.targets[idx]


class BatchSampler(Sampler):
    def __init__(self, num_samples, batch_size, shuffle=True, seed=42):
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1.")
        if batch_size < 2 and num_samples >= 2:
            raise ValueError("batch_size must be >= 2.")
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

    def __iter__(self):
        if self.shuffle:
            gen = torch.Generator()
            gen.manual_seed(self.seed + self._epoch)
            self._epoch += 1
            order = torch.randperm(self.num_samples, generator=gen).tolist()
        else:
            order = list(range(self.num_samples))

        n = self.num_samples
        bs = min(self.batch_size, n)
        if n >= 2 and bs >= 2 and (n % bs == 1):
            cut = n - (bs + 1)
            for s in range(0, cut, bs):
                yield order[s : s + bs]
            tail = order[cut:]
            mid = len(tail) // 2
            yield tail[:mid]
            yield tail[mid:]
        else:
            for s in range(0, n, bs):
                yield order[s : s + bs]

    def __len__(self):
        n = self.num_samples
        bs = min(self.batch_size, n)
        if n >= 2 and bs >= 2 and (n % bs == 1):
            return (n - (bs + 1)) // bs + 2
        return (n + bs - 1) // bs


def _check_config(cfg):
    metric = str(cfg.stop_metric).strip().lower()
    if metric == "corr":
        metric = "pearson"
    if metric not in ("mse", "pearson"):
        raise ValueError(f"stop_metric must be 'mse' or 'pearson', got '{cfg.stop_metric}'.")
    tuner = str(cfg.tuner).strip().lower()
    if tuner not in ("auto", "optuna", "random"):
        raise ValueError(f"tuner must be auto/optuna/random, got '{cfg.tuner}'.")
    if cfg.epochs < 1 or cfg.patience < 1:
        raise ValueError("epochs and patience must be >= 1.")
    if cfg.batch_size < 2:
        raise ValueError("batch_size must be >= 2.")
    if cfg.embed_dim < 2 or cfg.mlp_hidden < 2 or cfg.grn_dim < 2:
        raise ValueError("embed_dim, mlp_hidden, and grn_dim must be >= 2.")
    if cfg.grn_layers < 1:
        raise ValueError("grn_layers must be >= 1.")
    if cfg.lr <= 0 or not math.isfinite(cfg.lr):
        raise ValueError("lr must be positive and finite.")
    if cfg.weight_decay < 0 or not math.isfinite(cfg.weight_decay):
        raise ValueError("weight_decay must be >= 0 and finite.")
    if cfg.grad_clip < 0 or not math.isfinite(cfg.grad_clip):
        raise ValueError("grad_clip must be >= 0 and finite.")
    if not (0.0 <= cfg.dropout < 1.0):
        raise ValueError("dropout must be in [0, 1).")
    if not (0.0 < cfg.val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1).")
    if cfg.folds < 2:
        raise ValueError("folds must be >= 2.")
    return cfg.copy(stop_metric=metric, tuner=tuner)


def _build_model(dims, feature_names, grn_edges, cfg, device):
    mods = [Modality(name, int(dim)) for name, dim in dims.items()]
    genes = {k: list(v) for k, v in feature_names.items() if k.startswith("transcript")}
    return OmiGRN(
        modalities=mods,
        embed_dim=cfg.embed_dim,
        mlp_hidden=cfg.mlp_hidden,
        dropout=cfg.dropout,
        grn_edges=grn_edges,
        gene_names=genes if genes else None,
        grn_dim=cfg.grn_dim,
        grn_layers=cfg.grn_layers,
    ).to(device)


def predict_batch(model, modalities, batch_size=256, device="cpu"):
    dev = torch.device(device)
    first = next(iter(modalities.values()))
    n = int(first.shape[0])
    if n == 0:
        return np.empty(0, dtype=np.float32)

    model.eval()
    chunks = []
    step = max(1, int(batch_size))
    with torch.no_grad():
        for s in range(0, n, step):
            e = min(n, s + step)
            batch = {
                k: torch.from_numpy(np.ascontiguousarray(v[s:e], dtype=np.float32)).to(dev)
                for k, v in modalities.items()
            }
            preds = model(batch).reshape(-1)
            if not torch.isfinite(preds).all():
                raise FloatingPointError("Model returned non-finite predictions.")
            chunks.append(preds.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _fit_once(
    train_x,
    train_y_std,
    feature_names,
    grn_edges,
    cfg,
    device,
    seed,
    val_x=None,
    val_y_orig=None,
    y_scaler=None,
):
    set_seed(seed, device)
    dims = {k: arr.shape[1] for k, arr in train_x.items()}
    model = _build_model(dims, feature_names, grn_edges, cfg, device)
    n_params = model.num_params()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ds = _TensorSet(train_x, train_y_std)
    sampler = BatchSampler(len(ds), cfg.batch_size, shuffle=True, seed=seed + 10000)
    loader = DataLoader(ds, batch_sampler=sampler)

    best_epoch = 0
    best_mse = math.inf
    best_r = -math.inf
    best_state = None
    best_val_preds = None
    wait = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            preds = model({k: v.to(device) for k, v in bx.items()}).reshape(-1)
            loss = F.mse_loss(preds, by.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch} (lr={cfg.lr}).")
            loss.backward()
            if cfg.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        if val_x is not None and val_y_orig is not None and y_scaler is not None:
            vp_std = predict_batch(model, val_x, batch_size=cfg.batch_size, device=device)
            vp_orig = unscale_y(vp_std, y_scaler)
            vm = calc_metrics(val_y_orig, vp_orig)
            cur_mse = vm["mse"]
            cur_r = vm["pearson"]

            if cfg.stop_metric == "pearson":
                if math.isfinite(cur_r):
                    improved = best_state is None or (not math.isfinite(best_r)) or cur_r > best_r
                else:
                    improved = best_state is None or cur_mse < best_mse
            else:
                improved = best_state is None or cur_mse < best_mse

            if improved:
                best_epoch = epoch
                best_mse = cur_mse
                best_r = cur_r
                best_state = _cpu_state(model)
                best_val_preds = vp_orig
                wait = 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    break
        else:
            best_epoch = epoch
            best_state = _cpu_state(model)

    if best_state is None:
        best_state = _cpu_state(model)
        best_epoch = max(1, best_epoch)

    return {
        "best_epoch": int(best_epoch),
        "val_mse": float(best_mse),
        "val_pearson": float(best_r),
        "state": best_state,
        "val_preds": best_val_preds,
        "num_params": int(n_params),
        "matched_edges": model.matched_edges(),
    }


def _loss_score(fit_res, metric):
    if metric == "pearson":
        r = float(fit_res["val_pearson"])
        if math.isfinite(r):
            return -r
    return float(fit_res["val_mse"])


def _pick_tuner(name):
    if name != "auto":
        return name
    try:
        __import__("optuna")
        return "optuna"
    except ImportError:
        return "random"


def _random_cfg(rng, base):
    lr = float(10.0 ** rng.uniform(math.log10(5e-5), math.log10(1e-2)))
    bs = int(rng.choice([8, 16, 32, 64]))
    wd = float(10.0 ** rng.uniform(math.log10(1e-5), math.log10(1e-3)))
    pat = int(rng.choice([20, 40, 60, 80, 120, 160]))
    drop = float(rng.uniform(0.05, 0.50))
    gclip = float(rng.uniform(0.0, 5.0))
    edim = int(rng.choice([128, 256, 384, 512])) if base.tune_arch else base.embed_dim
    mhid = int(rng.choice([128, 256, 512, 768, 1024])) if base.tune_arch else base.mlp_hidden
    return base.copy(
        lr=lr,
        batch_size=bs,
        weight_decay=wd,
        patience=pat,
        dropout=drop,
        grad_clip=gclip,
        embed_dim=edim,
        mlp_hidden=mhid,
    )


def _hparams_dict(cfg):
    return {
        "lr": float(cfg.lr),
        "batch_size": int(cfg.batch_size),
        "weight_decay": float(cfg.weight_decay),
        "patience": int(cfg.patience),
        "embed_dim": int(cfg.embed_dim),
        "mlp_hidden": int(cfg.mlp_hidden),
        "grn_dim": int(cfg.grn_dim),
        "grn_layers": int(cfg.grn_layers),
        "dropout": float(cfg.dropout),
        "grad_clip": float(cfg.grad_clip),
        "refit_outer": bool(cfg.refit_outer),
        "stop_metric": str(cfg.stop_metric),
    }


def _tune_inner(inner_x, inner_y_std, val_x, val_y_orig, inner_y_scaler, feature_names, grn_edges, cfg, device, fold_seed):
    history = []
    if cfg.tune_trials <= 0:
        res = _fit_once(
            inner_x,
            inner_y_std,
            feature_names,
            grn_edges,
            cfg,
            device,
            fold_seed,
            val_x=val_x,
            val_y_orig=val_y_orig,
            y_scaler=inner_y_scaler,
        )
        history.append(
            {
                "trial": 0,
                "hparams": _hparams_dict(cfg),
                "best_epoch": res["best_epoch"],
                "val_mse": res["val_mse"],
                "val_pearson": res["val_pearson"],
            }
        )
        return cfg, res, history

    n_trials = int(cfg.tune_trials)
    backend = _pick_tuner(cfg.tuner)
    best_cfg = None
    best_fit = None
    best_score = math.inf

    def run_trial(idx, cand):
        nonlocal best_cfg, best_fit, best_score
        try:
            res = _fit_once(
                inner_x,
                inner_y_std,
                feature_names,
                grn_edges,
                cand,
                device,
                fold_seed + idx * 17,
                val_x=val_x,
                val_y_orig=val_y_orig,
                y_scaler=inner_y_scaler,
            )
            score = _loss_score(res, cfg.stop_metric)
        except FloatingPointError:
            history.append(
                {
                    "trial": idx,
                    "hparams": _hparams_dict(cand),
                    "best_epoch": 0,
                    "val_mse": float("nan"),
                    "val_pearson": float("nan"),
                    "status": "diverged",
                }
            )
            return 1e12

        history.append(
            {
                "trial": idx,
                "hparams": _hparams_dict(cand),
                "best_epoch": res["best_epoch"],
                "val_mse": res["val_mse"],
                "val_pearson": res["val_pearson"],
                "status": "ok",
            }
        )
        if best_fit is None or score < best_score:
            best_score = score
            best_cfg = cand
            best_fit = res
        return score

    if backend == "optuna":
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=fold_seed))
        init_params = {
            "lr": min(max(cfg.lr, 5e-5), 1e-2),
            "batch_size": cfg.batch_size if cfg.batch_size in (8, 16, 32, 64) else 64,
            "weight_decay": min(max(cfg.weight_decay, 1e-5), 1e-3),
            "patience": cfg.patience if cfg.patience in (20, 40, 60, 80, 120, 160) else 40,
            "dropout": min(max(cfg.dropout, 0.05), 0.50),
            "grad_clip": min(max(cfg.grad_clip, 0.0), 5.0),
        }
        if cfg.tune_arch:
            init_params["embed_dim"] = cfg.embed_dim if cfg.embed_dim in (128, 256, 384, 512) else 128
            init_params["mlp_hidden"] = cfg.mlp_hidden if cfg.mlp_hidden in (128, 256, 512, 768, 1024) else 128
        study.enqueue_trial(init_params)

        def obj(trial):
            lr = trial.suggest_float("lr", 5e-5, 1e-2, log=True)
            bs = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
            wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
            pat = trial.suggest_categorical("patience", [20, 40, 60, 80, 120, 160])
            drop = trial.suggest_float("dropout", 0.05, 0.50)
            gclip = trial.suggest_float("grad_clip", 0.0, 5.0)
            edim = trial.suggest_categorical("embed_dim", [128, 256, 384, 512]) if cfg.tune_arch else cfg.embed_dim
            mhid = trial.suggest_categorical("mlp_hidden", [128, 256, 512, 768, 1024]) if cfg.tune_arch else cfg.mlp_hidden
            cand = cfg.copy(
                lr=float(lr),
                batch_size=int(bs),
                weight_decay=float(wd),
                patience=int(pat),
                dropout=float(drop),
                grad_clip=float(gclip),
                embed_dim=int(edim),
                mlp_hidden=int(mhid),
            )
            return run_trial(trial.number, cand)

        study.optimize(obj, n_trials=n_trials)
    else:
        rng = np.random.default_rng(fold_seed)
        run_trial(0, cfg)
        for idx in range(1, n_trials):
            run_trial(idx, _random_cfg(rng, cfg))

    if best_cfg is None or best_fit is None:
        raise RuntimeError(f"All {n_trials} tuning trial(s) diverged.")
    return best_cfg, best_fit, history


def save_ckpt(payload, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(p)


def _edge_records(source):
    if source is None:
        return []
    df = read_edges(source)
    return [
        [str(r.Source).strip(), str(r.Target).strip(), float(r.Importance)]
        for r in df.itertuples(index=False)
    ]


def run_cv(
    data=None,
    config=None,
    modalities=None,
    targets=None,
    sample_ids=None,
    feature_names=None,
    target_name="phenotype",
    grn_edges=None,
    split_file=None,
    splits=None,
    output_dir=None,
    modality_names=None,
    verbose=True,
):
    if data is None:
        if modalities is None or targets is None or sample_ids is None:
            raise ValueError("Provide either OmicsData or (modalities, targets, sample_ids).")
        fnames = {}
        for k, arr in modalities.items():
            if feature_names is not None and k in feature_names:
                fnames[k] = [str(c) for c in feature_names[k]]
            else:
                fnames[k] = [f"{k}_{i}" for i in range(arr.shape[1])]
        data = OmicsData(
            sample_ids=np.asarray([str(s) for s in sample_ids], dtype=str),
            modalities={k: np.asarray(v, dtype=np.float32) for k, v in modalities.items()},
            targets=np.asarray(targets, dtype=np.float32),
            feature_names=fnames,
            target_name=str(target_name),
        )
    if config is None:
        config = TrainConfig()

    cfg = _check_config(config)
    if cfg.threads >= 1:
        torch.set_num_threads(int(cfg.threads))

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    active_mods = list(modality_names) if modality_names is not None else list(data.modalities.keys())
    if not active_mods:
        raise ValueError("No active modalities selected.")
    for name in active_mods:
        if name not in data.modalities:
            raise KeyError(f"Modality '{name}' not found in OmicsData.")

    edges_src = grn_edges if grn_edges is not None else cfg.grn_edges
    scenario = get_scenario(active_mods)
    has_tx = any(n.startswith("transcript") for n in active_mods)
    if has_tx and edges_src is None:
        raise ValueError(f"Scenario {scenario} requires grn_edges.")

    grn_records = _edge_records(edges_src) if has_tx else []
    sids = [str(s) for s in data.sample_ids.tolist()]
    y_all = np.asarray(data.targets, dtype=np.float64)
    if not np.isfinite(y_all).all():
        raise ValueError("Target array contains non-finite values.")

    raw_mods = {k: data.modalities[k] for k in active_mods}
    active_cols = {k: list(data.feature_names[k]) for k in active_mods}

    if splits is not None:
        fold_splits = parse_splits(splits, sids, val_ratio=cfg.val_ratio, seed=cfg.seed)
        split_meta = {"mode": "provided_splits", "num_folds": len(fold_splits)}
    elif split_file is not None:
        fold_splits, split_meta = read_splits(split_file, sids, val_ratio=cfg.val_ratio, seed=cfg.seed)
    else:
        fold_splits = make_splits(sids, folds=cfg.folds, val_ratio=cfg.val_ratio, seed=cfg.seed)
        split_meta = {"mode": "nested_kfold", "folds": cfg.folds, "val_ratio": cfg.val_ratio, "seed": cfg.seed}

    out_root = Path(output_dir) if output_dir is not None else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    results = []
    oof_rows = []

    for sp in fold_splits:
        f_num = sp.fold
        f_seed = cfg.seed + f_num * 1000
        f_dir = out_root / f"fold_{f_num}" if out_root is not None else None
        if f_dir is not None:
            f_dir.mkdir(parents=True, exist_ok=True)
            (f_dir / "splits.json").write_text(
                json.dumps(
                    {
                        "fold": f_num,
                        "train_ids": sp.train_ids,
                        "inner_train_ids": sp.inner_ids,
                        "val_ids": sp.val_ids,
                        "test_ids": sp.test_ids,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        in_raw = {k: arr[sp.inner_idx] for k, arr in raw_mods.items()}
        va_raw = {k: arr[sp.val_idx] for k, arr in raw_mods.items()}
        in_scalers = fit_scalers(in_raw)
        in_y_scaler = fit_y_scaler(y_all[sp.inner_idx])

        in_x = apply_scalers(in_raw, in_scalers)
        va_x = apply_scalers(va_raw, in_scalers)
        in_y_std = scale_y(y_all[sp.inner_idx], in_y_scaler)
        va_y_orig = y_all[sp.val_idx]

        sel_cfg, in_fit, history = _tune_inner(
            in_x,
            in_y_std,
            va_x,
            va_y_orig,
            in_y_scaler,
            active_cols,
            grn_records if has_tx else None,
            cfg,
            device,
            f_seed,
        )
        best_epoch = max(1, int(in_fit["best_epoch"]))
        sel_hparams = _hparams_dict(sel_cfg)

        if f_dir is not None:
            (f_dir / "selection.json").write_text(
                json.dumps(
                    {
                        "fold": f_num,
                        "best_epoch": best_epoch,
                        "val_mse": in_fit["val_mse"],
                        "val_pearson": in_fit["val_pearson"],
                        "hparams": sel_hparams,
                        "trials": history,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        if sel_cfg.refit_outer:
            tr_raw = {k: arr[sp.train_idx] for k, arr in raw_mods.items()}
            fin_scalers = fit_scalers(tr_raw)
            fin_y_scaler = fit_y_scaler(y_all[sp.train_idx])
            fin_x = apply_scalers(tr_raw, fin_scalers)
            fin_y_std = scale_y(y_all[sp.train_idx], fin_y_scaler)
            fin_fit = _fit_once(
                fin_x,
                fin_y_std,
                active_cols,
                grn_records if has_tx else None,
                sel_cfg.copy(epochs=best_epoch),
                device,
                f_seed + 999,
            )
            fin_state = fin_fit["state"]
            num_params = int(fin_fit["num_params"])
            saved_edges = list(fin_fit["matched_edges"])
            eval_tr_idx = sp.train_idx
        else:
            fin_scalers = in_scalers
            fin_y_scaler = in_y_scaler
            fin_x = in_x
            fin_state = in_fit["state"]
            num_params = int(in_fit["num_params"])
            saved_edges = list(in_fit["matched_edges"])
            eval_tr_idx = sp.inner_idx

        dims = {k: arr.shape[1] for k, arr in fin_x.items()}
        fin_model = _build_model(dims, active_cols, saved_edges if has_tx else None, sel_cfg, device)
        fin_model.load_state_dict(fin_state, strict=True)
        fin_model.eval()

        ckpt = {
            "scenario": scenario,
            "target": str(data.target_name),
            "fold": int(f_num),
            "best_epoch": int(best_epoch),
            "trainable_parameters": int(num_params),
            "model_config": {
                "modalities": [{"name": m.name, "dim": int(m.dim)} for m in fin_model.modalities],
                "embed_dim": int(sel_cfg.embed_dim),
                "mlp_hidden": int(sel_cfg.mlp_hidden),
                "dropout": float(sel_cfg.dropout),
                "grn_dim": int(sel_cfg.grn_dim),
                "grn_layers": int(sel_cfg.grn_layers),
            },
            "feature_names": active_cols,
            "grn_edges": saved_edges,
            "preprocessing": fin_scalers,
            "target_scaler": fin_y_scaler,
            "split_ids": {
                "train_ids": sp.train_ids,
                "inner_train_ids": sp.inner_ids,
                "val_ids": sp.val_ids,
                "test_ids": sp.test_ids,
            },
            "hparams": sel_hparams,
            "state_dict": fin_state,
        }

        ckpt_path = None
        if f_dir is not None:
            ckpt_file = f_dir / "model.pt"
            save_ckpt(ckpt, ckpt_file)
            ckpt_path = str(ckpt_file)

        tr_pred = unscale_y(predict_batch(fin_model, fin_x, batch_size=sel_cfg.batch_size, device=device), fin_y_scaler)
        tr_metrics = calc_metrics(y_all[eval_tr_idx], tr_pred)

        te_raw = {k: arr[sp.test_idx] for k, arr in raw_mods.items()}
        te_x = apply_scalers(te_raw, fin_scalers)
        te_pred = unscale_y(predict_batch(fin_model, te_x, batch_size=sel_cfg.batch_size, device=device), fin_y_scaler)
        te_y = y_all[sp.test_idx]
        te_metrics = calc_metrics(te_y, te_pred)

        va_pred = np.asarray(in_fit["val_preds"], dtype=np.float64)
        if f_dir is not None:
            pd.DataFrame({"sample_id": sp.val_ids, "true_value": va_y_orig, "predicted_value": va_pred}).to_csv(
                f_dir / "val_predictions.csv", index=False
            )
            pd.DataFrame({"sample_id": sp.test_ids, "true_value": te_y, "predicted_value": te_pred}).to_csv(
                f_dir / "test_predictions.csv", index=False
            )

        for sid, yt, yp in zip(sp.test_ids, te_y, te_pred):
            oof_rows.append({"fold": f_num, "sample_id": sid, "true_value": float(yt), "predicted_value": float(yp)})

        res = FoldOutput(
            target=data.target_name,
            scenario=scenario,
            fold=f_num,
            n_train=len(sp.train_idx),
            n_inner_train=len(sp.inner_idx),
            n_val=len(sp.val_idx),
            n_test=len(sp.test_idx),
            best_epoch=best_epoch,
            num_params=num_params,
            train_mse=tr_metrics["mse"],
            train_pearson=tr_metrics["pearson"],
            val_mse=in_fit["val_mse"],
            val_pearson=in_fit["val_pearson"],
            test_mse=te_metrics["mse"],
            test_rmse=te_metrics["rmse"],
            test_pearson=te_metrics["pearson"],
            hparams=sel_hparams,
            test_ids=sp.test_ids,
            test_targets=te_y.copy(),
            test_preds=te_pred.copy(),
            val_ids=sp.val_ids,
            val_targets=va_y_orig.copy(),
            val_preds=va_pred.copy(),
            ckpt_path=ckpt_path,
        )
        results.append(res)

        if verbose:
            print(
                f"[{data.target_name} | {scenario}] "
                f"Fold {f_num}/{len(fold_splits)}: best_epoch={best_epoch} "
                f"val_r={res.val_pearson:.4f} test_r={res.test_pearson:.4f} test_rmse={res.test_rmse:.4f}"
            )

    if out_root is not None:
        m_rows = [
            {
                "target": r.target,
                "scenario": r.scenario,
                "fold": r.fold,
                "n_train": r.n_train,
                "n_inner_train": r.n_inner_train,
                "n_val": r.n_val,
                "n_test": r.n_test,
                "best_epoch": r.best_epoch,
                "lr": r.hparams["lr"],
                "batch_size": r.hparams["batch_size"],
                "weight_decay": r.hparams["weight_decay"],
                "dropout": r.hparams["dropout"],
                "grad_clip": r.hparams["grad_clip"],
                "trainable_parameters": r.num_params,
                "train_mse": r.train_mse,
                "train_pearson": r.train_pearson,
                "val_mse": r.val_mse,
                "val_pearson": r.val_pearson,
                "test_mse": r.test_mse,
                "test_rmse": r.test_rmse,
                "test_pearson": r.test_pearson,
            }
            for r in results
        ]
        pd.DataFrame(m_rows, columns=list(METRIC_COLS)).to_csv(out_root / "metrics.csv", index=False)
        pd.DataFrame(oof_rows, columns=["fold", "sample_id", "true_value", "predicted_value"]).to_csv(
            out_root / "oof_predictions.csv", index=False
        )

        valid_r = [r.test_pearson for r in results if math.isfinite(r.test_pearson)]
        summary = {
            "target": str(data.target_name),
            "scenario": scenario,
            "num_samples": len(sids),
            "num_folds": len(results),
            "feature_dimensions": {k: len(v) for k, v in active_cols.items()},
            "trainable_parameters": results[0].num_params if results else 0,
            "mean_test_pearson": float(np.mean(valid_r)) if valid_r else float("nan"),
            "std_test_pearson": float(np.std(valid_r, ddof=1)) if len(valid_r) > 1 else 0.0,
            "mean_test_rmse": float(np.mean([r.test_rmse for r in results])),
            "mean_test_mse": float(np.mean([r.test_mse for r in results])),
            "split_metadata": split_meta,
            "training_config": cfg.to_dict(),
        }
        (out_root / "run_config.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return results


def load_ckpt(path, device="cpu", grn_edges=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Invalid OmiGRN checkpoint file: {path}")

    mcfg = ckpt["model_config"]
    mods = [Modality(str(m["name"]), int(m.get("dim", m.get("feature_dim")))) for m in mcfg["modalities"]]
    fnames = {str(k): list(v) for k, v in ckpt.get("feature_names", {}).items()}
    genes = {k: v for k, v in fnames.items() if k.startswith("transcript")}
    edges = grn_edges if grn_edges is not None else (ckpt.get("grn_edges") or None)

    dev = torch.device(device)
    model = OmiGRN(
        modalities=mods,
        embed_dim=int(mcfg["embed_dim"]),
        mlp_hidden=int(mcfg["mlp_hidden"]),
        dropout=float(mcfg["dropout"]),
        grn_edges=edges,
        gene_names=genes if genes else None,
        grn_dim=int(mcfg.get("grn_dim", mcfg["embed_dim"])),
        grn_layers=int(mcfg.get("grn_layers", 2)),
    ).to(dev)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, ckpt
