import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from .grn import clean_gene_id

_NA_STRINGS = frozenset({"", "na", "nan", "n/a", "null", "none", ".", "<na>"})
_NA_NUMBERS = (-999.0,)


class OmicsData:
    def __init__(self, sample_ids, modalities, targets, feature_names, target_name, removed=None):
        self.sample_ids = np.asarray(sample_ids, dtype=str)
        self.modalities = {k: np.asarray(v, dtype=np.float32) for k, v in modalities.items()}
        self.targets = np.asarray(targets, dtype=np.float32)
        self.feature_names = {k: list(v) for k, v in feature_names.items()}
        self.target_name = str(target_name)
        self.removed = {} if removed is None else dict(removed)


class CVFold:
    def __init__(self, fold, train_idx, inner_idx, val_idx, test_idx, train_ids, inner_ids, val_ids, test_ids):
        self.fold = int(fold)
        self.train_idx = np.asarray(train_idx, dtype=np.int64)
        self.inner_idx = np.asarray(inner_idx, dtype=np.int64)
        self.val_idx = np.asarray(val_idx, dtype=np.int64)
        self.test_idx = np.asarray(test_idx, dtype=np.int64)
        self.train_ids = list(train_ids)
        self.inner_ids = list(inner_ids)
        self.val_ids = list(val_ids)
        self.test_ids = list(test_ids)


def clean_pheno(values, context="phenotype"):
    s = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    num = pd.to_numeric(s, errors="coerce")
    is_na = s.isna() | s.astype(str).str.strip().str.lower().isin(_NA_STRINGS)
    bad = num.isna() & ~is_na
    if bad.any():
        examples = s[bad].astype(str).head(5).tolist()
        raise ValueError(f"Unrecognized phenotype values in {context}: {examples}.")
    num = num.astype(np.float64)
    return num.mask(~np.isfinite(num) | num.isin(_NA_NUMBERS))


def get_traits(pheno_path, sep="\t"):
    hdr = pd.read_csv(pheno_path, sep=sep, nrows=0)
    if len(hdr.columns) < 2:
        raise ValueError(f"Phenotype table '{pheno_path}' requires sample ID and >=1 trait column.")
    return [str(c).strip() for c in hdr.columns[1:]]


def read_matrix(path, sep="\t"):
    df = pd.read_csv(path, sep=sep, converters={0: lambda v: str(v).strip()})
    if len(df.columns) < 2:
        raise ValueError(f"Feature table '{path}' requires sample ID and >=1 feature column.")
    df = df.rename(columns={df.columns[0]: "sample_id"})
    df["sample_id"] = df["sample_id"].astype(str).str.strip()

    dup = df["sample_id"].duplicated()
    if dup.any():
        df = df[~dup].copy()
    df = df[df["sample_id"] != ""].copy()

    cols = [c for c in df.columns if c != "sample_id"]
    names = [str(c).strip() for c in cols]
    vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return df["sample_id"].tolist(), names, vals


def read_pheno(path, target=None, sep="\t"):
    df = pd.read_csv(path, sep=sep, converters={0: lambda v: str(v).strip()})
    if len(df.columns) < 2:
        raise ValueError(f"Phenotype table '{path}' requires sample ID and >=1 trait column.")
    rename = {df.columns[0]: "sample_id"}
    for col in df.columns[1:]:
        rename[col] = str(col).strip()
    df = df.rename(columns=rename)
    df["sample_id"] = df["sample_id"].astype(str).str.strip()

    traits = [c for c in df.columns if c != "sample_id"]
    if target is None:
        if len(traits) == 1:
            col = traits[0]
        else:
            raise ValueError(f"Multiple traits in '{path}' ({', '.join(traits)}); specify target.")
    else:
        col = str(target).strip()
        if col not in traits:
            raise ValueError(f"Trait '{col}' not found in '{path}'. Available: {', '.join(traits)}.")

    dup = df["sample_id"].duplicated()
    if dup.any():
        df = df[~dup].copy()
    df = df[df["sample_id"] != ""].copy()

    df[col] = clean_pheno(df[col], context=f"{path}:{col}")
    return df[["sample_id", col]], col


def _tx_list(paths):
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def load_data(
    geno_path=None,
    pheno_path="",
    target=None,
    geno_sep="\t",
    pheno_sep="\t",
    transcript_paths=None,
    transcript_sep="\t",
):
    tx_paths = _tx_list(transcript_paths)
    if geno_path is None and not tx_paths:
        raise ValueError("Provide at least one of geno_path or transcript_paths.")
    if not str(pheno_path):
        raise ValueError("pheno_path is required.")

    tables = []
    if geno_path is not None:
        sids, cols, vals = read_matrix(geno_path, sep=geno_sep)
        tables.append(("genotype", sids, cols, vals))
    for i, p in enumerate(tx_paths, start=1):
        sids, cols, vals = read_matrix(p, sep=transcript_sep)
        tables.append((f"transcript_{i}", sids, cols, vals))

    common = list(tables[0][1])
    removed = {}
    for name, sids, _, _ in tables[1:]:
        sid_set = set(sids)
        prev = set(common)
        common = [s for s in common if s in sid_set]
        removed[f"missing_{name}"] = sorted(prev - set(common))

    pheno_df, trait = read_pheno(pheno_path, target=target, sep=pheno_sep)
    p_set = set(pheno_df["sample_id"])
    prev = set(common)
    common = [s for s in common if s in p_set]
    removed["missing_target"] = sorted(prev - set(common))

    lookup = dict(zip(pheno_df["sample_id"].tolist(), pheno_df[trait].to_numpy()))
    nan_ids = [s for s in common if not np.isfinite(lookup[s])]
    if nan_ids:
        removed["nan_target"] = nan_ids
        bad = set(nan_ids)
        common = [s for s in common if s not in bad]

    if not common:
        raise ValueError(f"No overlapping samples with valid '{trait}' phenotypes.")

    modalities = {}
    feature_names = {}
    for name, sids, cols, vals in tables:
        idx_map = {s: i for i, s in enumerate(sids)}
        rows = np.asarray([idx_map[s] for s in common], dtype=np.int64)
        modalities[name] = vals[rows].astype(np.float32, copy=True)
        feature_names[name] = list(cols)

    targets = np.asarray([lookup[s] for s in common], dtype=np.float32)
    return OmicsData(
        sample_ids=np.asarray(common, dtype=str),
        modalities=modalities,
        targets=targets,
        feature_names=feature_names,
        target_name=trait,
        removed=removed,
    )


def _align_cols(mod_name, cols, vals, expected):
    if list(cols) == list(expected):
        return vals.astype(np.float32, copy=False)
    exact = {c: i for i, c in enumerate(cols)}
    if all(c in exact for c in expected):
        order = np.asarray([exact[c] for c in expected], dtype=np.int64)
        return vals[:, order].astype(np.float32, copy=True)

    if mod_name.startswith("transcript"):
        norm_map = {}
        for i, c in enumerate(cols):
            k = clean_gene_id(c)
            if k not in norm_map:
                norm_map[k] = i
        exp_norm = [clean_gene_id(c) for c in expected]
        if all(k in norm_map for k in exp_norm):
            order = np.asarray([norm_map[k] for k in exp_norm], dtype=np.int64)
            return vals[:, order].astype(np.float32, copy=True)

    missing = [c for c in expected if c not in exact]
    raise ValueError(f"Modality '{mod_name}' missing {len(missing)} expected feature(s), e.g. {missing[:5]}.")


def load_inputs(
    geno_path=None,
    transcript_paths=None,
    geno_sep="\t",
    transcript_sep="\t",
    expected_features=None,
):
    tx_paths = _tx_list(transcript_paths)
    if geno_path is None and not tx_paths:
        raise ValueError("Provide at least one of geno_path or transcript_paths.")

    tables = []
    if geno_path is not None:
        sids, cols, vals = read_matrix(geno_path, sep=geno_sep)
        tables.append(("genotype", sids, cols, vals))
    for i, p in enumerate(tx_paths, start=1):
        sids, cols, vals = read_matrix(p, sep=transcript_sep)
        tables.append((f"transcript_{i}", sids, cols, vals))

    common = list(tables[0][1])
    for _, sids, _, _ in tables[1:]:
        sid_set = set(sids)
        common = [s for s in common if s in sid_set]
    if not common:
        raise ValueError("No overlapping sample IDs across input feature files.")

    modalities = {}
    feature_names = {}
    for name, sids, cols, vals in tables:
        idx_map = {s: i for i, s in enumerate(sids)}
        rows = np.asarray([idx_map[s] for s in common], dtype=np.int64)
        mat = vals[rows]
        if expected_features is not None and name in expected_features:
            exp_cols = list(expected_features[name])
            mat = _align_cols(name, cols, mat, exp_cols)
            feature_names[name] = exp_cols
        else:
            mat = mat.astype(np.float32, copy=True)
            feature_names[name] = list(cols)
        modalities[name] = mat

    if expected_features is not None:
        missing = [m for m in expected_features if m not in modalities]
        if missing:
            raise ValueError(f"Missing required modality inputs: {missing}.")

    return common, feature_names, modalities


def fit_geno(train_x):
    x = np.asarray(train_x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("Genotype matrix must be non-empty 2D.")
    fin = np.isfinite(x)
    cnt = fin.sum(axis=0)
    tot = np.where(fin, x, 0.0).sum(axis=0, dtype=np.float64)
    means = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
    return {"impute_mean": means.astype(np.float32).tolist()}


def apply_geno(x, spec):
    arr = np.asarray(x, dtype=np.float32)
    means = np.asarray(spec["impute_mean"], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != means.shape[0]:
        raise ValueError(f"Genotype shape mismatch: got {arr.shape}, expected {means.shape[0]} columns.")
    fin = np.isfinite(arr)
    if fin.all():
        return arr.astype(np.float32, copy=True)
    return np.where(fin, arr, means[None, :]).astype(np.float32)


def fit_expr(train_x, min_std=1e-8):
    x = np.asarray(train_x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("Expression scaler requires at least 2 training samples.")
    fin = np.isfinite(x)
    cnt = fin.sum(axis=0)
    tot = np.where(fin, x, 0.0).sum(axis=0, dtype=np.float64)
    imp = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
    x_imp = np.where(fin, x, imp[None, :])

    mean = x_imp.mean(axis=0)
    raw_std = x_imp.std(axis=0, ddof=0)
    active = raw_std >= float(min_std)
    std = np.where(active, raw_std, 1.0)
    return {
        "impute_mean": imp.astype(np.float32).tolist(),
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "active_mask": active.astype(bool).tolist(),
        "ddof": 0,
    }


def apply_expr(x, spec):
    arr = np.asarray(x, dtype=np.float64)
    mean = np.asarray(spec["mean"], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != mean.shape[0]:
        raise ValueError(f"Expression shape mismatch: got {arr.shape}, expected {mean.shape[0]} genes.")
    imp = np.asarray(spec.get("impute_mean", spec["mean"]), dtype=np.float64)
    fin = np.isfinite(arr)
    x_imp = arr if fin.all() else np.where(fin, arr, imp[None, :])

    std_raw = np.asarray(spec.get("std", spec.get("scale")), dtype=np.float64)
    if "active_mask" in spec:
        active = np.asarray(spec["active_mask"], dtype=bool)
    elif "active" in spec:
        active = np.asarray(spec["active"], dtype=bool)
    else:
        active = std_raw >= 1e-8

    safe_std = np.where(active & (std_raw > 0.0), std_raw, 1.0)
    z = (x_imp - mean[None, :]) / safe_std[None, :]
    if (~active).any():
        z[:, ~active] = 0.0
    if not np.isfinite(z).all():
        raise FloatingPointError("Non-finite standardized expression values.")
    return z.astype(np.float32)


def fit_scalers(train_mods):
    scalers = {}
    for name, vals in train_mods.items():
        scalers[name] = fit_expr(vals) if name.startswith("transcript") else fit_geno(vals)
    return scalers


def apply_scalers(raw_mods, scalers):
    out = {}
    for name, spec in scalers.items():
        if name not in raw_mods:
            raise KeyError(f"Missing modality '{name}' required by preprocessing.")
        if name.startswith("transcript") or "mean" in spec:
            out[name] = apply_expr(raw_mods[name], spec)
        else:
            out[name] = apply_geno(raw_mods[name], spec)
    return out


def fit_y_scaler(train_y):
    y = np.asarray(train_y, dtype=np.float64)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("Target scaler requires at least 2 finite training labels.")
    mean = float(y.mean())
    raw_std = float(y.std(ddof=0))
    std = raw_std if raw_std > 1e-12 else 1.0
    return {"mean": mean, "std": std, "ddof": 0}


def scale_y(y, scaler):
    arr = np.asarray(y, dtype=np.float64)
    mean = float(scaler["mean"])
    std = float(scaler.get("std", scaler.get("scale", 1.0)))
    if std <= 0.0:
        std = 1.0
    return ((arr - mean) / std).astype(np.float32)


def unscale_y(preds, scaler):
    arr = np.asarray(preds, dtype=np.float64)
    mean = float(scaler["mean"])
    std = float(scaler.get("std", scaler.get("scale", 1.0)))
    return arr * std + mean


def _split_inner(train_idx, val_ratio, seed):
    n = len(train_idx)
    if n < 4:
        raise ValueError(f"Outer training fold needs at least 4 samples, got {n}.")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}.")
    n_val = max(2, int(round(n * val_ratio)))
    if n - n_val < 2:
        n_val = n - 2
    inner_idx, val_idx = train_test_split(train_idx, test_size=n_val, random_state=seed, shuffle=True)
    return np.asarray(inner_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def _check_fold(fold, sample_ids, train_idx, inner_idx, val_idx, test_idx):
    s_tr = set(train_idx.tolist())
    s_in = set(inner_idx.tolist())
    s_va = set(val_idx.tolist())
    s_te = set(test_idx.tolist())
    if s_tr & s_te:
        raise ValueError(f"Fold {fold}: train and test overlap.")
    if s_in & s_va:
        raise ValueError(f"Fold {fold}: inner_train and val overlap.")
    if (s_in | s_va) != s_tr:
        raise ValueError(f"Fold {fold}: inner_train union val must equal train.")
    if len(inner_idx) < 2 or len(val_idx) < 1 or len(test_idx) < 2:
        raise ValueError(f"Fold {fold}: insufficient samples after alignment.")
    return CVFold(
        fold=int(fold),
        train_idx=np.asarray(train_idx, dtype=np.int64),
        inner_idx=np.asarray(inner_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
        train_ids=[str(sample_ids[i]) for i in train_idx],
        inner_ids=[str(sample_ids[i]) for i in inner_idx],
        val_ids=[str(sample_ids[i]) for i in val_idx],
        test_ids=[str(sample_ids[i]) for i in test_idx],
    )


def make_splits(sample_ids, folds=10, val_ratio=0.1, seed=42):
    n = len(sample_ids)
    if folds < 2:
        raise ValueError(f"folds must be >= 2, got {folds}.")
    if n < max(6, 2 * folds):
        raise ValueError(f"Dataset size {n} is too small for {folds}-fold CV.")
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    idx = np.arange(n, dtype=np.int64)
    out = []
    for f, (tr_pos, te_pos) in enumerate(kf.split(idx), start=1):
        tr_idx = idx[tr_pos]
        te_idx = idx[te_pos]
        in_idx, va_idx = _split_inner(tr_idx, val_ratio=val_ratio, seed=seed + f)
        out.append(_check_fold(f, sample_ids, tr_idx, in_idx, va_idx, te_idx))
    return out


def _dedup_ids(items):
    seen = set()
    res = []
    for x in items:
        s = str(x).strip()
        if s and s not in seen:
            seen.add(s)
            res.append(s)
    return res


def parse_splits(raw_folds, sample_ids, val_ratio=0.1, seed=42, source="splits"):
    if not raw_folds:
        raise ValueError(f"No fold entries in '{source}'.")
    id2idx = {str(s): i for i, s in enumerate(sample_ids)}
    out = []
    for pos, entry in enumerate(raw_folds, start=1):
        if isinstance(entry, CVFold):
            out.append(entry)
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Fold {pos} in '{source}' must be a dict or CVFold.")
        f_num = int(entry.get("fold", pos))
        raw_te = entry.get("test_ids", entry.get("outer_test_ids"))
        if raw_te is None:
            raise ValueError(f"Fold {f_num} in '{source}' missing 'test_ids'.")
        te_ids = _dedup_ids(raw_te)

        raw_in = entry.get("inner_train_ids", entry.get("inner_ids"))
        raw_va = entry.get("val_ids", entry.get("validation_ids"))
        raw_tr = entry.get("train_ids", entry.get("outer_train_ids"))
        if raw_tr is None:
            if raw_in is not None and raw_va is not None:
                raw_tr = list(raw_in) + list(raw_va)
            else:
                raise ValueError(f"Fold {f_num} in '{source}' missing 'train_ids'.")
        tr_ids = _dedup_ids(raw_tr)
        if set(tr_ids) & set(te_ids):
            raise ValueError(f"Fold {f_num} in '{source}' has overlapping train_ids and test_ids.")

        tr_idx = np.asarray([id2idx[s] for s in tr_ids if s in id2idx], dtype=np.int64)
        te_idx = np.asarray([id2idx[s] for s in te_ids if s in id2idx], dtype=np.int64)

        if raw_in is not None and raw_va is not None:
            in_ids = _dedup_ids(raw_in)
            va_ids = _dedup_ids(raw_va)
            if set(in_ids) & set(va_ids):
                raise ValueError(f"Fold {f_num} in '{source}' has overlapping inner and val IDs.")
            if (set(in_ids) | set(va_ids)) != set(tr_ids):
                raise ValueError(f"Fold {f_num} in '{source}': inner union val must equal train_ids.")
            in_idx = np.asarray([id2idx[s] for s in in_ids if s in id2idx], dtype=np.int64)
            va_idx = np.asarray([id2idx[s] for s in va_ids if s in id2idx], dtype=np.int64)
        elif raw_in is not None or raw_va is not None:
            raise ValueError(f"Fold {f_num} in '{source}' must specify both inner and val IDs or neither.")
        else:
            in_idx, va_idx = _split_inner(tr_idx, val_ratio=val_ratio, seed=seed + f_num)

        out.append(_check_fold(f_num, sample_ids, tr_idx, in_idx, va_idx, te_idx))
    return out


def read_splits(split_path, sample_ids, val_ratio=0.1, seed=42):
    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    meta = {"mode": "custom_split_file"}
    if isinstance(payload, list):
        raw_folds = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("folds"), list):
            raw_folds = payload["folds"]
            for k in ("protocol", "seed", "population"):
                if k in payload:
                    meta[k] = payload[k]
        elif isinstance(payload.get("splits"), list):
            raw_folds = payload["splits"]
        elif any(k in payload for k in ("train_ids", "outer_train_ids", "test_ids")):
            raw_folds = [payload]
        else:
            raise ValueError(f"Unrecognized split JSON format in '{split_path}'.")
    else:
        raise ValueError(f"Split file '{split_path}' must contain a JSON object or list.")

    splits = parse_splits(raw_folds, sample_ids, val_ratio=val_ratio, seed=seed, source=str(split_path))
    meta["num_folds"] = len(splits)
    return splits, meta
