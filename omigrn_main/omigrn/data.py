"""
Data handling utilities for OmiGRN (genotype / transcriptome / phenotype loading).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


@dataclass
class DatasetBundle:
    """Container holding the prepared multimodal dataset."""

    sample_ids: np.ndarray
    modalities: Dict[str, np.ndarray]
    targets: np.ndarray
    removed_samples: Dict[str, List[str]]


def _load_feature_table(path: str, sep: str, prefix: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path, sep=sep)
    id_column = dataframe.columns[0]
    dataframe = dataframe.rename(columns={id_column: "sample_id"})
    dataframe["sample_id"] = dataframe["sample_id"].astype(str).str.strip()
    duplicated = dataframe["sample_id"].duplicated()
    if duplicated.any():
        dataframe = dataframe[~duplicated].copy()
    feature_columns = [col for col in dataframe.columns if col != "sample_id"]
    dataframe = dataframe.rename(columns={col: f"{prefix}{col}" for col in feature_columns})
    numeric_columns = [col for col in dataframe.columns if col != "sample_id"]
    dataframe[numeric_columns] = dataframe[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return dataframe


def _load_phenotype_table(path: str, sep: str, target: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path, sep=sep)
    id_column = dataframe.columns[0]
    dataframe = dataframe.rename(columns={id_column: "sample_id"})
    dataframe["sample_id"] = dataframe["sample_id"].astype(str).str.strip()
    if target not in dataframe.columns:
        available = ", ".join([col for col in dataframe.columns if col != "sample_id"])
        raise ValueError(
            f"Target column '{target}' not found in phenotype table. Available columns: {available}"
        )
    duplicated = dataframe["sample_id"].duplicated()
    if duplicated.any():
        dataframe = dataframe[~duplicated].copy()
    return dataframe[["sample_id", target]]


def prepare_dataset(
    geno_path: str | None,
    pheno_path: str,
    target: str,
    *,
    geno_sep: str = "\t",
    pheno_sep: str = "\t",
    transcript_paths: Sequence[str] | None = None,
    transcript_sep: str = "\t",
) -> DatasetBundle:
    """Load genotype and/or transcriptome features plus phenotype target."""

    modalities: Dict[str, np.ndarray] = {}
    removed_samples: Dict[str, List[str]] = {}

    if geno_path is None and not transcript_paths:
        raise ValueError("At least one of geno_path or transcript_paths must be provided.")

    working_df: pd.DataFrame | None = None
    if geno_path:
        geno_df = _load_feature_table(geno_path, geno_sep, prefix="geno_")
        working_df = geno_df.copy()

    if transcript_paths:
        for index, path in enumerate(transcript_paths, start=1):
            transcript_df = _load_feature_table(path, transcript_sep, prefix=f"tx{index}_")
            if working_df is None:
                working_df = transcript_df.copy()
                continue
            before_ids = set(working_df["sample_id"])
            working_df = working_df.merge(transcript_df, on="sample_id", how="inner")
            after_ids = set(working_df["sample_id"])
            removed = sorted(before_ids - after_ids)
            removed_samples[f"missing_transcript_{index}"] = removed

    if working_df is None:
        raise ValueError("No data loaded from genotype or transcript files.")

    pheno_df = _load_phenotype_table(pheno_path, pheno_sep, target)
    before_target_ids = set(working_df["sample_id"])
    working_df = working_df.merge(pheno_df, on="sample_id", how="inner")
    after_target_ids = set(working_df["sample_id"])
    removed_samples["missing_target"] = sorted(before_target_ids - after_target_ids)

    nan_targets = working_df[working_df[target].isna()]["sample_id"].tolist()
    if nan_targets:
        removed_samples["nan_target"] = nan_targets
        working_df = working_df.dropna(subset=[target]).copy()

    sample_ids = working_df["sample_id"].to_numpy()
    targets = working_df[target].to_numpy(dtype=np.float32)

    if geno_path:
        geno_columns = [col for col in working_df.columns if col.startswith("geno_")]
        if not geno_columns:
            raise ValueError("Genotype columns are missing from the merged dataset.")
        modalities["genotype"] = _clean_matrix(working_df[geno_columns].to_numpy(dtype=np.float32))

    if transcript_paths:
        for index, _ in enumerate(transcript_paths, start=1):
            prefix = f"tx{index}_"
            tx_columns = [col for col in working_df.columns if col.startswith(prefix)]
            if not tx_columns:
                raise ValueError(f"Transcript columns missing for transcript_{index}.")
            modalities[f"transcript_{index}"] = _clean_matrix(
                working_df[tx_columns].to_numpy(dtype=np.float32)
            )

    return DatasetBundle(
        sample_ids=sample_ids,
        modalities=modalities,
        targets=targets,
        removed_samples=removed_samples,
    )


def _clean_matrix(matrix: np.ndarray) -> np.ndarray:
    column_means = np.nanmean(matrix, axis=0)
    inds = np.where(np.isnan(matrix))
    matrix[inds] = np.take(column_means, inds[1])
    matrix = np.nan_to_num(matrix, nan=0.0)
    return matrix


class MultimodalDataset(Dataset):
    """Simple PyTorch dataset that returns modality tensors and targets."""

    def __init__(self, bundle: DatasetBundle, modality_names: Iterable[str]):
        self.targets = bundle.targets
        self.modalities = {name: bundle.modalities[name] for name in modality_names}
        self.sample_ids = bundle.sample_ids
        self.names = list(self.modalities.keys())

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        item = {
            name: np.asarray(self.modalities[name][index], dtype=np.float32)
            for name in self.names
        }
        target = np.float32(self.targets[index])
        return item, target, self.sample_ids[index]


def collate_batch(
    batch: Sequence[tuple[Mapping[str, np.ndarray], float, str]]
) -> tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    features: Dict[str, List[np.ndarray]] = {}
    targets: List[float] = []
    ids: List[str] = []

    for modalities, target, sample_id in batch:
        for name, array in modalities.items():
            features.setdefault(name, []).append(array)
        targets.append(target)
        ids.append(sample_id)

    stacked_features = {name: np.stack(arrays, axis=0) for name, arrays in features.items()}
    return stacked_features, np.asarray(targets, dtype=np.float32), ids
