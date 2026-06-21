"""
OmiGRN model: a dual-branch multi-omics network for crop phenotype prediction.

The genomic branch is a multi-scale genomic context mixer (MGCM, built from the
CSP / PKI blocks in ``blocks.py``); the transcriptomic branch is a GRN message-
passing network (``grn_encoder.py``). Each modality is encoded independently,
projected to a shared dimension, concatenated in the feature-fusion layer, and
mapped to the target trait by an MLP regression head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn

from .grn_encoder import GRNModalityEncoder
from .blocks import MGCM


@dataclass(frozen=True)
class ModalityConfig:
    """Description of a single modality."""

    name: str
    feature_dim: int


class ModalityEncoder(nn.Module):
    """MGCM-based encoder that projects raw modality features into embedding space."""

    def __init__(self, feature_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            MGCM(c1=1, c2=1, n=1),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class OmiGRN(nn.Module):
    """OmiGRN dual-branch multi-omics model.

    Each modality is encoded independently — the MGCM encoder for SNP / generic
    features, and the GRN-MPNN encoder for transcript features whose
    ``grn_gene_names`` (together with a GRN ``grn_edges_path``) are supplied. The
    per-modality embeddings are projected to a shared dimension, concatenated in
    the feature-fusion layer, and regressed to the target trait by an MLP head.
    """

    def __init__(
        self,
        modalities: Iterable[ModalityConfig],
        embed_dim: int = 256,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        grn_edges_path: str | None = None,
        grn_gene_names: Dict[str, Sequence[str]] | None = None,
        grn_hidden_dim: int | None = None,
        grn_layers: int = 2,
    ):
        super().__init__()
        self.modalities: List[ModalityConfig] = list(modalities)
        if not self.modalities:
            raise ValueError("At least one modality must be provided.")

        self.embed_dim = embed_dim

        encoders: Dict[str, nn.Module] = {}
        for modality in self.modalities:
            if grn_edges_path and grn_gene_names and modality.name in grn_gene_names:
                encoders[modality.name] = GRNModalityEncoder(
                    gene_names=grn_gene_names[modality.name],
                    edges_path=grn_edges_path,
                    embed_dim=embed_dim,
                    hidden_dim=grn_hidden_dim,
                    num_layers=grn_layers,
                    dropout=dropout,
                )
            else:
                encoders[modality.name] = ModalityEncoder(
                    modality.feature_dim,
                    embed_dim,
                    dropout,
                )

        self.encoders = nn.ModuleDict(encoders)

        # Feature-fusion layer (concatenation) followed by the MLP regression head:
        #   y = Linear(Dropout(GELU(Linear(LayerNorm(Z_fuse)))))
        concat_dim = embed_dim * len(self.modalities)
        self.head = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = []
        for modality in self.modalities:
            if modality.name not in inputs:
                raise KeyError(f"Missing modality '{modality.name}' in model inputs.")
            encoded = self.encoders[modality.name](inputs[modality.name])
            tokens.append(encoded)

        fused = torch.cat(tokens, dim=1)
        prediction = self.head(fused)
        return prediction.squeeze(-1)
