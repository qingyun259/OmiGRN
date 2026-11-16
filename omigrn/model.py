import torch
import torch.nn as nn

from .grn import GRNBranch
from .layers import MGCM


class Modality:
    def __init__(self, name, dim):
        self.name = str(name)
        self.dim = int(dim)

    def __repr__(self):
        return f"Modality(name={self.name!r}, dim={self.dim})"


class GenoBranch(nn.Module):
    def __init__(self, dim, embed_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            MGCM(c1=1, c2=1, n=1),
            nn.LayerNorm(dim),
            nn.Linear(dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.encoder(x)


def _is_tx(mod, gene_map):
    return mod.name.startswith("transcript") or bool(gene_map and mod.name in gene_map)


class OmiGRN(nn.Module):
    def __init__(
        self,
        modalities,
        embed_dim=128,
        mlp_hidden=128,
        dropout=0.1,
        grn_edges=None,
        gene_names=None,
        grn_dim=128,
        grn_layers=2,
    ):
        super().__init__()
        self.modalities = list(modalities)
        if not self.modalities:
            raise ValueError("At least one modality is required.")

        self.embed_dim = int(embed_dim)
        self.mlp_hidden = int(mlp_hidden)
        self.dropout = float(dropout)
        self.grn_dim = int(grn_dim if grn_dim is not None else embed_dim)
        self.grn_layers = int(grn_layers)

        if gene_names is not None and not isinstance(gene_names, dict):
            tx_mods = [m.name for m in self.modalities if m.name.startswith("transcript")]
            key = tx_mods[0] if tx_mods else self.modalities[-1].name
            gene_map = {key: list(gene_names)}
        else:
            gene_map = gene_names

        encoders = {}
        for mod in self.modalities:
            if _is_tx(mod, gene_map):
                if grn_edges is None or not gene_map or mod.name not in gene_map:
                    raise ValueError(
                        f"Transcript modality '{mod.name}' requires grn_edges and gene_names."
                    )
                encoders[mod.name] = GRNBranch(
                    gene_map[mod.name],
                    grn_edges,
                    embed_dim=self.embed_dim,
                    hidden_dim=self.grn_dim,
                    num_layers=self.grn_layers,
                    dropout=self.dropout,
                )
            else:
                encoders[mod.name] = GenoBranch(mod.dim, self.embed_dim, self.dropout)

        self.encoders = nn.ModuleDict(encoders)
        self.fusion_norms = nn.ModuleDict(
            {m.name: nn.LayerNorm(self.embed_dim) for m in self.modalities}
            if len(self.modalities) > 1
            else {}
        )
        cat_dim = self.embed_dim * len(self.modalities)
        self.head = nn.Sequential(
            nn.LayerNorm(cat_dim),
            nn.Linear(cat_dim, self.mlp_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def matched_edges(self):
        for enc in self.encoders.values():
            if isinstance(enc, GRNBranch):
                return list(enc.grn.matched_edges)
        return []

    def forward(self, inputs):
        for mod in self.modalities:
            if mod.name not in inputs:
                raise KeyError(f"Missing modality '{mod.name}' in inputs.")

        feats = []
        for mod in self.modalities:
            h = self.encoders[mod.name](inputs[mod.name])
            if mod.name in self.fusion_norms:
                h = self.fusion_norms[mod.name](h)
            feats.append(h)
        return self.head(torch.cat(feats, dim=1)).squeeze(-1)
