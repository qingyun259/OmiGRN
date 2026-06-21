from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

try:  # Optional dependency
    import dgl
    import dgl.function as fn
except ImportError:  # pragma: no cover - optional dependency
    dgl = None
    fn = None

_GRN_BUILD_LOGGED: set[tuple[str, int]] = set()


def _require_dgl() -> None:
    if dgl is None or fn is None:
        raise ImportError(
            "Missing dependency 'dgl'. Install DGL before enabling GRN-GNN, e.g. "
            "'pip install dgl' or the CUDA-specific build."
        )


def normalize_gene_name(value: str) -> str:
    name = str(value).strip()
    if name.startswith("gene:"):
        name = name[len("gene:") :]
    return name


def _prepare_gene_index(
    gene_names: Sequence[str],
) -> Tuple[Sequence[str], dict[str, int], bool]:
    normed = [normalize_gene_name(name) for name in gene_names]
    if len(set(normed)) == len(normed):
        return normed, {name: idx for idx, name in enumerate(normed)}, True

    raw = [str(name).strip() for name in gene_names]
    return raw, {name: idx for idx, name in enumerate(raw)}, False


def build_gene_graph_with_norm(
    edges_df: pd.DataFrame,
    gene_names: Sequence[str],
    *,
    undirected: bool = False,
) -> tuple["dgl.DGLGraph", int]:
    _require_dgl()

    if "Source" not in edges_df.columns or "Target" not in edges_df.columns:
        raise ValueError("edges_df must contain columns: Source, Target, Importance (optional)")

    genes, gene2id, use_norm = _prepare_gene_index(gene_names)
    mapper = normalize_gene_name if use_norm else str

    edges = edges_df.copy()
    edges["Source_norm"] = edges["Source"].map(mapper)
    edges["Target_norm"] = edges["Target"].map(mapper)
    edges = edges[edges["Source_norm"].isin(gene2id) & edges["Target_norm"].isin(gene2id)].copy()

    src = edges["Source_norm"].map(gene2id).astype(int).to_numpy()
    dst = edges["Target_norm"].map(gene2id).astype(int).to_numpy()

    if "Importance" in edges.columns:
        weights = pd.to_numeric(edges["Importance"], errors="coerce").fillna(1.0).to_numpy()
    else:
        weights = np.ones(len(edges), dtype=float)

    # Fix undirected bug (even if you don't enable it now)
    if undirected and len(edges) > 0:
        src0, dst0, w0 = src, dst, weights
        src = np.concatenate([src0, dst0], axis=0)
        dst = np.concatenate([dst0, src0], axis=0)
        weights = np.concatenate([w0, w0], axis=0)

    graph = dgl.graph((src, dst), num_nodes=len(genes))
    if len(weights) > 0:
        graph.edata["w"] = torch.tensor(np.log1p(weights), dtype=torch.float32).view(-1, 1)
    else:
        graph.edata["w"] = torch.zeros((graph.num_edges(), 1), dtype=torch.float32)

    return graph, int(len(edges))


class AttentiveReadout(nn.Module):
    def __init__(self, in_feats: int) -> None:
        super().__init__()
        self.key = nn.Linear(in_feats, in_feats)
        self.weight = nn.Sequential(nn.Linear(in_feats, 1, bias=False), nn.Sigmoid())
        self.value = nn.Linear(in_feats, in_feats)

    def forward(self, g, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with g.local_scope():
            k = self.key(h)
            g.ndata["a"] = self.weight(k)
            g.ndata["v"] = self.value(h)
            hg = dgl.readout_nodes(g, "v", weight="a", op="sum")
            return hg, g.ndata["a"]


class WeightedMPNNLayer(nn.Module):
    def __init__(self, d_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.lin = nn.Linear(d_hidden, d_hidden, bias=False)
        self.bn = nn.BatchNorm1d(d_hidden, affine=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h: torch.Tensor) -> torch.Tensor:
        with g.local_scope():
            g.ndata["h"] = h
            if "w" in g.edata and g.num_edges() > 0:
                g.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "agg"))
                agg = g.ndata["agg"]
            else:
                agg = torch.zeros_like(h)

            out = self.lin(agg)
            out = self.bn(out)
            out = torch.nn.functional.gelu(out)
            out = self.dropout(out)
            return out + h


class GRNEncoder(nn.Module):
    """
    Two-branch encoder:
      - Connected genes (deg>0 in GRN): GRN-GNN + attentive readout -> h_grn (B, hidden_dim)
      - Isolated genes (deg==0 in GRN): MLP on raw isolated expression -> h_iso (B, hidden_dim)
      - Gated fusion: h = gate*h_grn + (1-gate)*h_iso
    Output: (B, hidden_dim)
    """

    def __init__(
        self,
        gene_names: Sequence[str],
        edges_path: str | None,
        *,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _require_dgl()
        if not edges_path:
            raise ValueError("edges_path is required to build the GRN.")
        edges_df = pd.read_csv(edges_path)
        total_edges = int(edges_df.shape[0])
        graph, matched_edges = build_gene_graph_with_norm(edges_df, gene_names)

        self.base_graph_full = graph
        self.num_genes = graph.num_nodes()
        self.hidden_dim = hidden_dim

        # degrees on the GRN graph (no self-loop added here)
        deg = (graph.in_degrees() + graph.out_degrees()).to(torch.int64)
        conn_mask = deg > 0
        iso_mask = ~conn_mask

        self.conn_idx = torch.nonzero(conn_mask, as_tuple=False).view(-1).to(torch.int64)
        self.iso_idx = torch.nonzero(iso_mask, as_tuple=False).view(-1).to(torch.int64)

        conn_count = int(self.conn_idx.numel())
        iso_count = int(self.iso_idx.numel())

        if matched_edges == 0:
            print("[GRN] Warning: no edges matched transcript genes; using empty graph.")
        else:
            key = (str(edges_path), int(len(gene_names)))
            if key not in _GRN_BUILD_LOGGED:
                _GRN_BUILD_LOGGED.add(key)
                ratio = matched_edges / total_edges if total_edges else 0.0
                print(
                    "[GRN] Graph summary: "
                    f"genes={self.num_genes} | "
                    f"edges_matched={matched_edges}/{total_edges} ({ratio:.2%}) | "
                    f"connected_genes={conn_count} | "
                    f"isolated_genes={iso_count}"
                )

        # --- GNN branch (only if there are connected genes) ---
        if conn_count > 0:
            self.base_graph_conn = dgl.node_subgraph(self.base_graph_full, self.conn_idx)
            # node_subgraph preserves edge data (w) for kept edges
            self.node_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU())
            self.layers = nn.ModuleList(
                [WeightedMPNNLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)]
            )
            self.readout = AttentiveReadout(hidden_dim)
        else:
            self.base_graph_conn = None
            self.node_encoder = None
            self.layers = nn.ModuleList([])
            self.readout = None

        # --- MLP branch for isolated genes (only if iso_count>0) ---
        if iso_count > 0:
            self.iso_mlp = nn.Sequential(
                nn.LayerNorm(iso_count),
                nn.Linear(iso_count, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.iso_mlp = None

        # --- gated fusion (always defined) ---
        self.fuse_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Cache batched connected-graph by (device, batch_size)
        self._bg_cache: dict[tuple[torch.device, int], "dgl.DGLGraph"] = {}

    def _get_batched_conn_graph(self, device: torch.device, batch_size: int):
        if self.base_graph_conn is None:
            return None
        key = (device, batch_size)
        if key in self._bg_cache:
            return self._bg_cache[key]
        g = self.base_graph_conn.to(device)
        bg = dgl.batch([g for _ in range(batch_size)])
        self._bg_cache[key] = bg
        return bg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"GRNEncoder expects (batch, genes); got {tuple(x.shape)}")
        if x.size(1) != self.num_genes:
            raise ValueError(f"GRNEncoder expects {self.num_genes} genes, got {x.size(1)} features.")

        B = int(x.size(0))
        device = x.device
        d = self.hidden_dim

        # ---- GNN branch ----
        if self.base_graph_conn is not None and self.conn_idx.numel() > 0:
            bg = self._get_batched_conn_graph(device, B)
            # (B, G_conn) -> (B*G_conn,1)
            x_conn = x.index_select(1, self.conn_idx.to(device))
            bg.ndata["x"] = x_conn.reshape(-1, 1)
            h = self.node_encoder(bg.ndata["x"])
            for layer in self.layers:
                h = layer(bg, h)
            h_grn, _ = self.readout(bg, h)  # (B, d)
        else:
            h_grn = torch.zeros((B, d), device=device, dtype=x.dtype)

        # ---- MLP branch for isolated ----
        if self.iso_mlp is not None and self.iso_idx.numel() > 0:
            x_iso = x.index_select(1, self.iso_idx.to(device))  # (B, G_iso)
            h_iso = self.iso_mlp(x_iso)  # (B, d)
        else:
            h_iso = torch.zeros((B, d), device=device, dtype=x.dtype)

        # If one side is empty, just return the other (avoids weird gating on zeros)
        if self.iso_mlp is None or self.iso_idx.numel() == 0:
            return h_grn
        if self.base_graph_conn is None or self.conn_idx.numel() == 0:
            return h_iso

        # ---- gated fusion ----
        gate = self.fuse_gate(torch.cat([h_grn, h_iso], dim=-1))  # (B,1)
        h_fused = gate * h_grn + (1.0 - gate) * h_iso
        return h_fused


class GRNModalityEncoder(nn.Module):
    """Wrap GRNEncoder and project to the fusion embedding space."""

    def __init__(
        self,
        gene_names: Sequence[str],
        edges_path: str | None,
        *,
        embed_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.grn = GRNEncoder(
            gene_names=gene_names,
            edges_path=edges_path,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        if hidden_dim == embed_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.grn(x))
