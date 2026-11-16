from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    import dgl.function as fn
except ImportError:
    dgl = None
    fn = None


def has_dgl():
    return dgl is not None and fn is not None


def _require_dgl():
    if not has_dgl():
        raise ImportError("DGL is required for the GRN transcriptomic branch.")


def clean_gene_id(val):
    name = str(val).strip()
    return name[5:] if name.startswith("gene:") else name


def _index_genes(gene_names):
    cleaned = [clean_gene_id(g) for g in gene_names]
    if len(set(cleaned)) == len(cleaned):
        return cleaned, {g: i for i, g in enumerate(cleaned)}, True
    raw = [str(g).strip() for g in gene_names]
    return raw, {g: i for i, g in enumerate(raw)}, False


def read_edges(source):
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    else:
        rows = list(source)
        if not rows:
            df = pd.DataFrame(columns=["Source", "Target", "Importance"])
        elif len(rows[0]) >= 3:
            df = pd.DataFrame(rows, columns=["Source", "Target", "Importance"])
        elif len(rows[0]) == 2:
            df = pd.DataFrame(rows, columns=["Source", "Target"])
            df["Importance"] = 1.0
        else:
            raise ValueError("Edge rows must have 2 or 3 columns.")

    if "Source" not in df.columns or "Target" not in df.columns:
        raise ValueError("GRN table must contain 'Source' and 'Target' columns.")
    if "Importance" in df.columns:
        df["Importance"] = pd.to_numeric(df["Importance"], errors="coerce").fillna(1.0)
    else:
        df["Importance"] = 1.0
    return df[["Source", "Target", "Importance"]].copy()


def build_graph(edges, gene_names, undirected=False):
    _require_dgl()
    df = read_edges(edges)
    genes, gene2id, use_clean = _index_genes(gene_names)
    mapper = clean_gene_id if use_clean else (lambda v: str(v).strip())

    df["src_id"] = df["Source"].map(mapper)
    df["dst_id"] = df["Target"].map(mapper)
    matched = df[df["src_id"].isin(gene2id) & df["dst_id"].isin(gene2id)].copy()
    matched = matched.drop_duplicates(subset=["src_id", "dst_id"], keep="first")

    src = matched["src_id"].map(gene2id).astype(np.int64).to_numpy()
    dst = matched["dst_id"].map(gene2id).astype(np.int64).to_numpy()
    weights = matched["Importance"].to_numpy(dtype=np.float64)

    if undirected and len(matched) > 0:
        n_m = len(matched)
        src = np.concatenate([src, dst], axis=0)
        dst = np.concatenate([dst, src[:n_m]], axis=0)
        weights = np.concatenate([weights, weights], axis=0)

    g = dgl.graph((src, dst), num_nodes=len(genes))
    if len(weights) > 0:
        g.edata["w"] = torch.tensor(np.log1p(weights), dtype=torch.float32).view(-1, 1)
    else:
        g.edata["w"] = torch.zeros((g.num_edges(), 1), dtype=torch.float32)

    out_edges = matched[["src_id", "dst_id", "Importance"]].rename(
        columns={"src_id": "Source", "dst_id": "Target"}
    )
    return g, out_edges.reset_index(drop=True)


class AttnReadout(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.key = nn.Linear(dim, dim)
        self.weight = nn.Sequential(nn.Linear(dim, 1, bias=False), nn.Sigmoid())
        self.value = nn.Linear(dim, dim)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["a"] = self.weight(self.key(h))
            g.ndata["v"] = self.value(h)
            hg = dgl.readout_nodes(g, "v", weight="a", op="sum")
            return hg, g.ndata["a"]


class MPNNLayer(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.bn = nn.BatchNorm1d(dim, affine=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            if "w" in g.edata and g.num_edges() > 0:
                g.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "agg"))
                agg = g.ndata["agg"]
            else:
                agg = torch.zeros_like(h)

            out = self.lin(agg)
            if out.size(0) > 1 or not self.training:
                out = self.bn(out)
            out = self.dropout(F.gelu(out))
            return out + h


class GRNNet(nn.Module):
    def __init__(self, gene_names, edges, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        _require_dgl()
        if edges is None:
            raise ValueError("edges is required to initialize GRNNet.")

        g, matched = build_graph(edges, gene_names)
        self.num_genes = int(g.num_nodes())
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.matched_edges = [
            [str(r.Source), str(r.Target), float(r.Importance)]
            for r in matched.itertuples(index=False)
        ]

        deg = (g.in_degrees() + g.out_degrees()).to(torch.int64)
        self.conn_idx = torch.nonzero(deg > 0, as_tuple=False).view(-1).to(torch.int64)
        self.n_connected = int(self.conn_idx.numel())
        self.n_isolated = int(self.num_genes - self.n_connected)
        self.n_edges = int(len(matched))
        self.sub_g = dgl.node_subgraph(g, self.conn_idx) if self.n_connected > 0 else None

        self.node_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU())
        self.layers = nn.ModuleList([MPNNLayer(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.readout = AttnReadout(hidden_dim)

        self.expr_mlp = nn.Sequential(
            nn.Linear(self.num_genes, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.expr_norm = nn.LayerNorm(hidden_dim)
        self.fuse_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self._bg_cache = {}

    def _batched_graph(self, device, batch_size):
        if self.sub_g is None:
            return None
        key = (device, batch_size)
        if key not in self._bg_cache:
            g = self.sub_g.to(device)
            self._bg_cache[key] = dgl.batch([g for _ in range(batch_size)])
        return self._bg_cache[key]

    def forward(self, x):
        if x.dim() != 2 or x.size(1) != self.num_genes:
            raise ValueError(f"Expected input shape (batch, {self.num_genes}), got {tuple(x.shape)}")

        bs = int(x.size(0))
        dev = x.device
        if self.sub_g is not None and self.n_connected > 0:
            bg = self._batched_graph(dev, bs)
            x_conn = x.index_select(1, self.conn_idx.to(dev)).reshape(-1, 1)
            h = self.node_encoder(x_conn)
            for layer in self.layers:
                h = layer(bg, h)
            h_grn, _ = self.readout(bg, h)
        else:
            h_grn = torch.zeros((bs, self.hidden_dim), device=dev, dtype=x.dtype)

        h_grn = self.graph_norm(h_grn)
        h_expr = self.expr_norm(self.expr_mlp(x))
        gate = self.fuse_gate(torch.cat([h_grn, h_expr], dim=-1))
        return gate * h_grn + (1.0 - gate) * h_expr


class GRNBranch(nn.Module):
    def __init__(self, gene_names, edges, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        d = int(hidden_dim if hidden_dim is not None else embed_dim)
        self.grn = GRNNet(gene_names, edges, hidden_dim=d, num_layers=num_layers, dropout=dropout)
        self.proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(self.grn(x))
