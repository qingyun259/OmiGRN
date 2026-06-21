
import os
import re
import argparse
import traceback

import numpy as np
import pandas as pd

from arboreto.algo import grnboost2
from distributed import Client, LocalCluster


EXPR_SEP = "\t"

TF_SEP = "\t"
TF_ID_COLUMN = None

MODULE_SEP = "\t"
MODULE_GENE_COL = "GeneID"
MODULE_COL = "Module"

META_SEP = "\t"
META_SAMPLE_COL = "Sample"
META_TISSUE_COL = "leaf"

N_WORKERS = 4
THREADS_PER_WORKER = 1

MIN_NONZERO_FRAC = 0.05
MIN_NONZERO_ABS = 3
TOP_HV_GENES = 5000
TOP_EDGES = 50000
USE_FLOAT32 = True
MIN_MODULE_SIZE = 20

TF_SCOPE = "all"

WRITE_PER_MODULE = False

EDGE_FILTER_MODE = "per_target_topk"
TOPK_PER_TARGET = 10

EDGE_WEIGHT_TRANSFORM = "rank_in_target"


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def normalize_gene_id(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"^(gene:|Gene:|transcript:|Transcript:)", "", x)
    x = x.split("|")[0].split()[0]
    x = re.sub(r"(\.t\d+|\.\d+|_t\d+|_T\d+|-T\d+)$", "", x)
    return x

def read_table_auto(path: str, sep: str, index_col=None) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312", None):
        try:
            return pd.read_csv(path, sep=sep, index_col=index_col, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, sep=sep, index_col=index_col)

def clean_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df

def collapse_to_gene_level(df: pd.DataFrame) -> pd.DataFrame:
    raw_cols = list(df.columns)
    norm_cols = [normalize_gene_id(c) for c in raw_cols]
    df2 = df.copy()
    df2.columns = norm_cols
    if len(set(norm_cols)) < len(norm_cols):
        df2 = df2.groupby(axis=1, level=0).mean()
    return df2

def load_tf_list(tf_file: str, sep: str, id_col: str | None) -> pd.Index:
    tf_df = read_table_auto(tf_file, sep=sep, index_col=None)
    if id_col is None:
        cand = [c for c in tf_df.columns if any(k in str(c).lower() for k in ("gene", "id", "locus"))]
        use_col = cand[0] if cand else tf_df.columns[0]
    else:
        use_col = id_col
    ids = tf_df[use_col].dropna().astype(str).tolist()
    ids = [normalize_gene_id(x) for x in ids]
    return pd.Index(pd.unique(ids))

def min_nonzero_samples(n_samples: int) -> int:
    return max(MIN_NONZERO_ABS, int(MIN_NONZERO_FRAC * n_samples))

def filter_low_expression_genes(df: pd.DataFrame, min_nz: int, force_keep: set[str] | None = None) -> pd.DataFrame:
    force_keep = force_keep or set()
    nz = (df > 0).sum(axis=0)
    keep = set(nz[nz >= min_nz].index.tolist()) | force_keep
    return df.loc[:, sorted(keep)]

def select_highly_variable_genes(df: pd.DataFrame, top_n: int | None, force_keep: set[str]) -> pd.DataFrame:
    if top_n is None or df.shape[1] <= top_n:
        return df
    vars_ = df.var(axis=0).sort_values(ascending=False)
    hv = set(vars_.head(top_n).index.tolist())
    keep = hv | force_keep
    return df.loc[:, sorted(keep)]

def load_sample_meta(meta_file: str) -> pd.DataFrame | None:
    if not meta_file or not os.path.isfile(meta_file):
        return None
    meta = read_table_auto(meta_file, sep=META_SEP, index_col=None)
    if META_SAMPLE_COL not in meta.columns or META_TISSUE_COL not in meta.columns:
        return None
    return meta[[META_SAMPLE_COL, META_TISSUE_COL]].copy()

def write_list(path: str, items: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(str(x) + "\n")

def load_gene_modules(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"gene_modules 文件不存在：{path}")
    m = read_table_auto(path, sep=MODULE_SEP, index_col=None)
    if MODULE_GENE_COL not in m.columns or MODULE_COL not in m.columns:
        raise ValueError(f"gene_modules 文件需包含列：{MODULE_GENE_COL}, {MODULE_COL}")
    m = m[[MODULE_GENE_COL, MODULE_COL]].dropna()
    m[MODULE_GENE_COL] = m[MODULE_GENE_COL].astype(str).map(normalize_gene_id)
    m[MODULE_COL] = m[MODULE_COL].astype(str)
    m = m.drop_duplicates(subset=[MODULE_GENE_COL], keep="first")
    return m

def transform_edge_weights(net: pd.DataFrame) -> pd.DataFrame:
    if EDGE_WEIGHT_TRANSFORM == "none":
        return net
    net = net.copy()
    if EDGE_WEIGHT_TRANSFORM == "log1p":
        net["Importance"] = np.log1p(net["Importance"].astype(float))
        return net
    if EDGE_WEIGHT_TRANSFORM == "rank_in_target":
        net["rank"] = net.groupby("Target")["Importance"].rank(ascending=False, method="first")
        net["Importance"] = 1.0 / net["rank"]
        net = net.drop(columns=["rank"])
        return net
    return net

def filter_edges(net: pd.DataFrame) -> pd.DataFrame:
    if EDGE_FILTER_MODE == "global_topE":
        return net.sort_values("Importance", ascending=False).head(TOP_EDGES)
    if EDGE_FILTER_MODE == "per_target_topk":
        net = net.sort_values("Importance", ascending=False)
        net = net.groupby("Target", group_keys=False).head(TOPK_PER_TARGET)
        if TOP_EDGES is not None and len(net) > TOP_EDGES:
            net = net.head(TOP_EDGES)
        return net
    return net

def infer_grnboost2_for_module(df_group: pd.DataFrame, tf_names: list[str], module_genes: list[str], client: Client) -> pd.DataFrame:
    cols = sorted(set(module_genes) | set(tf_names))
    sub = df_group.loc[:, [c for c in cols if c in df_group.columns]]

    net = grnboost2(expression_data=sub, tf_names=tf_names, client_or_address=client)
    net = net.rename(columns={"TF": "Source", "target": "Target", "importance": "Importance"})
    net = net[net["Target"].isin(set(module_genes))].copy()
    return net


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Infer a module-aware GRN with GRNBoost2 (WGCNA modules + PlantTFDB TFs)."
    )
    p.add_argument("--expr", required=True, help="表达矩阵（行=样本，列=基因，第一列样本ID）。")
    p.add_argument("--tf-list", required=True, help="TF 列表文件（已映射到 gene ID）。")
    p.add_argument("--gene-modules", required=True, help="WGCNA 输出的 gene_modules.tsv（含 GeneID, Module 两列）。")
    p.add_argument("--outdir", required=True, help="输出目录。")
    p.add_argument("--expr-sep", default=EXPR_SEP, help="表达矩阵分隔符（默认制表符）。")
    p.add_argument("--tf-sep", default=TF_SEP, help="TF 文件分隔符（默认制表符）。")
    p.add_argument("--tf-id-column", default=TF_ID_COLUMN, help="TF 文件中基因ID所在列名（默认自动识别）。")
    p.add_argument("--module-sep", default=MODULE_SEP, help="gene_modules 文件分隔符（默认制表符）。")
    p.add_argument("--sample-meta", default="", help="可选样本元数据文件（含样本列与组织列）。")
    p.add_argument("--meta-sample-col", default=META_SAMPLE_COL, help="元数据中的样本列名。")
    p.add_argument("--meta-tissue-col", default=META_TISSUE_COL, help="元数据中的组织/分组列名。")
    p.add_argument("--workers", type=int, default=N_WORKERS, help="Dask worker 数。")
    p.add_argument("--threads-per-worker", type=int, default=THREADS_PER_WORKER, help="每个 worker 的线程数。")
    p.add_argument("--min-nonzero-frac", type=float, default=MIN_NONZERO_FRAC, help="基因保留所需非零样本比例。")
    p.add_argument("--min-nonzero-abs", type=int, default=MIN_NONZERO_ABS, help="基因保留所需非零样本数下限。")
    p.add_argument("--top-hv-genes", type=int, default=TOP_HV_GENES, help="保留高变基因数；0 表示不做高变筛选。")
    p.add_argument("--min-module-size", type=int, default=MIN_MODULE_SIZE, help="参与建网的最小模块基因数。")
    p.add_argument("--no-float32", action="store_true", help="关闭 float32（改用 float64）。")
    p.add_argument("--tf-scope", choices=["all", "module"], default=TF_SCOPE, help="模块内推断时的 TF 候选范围。")
    p.add_argument("--edge-filter-mode", choices=["global_topE", "per_target_topk"], default=EDGE_FILTER_MODE, help="边筛选策略。")
    p.add_argument("--topk-per-target", type=int, default=TOPK_PER_TARGET, help="per_target_topk 模式下每个 target 的入边数。")
    p.add_argument("--top-edges", type=int, default=TOP_EDGES, help="全局边数上限；0 表示不限制。")
    p.add_argument("--edge-weight-transform", choices=["none", "log1p", "rank_in_target"], default=EDGE_WEIGHT_TRANSFORM, help="边权变换方式。")
    p.add_argument("--write-per-module", action="store_true", help="额外保存每个模块的网络文件。")
    return p.parse_args()


def _apply_args(args: argparse.Namespace) -> None:
    """把命令行参数写回模块级配置（供上面的工具函数读取）。"""
    global INPUT_EXPR, EXPR_SEP, TF_LIST_FILE, TF_SEP, TF_ID_COLUMN
    global GENE_MODULE_FILE, MODULE_SEP, SAMPLE_META_FILE, META_SAMPLE_COL, META_TISSUE_COL
    global OUTDIR, N_WORKERS, THREADS_PER_WORKER
    global MIN_NONZERO_FRAC, MIN_NONZERO_ABS, TOP_HV_GENES, TOP_EDGES, USE_FLOAT32, MIN_MODULE_SIZE
    global TF_SCOPE, WRITE_PER_MODULE, EDGE_FILTER_MODE, TOPK_PER_TARGET, EDGE_WEIGHT_TRANSFORM

    INPUT_EXPR = args.expr
    EXPR_SEP = args.expr_sep
    TF_LIST_FILE = args.tf_list
    TF_SEP = args.tf_sep
    TF_ID_COLUMN = args.tf_id_column
    GENE_MODULE_FILE = args.gene_modules
    MODULE_SEP = args.module_sep
    SAMPLE_META_FILE = args.sample_meta
    META_SAMPLE_COL = args.meta_sample_col
    META_TISSUE_COL = args.meta_tissue_col
    OUTDIR = args.outdir
    N_WORKERS = args.workers
    THREADS_PER_WORKER = args.threads_per_worker
    MIN_NONZERO_FRAC = args.min_nonzero_frac
    MIN_NONZERO_ABS = args.min_nonzero_abs
    TOP_HV_GENES = args.top_hv_genes if args.top_hv_genes and args.top_hv_genes > 0 else None
    TOP_EDGES = args.top_edges if args.top_edges and args.top_edges > 0 else None
    USE_FLOAT32 = not args.no_float32
    MIN_MODULE_SIZE = args.min_module_size
    TF_SCOPE = args.tf_scope
    WRITE_PER_MODULE = args.write_per_module
    EDGE_FILTER_MODE = args.edge_filter_mode
    TOPK_PER_TARGET = args.topk_per_target
    EDGE_WEIGHT_TRANSFORM = args.edge_weight_transform


def main():
    args = parse_args()
    _apply_args(args)

    ensure_outdir(OUTDIR)

    print("读取表达矩阵...")
    df = read_table_auto(INPUT_EXPR, sep=EXPR_SEP, index_col=0)
    df = clean_duplicates(df)
    if USE_FLOAT32:
        df = df.astype(np.float32)

    df = collapse_to_gene_level(df)
    print(f"Gene-level 矩阵维度: {df.shape} (Samples x Genes)")

    print("读取 TF 列表...")
    tf_all = load_tf_list(TF_LIST_FILE, sep=TF_SEP, id_col=TF_ID_COLUMN)
    tf_all_set = set(tf_all.tolist())

    expr_genes_set = set(df.columns.tolist())
    tf_present = sorted(tf_all_set & expr_genes_set)
    tf_missing = sorted(tf_all_set - expr_genes_set)

    min_nz = min_nonzero_samples(df.shape[0])
    if tf_present:
        nz_tf = (df[tf_present] > 0).sum(axis=0)
        tf_low_expr = nz_tf[nz_tf < min_nz].index.tolist()
        tf_good = nz_tf[nz_tf >= min_nz].index.tolist()
    else:
        tf_low_expr, tf_good = [], []

    write_list(os.path.join(OUTDIR, "TF_used.txt"), tf_good)
    write_list(os.path.join(OUTDIR, "TF_low_expr.txt"), tf_low_expr)
    write_list(os.path.join(OUTDIR, "TF_missing.txt"), tf_missing)

    print(f"TF 总数(归一化): {len(tf_all_set)} | 匹配到矩阵: {len(tf_present)} | 缺失: {len(tf_missing)}")
    print(f"TF 低表达(<{min_nz}样本>0): {len(tf_low_expr)} | 可用 TF: {len(tf_good)}")

    df = filter_low_expression_genes(df, min_nz=min_nz, force_keep=set(tf_good))
    df = select_highly_variable_genes(df, top_n=TOP_HV_GENES, force_keep=set(tf_good))
    print(f"过滤后用于建网的基因数: {df.shape[1]}")

    mods = load_gene_modules(GENE_MODULE_FILE)
    mods = mods[mods[MODULE_GENE_COL].isin(set(df.columns))].copy()
    if mods.empty:
        raise ValueError("模块文件与表达矩阵（过滤后）没有任何基因交集，请检查 ID 归一化/过滤阈值。")

    meta = load_sample_meta(SAMPLE_META_FILE)
    groups: dict[str, pd.DataFrame] = {}
    if meta is None:
        groups["all"] = df
    else:
        meta = meta.dropna()
        df2 = df.copy()
        df2.index = df2.index.astype(str)
        meta2 = meta.copy()
        meta2[META_SAMPLE_COL] = meta2[META_SAMPLE_COL].astype(str)
        common = sorted(set(meta2[META_SAMPLE_COL]) & set(df2.index))
        if not common:
            groups["all"] = df
        else:
            meta2 = meta2[meta2[META_SAMPLE_COL].isin(common)]
            for tissue, sub in meta2.groupby(META_TISSUE_COL):
                sample_ids = sub[META_SAMPLE_COL].tolist()
                groups[str(tissue)] = df2.loc[sample_ids, :]

    cluster = None
    client = None
    try:
        cluster = LocalCluster(n_workers=N_WORKERS, threads_per_worker=THREADS_PER_WORKER)
        client = Client(cluster)
        print("Dask 启动成功。")

        for gname, gdf in groups.items():
            print(f"\n=== 组: {gname} | samples={gdf.shape[0]} genes={gdf.shape[1]} ===")

            tf_names_all = sorted(set(tf_good) & set(gdf.columns))
            if len(tf_names_all) == 0:
                print(f"[{gname}] 无可用 TF，跳过。")
                continue

            all_edges = []
            for module, subm in mods.groupby(MODULE_COL):
                module_genes = subm[MODULE_GENE_COL].tolist()
                if len(module_genes) < MIN_MODULE_SIZE:
                    continue

                if TF_SCOPE == "module":
                    tf_names = sorted(set(tf_names_all) & set(module_genes))
                else:
                    tf_names = tf_names_all

                if len(tf_names) == 0:
                    continue

                print(f"[{gname}] module={module} | genes={len(module_genes)} | TFs={len(tf_names)} ...")
                net_m = infer_grnboost2_for_module(gdf, tf_names=tf_names, module_genes=module_genes, client=client)
                if net_m.empty:
                    continue
                net_m["Module"] = module
                all_edges.append(net_m)

                if WRITE_PER_MODULE:
                    out_m = os.path.join(OUTDIR, f"network_{gname}__module_{module}.csv")
                    net_m.sort_values("Importance", ascending=False).to_csv(out_m, index=False)

            if not all_edges:
                print(f"[{gname}] 没有得到任何模块网络。")
                continue

            net = pd.concat(all_edges, ignore_index=True)

            net = transform_edge_weights(net)

            net = filter_edges(net)

            net = net.sort_values("Importance", ascending=False)
            out_csv = os.path.join(OUTDIR, f"network_{gname}.csv")
            net.to_csv(out_csv, index=False)
            print(f"[{gname}] 输出：{out_csv} | edges={len(net)}")

        print("\n全部完成。")

    except Exception as e:
        print(f"运行出错: {e}")
        traceback.print_exc()

    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()

if __name__ == "__main__":
    main()
