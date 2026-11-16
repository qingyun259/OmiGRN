import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

HEADER_KEYS = {
    "public_gene",
    "source_gene",
    "target_gene",
    "old_gene",
    "new_gene",
    "old_id",
    "new_id",
    "source_id",
    "target_id",
    "from_id",
    "to_id",
    "query_gene",
    "related_gene",
    "gene_id",
    "source",
    "target",
}

NA_TOKENS = {"", "-", ".", "na", "n/a", "nan", "none", "null", "<na>"}


def clean_gene_id(val):
    name = str(val).strip()
    return name[5:] if name.startswith("gene:") else name


def open_text(path):
    p = Path(path)
    if p.suffix.lower() == ".gz":
        return gzip.open(p, "rt", encoding="utf-8-sig", newline="")
    return p.open("r", encoding="utf-8-sig", newline="")


def guess_sep(line, sep=None):
    if sep and sep != "auto":
        return "\t" if sep == "\\t" else sep
    if "\t" in line:
        return "\t"
    if "," in line:
        return ","
    return "\t"


def read_genes(path, sep="\t"):
    with open_text(path) as fh:
        header = ""
        for line in fh:
            if line.strip():
                header = line.rstrip("\r\n")
                break
    if not header:
        raise ValueError(f"Empty transcriptome file: {path}.")

    delim = guess_sep(header, sep)
    cols = next(csv.reader([header], delimiter=delim))
    if len(cols) <= 1 and delim == "\t" and "," in header:
        cols = next(csv.reader([header], delimiter=","))
    if len(cols) <= 1:
        raise ValueError(f"Expected sample ID plus >=1 gene columns in {path}.")

    genes = [clean_gene_id(c) for c in cols[1:]]
    if any(not g for g in genes):
        raise ValueError(f"Empty gene ID in header of {path}.")
    if len(genes) != len(set(genes)):
        counts = Counter(genes)
        dups = [g for g, n in counts.items() if n > 1][:5]
        raise ValueError(f"Duplicate gene columns in {path}: {dups}.")
    return genes


def read_interactions(path):
    raw_rows = 0
    edges = set()

    with open_text(path) as fh:
        first_line = None
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            first_line = line
            break

        if first_line is None:
            raise ValueError(f"Interaction file contains no records: {path}.")

        delim = guess_sep(first_line)

        def iter_lines():
            yield first_line
            for line in fh:
                s = line.strip()
                if s and not s.startswith("#"):
                    yield line

        reader = csv.reader(iter_lines(), delimiter=delim)
        first_row = [c.strip() for c in next(reader)]
        if len(first_row) < 2:
            raise ValueError(f"Expected at least 2 columns in {path}.")

        is_funtfbs = len(first_row) >= 4 and first_row[1].lower() == "regulates"
        src_idx, dst_idx = 0, (2 if is_funtfbs else 1)

        src_keys = {"source", "tf", "regulator", "from", "tf_id", "regulator_id"}
        dst_keys = {"target", "to", "target_id", "gene_id", "target_gene"}
        has_header = (not is_funtfbs) and (
            first_row[0].lower() in src_keys or first_row[1].lower() in dst_keys
        )

        if has_header:
            low = [c.lower() for c in first_row]
            for key in ("source", "tf", "regulator", "from", "tf_id", "regulator_id"):
                if key in low:
                    src_idx = low.index(key)
                    break
            for key in ("target", "to", "target_id", "target_gene"):
                if key in low:
                    dst_idx = low.index(key)
                    break
        else:
            src = clean_gene_id(first_row[src_idx])
            dst = clean_gene_id(first_row[dst_idx])
            if src and dst:
                raw_rows += 1
                edges.add((src, dst))

        for row in reader:
            if not row or len(row) <= max(src_idx, dst_idx):
                continue
            src = clean_gene_id(row[src_idx])
            dst = clean_gene_id(row[dst_idx])
            if not src or not dst:
                continue
            raw_rows += 1
            edges.add((src, dst))

    if not edges:
        raise ValueError(f"No valid directed edges parsed from {path}.")
    return edges, raw_rows


def read_id_map(path, source_col=None, target_col=None):
    forward = defaultdict(set)
    total_rows = 0
    kept_rows = 0

    with open_text(path) as fh:
        lines = [line for line in fh if line.strip() and not line.lstrip().startswith("#")]

    if not lines:
        raise ValueError(f"ID mapping file is empty: {path}.")

    delim = guess_sep(lines[0])
    reader = csv.reader(lines, delimiter=delim)
    first_row = [c.strip() for c in next(reader)]
    if len(first_row) < 2:
        raise ValueError(f"ID mapping file must have at least 2 columns: {path}.")

    low_first = [c.lower() for c in first_row]
    has_header = (
        source_col is not None
        or target_col is not None
        or any(tok in HEADER_KEYS for tok in low_first)
    )

    src_idx = 0
    dst_idx = 1

    if has_header:
        if source_col is not None:
            if source_col not in first_row:
                raise ValueError(f"Source column '{source_col}' not found in {first_row}.")
            src_idx = first_row.index(source_col)
        if target_col is not None:
            if target_col not in first_row:
                raise ValueError(f"Target column '{target_col}' not found in {first_row}.")
            dst_idx = first_row.index(target_col)
        rows = list(reader)
    else:
        rows = [first_row] + list(reader)

    for row in rows:
        if not row or len(row) <= max(src_idx, dst_idx):
            continue
        total_rows += 1
        old_id = clean_gene_id(row[src_idx])
        raw_dst = row[dst_idx].strip()
        if old_id.lower() in NA_TOKENS or raw_dst.lower() in NA_TOKENS:
            continue
        targets = []
        for tok in raw_dst.replace(";", ",").split(","):
            gid = clean_gene_id(tok)
            if gid and gid.lower() not in NA_TOKENS:
                targets.append(gid)
        if not targets:
            continue
        kept_rows += 1
        forward[old_id].update(targets)

    reverse = defaultdict(set)
    for old_id, targets in forward.items():
        for new_id in targets:
            reverse[new_id].add(old_id)

    pairs = {
        old_id: next(iter(targets))
        for old_id, targets in forward.items()
        if len(targets) == 1 and len(reverse[next(iter(targets))]) == 1
    }
    stats = {
        "mapping_rows_read": total_rows,
        "mapping_rows_considered": kept_rows,
        "unique_source_genes_in_map": len(forward),
        "one_to_one_pairs": len(pairs),
    }
    return pairs, stats


def get_parser():
    p = argparse.ArgumentParser(
        description="Map PlantRegMap FunTFBS regulatory interactions to an expression matrix"
    )
    p.add_argument("--interactions", required=True, help="PlantRegMap FunTFBS interaction file or edge list")
    p.add_argument("--transcripts", required=True, help="Expression matrix with sample IDs in the first column")
    p.add_argument("--transcript-sep", default="\t", help="Delimiter for --transcripts")
    p.add_argument("--id-map", default=None, help="Optional 1-to-1 gene ID mapping table")
    p.add_argument("--map-source-col", default=None, help="Source gene ID column in --id-map")
    p.add_argument("--map-target-col", default=None, help="Target gene ID column in --id-map")
    p.add_argument("--output", required=True, help="Output GRN CSV path")
    p.add_argument("--summary-json", default=None, help="Optional summary JSON path")
    return p


def main(argv=None):
    args = get_parser().parse_args(argv)
    interactions_path = Path(args.interactions)
    tx_path = Path(args.transcripts)
    id_map_path = Path(args.id_map) if args.id_map is not None else None
    out_path = Path(args.output)
    summary_path = Path(args.summary_json) if args.summary_json is not None else None

    genes = read_genes(tx_path, sep=args.transcript_sep)
    gene_set = set(genes)

    pub_edges, raw_rows = read_interactions(interactions_path)
    pub_nodes = {g for edge in pub_edges for g in edge}

    direct = pub_nodes & gene_set
    aliases = {}
    map_stats = {}
    if id_map_path is not None:
        aliases, map_stats = read_id_map(
            id_map_path,
            source_col=args.map_source_col,
            target_col=args.map_target_col,
        )

    node_map = {g: g for g in direct}
    for g in pub_nodes - direct:
        mapped = aliases.get(g)
        if mapped is not None and mapped in gene_set:
            node_map[g] = mapped

    dst_counts = Counter(node_map.values())
    collisions = sorted(src for src, dst in node_map.items() if dst_counts[dst] > 1)
    node_map = {src: dst for src, dst in node_map.items() if dst_counts[dst] == 1}

    kept_edges = sorted(
        {
            (node_map[src], node_map[dst])
            for src, dst in pub_edges
            if src in node_map and dst in node_map
        }
    )
    if not kept_edges:
        raise ValueError("Zero directed edges matched --transcripts; check gene IDs or --id-map.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["Source", "Target", "Importance"])
        for src, dst in kept_edges:
            writer.writerow([src, dst, "1.0"])

    tfs = sorted({src for src, _ in kept_edges})
    targets = sorted({dst for _, dst in kept_edges})
    connected = sorted({g for edge in kept_edges for g in edge})
    n_self = sum(1 for src, dst in kept_edges if src == dst)
    n_iso = len(genes) - len(connected)
    iso_pct = (100.0 * n_iso / len(genes)) if genes else 0.0

    if summary_path is not None:
        summary = {
            "interactions_input": str(interactions_path),
            "transcripts_input": str(tx_path),
            "id_map_input": str(id_map_path) if id_map_path is not None else None,
            "output_csv": str(out_path),
            "raw_interaction_rows": raw_rows,
            "unique_public_edges": len(pub_edges),
            "unique_public_nodes": len(pub_nodes),
            "direct_matched_nodes": len(direct),
            "mapped_public_nodes": len(node_map),
            "collision_nodes_excluded": len(collisions),
            "expressed_genes": len(genes),
            "tfs": len(tfs),
            "target_genes": len(targets),
            "connected_nodes": len(connected),
            "isolated_genes": n_iso,
            "isolated_genes_pct": round(iso_pct, 2),
            "directed_edges": len(kept_edges),
            "autoregulatory_edges": n_self,
            "uniform_importance": 1.0,
            **map_stats,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"Saved {len(kept_edges):,} edges to {out_path} "
        f"(genes={len(genes):,}, tfs={len(tfs):,}, connected={len(connected):,}, "
        f"isolated={n_iso:,} [{iso_pct:.2f}%], self={n_self:,})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
