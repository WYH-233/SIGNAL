from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model_training" / "model_core"))
from model_k_2 import Encoder, GRACE


def chromosomes_from_arg(value: str) -> list[str]:
    if value == "autosome_x":
        return [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    return [x if x.startswith("chr") else f"chr{x}" for x in value.split(",")]


def export_attention(cache_dir: Path, checkpoint: Path, outdir: Path, chromosomes: list[str], device: torch.device,
                     input_dim: int, hidden_dim: int, output_dim: int, heads: int, dropout: float,
                     edge_dim: int, projection_dim: int, temperature: float) -> list[Path]:
    encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                      heads_num=heads, dropout=dropout, edge_dim=edge_dim)
    model = GRACE(encoder=encoder, num_hidden=output_dim,
                  num_proj_hidden=projection_dim, tau=temperature).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    epoch = int("".join([c for c in checkpoint.stem if c.isdigit()]) or 0)

    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = outdir / f"weights_combined_{epoch}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    attention_files = []
    for chrom in chromosomes:
        cache = torch.load(cache_dir / f"{chrom}_graph.pt", map_location="cpu", weights_only=False)
        data = cache["data"]
        idx2region = cache["idx2region"]
        with torch.no_grad():
            non_self = data.edge_index[0] != data.edge_index[1]
            edge_index = data.edge_index[:, non_self]
            if edge_index.shape[1] == 0:
                raise ValueError(f"No non-self edges available for {chrom}")
            new_data = Data(x=data.x.float(), edge_index=edge_index, edge_attr=data.edge_attr[non_self].float())
            model(new_data.to(device))
            unnorm1 = model.encoder.unnormalized_att_weights_first
            unnorm2 = model.encoder.unnormalized_att_weights_second
            unnorm_mean = (unnorm1 + unnorm2) / 2
            scores = unnorm_mean.mean(dim=1).cpu().numpy()
            edges = edge_index.cpu().numpy()
            raw = pd.DataFrame({
                "region1": [idx2region[int(i)] for i in edges[0]],
                "region2": [idx2region[int(i)] for i in edges[1]],
                "attention": scores,
            })
            raw.to_csv(raw_dir / f"{chrom}_attention_score.csv", sep="\t", index=False, header=False)
            bedpe = attention_to_bedpe(raw)
            attention_path = outdir / f"{chrom}_attention_score_true.bedpe"
            bedpe.to_csv(attention_path, sep="\t", index=False, header=False)
            attention_files.append(attention_path)
    return attention_files


def attention_to_bedpe(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("No non-self edges available for attention export")
    if not np.isfinite(raw["attention"].to_numpy(dtype=float)).all():
        raise ValueError("Attention scores must be finite")
    result = pd.DataFrame(index=raw.index)
    for anchor in (1, 2):
        regions = raw[f"region{anchor}"].astype(str).str.extract(r"^([^:]+):(\d+)-(\d+)$")
        if regions.isna().any().any():
            raise ValueError(f"Invalid region{anchor} ID; expected chrom:start-end")
        result[f"chr{anchor}"] = regions[0]
        result[f"start{anchor}"] = regions[1].astype("int64")
        result[f"end{anchor}"] = regions[2].astype("int64")
        if (result[f"end{anchor}"] <= result[f"start{anchor}"]).any():
            raise ValueError("Region end must exceed region start")
    result["attention"] = raw["attention"].to_numpy()
    return result


def normalize_and_cluster(attention_dir: Path, outdir: Path, random_state: int = 42,
                          attention_files: list[Path] | None = None) -> None:
    paths = sorted(attention_dir.glob("chr*_attention_score_true.bedpe")) if attention_files is None else attention_files
    if not paths:
        raise FileNotFoundError(f"No attention BEDPE files found in {attention_dir}")
    norm_dir = outdir / "max_min_norm"
    kmeans_dir = norm_dir / "kmeans_2_groups"
    high_dir = kmeans_dir / "high_groups"
    norm_dir.mkdir(parents=True, exist_ok=True)
    kmeans_dir.mkdir(parents=True, exist_ok=True)
    high_dir.mkdir(parents=True, exist_ok=True)

    all_grouped = []
    all_high = []
    for path in paths:
        cols = ["chr1", "start1", "end1", "chr2", "start2", "end2", "attention"]
        df = pd.read_csv(path, sep="\t", names=cols, header=None)
        if len(df) < 2 or df["attention"].nunique() < 2:
            raise ValueError(f"Cannot form two attention groups from {path}: at least two distinct scores are required")
        if not np.isfinite(df["attention"].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite attention scores in {path}")
        min_score = df["attention"].min()
        max_score = df["attention"].max()
        df["attention"] = (df["attention"] - min_score) / (max_score - min_score + 1e-12)
        norm_file = norm_dir / path.name.replace(".bedpe", "_norm.bedpe")
        df.to_csv(norm_file, sep="\t", index=False, header=False)

        grouped = df.copy()
        labels = KMeans(n_clusters=2, random_state=random_state, n_init=10).fit_predict(grouped[["attention"]].values)
        grouped["group"] = labels
        means = grouped.groupby("group")["attention"].mean().sort_values()
        label_map = {means.index[0]: "Low", means.index[1]: "High"}
        grouped["group"] = grouped["group"].map(label_map)
        chrom = path.name.split("_")[0]
        grouped.to_csv(kmeans_dir / f"{chrom}_kmeans2_groups.bedpe", sep="\t", index=False, header=False)
        high = grouped[grouped["group"] == "High"].iloc[:, :-1].copy()
        high.to_csv(high_dir / f"{chrom}_high_group.bedpe", sep="\t", index=False, header=False)
        all_grouped.append(grouped)
        all_high.append(high)

    if all_grouped:
        pd.concat(all_grouped).to_csv(kmeans_dir / "all_chromosomes_kmeans2_groups.bedpe", sep="\t", index=False, header=False)
    if all_high:
        pd.concat(all_high).to_csv(high_dir / "all_chromosomes_high_groups.bedpe", sep="\t", index=False, header=False)


def main():
    parser = argparse.ArgumentParser(description="Export attention and high group loops.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--chromosomes", default="autosome_x")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--edge-dim", type=int, default=3)
    parser.add_argument("--projection-dim", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--kmeans-seed", type=int, default=42)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    chromosomes = chromosomes_from_arg(args.chromosomes)
    attention_files = export_attention(args.cache_dir, args.checkpoint, args.outdir, chromosomes, device,
                     args.input_dim, args.hidden_dim, args.output_dim, args.heads,
                     args.dropout, args.edge_dim, args.projection_dim, args.temperature)
    normalize_and_cluster(args.outdir, args.outdir, random_state=args.kmeans_seed,
                          attention_files=attention_files)


if __name__ == "__main__":
    main()
