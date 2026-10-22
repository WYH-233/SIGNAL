from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.stats import zscore


def chromosomes_from_arg(value: str) -> list[str]:
    if value == "autosome_x":
        return [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    return [x if x.startswith("chr") else f"chr{x}" for x in value.split(",")]


def peak_to_bin_index(start: int, end: int, binsize: int) -> int:
    return ((int(start) + int(end)) // 2) // binsize


def load_hic_matrix(hic_file: Path, device: torch.device, binsize: int):
    df = pd.read_csv(hic_file, sep="\t", header=None, names=["bin1", "bin2", "value"])
    max_bin = max(df["bin1"].max(), df["bin2"].max())
    num_bins = max_bin // binsize + 1
    row = (df["bin1"] // binsize).values
    col = (df["bin2"] // binsize).values
    val = df["value"].values
    mat = coo_matrix((val, (row, col)), shape=(num_bins, num_bins))
    mat = mat + mat.T
    mat.setdiag(mat.diagonal() / 2)
    mat = mat.tocoo()
    dense = torch.zeros((num_bins, num_bins), dtype=torch.float32, device=device)
    dense[torch.tensor(mat.row, device=device), torch.tensor(mat.col, device=device)] = torch.tensor(
        mat.data, dtype=torch.float32, device=device
    )
    return dense


def prepare_chromosome(
    chrom: str,
    metacell_matrix: Path,
    merged_loops: Path,
    hic_template: str,
    outdir: Path,
    device: torch.device,
    binsize: int,
) -> None:
    chrom_dir = outdir / chrom
    chrom_dir.mkdir(parents=True, exist_ok=True)

    access = pd.read_csv(metacell_matrix, index_col=0)
    chrom_access = access[access.index.str.startswith(f"{chrom}:")].copy()
    access_file = chrom_dir / f"metacell_accessibility_matrix_{chrom}.csv"
    zscore_file = chrom_dir / f"metacell_accessibility_matrix_{chrom}_zscore.csv"
    chrom_access.to_csv(access_file)
    chrom_access.apply(zscore, nan_policy="omit").to_csv(zscore_file)

    loops = pd.read_csv(merged_loops, sep="\t", header=0)
    chrom_loops = loops[(loops["chr1"] == chrom) & (loops["chr2"] == chrom)].copy()
    interaction_file = chrom_dir / f"{chrom}_interactions.bed"
    strength_file = chrom_dir / f"{chrom}_interactions_with_hic_strength_vc.bed"
    chrom_loops.to_csv(interaction_file, sep="\t", index=False)

    chrom_id = chrom.replace("chr", "")
    hic_file = Path(hic_template.format(chrom=chrom, chrom_id=chrom_id))
    hic = load_hic_matrix(hic_file, device=device, binsize=binsize)
    bin1 = torch.tensor([peak_to_bin_index(s, e, binsize) for s, e in zip(chrom_loops["start1"], chrom_loops["end1"])], device=device)
    bin2 = torch.tensor([peak_to_bin_index(s, e, binsize) for s, e in zip(chrom_loops["start2"], chrom_loops["end2"])], device=device)
    values = torch.zeros(len(chrom_loops), dtype=torch.float32, device=device)
    valid = (bin1 < hic.shape[0]) & (bin2 < hic.shape[1])
    values[valid] = hic[bin1[valid], bin2[valid]]
    chrom_loops["hic_oe"] = values.cpu().numpy()
    chrom_loops["hic_observed"] = values.cpu().numpy()
    chrom_loops.to_csv(strength_file, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description="Prepare per-chromosome signal model inputs.")
    parser.add_argument("--metacell-matrix", type=Path, required=True)
    parser.add_argument("--merged-loops", type=Path, required=True)
    parser.add_argument("--hic-template", required=True, help="Use {chrom_id} or {chrom}, e.g. ..._{chrom_id}_5000bp_observed_VC.txt")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--chromosomes", default="autosome_x")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--binsize", type=int, default=5000, help="Hi-C bin size in bp")
    args = parser.parse_args()

    if args.binsize < 1:
        parser.error("--binsize must be greater than 0")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    for chrom in chromosomes_from_arg(args.chromosomes):
        print(f"[prepare] {chrom}")
        prepare_chromosome(chrom, args.metacell_matrix, args.merged_loops, args.hic_template,
                           args.outdir, device, args.binsize)


if __name__ == "__main__":
    main()
