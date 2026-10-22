from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def invert_dod_to_strength(input_file: Path, output_file: Path) -> None:
    df = pd.read_csv(input_file, sep="\t")

    required = ["ChromID", "Start", "End", "DoD"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"输入文件缺少字段: {missing}")

    max_dod = df["DoD"].dropna().max()
    if np.isnan(max_dod):
        raise ValueError("DoD 列全为 NA，无法计算 strength。")

    print(f"max(DoD) = {max_dod:.4f}")

    df["strength"] = df["DoD"].apply(
        lambda d: 0.0 if pd.isna(d) else max_dod - d
    )


    df["chr"] = df["ChromID"].astype(str)
    if not df["chr"].iloc[0].startswith("chr"):
        df["chr"] = "chr" + df["chr"]

    bed_df = df[["chr", "Start", "End", "strength"]].copy()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    bed_df.to_csv(output_file, sep="\t", header=False, index=False)
    print(f"Saved: {output_file} ({bed_df.shape[0]} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert mdknn DoD table to TAD strength BED."
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="mdknn DoD output file (already chr-prefixed, e.g. mdknn_*_chr.txt)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output TAD strength BED file (e.g. mdknn_*_inverted_strength_compAB.bed)",
    )
    args = parser.parse_args()
    invert_dod_to_strength(args.input, args.output)


if __name__ == "__main__":
    main()
