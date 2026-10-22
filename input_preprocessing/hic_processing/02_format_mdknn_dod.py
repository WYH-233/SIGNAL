from __future__ import annotations

import argparse

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add chr prefix to mdknn DoD ChromID.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input DoD file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output DoD file with chr prefix")
    args = parser.parse_args()

    print(f"Reading: {args.input}")
    df = pd.read_csv(args.input, sep="\t")

    print("Adding 'chr' prefix to ChromID ...")
    df["ChromID"] = "chr" + df["ChromID"].astype(str)

    print(f"Saving: {args.output}")
    df.to_csv(args.output, sep="\t", index=False)
    print("Done.")
