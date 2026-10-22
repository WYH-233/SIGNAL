from __future__ import annotations

import argparse
import os

import bioframe
import cooler
import pandas as pd
from cooltools import insulation


def extract_TADs(insulation_table, is_boundary_col, max_TAD_length=100_000_000):
    tads = bioframe.merge(insulation_table[insulation_table[is_boundary_col] == False])
    return tads[(tads["end"] - tads["start"]) <= max_TAD_length].reset_index(drop=True)[['chrom', 'start', 'end']]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call TAD boundaries from .cool via cooltools insulation.")
    parser.add_argument("--cool-file", type=str, required=True, help="Path to .cool file")
    parser.add_argument("--output-dir", type=str, default="./mcooltools_ouput", help="Output directory")
    parser.add_argument("--resolution", type=int, default=5000, help="Resolution in bp")
    parser.add_argument("--window-multiplier", type=int, default=20, help="Window = multiplier * resolution")
    args = parser.parse_args()

    resolution = args.resolution
    windows = [args.window_multiplier * resolution]

    print(f"Loading: {args.cool_file}")
    clr = cooler.Cooler(args.cool_file)
    insulation_table = insulation(clr, windows, verbose=True)

    first_window_summary = insulation_table.columns[[str(windows[-1]) in i for i in insulation_table.columns]]
    columns_to_save = ['chrom', 'start', 'end', 'region', 'is_bad_bin'] + list(first_window_summary)
    subset_table = insulation_table[columns_to_save]
    subset_table.to_csv(f"resolution_{resolution}_w_{windows[-1]}_insulation_table.csv", index=False)

    os.makedirs(args.output_dir, exist_ok=True)
    for w in windows:
        is_boundary_col = f'is_boundary_{w}'
        tads_table = extract_TADs(insulation_table, is_boundary_col)
        output_path = os.path.join(args.output_dir, f"TADs_{resolution}_window_{w}.bed")
        tads_table.to_csv(output_path, sep='\t', header=False, index=False)
        print(f"Saved TADs: {output_path}")

    print("Done.")
