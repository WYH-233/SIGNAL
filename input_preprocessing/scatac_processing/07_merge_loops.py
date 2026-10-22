from __future__ import annotations

import argparse

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Cicero and ArchR peak interaction loops.")
    parser.add_argument("--cicero-bed", type=str, default="cicero_0_interacting_peaks_1mb_0922_one_direction.bed",
                        help="Cicero interacting peaks BED")
    parser.add_argument("--archr-bed", type=str, default="archr_interacting_peaks_0_1mb_with_correlation.bed",
                        help="ArchR interacting peaks BED")
    parser.add_argument("--output", "-o", type=str, default="merged_interacting_peaks_with_scores.bed",
                        help="Output merged BED path")
    args = parser.parse_args()

    cicero_df = pd.read_csv(args.cicero_bed, sep='\t', header=None,
                            names=['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'cicero_score'])
    archr_df = pd.read_csv(args.archr_bed, sep='\t', header=None,
                           names=['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'archr_score'])

    merged_df = pd.merge(cicero_df, archr_df,
                         on=['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2'],
                         how='outer')

    merged_df['cicero_score'] = merged_df['cicero_score'].fillna(0)
    merged_df['archr_score'] = merged_df['archr_score'].fillna(0)

    merged_df.to_csv(args.output, sep='\t', header=True, index=False)
    print(f"合并完成，结果已保存到 {args.output}（包含列名）")
