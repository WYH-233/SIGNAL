from __future__ import annotations

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Convert ArchR co-accessibility CSV to BEDPE.")
    parser.add_argument("--sample", "-s", type=str, default="A549_C3", help="Sample name")
    parser.add_argument("--coaccess-csv", type=str, default=None,
                        help="ArchR co-accessibility CSV (default: {sample}_archr_coaccessibility_0_1mb.csv)")
    parser.add_argument("--peaks-bed", type=str, default=None,
                        help="Peaks BED file (default: {sample}_peaks.bed)")
    parser.add_argument("--output", "-o", type=str,
                        default="archr_interacting_peaks_0_1mb_with_correlation.bed",
                        help="Output BEDPE file")
    args = parser.parse_args()

    coaccess_csv = args.coaccess_csv or f"{args.sample}_archr_coaccessibility_0_1mb.csv"
    peaks_bed = args.peaks_bed or f"{args.sample}_peaks.bed"


    peaks = []
    with open(peaks_bed, 'r') as f:
        for line in f:
            chrom, start, end = line.strip().split('\t')
            peaks.append({
                'chrom': chrom,
                'start': int(start),
                'end': int(end),
                'mid': (int(start) + int(end)) // 2
            })

    mid_to_peak = {}
    for peak in peaks:
        mid_to_peak[(peak['chrom'], peak['mid'])] = peak


    interactions = pd.read_csv(coaccess_csv)
    results = []
    for _, row in interactions.iterrows():
        chrom = row['seqnames']
        mid1 = row['start']
        mid2 = row['end']
        corr = row['correlation']

        peak1 = mid_to_peak.get((chrom, mid1))
        peak2 = mid_to_peak.get((chrom, mid2))

        if peak1 and peak2 and corr > 0:
            results.append([
                peak1['chrom'], peak1['start'], peak1['end'],
                peak2['chrom'], peak2['start'], peak2['end'],
                corr
            ])

    with open(args.output, 'w') as f:
        for row in results:
            f.write('\t'.join(map(str, row)) + '\n')

    print(f"转换完成: {len(results)} 对 (correlation > 0), 保存至 {args.output}")


if __name__ == "__main__":
    main()
