from __future__ import annotations

import argparse
import csv


def anchor_key(chrom, start, end):
    chrom_clean = chrom.lower().replace("chr", "")
    chrom_num = int(chrom_clean) if chrom_clean.isdigit() else {"x": 23, "y": 24, "m": 25}.get(chrom_clean, 1000)
    return (chrom_num, int(start), int(end))


def main():
    parser = argparse.ArgumentParser(description="Convert Cicero CSV to filtered one-direction BEDPE.")
    parser.add_argument("--input", "-i", type=str, default="cicero_output/cicero_connections_1mb.csv",
                        help="Input Cicero CSV")
    parser.add_argument("--output", "-o", type=str,
                        default="cicero_0_interacting_peaks_1mb_one_direction.bed",
                        help="Output BEDPE file")
    args = parser.parse_args()

    rows = []
    with open(args.input, "r") as fin:
        reader = csv.reader(fin)
        next(reader)
        for row in reader:
            peak1 = row[0].strip('"').split("_")
            peak2 = row[1].strip('"').split("_")
            coaccess = float(row[2].strip('"'))

            if coaccess <= 0:
                continue
            if len(peak1) != 3 or len(peak2) != 3:
                print(f"忽略格式异常的行: {row}")
                continue

            chrom1, start1, end1 = peak1
            chrom2, start2, end2 = peak2


            if anchor_key(chrom1, start1, end1) < anchor_key(chrom2, start2, end2):
                rows.append([chrom1, start1, end1, chrom2, start2, end2, str(coaccess)])

    with open(args.output, "w") as fout:
        for row in rows:
            fout.write("\t".join(row) + "\n")

    print(f"转换完成: {len(rows)} 条 (coaccess > 0, 单向), 保存至 {args.output}")


if __name__ == "__main__":
    main()
