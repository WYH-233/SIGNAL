from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import hicstraw


def process_chromosome(args):
    chrom, hicfile, res, output_dir = args
    print(f"Processing {chrom} ...")
    result = hicstraw.straw('observed', 'VC', hicfile, chrom, chrom, 'BP', res)
    output_filename = os.path.join(output_dir, f"hic_results_{chrom}_{res}bp_observed_VC.txt")
    with open(output_filename, "w") as result_file:
        for item in result:
            result_file.write(f"{item.binX}\t{item.binY}\t{item.counts}\n")
    return f"Finished {chrom}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split .hic observed/VC by chromosome.")
    parser.add_argument("--hic-file", type=str, required=True, help="Path to .hic file")
    parser.add_argument("--output-dir", type=str, default="./chr_observed_VC",
                        help="Output directory for per-chromosome files")
    parser.add_argument("--res", type=int, default=5000, help="Resolution in bp")
    args = parser.parse_args()

    hicfile = args.hic_file
    output_dir = args.output_dir
    res = args.res
    os.makedirs(output_dir, exist_ok=True)

    hic = hicstraw.HiCFile(hicfile)
    hic_chroms = [chrom.name for chrom in hic.getChromosomes()]
    print("Chromosomes inside hic:")
    print(hic_chroms)

    has_chr = any(c.startswith("chr") for c in hic_chroms)
    if has_chr:
        chromosomes_to_process = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    else:
        chromosomes_to_process = [str(i) for i in range(1, 23)] + ["X", "Y"]

    chromosomes_to_process = [c for c in chromosomes_to_process if c in hic_chroms]
    print("\nChromosomes to process:")
    print(chromosomes_to_process)

    if len(chromosomes_to_process) == 0:
        raise ValueError("No standard chromosomes found in hic file.")

    n_processes = min(8, len(chromosomes_to_process))
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_chromosome, [(chrom, hicfile, res, output_dir) for chrom in chromosomes_to_process])

    print("\nAll done!")
    for result in results:
        print(result)
