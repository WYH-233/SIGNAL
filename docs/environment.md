# Software environment

The signal-model entry point combines Python and PyTorch Geometric. Optional input-preparation scripts also require R/Bioconductor and command-line tools. Record exact tested versions for the final manuscript release; this package does not yet contain a fully locked environment.

## Python

Install PyTorch and PyTorch Geometric builds appropriate for the local CPU/CUDA system. Then install `requirements.txt`. Keep SciPy below 1.12 while `input_preprocessing/tools/mdknn_1.py` imports the removed `scipy.stats.itemfreq` API.

## R/Bioconductor

For the input-preparation stages, install: `ArchR`, `cicero`, `monocle3`, `Signac`, `Seurat`, `Matrix`, `SummarizedExperiment`, `GenomicRanges`, `IRanges`, `RANN`, `data.table` and `dplyr`.

## Command-line tools

- `hic2cool` and `cooler` for `.hic` conversion and matrix balancing.
- `macs2` when invoked by ArchR peak calling.
