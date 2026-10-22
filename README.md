# SIGNAL

# SIGNAL

SIGNAL is an unsupervised graph contrastive learning framework for inferring cis-regulatory interactions by integrating single-cell chromatin accessibility with multi-scale three-dimensional chromatin organization. SIGNAL constructs a peak-to-peak interaction graph using scATAC-seq-derived accessibility and co-accessibility information together with Hi-C contact frequency. TAD organization, quantified using the degree of disorder (DoD), guides graph augmentation. A shared edge-aware graph attention network (EGAT) is trained using an InfoNCE objective, and candidate interactions are subsequently prioritized using attention-derived scores to generate a cis-regulatory network.

## Repository layout

| Path | Contents |
|---|---|
| `input_preprocessing/` | Hi-C/scATAC-seq preprocessing, MD-KNN and per-chromosome graph-input construction |
| `model_training/` | GRACE/GATv2 model definition and training entry point |
| `output_generation/` | Attention-score export and high/low loop grouping |
| `SIGNAL.py` | Parameterized command-line interface for input construction, training, export and combined runs |
| `data/README.md` | Input-data manifest; data files are not bundled |
| `docs/` | Workflow and environment notes |

## Requirements and installation

The model and command-line interface use Python. PyTorch and PyTorch Geometric must be installed with builds compatible with the target CPU/CUDA platform; the remaining Python packages are listed in [`requirements.txt`](requirements.txt). Some input-preparation stages additionally require R/Bioconductor packages and command-line tools. The full dependency groups and current compatibility notes are in [`docs/environment.md`](docs/environment.md).

```bash
git clone <repository-url>
cd <repository-directory>
# Install a platform-compatible PyTorch and PyTorch Geometric build first.
python -m pip install -r requirements.txt
python SIGNAL.py --help
```

The Python/R dependency versions have not yet been frozen into a complete tested environment. Record the versions used for the final manuscript release and publish an environment lock file or container definition with that release.

The MD-KNN implementation in `input_preprocessing/tools/mdknn_1.py` uses `scipy.stats.itemfreq`, removed in SciPy 1.12. The environment therefore pins `scipy<1.12` unless that implementation is updated.

## Running the algorithm

Clone the repository and prepare the inputs listed in `data/README.md`. The main entry point accepts file paths, chromosome selection and model settings as command-line parameters. First build per-chromosome graph inputs:

```bash
python SIGNAL.py build-inputs \
  --metacell-matrix /path/to/metacell_accessibility.csv \
  --merged-loops /path/to/merged_accessibility_links.tsv \
  --hic-template '/path/to/hic_results_{chrom_id}_5000bp_observed_VC.txt' \
  --outdir /path/to/graph_inputs \
  --chromosomes chr1,chr2,chrX \
  --binsize 5000 \
  --device cpu
```

Then train the signal model and export high-group loops in one command:

```bash
python SIGNAL.py run \
  --input-dir /path/to/graph_inputs \
  --tad-strength-bed /path/to/tad_strength.bed \
  --outdir /path/to/model_output \
  --chromosomes chr1,chr2,chrX \
  --epochs 3000 \
  --save-every 10 \
  --lr 0.0001 \
  --weight-decay 0.0005 \
  --mask-prob-1 0.2 \
  --mask-prob-2 0.5 \
  --input-dim 256 \
  --hidden-dim 32 \
  --output-dim 16 \
  --seed 42 \
  --kmeans-seed 42 \
  --device cpu
```

`python SIGNAL.py --help` lists all stages. `build-inputs`, `train` and `export` are also available separately; `run` selects the latest saved checkpoint by default. Use `--checkpoint-epoch` to export a specific epoch. Model dimensions, contrastive mask probabilities, optimizer settings, random seeds and Hi-C bin size are exposed as arguments. Full-data runs can require substantial memory, disk space and compute time. GPU use is optional; choose a PyTorch device supported by the installed build.

| Stage | Main user-supplied parameters |
|---|---|
| `build-inputs` | Accessibility matrix, merged links, Hi-C path template, output directory, chromosomes, Hi-C bin size and device |
| `train` | Graph-input directory, TAD-strength BED, output directory, chromosomes, epochs, checkpoint interval, learning rate, weight decay, seed, graph dimensions and view-mask probabilities |
| `export` | Graph cache, checkpoint, output directory, chromosomes, model dimensions, device and K-means seed |
| `run` | All `train` parameters plus an optional checkpoint epoch for export |

The CLI begins with prepared accessibility/link tables and chromosome-wise Hi-C contacts, or with graph-input files already built. Input schemas are described in [`data/README.md`](data/README.md). The included R scripts in `input_preprocessing/` provide upstream ArchR/Cicero processing components; those steps currently expect their documented working-directory file layout.

The combined `run` command writes model checkpoints and graph caches under `<outdir>/saved_models/` and `<outdir>/cached_graph_data/`; loop scores and high-group BEDPE tables are written under `<outdir>/high_groups/`.


