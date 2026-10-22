from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUILD_INPUTS = ROOT / "input_preprocessing" / "build_model_inputs" / "01_prepare_signal_model_inputs.py"
TRAIN_MODEL = ROOT / "model_training" / "train_signal_model.py"
EXPORT_LOOPS = ROOT / "output_generation" / "export_high_group_loops.py"


def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metacell-matrix", type=Path, required=True,
                        help="Formatted accessibility matrix (CSV; rows are genomic peaks)")
    parser.add_argument("--merged-loops", type=Path, required=True,
                        help="Merged co-accessibility interactions (tab-separated)")
    parser.add_argument("--hic-template", required=True,
                        help="Per-chromosome Hi-C contact path template; supports {chrom} and {chrom_id}")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for graph inputs")
    parser.add_argument("--chromosomes", default="autosome_x",
                        help="Comma-separated chromosomes, e.g. chr1,chr2,chrX; default: autosome_x")
    parser.add_argument("--device", default="cpu", help="Input-building device, e.g. cpu or cuda:0")
    parser.add_argument("--binsize", type=int, default=5000, help="Hi-C bin size in bp")


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", type=Path, required=True, help="Per-chromosome graph-input directory")
    parser.add_argument("--tad-strength-bed", type=Path, required=True, help="TAD-strength BED file")
    parser.add_argument("--outdir", type=Path, required=True, help="Model output directory")
    parser.add_argument("--chromosomes", default="autosome_x",
                        help="Comma-separated chromosomes, e.g. chr1,chr2,chrX; default: autosome_x")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--early-stop-window", type=int, default=100,
                        help="Epochs without a new lowest training loss before stopping; 0 disables early stopping")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu or cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--edge-dim", type=int, default=3)
    parser.add_argument("--projection-dim", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--mask-prob-1", type=float, default=0.2)
    parser.add_argument("--mask-prob-2", type=float, default=0.5)
    parser.add_argument("--zscore-edge-features", action="store_true",
                        help="Apply z-score normalization to edge features")


def add_export_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, required=True, help="Cached graph directory from training")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint to export")
    parser.add_argument("--outdir", type=Path, required=True, help="Loop score and group output directory")
    parser.add_argument("--chromosomes", default="autosome_x",
                        help="Comma-separated chromosomes, e.g. chr1,chr2,chrX; default: autosome_x")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda:0")
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--edge-dim", type=int, default=3)
    parser.add_argument("--projection-dim", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--kmeans-seed", type=int, default=42)


def command_for_build(args: argparse.Namespace) -> list[str]:
    return [sys.executable, str(BUILD_INPUTS),
            "--metacell-matrix", str(args.metacell_matrix),
            "--merged-loops", str(args.merged_loops),
            "--hic-template", args.hic_template,
            "--outdir", str(args.outdir),
            "--chromosomes", args.chromosomes,
            "--device", args.device,
            "--binsize", str(args.binsize)]


def command_for_train(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(TRAIN_MODEL),
               "--input-dir", str(args.input_dir),
               "--tad-strength-bed", str(args.tad_strength_bed),
               "--outdir", str(args.outdir),
               "--chromosomes", args.chromosomes,
               "--epochs", str(args.epochs),
               "--early-stop-window", str(args.early_stop_window),
               "--save-every", str(args.save_every),
               "--device", args.device,
               "--seed", str(args.seed),
               "--lr", str(args.lr),
               "--weight-decay", str(args.weight_decay),
               "--input-dim", str(args.input_dim),
               "--hidden-dim", str(args.hidden_dim),
               "--output-dim", str(args.output_dim),
               "--heads", str(args.heads),
               "--dropout", str(args.dropout),
               "--edge-dim", str(args.edge_dim),
               "--projection-dim", str(args.projection_dim),
               "--temperature", str(args.temperature),
               "--mask-prob-1", str(args.mask_prob_1),
               "--mask-prob-2", str(args.mask_prob_2)]
    if args.zscore_edge_features:
        command.append("--zscore-edge-features")
    return command


def command_for_export(args: argparse.Namespace) -> list[str]:
    return [sys.executable, str(EXPORT_LOOPS),
            "--cache-dir", str(args.cache_dir),
            "--checkpoint", str(args.checkpoint),
            "--outdir", str(args.outdir),
            "--chromosomes", args.chromosomes,
            "--device", args.device,
            "--input-dim", str(args.input_dim),
            "--hidden-dim", str(args.hidden_dim),
            "--output-dim", str(args.output_dim),
            "--heads", str(args.heads),
            "--dropout", str(args.dropout),
            "--edge-dim", str(args.edge_dim),
            "--projection-dim", str(args.projection_dim),
            "--temperature", str(args.temperature),
            "--kmeans-seed", str(args.kmeans_seed)]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def latest_checkpoint(model_dir: Path, max_epoch: int) -> Path:
    checkpoints = []
    for path in model_dir.glob("model_epoch_*.pt"):
        match = re.fullmatch(r"model_epoch_(\d+)\.pt", path.name)
        if match and int(match.group(1)) <= max_epoch:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        raise FileNotFoundError(f"No model_epoch_*.pt checkpoint found in {model_dir}")
    return max(checkpoints, key=lambda item: item[0])[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="SIGNAL.py",
        description="Run SIGNAL to prepare graph inputs, train the model and export chromatin-loop groups.",
        epilog="Example: python SIGNAL.py run --input-dir graph_inputs --tad-strength-bed tad_strength.bed --outdir results --epochs 3000",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-inputs", help="Build per-chromosome graph inputs")
    add_build_arguments(build_parser)

    train_parser = subparsers.add_parser("train", help="Train the graph-attention model")
    add_train_arguments(train_parser)

    export_parser = subparsers.add_parser("export", help="Export attention scores and high-group loops")
    add_export_arguments(export_parser)

    run_parser = subparsers.add_parser("run", help="Train the model and export loops in one command")
    add_train_arguments(run_parser)
    run_parser.add_argument("--kmeans-seed", type=int, default=42)
    run_parser.add_argument("--checkpoint-epoch", type=int, default=None,
                            help="Checkpoint epoch to export; default: latest saved epoch")
    run_parser.add_argument("--export-device", default=None,
                            help="Inference device; defaults to --device")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-inputs":
        run(command_for_build(args))
    elif args.command == "train":
        run(command_for_train(args))
    elif args.command == "export":
        args.outdir.mkdir(parents=True, exist_ok=True)
        run(command_for_export(args))
    elif args.command == "run":
        run(command_for_train(args))
        model_dir = args.outdir / "saved_models"
        training_log = (model_dir / "train_log.txt").read_text(encoding="utf-8")
        completed_epochs = re.findall(r"Epoch\s+(\d+),\s+Avg Loss:", training_log)
        if not completed_epochs:
            raise RuntimeError("Training log contains no completed epochs")
        last_epoch = int(completed_epochs[-1])
        if args.checkpoint_epoch is None:
            checkpoint = latest_checkpoint(model_dir, last_epoch)
        else:
            if args.checkpoint_epoch > last_epoch:
                raise ValueError(f"Requested checkpoint exceeds the last completed training epoch ({last_epoch})")
            checkpoint = model_dir / f"model_epoch_{args.checkpoint_epoch:03d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Requested checkpoint does not exist: {checkpoint}")
        export_args = argparse.Namespace(
            cache_dir=args.outdir / "cached_graph_data",
            checkpoint=checkpoint,
            outdir=args.outdir / "high_groups",
            chromosomes=args.chromosomes,
            device=args.export_device or args.device,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            heads=args.heads,
            dropout=args.dropout,
            edge_dim=args.edge_dim,
            projection_dim=args.projection_dim,
            temperature=args.temperature,
            kmeans_seed=args.kmeans_seed,
        )
        export_args.outdir.mkdir(parents=True, exist_ok=True)
        run(command_for_export(export_args))


if __name__ == "__main__":
    main()
