from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent / "model_core"))
from graph_utils import build_view, generate_mask_tad_strength, load_graph_data
from model_k_2 import Encoder, GRACE


def chromosomes_from_arg(value: str) -> list[str]:
    if value == "autosome_x":
        return [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    return [x if x.startswith("chr") else f"chr{x}" for x in value.split(",")]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    parser = argparse.ArgumentParser(description="Train signal model from prepared chromosome graphs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tad-strength-bed", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--chromosomes", default="autosome_x")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--early-stop-window", type=int, default=100,
                        help="Stop after this many epochs without a new lowest training loss; 0 disables early stopping")
    parser.add_argument("--device", default="cuda:0")
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
    parser.add_argument("--mask-prob-1", type=float, default=0.2,
                        help="Edge-mask probability for the first contrastive view")
    parser.add_argument("--mask-prob-2", type=float, default=0.5,
                        help="Edge-mask probability for the second contrastive view")
    parser.add_argument("--zscore-edge-features", action="store_true", default=False,
                        help="Whether to z-score normalize edge features (default: False)")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.early_stop_window < 0:
        parser.error("--early-stop-window must be nonnegative")
    if args.save_every < 1:
        parser.error("--save-every must be at least 1")
    if not 0 <= args.mask_prob_1 <= 1 or not 0 <= args.mask_prob_2 <= 1:
        parser.error("mask probabilities must be between 0 and 1")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than 0")

    set_random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    save_dir = args.outdir / "saved_models"
    cache_dir = args.outdir / "cached_graph_data"
    save_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=save_dir / "train_log.txt",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    chromosomes = chromosomes_from_arg(args.chromosomes)
    all_data, keep_masks_view1, keep_masks_view2 = [], [], []
    for chrom in chromosomes:
        data, _, region2idx, idx2region = load_graph_data(
            args.input_dir / chrom / f"metacell_accessibility_matrix_{chrom}_zscore.csv",
            args.input_dir / chrom / f"{chrom}_interactions_with_hic_strength_vc.bed",
            zscore_edge=args.zscore_edge_features,
        )
        keep_mask_view1, normalized_tad_drop_score = generate_mask_tad_strength(
            str(args.tad_strength_bed), region2idx, data.edge_index, data.num_nodes, p=args.mask_prob_1)
        keep_mask_view2, _ = generate_mask_tad_strength(
            str(args.tad_strength_bed), region2idx, data.edge_index, data.num_nodes, p=args.mask_prob_2)
        torch.save({"data": data, "idx2region": idx2region}, cache_dir / f"{chrom}_graph.pt")
        all_data.append(data)
        keep_masks_view1.append(keep_mask_view1)
        keep_masks_view2.append(keep_mask_view2)

    encoder = Encoder(input_dim=args.input_dim, hidden_dim=args.hidden_dim,
                      output_dim=args.output_dim, heads_num=args.heads,
                      dropout=args.dropout, edge_dim=args.edge_dim)
    model = GRACE(encoder=encoder, num_hidden=args.output_dim,
                  num_proj_hidden=args.projection_dim, tau=args.temperature).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    no_improvement = 0
    convergence_path = save_dir / f"converged_epoch_windows_{args.early_stop_window}.txt"
    if convergence_path.exists():
        convergence_path.unlink()

    model.train()
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        paired = list(zip(all_data, keep_masks_view1, keep_masks_view2))
        random.shuffle(paired)
        total_loss = 0.0
        for data, keep_mask_view1, keep_mask_view2 in paired:
            optimizer.zero_grad()
            view1 = build_view(data, keep_mask_view1).to(device)
            view2 = build_view(data, keep_mask_view2).to(device)
            z1 = model(view1)
            z2 = model(view2)
            loss = model.loss(z1, z2).to(device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(paired)
        print(f"Epoch {epoch:04d}, Avg Loss: {avg_loss:.6f}, Time: {time.time() - start:.2f}s")
        logging.info("Epoch %04d, Avg Loss: %.6f", epoch, avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
        else:
            no_improvement += 1
        if args.early_stop_window > 0 and no_improvement >= args.early_stop_window:
            message = f"Converged at epoch {epoch}, loss = {avg_loss:.6f}"
            convergence_path.write_text(message + "\n", encoding="utf-8")
            print(message)
            logging.info(message)
            torch.save(model.state_dict(), save_dir / f"model_epoch_{epoch:03d}.pt")
            break
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), save_dir / f"model_epoch_{epoch:03d}.pt")

    if epoch % args.save_every != 0:
        torch.save(model.state_dict(), save_dir / f"model_epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
