from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def load_graph_data(feature_path: str, interaction_path: str, zscore_edge: bool = False):







    feat_df = pd.read_csv(feature_path, sep=",", index_col=0)
    feat_df.index.name = "region"

    interactions = pd.read_csv(interaction_path, sep="\t", header=0)
    interactions["region1"] = (
        interactions["chr1"].astype(str) + ":" +
        interactions["start1"].astype(str) + "-" +
        interactions["end1"].astype(str)
    )
    interactions["region2"] = (
        interactions["chr2"].astype(str) + ":" +
        interactions["start2"].astype(str) + "-" +
        interactions["end2"].astype(str)
    )

    valid_regions = set(feat_df.index)
    filtered = interactions[
        interactions["region1"].isin(valid_regions) &
        interactions["region2"].isin(valid_regions)
    ].copy()

    used_regions = sorted(set(filtered["region1"]) | set(filtered["region2"]))
    region2idx = {region: idx for idx, region in enumerate(used_regions)}
    idx2region = {idx: region for region, idx in region2idx.items()}

    x = torch.tensor(feat_df.loc[used_regions].values, dtype=torch.float)

    edge_index = []
    coaccess_list = []
    hic_list = []
    archr_list = []

    for _, row in filtered.iterrows():
        for src_region, dst_region in [(row["region1"], row["region2"]),
                                        (row["region2"], row["region1"])]:
            src = region2idx[src_region]
            dst = region2idx[dst_region]
            edge_index.append([src, dst])
            coaccess_list.append(row["cicero_score"])
            hic_list.append(row["hic_observed"])
            archr_list.append(row["archr_score"])

    edge_attr_np = np.stack([
        np.array(coaccess_list, dtype=float),
        np.array(archr_list, dtype=float),
        np.array(hic_list, dtype=float),
    ], axis=1)

    if zscore_edge:
        mean = edge_attr_np.mean(axis=0, keepdims=True)
        std = edge_attr_np.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        edge_attr_np = (edge_attr_np - mean) / std

    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).T

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, used_regions, region2idx, idx2region


def generate_mask_tad_strength(
    bed_path: str,
    region2idx: dict,
    edge_index: torch.Tensor,
    num_nodes: int,
    p: float = 0.2,
    threshold: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:





    bed_df = pd.read_csv(bed_path, sep="\t", header=None)
    bed_df.columns = ["chr", "start", "end", "strength"]

    communities = []
    com_cs = []
    for _, row in bed_df.iterrows():
        region_start, region_end, chrom, strength = row["start"], row["end"], row["chr"], row["strength"]
        members = []
        for region in region2idx:
            match = re.match(r"(chr[\w]+):(\d+)-(\d+)", region)
            if match:
                r_chr, r_start, r_end = match.groups()
                r_start, r_end = int(r_start), int(r_end)
                if r_chr == chrom and not (r_end <= region_start or r_start >= region_end):
                    members.append(region2idx[region])
        if members:
            communities.append(members)
            com_cs.append(strength)

    com_cs = np.array(com_cs)


    node_cs = np.zeros(num_nodes)
    for com, strength in zip(communities, com_cs):
        for node in com:
            node_cs[node] = strength


    com = np.full(num_nodes, -1, dtype=int)
    for com_idx, members in enumerate(communities):
        for node in members:
            com[node] = com_idx

    unmatched_nodes = np.flatnonzero(com == -1)
    if unmatched_nodes.size:
        idx2region = {idx: region for region, idx in region2idx.items()}
        examples = ", ".join(idx2region.get(int(idx), f"node {idx}") for idx in unmatched_nodes[:10])
        raise ValueError(
            f"{unmatched_nodes.size} of {num_nodes} peaks have no matching TAD in {bed_path}. "
            f"Examples: {examples}. Check chromosome names, genome assembly, and coordinates."
        )

    def edge_mod(x):
        return com_cs[x[0]] if x[0] == x[1] else -(float(com_cs[x[0]]) + float(com_cs[x[1]]))

    def normalize(x):
        return (np.max(x) - x) / (np.max(x) - np.min(x))

    edge_weight = np.asarray([edge_mod([com[u.item()], com[v.item()]]) for u, v in edge_index.T])
    edge_drop_probability = normalize(edge_weight)

    edge_drop_probability = torch.from_numpy(edge_drop_probability)
    normalized_tad_drop_score = edge_drop_probability.clone()

    edge_drop_probability = edge_drop_probability / edge_drop_probability.mean() * (1. - p)
    edge_drop_probability = edge_drop_probability.where(
        edge_drop_probability > (1. - threshold),
        torch.ones_like(edge_drop_probability) * (1. - threshold)
    )
    edge_drop_probability = edge_drop_probability.where(
        edge_drop_probability < 1,
        torch.ones_like(edge_drop_probability) * 1
    )

    drop_mask = torch.bernoulli(edge_drop_probability).to(torch.bool)
    keep_mask = ~drop_mask
    return keep_mask, normalized_tad_drop_score


def build_view(data: Data, keep_mask: torch.Tensor) -> Data:

    return Data(
        x=data.x.float(),
        edge_index=data.edge_index[:, keep_mask],
        edge_attr=data.edge_attr[keep_mask].float(),
    )
