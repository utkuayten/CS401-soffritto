# utils_gat_intra.py
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data


def load_first_npz_array(path: str) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    return z[z.files[0]]


def _npz_load(path: str):
    return np.load(path, allow_pickle=False)


def _get_chrom_key(npz_obj, chrom) -> str:
    """
    Accepts chrom like "9" or 9 or "chr9" and tries common variants.
    """
    c = str(chrom)
    candidates = [c, f"chr{c}", c.replace("chr", ""), f"chr{c.replace('chr','')}"]
    for k in candidates:
        if k in npz_obj:
            return k
    raise KeyError(
        f"Chromosome key {chrom!r} not found. "
        f"Available keys (sample): {list(npz_obj.keys())[:12]}"
    )


def build_chr_multiscale_edge_index(num_nodes: int, hop_list=(1, 2, 4, 8)) -> torch.Tensor:
    """
    Undirected multi-scale 1D edges: i <-> i+h for h in hop_list
    Returns edge_index [2, E] (torch.long).
    """
    src_all = []
    dst_all = []
    for h in hop_list:
        h = int(h)
        if h <= 0 or h >= num_nodes:
            continue
        src = torch.arange(0, num_nodes - h, dtype=torch.long)
        dst = src + h
        src_all.append(torch.cat([src, dst], dim=0))
        dst_all.append(torch.cat([dst, src], dim=0))
    edge_index = torch.stack([torch.cat(src_all), torch.cat(dst_all)], dim=0)
    return edge_index


def load_gat_intra_cell_line_train(
        features_file: str,
        labels_file: str,
        train_chromosomes,
        test_chromosome,
        hop_list=(1, 2, 4, 8),
):
    """
    Intra-cell-line ONLY.

    Args:
      features_file: path to *.npz (chrom -> (N,F))
      labels_file:   path to *.npz (chrom -> (N,C)) where rows sum to 1
      train_chromosomes: iterable of chrom identifiers (e.g., ["1","2",...])
      test_chromosome: one chrom identifier (e.g., "9")
      hop_list: multi-scale neighborhood hops for edges (default: (1,2,4,8))

    Returns:
      train_data_dict: {chrom(str): Data(x,y,edge_index)}
      test_data: Data(x,y,edge_index)
      scaler: fitted StandardScaler
    """
    Xnpz = _npz_load(features_file)
    Ynpz = _npz_load(labels_file)

    # Fit scaler on concatenated train chromosomes
    X_train_list = []
    for chrom in train_chromosomes:
        ck = _get_chrom_key(Xnpz, chrom)
        X_train_list.append(Xnpz[ck])
    X_train = np.concatenate(X_train_list, axis=0)

    scaler = StandardScaler()
    scaler.fit(X_train)

    # Build PyG Data per training chromosome (scaled features)
    train_data_dict = {}
    for chrom in train_chromosomes:
        ck = _get_chrom_key(Xnpz, chrom)
        yk = _get_chrom_key(Ynpz, chrom)

        Xc = scaler.transform(Xnpz[ck]).astype(np.float32)
        Yc = Ynpz[yk].astype(np.float32)

        x = torch.from_numpy(Xc)
        y = torch.from_numpy(Yc)
        edge_index = build_chr_multiscale_edge_index(x.size(0), hop_list=hop_list)

        train_data_dict[str(chrom)] = Data(x=x, y=y, edge_index=edge_index)

    # Test chromosome
    ck = _get_chrom_key(Xnpz, test_chromosome)
    yk = _get_chrom_key(Ynpz, test_chromosome)

    Xt = scaler.transform(Xnpz[ck]).astype(np.float32)
    Yt = Ynpz[yk].astype(np.float32)

    xt = torch.from_numpy(Xt)
    yt = torch.from_numpy(Yt)
    edge_index_t = build_chr_multiscale_edge_index(xt.size(0), hop_list=hop_list)

    test_data = Data(x=xt, y=yt, edge_index=edge_index_t)

    return train_data_dict, test_data, scaler


def to_device_data_dict(train_data_dict, device: str):
    return {k: v.to(device) for k, v in train_data_dict.items()}


def to_device_data(data: Data, device: str):
    return data.to(device)