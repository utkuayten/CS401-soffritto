# =========================
# train_ablation.py
# Intra-cell-line full-chromosome GAT+BiLSTM training with feature-channel ablations
# (based on train_intra_cell.py)
# =========================
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv as _GAT
except Exception:
    from torch_geometric.nn import GATConv as _GAT

from torch_geometric.data import Data

# ============================================================
# Data utilities (self-contained so you don't have to edit utils.py)
# ============================================================

DEFAULT_ALL_COLS: List[str] = [
    "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1",
    "H3K4me3", "H3K9me3", "GC_content", "gene_density", "2-stage"
]


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


def build_chr_multiscale_edge_index(num_nodes: int, hop_list: Sequence[int] = (1, 2, 4, 8)) -> torch.Tensor:
    """
    Undirected multi-scale 1D edges: i <-> i+h for h in hop_list
    Returns edge_index [2, E] (torch.long).
    """
    src_all, dst_all = [], []
    for h in hop_list:
        h = int(h)
        if h <= 0 or h >= num_nodes:
            continue
        src = torch.arange(0, num_nodes - h, dtype=torch.long)
        dst = src + h
        src_all.append(torch.cat([src, dst], dim=0))
        dst_all.append(torch.cat([dst, src], dim=0))

    # safe fallback: if hop_list is empty/invalid, add self loops
    if not src_all:
        idx = torch.arange(num_nodes, dtype=torch.long)
        return torch.stack([idx, idx], dim=0)

    return torch.stack([torch.cat(src_all), torch.cat(dst_all)], dim=0)


def _validate_selected_cols(selected_cols: Sequence[str], all_cols: Sequence[str]) -> List[int]:
    if not selected_cols:
        raise ValueError("selected_cols is empty. Provide at least one feature name.")
    all_cols = list(all_cols)
    col_to_idx = {c: i for i, c in enumerate(all_cols)}

    missing = [c for c in selected_cols if c not in col_to_idx]
    if missing:
        raise ValueError(
            f"Unknown feature(s) in --selected_cols: {missing}. "
            f"Valid names are: {all_cols}"
        )

    # preserve user order; drop duplicates while preserving order
    seen = set()
    selected_unique = []
    for c in selected_cols:
        if c not in seen:
            selected_unique.append(c)
            seen.add(c)

    return [col_to_idx[c] for c in selected_unique]


def load_gat_intra_cell_line_train_selected(
        features_file: str,
        labels_file: str,
        train_chromosomes: Iterable[str],
        test_chromosome: str,
        hop_list: Sequence[int] = (1, 2, 4, 8),
        selected_cols: Sequence[str] = DEFAULT_ALL_COLS,
        all_cols: Sequence[str] = DEFAULT_ALL_COLS,
):
    """
    Intra-cell-line ONLY, with feature-channel selection by column names.

    Assumptions:
      - features_file is an .npz mapping chrom -> (N,F) float array
      - labels_file   is an .npz mapping chrom -> (N,C) float array (rows sum to 1)
      - The feature dimension F is ordered according to `all_cols`.

    Returns:
      train_data_dict: {chrom(str): Data(x,y,edge_index)}
      test_data: Data(x,y,edge_index)
      scaler: fitted StandardScaler (fit on selected feature channels only)
      selected_cols_final: list[str] after de-duplication preserving order
    """
    Xnpz = _npz_load(features_file)
    Ynpz = _npz_load(labels_file)

    idxs = _validate_selected_cols(selected_cols, all_cols)
    selected_cols_final = []
    seen = set()
    for c in selected_cols:
        if c in all_cols and c not in seen:
            selected_cols_final.append(c)
            seen.add(c)

    # Fit scaler on concatenated train chromosomes (selected channels only)
    X_train_list = []
    for chrom in train_chromosomes:
        ck = _get_chrom_key(Xnpz, chrom)
        Xc = Xnpz[ck]
        if Xc.ndim != 2:
            raise ValueError(f"Expected features array [N,F] for chrom {chrom}, got shape {Xc.shape}")
        if Xc.shape[1] < max(idxs) + 1:
            raise ValueError(
                f"Features for chrom {chrom} have F={Xc.shape[1]} but need at least {max(idxs)+1} "
                f"based on selected_cols={selected_cols_final} and all_cols={list(all_cols)}"
            )
        X_train_list.append(Xc[:, idxs])
    X_train = np.concatenate(X_train_list, axis=0)

    scaler = StandardScaler()
    scaler.fit(X_train)

    # Build PyG Data per training chromosome (scaled features)
    train_data_dict: Dict[str, Data] = {}
    for chrom in train_chromosomes:
        ck = _get_chrom_key(Xnpz, chrom)
        yk = _get_chrom_key(Ynpz, chrom)

        Xc = scaler.transform(Xnpz[ck][:, idxs]).astype(np.float32)
        Yc = Ynpz[yk].astype(np.float32)

        x = torch.from_numpy(Xc)
        y = torch.from_numpy(Yc)
        edge_index = build_chr_multiscale_edge_index(x.size(0), hop_list=hop_list)

        train_data_dict[str(chrom)] = Data(x=x, y=y, edge_index=edge_index)

    # Test chromosome
    ck = _get_chrom_key(Xnpz, test_chromosome)
    yk = _get_chrom_key(Ynpz, test_chromosome)

    Xt = scaler.transform(Xnpz[ck][:, idxs]).astype(np.float32)
    Yt = Ynpz[yk].astype(np.float32)

    xt = torch.from_numpy(Xt)
    yt = torch.from_numpy(Yt)
    edge_index_t = build_chr_multiscale_edge_index(xt.size(0), hop_list=hop_list)

    test_data = Data(x=xt, y=yt, edge_index=edge_index_t)

    return train_data_dict, test_data, scaler, selected_cols_final


def to_device_data_dict(train_data_dict: Dict[str, Data], device: str) -> Dict[str, Data]:
    return {k: v.to(device) for k, v in train_data_dict.items()}


def to_device_data(data: Data, device: str) -> Data:
    return data.to(device)


# ============================================================
# Model (same as train_intra_cell.py)
# ============================================================

def build_full_edges(n: int, hop_list: Sequence[int] = (1, 2, 4), device=None) -> torch.Tensor:
    """
    Undirected multi-hop edges over the ENTIRE chromosome of length n.
    WARNING: This can be extremely large and may OOM for big n / big hop_list.
    """
    device = device or "cpu"
    src_chunks, dst_chunks = [], []

    for hop in hop_list:
        hop = int(hop)
        if hop <= 0 or hop >= n:
            continue
        i = torch.arange(0, n - hop, device=device, dtype=torch.long)
        j = i + hop
        src_chunks.append(torch.cat([i, j], 0))
        dst_chunks.append(torch.cat([j, i], 0))

    # self loops help stability
    idx = torch.arange(n, device=device, dtype=torch.long)
    src_chunks.append(idx)
    dst_chunks.append(idx)

    src = torch.cat(src_chunks, 0)
    dst = torch.cat(dst_chunks, 0)
    return torch.stack([src, dst], 0)


class LocalGATEncoder(nn.Module):
    """GAT applied over the whole chromosome graph."""
    def __init__(self, in_dim: int, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.gat = _GAT(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.out_dim = hidden_dim * heads
        self.norm = nn.LayerNorm(self.out_dim)
        self.proj = nn.Linear(in_dim, self.out_dim) if in_dim != self.out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h0 = self.proj(x)
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.gat(h, edge_index)
        h = F.elu(h)
        return self.norm(h + h0)


class GATxSoffritto(nn.Module):
    """
    Full-chrom pipeline:
      x [N,F] -> GAT over full graph -> z [N,D] -> BiLSTM -> fc -> log_softmax
    """
    def __init__(
            self,
            in_dim: int,
            gat_hidden: int,
            gat_heads: int,
            lstm_hidden: int,
            lstm_layers: int,
            out_dim: int,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.gat = LocalGATEncoder(in_dim, gat_hidden, gat_heads, dropout)
        d = self.gat.out_dim

        # Unbatched mode: input [seq_len, input_size]
        self.lstm = nn.LSTM(d, lstm_hidden, lstm_layers, bidirectional=True)
        self.fc = nn.Linear(2 * lstm_hidden, out_dim)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.hidden = None

    def init_hidden(self, device: torch.device):
        h0 = torch.zeros(2 * self.lstm_layers, self.lstm_hidden, device=device)
        c0 = torch.zeros(2 * self.lstm_layers, self.lstm_hidden, device=device)
        return (h0, c0)

    def reset_hidden(self, device: torch.device):
        self.hidden = self.init_hidden(device)

    def forward_full(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.hidden is None:
            self.hidden = self.init_hidden(x.device)

        z = self.gat(x, edge_index)                   # [N, D]
        out, self.hidden = self.lstm(z, self.hidden)  # [N, 2H]

        # detach to avoid keeping graph around
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        logits = self.fc(out)                         # [N, C]
        return self.log_softmax(logits)


# ============================================================
# Trainer (based on GAT_intracell in train_intra_cell.py)
# ============================================================

class GAT_IntraCell_AblationTrainer:
    def __init__(
            self,
            features_file: str,
            labels_file: str,
            train_chromosomes: Sequence[str],
            test_chromosome: str,
            hop_list: Sequence[int] = (1, 2, 4),

            # model
            gat_hidden: int = 8,
            gat_heads: int = 2,
            lstm_hidden: int = 64,
            lstm_layers: int = 2,
            dropout: float = 0.10,

            # optimization
            lr: float = 1e-3,
            weight_decay: float = 1e-6,
            epochs: int = 100,
            grad_clip: float = 1.0,

            # early stopping
            patience: int = 20,
            min_delta: float = 1e-5,
            chroms_per_epoch: int | None = None,

            # ablation
            selected_cols: Sequence[str] = DEFAULT_ALL_COLS,
            all_cols: Sequence[str] = DEFAULT_ALL_COLS,

            device: str | None = None,
            seed: int = 0,
    ):
        self.features_file = features_file
        self.labels_file = labels_file
        self.train_chromosomes = list(train_chromosomes)
        self.test_chromosome = str(test_chromosome)
        self.hop_list = tuple(int(h) for h in hop_list)

        self.gat_hidden = int(gat_hidden)
        self.gat_heads = int(gat_heads)
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)
        self.dropout = float(dropout)

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.grad_clip = float(grad_clip)

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.chroms_per_epoch = chroms_per_epoch if chroms_per_epoch is None else int(chroms_per_epoch)

        self.selected_cols = list(selected_cols)
        self.all_cols = list(all_cols)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(seed)

        self.model: GATxSoffritto | None = None
        self.train_data: Dict[str, Data] | None = None
        self.test_data: Data | None = None
        self.scaler: StandardScaler | None = None
        self.selected_cols_final: List[str] | None = None

        self.best_state = None
        self.best_test_kl = float("inf")

        # cache full edges per chromosome length (and hop_list)
        self._edge_cache: Dict[Tuple[int, Tuple[int, ...]], torch.Tensor] = {}

    def _seed_all(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def prepare_data(self):
        self._seed_all()
        train_dict, test_data, scaler, selected_cols_final = load_gat_intra_cell_line_train_selected(
            features_file=self.features_file,
            labels_file=self.labels_file,
            train_chromosomes=self.train_chromosomes,
            test_chromosome=self.test_chromosome,
            hop_list=(1,),  # ignored for full-chrom edges
            selected_cols=self.selected_cols,
            all_cols=self.all_cols,
        )
        self.train_data = to_device_data_dict(train_dict, self.device)
        self.test_data = to_device_data(test_data, self.device)
        self.scaler = scaler
        self.selected_cols_final = selected_cols_final

        any_chrom = next(iter(self.train_data.keys()))
        in_dim = int(self.train_data[any_chrom].x.shape[1])
        out_dim = int(self.train_data[any_chrom].y.shape[1])
        return in_dim, out_dim

    def build_model(self, in_dim: int, out_dim: int):
        self.model = GATxSoffritto(
            in_dim=in_dim,
            gat_hidden=self.gat_hidden,
            gat_heads=self.gat_heads,
            lstm_hidden=self.lstm_hidden,
            lstm_layers=self.lstm_layers,
            out_dim=out_dim,
            dropout=self.dropout,
        ).to(self.device)

    def _get_full_edges(self, n: int, device: torch.device) -> torch.Tensor:
        key = (int(n), self.hop_list)
        ei = self._edge_cache.get(key)
        if ei is None or ei.device != device:
            ei = build_full_edges(n, hop_list=self.hop_list, device=device)
            self._edge_cache[key] = ei
        return ei

    def _full_chrom_loss(self, x: torch.Tensor, y: torch.Tensor, train: bool) -> float:
        assert self.model is not None
        self.model.reset_hidden(x.device)

        loss_fn = nn.KLDivLoss(reduction="batchmean")

        n = int(x.size(0))
        edge_index = self._get_full_edges(n, x.device)

        log_q = self.model.forward_full(x, edge_index)  # [N, C]
        loss = loss_fn(log_q, y)

        if train:
            loss.backward()

        return float(loss.detach())

    def fit(self):
        in_dim, out_dim = self.prepare_data()
        self.build_model(in_dim, out_dim)

        assert self.model is not None
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        bad_epochs = 0
        train_keys = list(self.train_data.keys())  # type: ignore[arg-type]

        for ep in range(1, self.epochs + 1):
            self.model.train()
            opt.zero_grad(set_to_none=True)

            keys = train_keys
            if self.chroms_per_epoch is not None and self.chroms_per_epoch < len(train_keys):
                g = torch.Generator(device="cpu")
                g.manual_seed(self.seed + ep)
                idx = torch.randperm(len(train_keys), generator=g)[: self.chroms_per_epoch].tolist()
                keys = [train_keys[i] for i in idx]

            train_loss = 0.0
            for k in keys:
                d = self.train_data[k]  # type: ignore[index]
                train_loss += self._full_chrom_loss(d.x, d.y, train=True)
            train_loss /= max(1, len(keys))

            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            opt.step()

            test_kl = self.evaluate_kl()

            improved = (self.best_test_kl - test_kl) > self.min_delta
            if improved:
                self.best_test_kl = test_kl
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if ep == 1 or ep % 10 == 0:
                feats = ",".join(self.selected_cols_final or [])
                print(
                    f"epoch {ep:03d} | train_KL={train_loss:.6f} | test_KL={test_kl:.6f} "
                    f"| best={self.best_test_kl:.6f} | feats=[{feats}]"
                )

            if bad_epochs >= self.patience:
                print(f"Early stop at epoch {ep:03d} (best test_KL={self.best_test_kl:.6f})")
                break

            opt.zero_grad(set_to_none=True)

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return self

    @torch.no_grad()
    def evaluate_kl(self) -> float:
        assert self.model is not None and self.test_data is not None
        self.model.eval()
        d = self.test_data
        return float(self._full_chrom_loss(d.x, d.y, train=False))

    @torch.no_grad()
    def predict_test_probs(self) -> np.ndarray:
        assert self.model is not None and self.test_data is not None
        self.model.eval()
        d = self.test_data
        x = d.x
        n = int(x.size(0))

        self.model.reset_hidden(x.device)
        edge_index = self._get_full_edges(n, x.device)

        log_q = self.model.forward_full(x, edge_index)  # [N, C]
        return log_q.exp().detach().cpu().numpy()


# ============================================================
# CLI
# ============================================================

def _parse_int_list(values: Sequence[str]) -> List[int]:
    return [int(v) for v in values]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train full-chrom GAT+BiLSTM with feature ablations.")

    # paths
    parser.add_argument("--features_file", type=str, required=True, help="Path to features .npz (chrom -> [N,F]).")
    parser.add_argument("--labels_file", type=str, required=True, help="Path to labels .npz (chrom -> [N,C]).")

    # chromosomes
    parser.add_argument(
        "--train_chromosomes",
        nargs="+",
        type=str,
        required=True,
        help='Training chromosomes (e.g. --train_chromosomes 1 2 3 ... 22).',
    )
    parser.add_argument("--test_chromosome", type=str, required=True, help='Test chromosome (e.g. "9").')

    # graph hops
    parser.add_argument(
        "--hop_list",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="Hop list for full-chrom edges (default: 1 2 4).",
    )

    # model hparams
    parser.add_argument("--gat_hidden", type=int, default=8)
    parser.add_argument("--gat_heads", type=int, default=2)
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)

    # optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # early stopping
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--chroms_per_epoch", type=int, default=None)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    # outputs
    parser.add_argument(
        "--pred_out",
        type=str,
        default="pred_test_probs.npy",
        help="Where to save test predictions (numpy array) (default: pred_test_probs.npy).",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default=None,
        help="Optional path to save best model state_dict (.pt).",
    )
    parser.add_argument(
        "--run_meta_out",
        type=str,
        default=None,
        help="Optional path to save a JSON with run metadata (selected_cols, best_test_kl, args).",
    )

    # Feature selection (USER REQUEST)
    parser.add_argument(
        "--selected_cols",
        nargs="+",
        type=str,
        default=DEFAULT_ALL_COLS,
        help="List of feature column names to use for training (default: all 9 features).",
    )
    parser.add_argument(
        "--all_cols",
        nargs="+",
        type=str,
        default=DEFAULT_ALL_COLS,
        help="Full ordered list of feature columns in your .npz (default: the standard 9).",
    )

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    trainer = GAT_IntraCell_AblationTrainer(
        features_file=args.features_file,
        labels_file=args.labels_file,
        train_chromosomes=args.train_chromosomes,
        test_chromosome=args.test_chromosome,
        hop_list=tuple(args.hop_list),

        gat_hidden=args.gat_hidden,
        gat_heads=args.gat_heads,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,

        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        grad_clip=args.grad_clip,

        patience=args.patience,
        min_delta=args.min_delta,
        chroms_per_epoch=args.chroms_per_epoch,

        selected_cols=args.selected_cols,
        all_cols=args.all_cols,

        device=args.device,
        seed=args.seed,
    )

    trainer.fit()

    pred = trainer.predict_test_probs()
    np.save(args.pred_out, pred)
    print(f"Saved predictions: {args.pred_out} | shape={pred.shape}")

    if args.model_out:
        assert trainer.model is not None
        torch.save(trainer.model.state_dict(), args.model_out)
        print(f"Saved model state_dict: {args.model_out}")

    if args.run_meta_out:
        meta = {
            "best_test_kl": trainer.best_test_kl,
            "selected_cols": trainer.selected_cols_final,
            "all_cols": args.all_cols,
            "args": vars(args),
        }
        with open(args.run_meta_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved run metadata: {args.run_meta_out}")


if __name__ == "__main__":
    main()
