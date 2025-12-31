# gat_intracell_keep_testshape.py
# ✅ NO COARSENING.
# ✅ NO POSITIONAL ENCODING.
# Keep original bins as nodes (N unchanged, predictions are [N, C]).
# Widen receptive field by creating edges to the desired hop distances.

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv as _GAT
except Exception:
    from torch_geometric.nn import GATConv as _GAT

from utils import (
    load_gat_intra_cell_line_train,
    to_device_data_dict,
    to_device_data,
)

def build_multiscale_edges(
        n: int,
        hop_list=(1, 2, 4, 8),
        add_self_loops: bool = True,
        device=None,
) -> torch.Tensor:
    """
    Build 1D chain multi-hop undirected edges for n nodes.
    Returns edge_index [2, E] (dtype long).
    """
    device = device or "cpu"
    src_chunks, dst_chunks = [], []

    for hop in hop_list:
        if hop <= 0 or hop >= n:
            continue
        i = torch.arange(0, n - hop, device=device, dtype=torch.long)
        j = i + hop
        # undirected
        src_chunks.append(torch.cat([i, j], dim=0))
        dst_chunks.append(torch.cat([j, i], dim=0))

    if add_self_loops:
        idx = torch.arange(n, device=device, dtype=torch.long)
        src_chunks.append(idx)
        dst_chunks.append(idx)

    if len(src_chunks) == 0:
        if n <= 0:
            return torch.empty((2, 0), device=device, dtype=torch.long)
        idx = torch.arange(n, device=device, dtype=torch.long)
        return torch.stack([idx, idx], dim=0)

    src = torch.cat(src_chunks, dim=0)
    dst = torch.cat(dst_chunks, dim=0)
    return torch.stack([src, dst], dim=0)



class GATResBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, heads: int = 8, dropout: float = 0.10):
        super().__init__()
        self.dropout = dropout
        self.gat = _GAT(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        out_dim = hidden_dim * heads
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.gat(h, edge_index)
        h = F.relu(h)
        return self.norm(h + self.proj(x))


class GatedHead(nn.Module):
    def __init__(self, d: int, out_dim: int, dropout: float = 0.10, widen: int = 4):
        super().__init__()
        h = d * widen
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 2 * h)
        self.proj = nn.Linear(h, out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = self.norm(x)
        a, b = self.fc(x).chunk(2, dim=-1)
        h = a * torch.sigmoid(b)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.proj(h)


class GATNodePredictor(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            heads: int = 8,
            layers: int = 3,
            dropout: float = 0.10,
            widen: int = 4,
    ):
        super().__init__()
        stem_dim = hidden_dim * heads

        self.stem = nn.Sequential(
            nn.Linear(in_dim, stem_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(stem_dim),
        )

        blocks = []
        d = stem_dim
        for _ in range(layers):
            blocks.append(GATResBlock(d, hidden_dim, heads=heads, dropout=dropout))
            d = hidden_dim * heads
        self.blocks = nn.ModuleList(blocks)

        self.head = GatedHead(d, out_dim, dropout=dropout, widen=widen)

    def forward(self, x, edge_index):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x, edge_index)
        logits = self.head(x)
        return F.log_softmax(logits, dim=-1)


class GAT_intracell:
    def __init__(
            self,
            features_file: str,
            labels_file: str,
            train_chromosomes,
            test_chromosome,

            hop_list=(1, 2, 4, 8),

            hidden_dim: int = 32,
            heads: int = 8,
            layers: int = 2,
            dropout: float = 0.10,
            widen: int = 4,

            lr: float = 1e-3,
            weight_decay: float = 1e-5,
            epochs: int = 200,
            grad_clip: float = 1.0,

            patience: int = 30,
            min_delta: float = 1e-5,
            chroms_per_epoch: int | None = 6,

            device: str | None = None,
    ):
        self.features_file = features_file
        self.labels_file = labels_file
        self.train_chromosomes = train_chromosomes
        self.test_chromosome = test_chromosome
        self.hop_list = tuple(hop_list)

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.widen = widen

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.patience = patience
        self.min_delta = min_delta
        self.chroms_per_epoch = chroms_per_epoch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: nn.Module | None = None
        self.train_data = None
        self.test_data = None
        self.scaler = None

        self.best_state = None
        self.best_test_kl = float("inf")

    def prepare_data(self):
        train_dict, test_data, scaler = load_gat_intra_cell_line_train(
            features_file=self.features_file,
            labels_file=self.labels_file,
            train_chromosomes=self.train_chromosomes,
            test_chromosome=self.test_chromosome,
            hop_list=self.hop_list,
        )

        for _, d in train_dict.items():
            d.edge_index = build_multiscale_edges(
                int(d.x.size(0)),
                hop_list=self.hop_list,
                device=d.x.device,
            )

        test_data.edge_index = build_multiscale_edges(
            int(test_data.x.size(0)),
            hop_list=self.hop_list,
            device=test_data.x.device,
        )

        self.train_data = to_device_data_dict(train_dict, self.device)
        self.test_data = to_device_data(test_data, self.device)
        self.scaler = scaler

        any_chrom = next(iter(self.train_data.keys()))
        in_dim = int(self.train_data[any_chrom].x.shape[1])
        out_dim = int(self.train_data[any_chrom].y.shape[1])
        return in_dim, out_dim

    def build_model(self, in_dim, out_dim):
        self.model = GATNodePredictor(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=out_dim,
            heads=self.heads,
            layers=self.layers,
            dropout=self.dropout,
            widen=self.widen,
        ).to(self.device)

    def fit(self):
        in_dim, out_dim = self.prepare_data()
        self.build_model(in_dim, out_dim)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        bad_epochs = 0
        train_keys = list(self.train_data.keys())

        for ep in range(1, self.epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            if self.chroms_per_epoch is None or self.chroms_per_epoch >= len(train_keys):
                keys = train_keys
            else:
                g = torch.Generator()
                g.manual_seed(ep)
                idx = torch.randperm(len(train_keys), generator=g)[: self.chroms_per_epoch].tolist()
                keys = [train_keys[i] for i in idx]

            total_loss = 0.0
            for k in keys:
                d = self.train_data[k]
                log_q = self.model(d.x, d.edge_index)  # [N,C]
                total_loss = total_loss + loss_fn(log_q, d.y)

            total_loss = total_loss / max(1, len(keys))
            total_loss.backward()

            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            optimizer.step()
            scheduler.step()

            test_kl = self.evaluate_kl()

            improved = (self.best_test_kl - test_kl) > self.min_delta
            if improved:
                self.best_test_kl = test_kl
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {ep:03d} | lr={lr_now:.2e} | train_KL={total_loss.item():.6f} "
                f"| test_KL={test_kl:.6f} | best={self.best_test_kl:.6f} | chroms={len(keys)}"
            )

            if bad_epochs >= self.patience:
                print(f"Early stop at epoch {ep:03d} (best test_KL={self.best_test_kl:.6f})")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return self

    @torch.no_grad()
    def evaluate_kl(self) -> float:
        self.model.eval()
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        log_q = self.model(self.test_data.x, self.test_data.edge_index)
        loss = loss_fn(log_q, self.test_data.y)
        return float(loss)

    @torch.no_grad()
    def predict_test_probs(self) -> np.ndarray:
        self.model.eval()
        log_q = self.model(self.test_data.x, self.test_data.edge_index)  # [N,C]
        return log_q.exp().detach().cpu().numpy()


# Example usage
if __name__ == "__main__":
    all_chroms = [f"chr{i}" for i in range(1, 23)]
    train_chroms = [c for c in all_chroms if c not in ("chr6", "chr9")]
    test_chrom = "chr9"

    # currently set (1,10,20) -> that means 1 hop, 10 hops, 20 hops only.
    hop_list_10kb = ((1,))

    trainer = GAT_intracell(
        features_file="GAT/data/H1_features.npz",
        labels_file="GAT/data/H1_labels.npz",
        train_chromosomes=train_chroms,
        test_chromosome=test_chrom,
        hop_list=hop_list_10kb,

        hidden_dim=16,
        heads=4,
        layers=2,
        dropout=0.10,
        widen=4,

        lr=1e-3,
        weight_decay=1e-5,
        epochs=200,
        chroms_per_epoch=None,
    )

    trainer.fit()
    probs = trainer.predict_test_probs()
    print("test probs shape:", probs.shape)

    out_path = "GAT/predictions/H1_predictions.npz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, test_chromosome=test_chrom, probs=probs)
    print("saved:", out_path)