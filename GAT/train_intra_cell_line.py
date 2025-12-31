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

# ------------------------------------------------------------
# 1) Directed multi-hop edges (forward/backward)  -> BiLSTM-like
# ------------------------------------------------------------
def build_multiscale_edges_directed(
        n: int,
        hop_list=(1, 2, 4, 8),
        direction: str = "forward",   # "forward" (i->i+hop) or "backward" (i+hop->i)
        add_self_loops: bool = True,
        device=None,
) -> torch.Tensor:
    device = device or "cpu"
    src_chunks, dst_chunks = [], []

    for hop in hop_list:
        if hop <= 0 or hop >= n:
            continue
        i = torch.arange(0, n - hop, device=device, dtype=torch.long)
        j = i + hop

        if direction == "forward":
            src_chunks.append(i)
            dst_chunks.append(j)
        elif direction == "backward":
            src_chunks.append(j)
            dst_chunks.append(i)
        else:
            raise ValueError(f"direction must be 'forward' or 'backward', got: {direction}")

    if add_self_loops:
        idx = torch.arange(n, device=device, dtype=torch.long)
        src_chunks.append(idx)
        dst_chunks.append(idx)

    if len(src_chunks) == 0:
        idx = torch.arange(max(n, 0), device=device, dtype=torch.long)
        return torch.stack([idx, idx], dim=0) if n > 0 else torch.empty((2, 0), device=device, dtype=torch.long)

    src = torch.cat(src_chunks, dim=0)
    dst = torch.cat(dst_chunks, dim=0)
    return torch.stack([src, dst], dim=0)


# ------------------------------------------------------------
# 2) Positional features (very light)
#    LSTM has ordering inherently; GAT benefits from explicit pos.
# ------------------------------------------------------------
def add_pos_features(x: torch.Tensor, n_freqs: int = 4) -> torch.Tensor:
    """
    x: [N,F]
    returns [N, F + (1 + 2*n_freqs)]
      - one linear position in [-1,1]
      - sin/cos at multiple frequencies
    """
    n = x.size(0)
    if n <= 1:
        pos = torch.zeros((n, 1), device=x.device, dtype=x.dtype)
        return torch.cat([x, pos], dim=1)

    t = torch.linspace(-1.0, 1.0, n, device=x.device, dtype=x.dtype).unsqueeze(1)  # [N,1]
    feats = [t]
    # sin/cos bank
    for k in range(1, n_freqs + 1):
        w = (2.0 ** (k - 1)) * torch.pi
        feats.append(torch.sin(w * t))
        feats.append(torch.cos(w * t))
    p = torch.cat(feats, dim=1)
    return torch.cat([x, p], dim=1)


# ------------------------------------------------------------
# 3) Soffritto-like "gated memory" inside message passing blocks
#    Use attention message + GRU update (LSTM-ish).
# ------------------------------------------------------------
class GATGRUBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, heads: int = 8, dropout: float = 0.10):
        super().__init__()
        self.dropout = dropout
        self.gat = _GAT(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        out_dim = hidden_dim * heads

        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.gru = nn.GRUCell(out_dim, out_dim)  # gated update
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        # x: [N, D]
        h_prev = self.proj(x)  # [N, out_dim]
        m = F.dropout(x, p=self.dropout, training=self.training)
        m = self.gat(m, edge_index)            # [N, out_dim]
        m = F.elu(m)
        h = self.gru(m, h_prev)                # [N, out_dim]
        return self.norm(h)


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


# ------------------------------------------------------------
# 4) Bi-directional GAT encoder (BiLSTM-like)
# ------------------------------------------------------------
class BiGATNodePredictor(nn.Module):
    """
    Two towers:
      - forward tower reads edges i -> i+hop
      - backward tower reads edges i -> i-hop
    Then concat and predict per node.
    """
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

        # separate stems (optional, helps specialization)
        self.stem_f = nn.Sequential(
            nn.Linear(in_dim, stem_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(stem_dim),
        )
        self.stem_b = nn.Sequential(
            nn.Linear(in_dim, stem_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(stem_dim),
        )

        self.blocks_f = nn.ModuleList([GATGRUBlock(stem_dim, hidden_dim, heads=heads, dropout=dropout) for _ in range(layers)])
        self.blocks_b = nn.ModuleList([GATGRUBlock(stem_dim, hidden_dim, heads=heads, dropout=dropout) for _ in range(layers)])

        d = 2 * stem_dim
        self.head = GatedHead(d, out_dim, dropout=dropout, widen=widen)

    def forward(self, x, edge_index_fwd, edge_index_bwd):
        xf = self.stem_f(x)
        for b in self.blocks_f:
            xf = b(xf, edge_index_fwd)

        xb = self.stem_b(x)
        for b in self.blocks_b:
            xb = b(xb, edge_index_bwd)

        h = torch.cat([xf, xb], dim=-1)     # BiLSTM-like concat
        logits = self.head(h)
        return F.log_softmax(logits, dim=-1)


# ------------------------------------------------------------
# Trainer (same style as your current one)
# ------------------------------------------------------------
class GAT_intracell:
    def __init__(
            self,
            features_file: str,
            labels_file: str,
            train_chromosomes,
            test_chromosome,

            # Soffritto-ish receptive field: use dilations / multi-hop
            hop_list=(1, 2, 4, 8, 16),

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

            # positional features
            pos_freqs: int = 4,

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

        self.pos_freqs = pos_freqs

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
            hop_list=self.hop_list,  # loader may ignore; we rebuild edges anyway
        )

        # Add edges + pos features BEFORE moving to device
        for _, d in train_dict.items():
            d.x = add_pos_features(d.x, n_freqs=self.pos_freqs)
            n = int(d.x.size(0))
            d.edge_index_fwd = build_multiscale_edges_directed(n, self.hop_list, direction="forward", device=d.x.device)
            d.edge_index_bwd = build_multiscale_edges_directed(n, self.hop_list, direction="backward", device=d.x.device)

        test_data.x = add_pos_features(test_data.x, n_freqs=self.pos_freqs)
        n = int(test_data.x.size(0))
        test_data.edge_index_fwd = build_multiscale_edges_directed(n, self.hop_list, direction="forward", device=test_data.x.device)
        test_data.edge_index_bwd = build_multiscale_edges_directed(n, self.hop_list, direction="backward", device=test_data.x.device)

        self.train_data = to_device_data_dict(train_dict, self.device)
        self.test_data = to_device_data(test_data, self.device)
        self.scaler = scaler

        any_chrom = next(iter(self.train_data.keys()))
        in_dim = int(self.train_data[any_chrom].x.shape[1])
        out_dim = int(self.train_data[any_chrom].y.shape[1])
        return in_dim, out_dim

    def build_model(self, in_dim, out_dim):
        self.model = BiGATNodePredictor(
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
            optimizer.zero_grad(set_to_none=True)

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
                log_q = self.model(d.x, d.edge_index_fwd, d.edge_index_bwd)  # [N,C]
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
            if (ep == 1) or (ep % 10 == 0):
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
        log_q = self.model(self.test_data.x, self.test_data.edge_index_fwd, self.test_data.edge_index_bwd)
        loss = loss_fn(log_q, self.test_data.y)
        return float(loss)

    @torch.no_grad()
    def predict_test_probs(self) -> np.ndarray:
        self.model.eval()
        log_q = self.model(self.test_data.x, self.test_data.edge_index_fwd, self.test_data.edge_index_bwd)
        return log_q.exp().detach().cpu().numpy()


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    all_chroms = [f"chr{i}" for i in range(1, 23)]
    train_chroms = [c for c in all_chroms if c not in ("chr6", "chr9")]
    test_chrom = "chr9"

    # Better than (1,) if you want sequence-like larger context:
    # For 1kb bins: (1,2,4,8,16) ~ up to 16kb hops per layer stack
    hop_list = (1, 2, 4, 8, 16)

    trainer = GAT_intracell(
        features_file="GAT/data/H1_features.npz",
        labels_file="GAT/data/H1_labels.npz",
        train_chromosomes=train_chroms,
        test_chromosome=test_chrom,
        hop_list=hop_list,

        hidden_dim=16,
        heads=4,
        layers=2,
        dropout=0.10,
        widen=4,

        lr=1e-3,
        weight_decay=1e-5,
        epochs=200,
        chroms_per_epoch=None,

        pos_freqs=4,    # positional features strength
        device=None,
    )

    trainer.fit()
    probs = trainer.predict_test_probs()
    print("test probs shape:", probs.shape)

    out_path = "GAT/predictions/H1_predictions.npz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, test_chromosome=test_chrom, probs=probs)
    print("saved:", out_path)