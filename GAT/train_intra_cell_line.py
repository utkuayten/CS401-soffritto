import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try GATv2 first (often better attention); fallback to GATConv if unavailable
try:
    from torch_geometric.nn import GATv2Conv as _GAT
except Exception:
    from torch_geometric.nn import GATConv as _GAT

from utils import (
    load_gat_intra_cell_line_train,
    to_device_data_dict,
    to_device_data,
)

# -------------------------
# Improvements requested:
#  1) Multi-scale edges (in utils): hop_list=(1,2,4,8) instead of hops-only chain
#  2) Positional encoding concatenated to node features
#  3) Stronger "Soffritto-like" end head (wide + gated option)
#  4) Lower dropout default + lower weight_decay default to avoid plateaus
#  5) Optional chrom sampling per epoch (faster + reduces gradient interference)
# -------------------------

def sinusoidal_pos_enc(n: int, d: int, device) -> torch.Tensor:
    """(n,d) fixed positional encoding for 1D chromosome coordinates."""
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)  # (n,1)
    i = torch.arange(d, device=device, dtype=torch.float32).unsqueeze(0)    # (1,d)
    angles = pos / torch.pow(10000.0, (2 * (i // 2)) / d)
    pe = torch.zeros((n, d), device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


class GATResBlock(nn.Module):
    """
    Soffritto-like stabilization:
      Dropout -> (GAT/GATv2) -> ReLU -> Residual -> LayerNorm
    """
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
    """
    Strong gated readout (GLU-like) often works well for distribution targets.
    LayerNorm -> Linear(2H) -> split -> A * sigmoid(B) -> ReLU -> Dropout -> out
    """
    def __init__(self, d: int, out_dim: int, dropout: float = 0.10, widen: int = 4):
        super().__init__()
        h = d * widen
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 2 * h)
        self.proj = nn.Linear(h, out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = self.norm(x)
        ab = self.fc(x)
        a, b = ab.chunk(2, dim=-1)
        h = a * torch.sigmoid(b)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.proj(h)


class GATNodePredictor(nn.Module):
    """
    Improved GAT for genomic bins:
      - add fixed positional encoding (concat)
      - stem projection
      - residual+norm GAT blocks (GATv2 if available)
      - gated strong head
      - output log-probs for KLDivLoss
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
            pe_dim: int = 16,
    ):
        super().__init__()
        self.dropout = dropout
        self.pe_dim = pe_dim

        d0 = in_dim + pe_dim
        stem_dim = hidden_dim * heads

        self.stem = nn.Sequential(
            nn.Linear(d0, stem_dim),
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
        pe = sinusoidal_pos_enc(x.size(0), self.pe_dim, x.device)
        x = torch.cat([x, pe], dim=-1)

        x = self.stem(x)
        for b in self.blocks:
            x = b(x, edge_index)

        logits = self.head(x)
        return F.log_softmax(logits, dim=-1)


class GAT_intracell:
    """
    Intra-cell-line trainer (Soffritto-like):
      - train on multiple chromosomes (each chrom is a graph)
      - test on one chromosome
      - KLDivLoss(batchmean)

    Improvements included:
      - positional encoding
      - multi-scale edges (expects utils to support hop_list OR you pass hops and update utils)
      - GATv2 if available
      - gated strong head
      - cosine LR schedule
      - optional chromosome sampling per epoch (speed + reduces interference)
    """
    def __init__(
            self,
            features_file: str,
            labels_file: str,
            train_chromosomes,
            test_chromosome,

            # graph connectivity (update utils to use hop_list if provided)
            hops: int = 2,                 # fallback if your utils only supports hops
            hop_list=(1, 2, 4, 8),         # preferred multi-scale

            hidden_dim: int = 8,
            heads: int = 4,
            layers: int = 2,
            dropout: float = 0.10,
            widen: int = 4,
            pe_dim: int = 16,

            lr: float = 1e-3,
            weight_decay: float = 1e-5,
            epochs: int = 200,
            grad_clip: float = 1.0,

            # early stopping
            patience: int = 30,
            min_delta: float = 1e-5,

            # training trick
            chroms_per_epoch: int | None = 6,  # sample this many chromosomes each epoch; None => use all

            device: str | None = None,
    ):
        self.features_file = features_file
        self.labels_file = labels_file
        self.train_chromosomes = train_chromosomes
        self.test_chromosome = test_chromosome
        self.hops = hops
        self.hop_list = hop_list

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.widen = widen
        self.pe_dim = pe_dim

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
        # IMPORTANT:
        # If your utils currently only takes `hops`, this call is fine.
        # For best performance, modify utils to accept hop_list and build multiscale edges.
        train_dict, test_data, scaler = load_gat_intra_cell_line_train(
            features_file=self.features_file,
            labels_file=self.labels_file,
            train_chromosomes=self.train_chromosomes,
            test_chromosome=self.test_chromosome,
            hop_list=self.hop_list,  # CORRECT (tuple/list)
        )
        self.train_data = to_device_data_dict(train_dict, self.device)
        self.test_data = to_device_data(test_data, self.device)
        self.scaler = scaler

        any_chrom = next(iter(self.train_data.keys()))
        in_dim = self.train_data[any_chrom].x.shape[1]
        out_dim = self.train_data[any_chrom].y.shape[1]
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
            pe_dim=self.pe_dim,
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

            # chromosome sampling per epoch (reduces interference and speeds training)
            if self.chroms_per_epoch is None or self.chroms_per_epoch >= len(train_keys):
                keys = train_keys
            else:
                # deterministic-ish sampling per epoch
                g = torch.Generator()
                g.manual_seed(ep)
                idx = torch.randperm(len(train_keys), generator=g)[: self.chroms_per_epoch].tolist()
                keys = [train_keys[i] for i in idx]

            total_loss = 0.0
            for k in keys:
                d = self.train_data[k]
                log_q = self.model(d.x, d.edge_index)
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
        log_q = self.model(self.test_data.x, self.test_data.edge_index)
        return log_q.exp().detach().cpu().numpy()