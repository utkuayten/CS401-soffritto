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

from utils import load_gat_intra_cell_line_train, to_device_data_dict, to_device_data


def build_chunk_edges(n: int, hop_list=(1, 2, 4), device=None) -> torch.Tensor:
    """Undirected multi-hop edges inside a CHUNK of length n."""
    device = device or "cpu"
    src_chunks, dst_chunks = [], []
    for hop in hop_list:
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
    """Small local GAT over a chunk; acts like a learnable 'conv'."""
    def __init__(self, in_dim: int, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.gat = _GAT(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.out_dim = hidden_dim * heads
        self.norm = nn.LayerNorm(self.out_dim)
        self.proj = nn.Linear(in_dim, self.out_dim) if in_dim != self.out_dim else nn.Identity()

    def forward(self, x, edge_index):
        # x: [T, F] for chunk length T
        h0 = self.proj(x)
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.gat(h, edge_index)
        h = F.elu(h)
        return self.norm(h + h0)


class GATxSoffritto(nn.Module):
    """
    Chunk-level pipeline:
      chunk x [T,F] -> local GAT -> z [T,D] -> BiLSTM(stateful) -> fc -> log_softmax
    """
    def __init__(self, in_dim: int, gat_hidden: int, gat_heads: int,
                 lstm_hidden: int, lstm_layers: int, out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.gat = LocalGATEncoder(in_dim, gat_hidden, gat_heads, dropout)
        d = self.gat.out_dim

        # Important: match Soffritto style (unbatched sequence input)
        self.lstm = nn.LSTM(d, lstm_hidden, lstm_layers, bidirectional=True)
        self.fc = nn.Linear(2 * lstm_hidden, out_dim)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.hidden = None  # stateful across chunks

    def init_hidden(self, device):
        # Unbatched mode hidden shape: (num_layers*num_directions, hidden)
        h0 = torch.zeros(2 * self.lstm_layers, self.lstm_hidden, device=device)
        c0 = torch.zeros(2 * self.lstm_layers, self.lstm_hidden, device=device)
        return (h0, c0)

    def reset_hidden(self, device):
        self.hidden = self.init_hidden(device)

    def forward_chunk(self, x_chunk: torch.Tensor, edge_index: torch.Tensor):
        """
        x_chunk: [T, F]
        returns log_probs: [T, C]
        """
        if self.hidden is None:
            self.hidden = self.init_hidden(x_chunk.device)

        # local GAT encodes within chunk
        z = self.gat(x_chunk, edge_index)   # [T, D]

        # LSTM expects [seq_len, input_size] in unbatched mode => OK
        out, self.hidden = self.lstm(z, self.hidden)

        # truncate BPTT exactly like Soffritto
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        logits = self.fc(out)               # [T, C]
        return self.log_softmax(logits)


class GAT_intracell:
    def __init__(
            self,
            features_file: str,
            labels_file: str,
            train_chromosomes,
            test_chromosome,

            hop_list=(1, 2, 4),

            # GAT encoder params
            gat_hidden: int = 8,
            gat_heads: int = 2,

            # LSTM params (Soffritto-like)
            num_hiddens: int = 64,
            num_layers: int = 2,

            dropout: float = 0.10,

            lr: float = 1e-3,
            weight_decay: float = 1e-6,
            epochs: int = 100,
            grad_clip: float = 1.0,

            # Soffritto "batch_size" is actually chunk_len
            chunk_len: int = 512,

            patience: int = 20,
            min_delta: float = 1e-5,
            chroms_per_epoch: int | None = None,

            device: str | None = None,
    ):
        self.features_file = features_file
        self.labels_file = labels_file
        self.train_chromosomes = train_chromosomes
        self.test_chromosome = test_chromosome
        self.hop_list = tuple(hop_list)

        self.gat_hidden = gat_hidden
        self.gat_heads = gat_heads
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.dropout = dropout

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.chunk_len = int(chunk_len)

        self.patience = patience
        self.min_delta = min_delta
        self.chroms_per_epoch = chroms_per_epoch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.train_data = None
        self.test_data = None
        self.best_state = None
        self.best_test_kl = float("inf")

    def prepare_data(self):
        train_dict, test_data, scaler = load_gat_intra_cell_line_train(
            features_file=self.features_file,
            labels_file=self.labels_file,
            train_chromosomes=self.train_chromosomes,
            test_chromosome=self.test_chromosome,
            hop_list=(1,),  # ignored; we build per-chunk edges
        )
        self.train_data = to_device_data_dict(train_dict, self.device)
        self.test_data = to_device_data(test_data, self.device)

        any_chrom = next(iter(self.train_data.keys()))
        in_dim = int(self.train_data[any_chrom].x.shape[1])
        out_dim = int(self.train_data[any_chrom].y.shape[1])
        return in_dim, out_dim

    def build_model(self, in_dim, out_dim):
        self.model = GATxSoffritto(
            in_dim=in_dim,
            gat_hidden=self.gat_hidden,
            gat_heads=self.gat_heads,
            lstm_hidden=self.num_hiddens,
            lstm_layers=self.num_layers,
            out_dim=out_dim,
            dropout=self.dropout,
        ).to(self.device)

    def _stream_chrom_loss(self, x: torch.Tensor, y: torch.Tensor, train: bool):
        assert self.model is not None
        self.model.reset_hidden(x.device)

        loss_fn = nn.KLDivLoss(reduction="batchmean")

        total_kl_sum = 0.0
        total_n = 0

        n = x.size(0)
        for s in range(0, n, self.chunk_len):
            e = min(s + self.chunk_len, n)
            x_chunk = x[s:e]
            y_chunk = y[s:e]

            edge_index = build_chunk_edges(e - s, hop_list=self.hop_list, device=x.device)
            log_q = self.model.forward_chunk(x_chunk, edge_index)

            # batchmean = (sum_kl_chunk / chunk_n)
            loss = loss_fn(log_q, y_chunk)

            if train:
                loss.backward()

            chunk_n = (e - s)
            total_kl_sum += float(loss.detach()) * chunk_n
            total_n += chunk_n

        return total_kl_sum / max(1, total_n)

    def fit(self):
        in_dim, out_dim = self.prepare_data()
        self.build_model(in_dim, out_dim)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        bad_epochs = 0
        train_keys = list(self.train_data.keys())

        for ep in range(1, self.epochs + 1):
            self.model.train()
            opt.zero_grad(set_to_none=True)

            # optionally sample chroms like your previous setup
            keys = train_keys
            if self.chroms_per_epoch is not None and self.chroms_per_epoch < len(train_keys):
                g = torch.Generator(device="cpu")
                g.manual_seed(ep)
                idx = torch.randperm(len(train_keys), generator=g)[: self.chroms_per_epoch].tolist()
                keys = [train_keys[i] for i in idx]

            # train: accumulate grads across chromosomes (like soffritto loops)
            train_loss = 0.0
            for k in keys:
                d = self.train_data[k]
                train_loss += self._stream_chrom_loss(d.x, d.y, train=True)
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
                print(f"epoch {ep:03d} | train_KL={train_loss:.6f} | test_KL={test_kl:.6f} | best={self.best_test_kl:.6f}")

            if bad_epochs >= self.patience:
                print(f"Early stop at epoch {ep:03d} (best test_KL={self.best_test_kl:.6f})")
                break

            opt.zero_grad(set_to_none=True)

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return self

    @torch.no_grad()
    def evaluate_kl(self):
        self.model.eval()
        d = self.test_data
        return float(self._stream_chrom_loss(d.x, d.y, train=False))

    @torch.no_grad()
    def predict_test_probs(self):
        """
        Stream test chrom and concatenate outputs => [N, C]
        """
        self.model.eval()
        d = self.test_data
        x = d.x
        n = x.size(0)

        self.model.reset_hidden(x.device)
        chunks = []

        for s in range(0, n, self.chunk_len):
            e = min(s + self.chunk_len, n)
            edge_index = build_chunk_edges(e - s, hop_list=self.hop_list, device=x.device)
            log_q = self.model.forward_chunk(x[s:e], edge_index)
            chunks.append(log_q.exp().detach().cpu())

        return torch.cat(chunks, dim=0).numpy()