# run_bert_genomic.py

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader.data_loader import Dataset_Custom  # <-- change this

class GenomicBERT(nn.Module):
    def __init__(
        self,
        enc_in: int = 9,
        c_out: int = 16,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
        rt2_idx: int = 8,   # index of 2RT feature among the 9 inputs
    ):
        super().__init__()
        self.enc_in = enc_in
        self.c_out = c_out
        self.d_model = d_model
        self.rt2_idx = rt2_idx

        # 9 features -> d_model
        self.feat_proj = nn.Linear(enc_in, d_model)

        # learned positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))

        # Transformer encoder (BERT-like)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,   # [B, L, D]
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # nonlinear head: [context + 2RT] -> 16 bins
        hidden_head = 128
        self.head = nn.Sequential(
            nn.Linear(d_model + 1, hidden_head),
            nn.GELU(),
            nn.Linear(hidden_head, c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        assert C == self.enc_in, f"Expected enc_in={self.enc_in}, got {C}"

        # take 2RT scalar from last time step
        rt2 = x[:, -1, self.rt2_idx:self.rt2_idx+1]   # [B, 1]

        # feature projection + position
        h = self.feat_proj(x)                         # [B, L, D]
        pos = self.pos_embed[:, :L, :]                # [1, L, D]
        h = h + pos

        # BERT encoder
        h_enc = self.encoder(h)                       # [B, L, D]

        # use last token as "CLS"
        h_last = h_enc[:, -1, :]                      # [B, D]

        # concat context rep + 2RT scalar
        h_cat = torch.cat([h_last, rt2], dim=-1)      # [B, D+1]
        logits = self.head(h_cat)                     # [B, 16]
        log_probs = F.log_softmax(logits, dim=-1)     # [B, 16]

        return log_probs.unsqueeze(1)                 # [B, 1, 16]


def build_loaders(
    root_path,
    data_path,
    train_chroms,
    val_chroms,
    seq_len=32,
    label_len=16,
    pred_len=1,
    batch_size=256,
):
    size = [seq_len, label_len, pred_len]

    train_ds = Dataset_Custom(
        root_path=root_path,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=val_chroms,
        flag="train",
        size=size,
        features="M",
        data_path=data_path,
        target="target_1",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="w",
        selected_cols=[
                'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1',
                'H3K4me3', 'H3K9me3', 'GC_content', 'gene_density', '2-stage'
            ]
    )

    val_ds = Dataset_Custom(
        root_path=root_path,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=val_chroms,
        flag="val",
        size=size,
        features="M",
        data_path=data_path,
        target="target_1",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="w",
        selected_cols=[
            'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1',
            'H3K4me3', 'H3K9me3', 'GC_content', 'gene_density', '2-stage'
        ]
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False
    )

    print("train len:", len(train_ds))
    print("val   len:", len(val_ds))
    return train_loader, val_loader


def main():
    # ======= CONFIG YOU NEED TO SET =======
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    ROOT_PATH = str(PROJECT_ROOT / "/CS401-soffritto/GenomicBert/data")   # <- change if needed
    DATA_PATH = "H1_genomic.csv"                               # <- or your csv name

    # same splits you used for Informer
    TRAIN_CHROMS = [1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22]
    VAL_CHROMS   = [13]

    seq_len   = 32
    label_len = 16
    pred_len  = 1
    enc_in    = 9     # 9 genomic features (incl. 2RT)
    c_out     = 16    # 16 RT bins

    batch_size = 256
    epochs     = 5
    lr         = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # loaders from your Dataset_Custom
    train_loader, val_loader = build_loaders(
        root_path=ROOT_PATH,
        data_path=DATA_PATH,
        train_chroms=TRAIN_CHROMS,
        val_chroms=VAL_CHROMS,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        batch_size=batch_size,
    )

    # model
    model = GenomicBERT(
        enc_in=enc_in,
        c_out=c_out,
        d_model=256,
        n_heads=4,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
        max_len=seq_len,
        rt2_idx=8,      # if 2RT is at a different index, change this
    ).to(device)

    # KLDiv between model log-probs and target probs (or one-hot)
    criterion = nn.KLDivLoss(reduction="batchmean", log_target=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ====== TRAIN ======
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.float().to(device)  # [B, seq_len, 9]
            batch_y = batch_y.float().to(device)  # [B, label_len+pred_len, 16]

            # we only train on last pred_len step
            target = batch_y[:, -pred_len:, :]     # [B, 1, 16]

            optimizer.zero_grad()
            log_probs = model(batch_x)            # [B, 1, 16]

            loss = criterion(log_probs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # ====== VALIDATION ======
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                target = batch_y[:, -pred_len:, :]

                log_probs = model(batch_x)
                loss = criterion(log_probs, target)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch}: train_loss={avg_train:.6f}  val_loss={avg_val:.6f}")

    # quick sanity forward
    model.eval()
    batch_x, batch_y, _, _ = next(iter(val_loader))
    batch_x = batch_x.float().to(device)
    with torch.no_grad():
        out = model(batch_x)  # [B, 1, 16]
    print("Sanity output shape:", out.shape)


if __name__ == "__main__":
    main()