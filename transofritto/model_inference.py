#!/usr/bin/env python3
import os, sys
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from torch.utils.data import DataLoader
import torch.nn as nn


# allow imports from root
THIS_DIR    = os.path.dirname(__file__)
PROJECT_ROOT= os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from transofritto.informer.exp.exp_informer import Exp_Informer
from transofritto.informer.data.data_loader import Dataset_Custom  # ← adjust this import

def load_checkpoint(exp, path, device):
    ckpt = torch.load(path, map_location=device)
    sd   = ckpt.get("state_dict", ckpt)
    new  = OrderedDict((k.replace("module.",""),v) for k,v in sd.items())
    exp.model.load_state_dict(new)

def main():
    # ─── settings (must match your training) ─────────────────────────────────
    checkpoint_path = "transofritto/best_model/checkpoint.pth"
    enc_in, dec_in, c_out = 9, 16, 16
    seq_len, label_len, pred_len = 32, 16, 1
    # build args for Exp_Informer (fill in your arch params) …
    args = type("A",(),{})()
    for k,v in dict(
        model="informer", data="custom", features="MS", target="target_1", freq="g",
        root_path="data", data_path="H1_genomic.csv", checkpoints="./checkpoints/",
        enc_in=enc_in, dec_in=dec_in, c_out=c_out,
        seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        e_layers=2, d_layers=2, d_model=512, n_heads=8, d_ff=2048,
        dropout=0.14, attn="prob", factor=5, embed="timeF", activation="gelu",
        mix=False, distil=True, output_attention=False, use_multi_gpu=False,
        use_gpu=False, gpu=0, devices="0", num_workers=0
    ).items(): setattr(args, k, v)

    # pick device
    if   torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available():         device = torch.device("cuda")
    else:                                   device = torch.device("cpu")

    # build and load model
    exp = Exp_Informer(args)
    exp.device = device
    exp.model.to(device)
    load_checkpoint(exp, checkpoint_path, device)
    exp.model.eval()

    # prepare your test loader
    ds = Dataset_Custom(
        root_path="data", flag='test',
        size=(seq_len, label_len, pred_len),
        features="MS", data_path="H1_genomic.csv",
        target="target_1", scale=True, inverse=False, timeenc=0, freq='w',
        cols=[f"target_{i+1}" for i in range(c_out)]
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_preds = []
    all_reals = []

    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in loader:
            # cast to float32 then move to device
            x_enc      = seq_x.float().to(device)
            x_enc_mark = seq_x_mark.float().to(device)
            x_dec      = seq_y[:, :label_len].float().to(device)      # history portion
            x_dec_mark = seq_y_mark[:, :label_len].float().to(device)

            # forward
            out = exp.model(x_enc, x_enc_mark, x_dec, x_dec_mark)     # [B, pred_len, c_out]
            preds = out.cpu().numpy()                                 # (B,1,c_out)

            # grab the *true* next-step values from seq_y
            reals = seq_y[:, label_len:, :].cpu().numpy()             # (B,1,c_out)

            all_preds.append(preds)
            all_reals.append(reals)

    # stack and reshape to (N, c_out)
    preds = np.concatenate(all_preds, axis=0)
    reals = np.concatenate(all_reals, axis=0)

    print(preds.shape, reals.shape)

    probs = np.exp(preds)
    probs = probs / probs.sum(axis=2, keepdims=True)
    preds = probs

    preds_flat = torch.from_numpy(preds.squeeze(1)).float()
    reals_flat = torch.from_numpy(reals.squeeze(1)).float()

    kl_fn = nn.KLDivLoss(reduction="batchmean")
    mean_kl = kl_fn(preds_flat.log(), reals_flat)

    print(f"Mean KL divergence: {mean_kl.item():.6f}")

    real_lists = [row[0].tolist() for row in reals]
    pred_lists = [row[0].tolist() for row in preds]

    df = pd.DataFrame({
        "real": real_lists,
        "pred": pred_lists
    })

    df.to_csv("transofritto/best_model/results/results.csv", index=False)
    print("Saved real vs. predicted distributions to results.csv")

if __name__ == "__main__":
    main()
