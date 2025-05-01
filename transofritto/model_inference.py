#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
import random

# reproducibility
torch.manual_seed(42)
numpy = np  # alias
np.random.seed(42)
random.seed(42)
# enforce deterministic operations when possible
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch.utils.data import DataLoader
import torch.nn as nn

# allow imports from project root
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from transofritto.informer.exp.exp_informer import Exp_Informer
from transofritto.informer.data.data_loader import Dataset_Custom


def load_checkpoint(exp, path, device):
    """Load state_dict into Exp_Informer, stripping DataParallel prefixes and handling directories."""
    if os.path.isdir(path):
        ckpt_path = os.path.join(path, 'checkpoint.pth')
    else:
        ckpt_path = path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get('state_dict', ckpt)
    clean = OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())
    exp.model.load_state_dict(clean)


def run_inference():
    # Hardcoded settings
    checkpoint = 'transofritto/best_model/checkpoint.pth'
    split = 'val'
    batch_size = 256

    # Hyperparameters from best Optuna trial (best: trial 15)
    # Format: seq_len,label_len,pred_len,d_model,e_layers,d_layers,d_ff,dropout,factor,learning_rate,mix,n_heads
    hp = {
        'seq_len': 27,
        'label_len': 13,
        'pred_len': 1,
        'd_model': 128,
        'e_layers': 1,
        'd_layers': 2,
        'n_heads': 8,
        'd_ff': 512,
        'dropout': 0.15460122489925412,
        'factor': 3,
        'learning_rate': 0.0009593857258681762,
        'mix': True
    }

    # Build Exp_Informer args
    args = type('A', (), {})()
    constants = dict(
        model='informer', data='custom', features='M', target='target_1', freq='w',
        root_path='data', data_path='H1_genomic.csv', checkpoints='./checkpoints/',
        enc_in=9, dec_in=16, c_out=16,
        num_workers=0, use_multi_gpu=False, use_gpu=False,
        devices='0', gpu=0, inverse=False, output_attention=False,
        distil=True, attn='prob', factor=3, embed='timeF', activation='gelu'
    )
    params = {**constants, **hp, 'batch_size': batch_size}
    for k, v in params.items(): setattr(args, k, v)

    # Force CPU
    device = torch.device('cpu')
    args.device = device

    # Initialize and load model
    exp = Exp_Informer(args)
    exp.device = device
    exp.model.to(device)
    load_checkpoint(exp, checkpoint, device)
    exp.model.eval()

    # Prepare data loader
    data_obj, _ = exp._get_data(flag=split)
    loader = DataLoader(data_obj, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    all_preds, all_reals = [], []
    with torch.no_grad():
        for bx, by, bxm, bym in loader:
            pred_log, true = exp._process_one_batch(data_obj, bx, by, bxm, bym)
            all_preds.append(torch.exp(pred_log).numpy())
            all_reals.append(true.numpy())

    # Concatenate and reshape
    preds = np.concatenate(all_preds, axis=0).reshape(-1, args.c_out)
    reals = np.concatenate(all_reals, axis=0).reshape(-1, args.c_out)
    print(f"preds shape: {preds.shape}  reals shape: {reals.shape}")

    # Compute KL divergence
    kl_fn = nn.KLDivLoss(reduction='batchmean')
    mean_kl = kl_fn(torch.from_numpy(preds).log(), torch.from_numpy(reals))
    print(f"Mean KL divergence: {mean_kl.item():.6f}")

    # Save results
    df = pd.DataFrame({'real': reals.tolist(), 'pred': preds.tolist()})
    out_csv = os.path.join(os.path.dirname(checkpoint), 'results.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")


if __name__ == '__main__':
    run_inference()
