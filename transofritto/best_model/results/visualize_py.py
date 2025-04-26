#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def main():
    # Path to your results CSV
    fname = 'transofritto/best_model/results/results.csv'
    if not os.path.isfile(fname):
        print(f"Error: '{fname}' not found.")
        return

    # Load the CSV
    df = pd.read_csv(fname)

    # Parse the 'real' and 'pred' columns into numpy arrays
    def parse_col(col):
        return np.vstack(
            df[col]
              .str.strip('[]')
              .str.split(',')
              .apply(lambda lst: [float(x) for x in lst])
              .to_list()
        )
    real = parse_col('real')  # shape (N, c_out)
    pred = parse_col('pred')  # shape (N, c_out)

    # Plot side-by-side heatmaps in light gray
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    im0 = axes[0].imshow(real, aspect='auto', cmap='gray', alpha=0.8)
    axes[0].set_title('True Distributions')
    axes[0].set_xlabel('Fraction Index')
    axes[0].set_ylabel('Sample Index')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred, aspect='auto', cmap='gray', alpha=0.8)
    axes[1].set_title('Predicted Distributions')
    axes[1].set_xlabel('Fraction Index')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.patch.set_facecolor('lightgray')
    plt.tight_layout()
    plt.show()

    # Compute mean KL divergence using torch.nn.KLDivLoss
    # Convert to torch tensors
    P = torch.from_numpy(real).float()   # true distributions
    Q = torch.from_numpy(pred).float()   # predicted distributions
    log_Q = torch.log(Q)

    # KLDivLoss expects input=log_probs, target=probs
    kl_fn = nn.KLDivLoss(reduction='batchmean')
    mean_kl = kl_fn(log_Q, P).item()
    print(f"Mean KL divergence: {mean_kl:.6f}")

if __name__ == "__main__":
    main()
