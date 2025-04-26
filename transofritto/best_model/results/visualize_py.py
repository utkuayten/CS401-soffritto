#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    fname = 'transofritto/best_model/results/results.csv'
    if not os.path.isfile(fname):
        print(f"Error: '{fname}' not found. Please place it in the current directory.")
        return

    # 1. Load CSV
    df = pd.read_csv(fname)

    # 2. Parse the 'real' and 'pred' columns into numpy arrays
    real = np.vstack(
        df['real']
          .str.strip('[]')
          .str.split(',')
          .apply(lambda lst: [float(x) for x in lst])
          .to_list()
    )
    pred = np.vstack(
        df['pred']
          .str.strip('[]')
          .str.split(',')
          .apply(lambda lst: [float(x) for x in lst])
          .to_list()
    )

    # 3. Plot side-by-side heatmaps (light gray colormap, subplots)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    # True distributions
    im0 = axes[0].imshow(real, aspect='auto', cmap='gray', alpha=0.7)
    axes[0].set_title('True Distributions')
    axes[0].set_xlabel('Fraction Index')
    axes[0].set_ylabel('Sample Index')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Predicted distributions
    im1 = axes[1].imshow(pred, aspect='auto', cmap='gray', alpha=0.7)
    axes[1].set_title('Predicted Distributions')
    axes[1].set_xlabel('Fraction Index')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Use a light gray background
    fig.patch.set_facecolor('lightgray')
    plt.tight_layout()
    plt.show()

    # 4. Compute mean KL divergence using numpy
    eps = 1e-8
    P = real + eps
    Q = pred + eps
    kl_per_sample = np.sum(P * (np.log(P) - np.log(Q)), axis=1)
    mean_kl = kl_per_sample.mean()
    print(f"Mean KL divergence: {mean_kl:.6f}")

if __name__ == "__main__":
    main()
