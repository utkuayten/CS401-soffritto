import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
from scipy.stats import spearmanr, ks_2samp, wasserstein_distance

def metric(x, y, eps=1e-8):
    # Ensure tensor input
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    # Fix shape: squeeze (N, 1, 16) → (N, 16)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x.squeeze(1)
    if y.ndim == 3 and y.shape[1] == 1:
        y = y.squeeze(1)

    # Convert to probability space
    p = x.exp().clamp(min=eps)         # log(p) → p
    q = y.clamp(min=eps)
    q = q / q.sum(dim=1, keepdim=True) # Ensure normalization

    # Convert to NumPy for metrics
    p_np = p.cpu().numpy()
    q_np = q.cpu().numpy()

    # Classical metrics
    mae = mean_absolute_error(q_np.flatten(), p_np.flatten())
    mse = mean_squared_error(q_np.flatten(), p_np.flatten())
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(q_np.flatten(), p_np.flatten())
    mspe = np.mean(((q_np - p_np) / (q_np + eps))**2)

    # KL divergence: KL(q || p)
    kl = F.kl_div(x, q, reduction='batchmean').item()

    # Spearman correlation (mean per row)
    rho_list = []
    for pi, qi in zip(p_np, q_np):
        corr = spearmanr(pi, qi).correlation
        if not np.isnan(corr):
            rho_list.append(corr)
    spearman_r = np.mean(rho_list) if rho_list else float('nan')

    # Wasserstein distance
    support = np.arange(p_np.shape[1])
    wass = np.mean([
        wasserstein_distance(support, support, pi / pi.sum(), qi / qi.sum())
        for pi, qi in zip(p_np, q_np)
    ])

    # KS statistic
    ks = np.mean([
        ks_2samp(pi, qi).statistic
        for pi, qi in zip(p_np, q_np)
    ])

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'MSPE': mspe,
        'KL': kl,
        'SpearmanR': spearman_r,
        'Wasserstein': wass,
        'KSstat': ks
    }
