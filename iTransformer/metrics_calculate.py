import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

class GenomicMetrics:
    def __init__(self, true_path, pred_path, eps=1e-8):
        # Load and squeeze out the singleton dimension
        raw_true   = np.load(true_path)   # shape: (N, 1, 16)
        raw_pred   = np.load(pred_path)   # shape: (N, 1, 16)
        true       = np.squeeze(raw_true, axis=1)   # → (N, 16)
        log_pred   = np.squeeze(raw_pred, axis=1)   # → (N, 16), log-probabilities

        # 1) Normalize true profiles to sum to 1
        self.true = np.clip(true, 0, None)
        self.true /= (self.true.sum(axis=1, keepdims=True) + eps)

        # 2) Store log-probs and compute probabilities
        self.log_pred = log_pred
        self.pred     = np.exp(self.log_pred)  # now sums to 1 per row

        # Prep
        self.N, self.K    = self.true.shape
        self.positions    = np.arange(self.K)
        self.eps          = eps

    def kl_divergence(self):
        """
        Uses PyTorch KLDivLoss: input = log-probs, target = probs
        """
        p     = torch.from_numpy(self.true).float()
        log_q = torch.from_numpy(self.log_pred).float()
        loss  = F.kl_div(log_q, p, reduction='batchmean')
        return loss.item()

    def mse(self):
        return np.mean((self.true - self.pred) ** 2)

    def pearson_r(self):
        return stats.pearsonr(self.true.ravel(), self.pred.ravel())[0]

    def spearman_r(self):
        return stats.spearmanr(self.true.ravel(), self.pred.ravel())[0]

    def wasserstein(self):
        dists = [
            wasserstein_distance(self.positions, self.positions, t, p)
            for t, p in zip(self.true, self.pred)
        ]
        return np.mean(dists)

    def ks_statistic(self):
        ks_vals = [
            np.max(np.abs(np.cumsum(t) - np.cumsum(p)))
            for t, p in zip(self.true, self.pred)
        ]
        return np.mean(ks_vals)

    def all_metrics(self):
        return {
            'KL Divergence':        self.kl_divergence(),
            'Mean Squared Error':   self.mse(),
            "Pearson's r":          self.pearson_r(),
            "Spearman's ρ":         self.spearman_r(),
            'Wasserstein Dist.':    self.wasserstein(),
            'KS Statistic':         self.ks_statistic(),
        }

    def plot_heatmaps(self, cmap='gray_r'):
        """
        Displays heatmaps of the true vs. predicted 16-fraction profiles.
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        im0 = axes[0].imshow(self.true, aspect='auto', cmap=cmap)
        axes[0].set_title('True 16-Fraction Profiles')
        axes[0].set_ylabel('Genomic Bin')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(self.pred, aspect='auto', cmap=cmap)
        axes[1].set_title('Predicted 16-Fraction Profiles')
        axes[1].set_ylabel('Genomic Bin')
        axes[1].set_xlabel('Fraction Index')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

# ── Example Usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    gm = GenomicMetrics(
        '/Users/ozgun/DataspellProjects/CS401-soffritto/results/test_iTransformer_custom_ftM_sl96_ll48_pl1_dm512_nh4_el2_dl2_df2048_fc1_ebtimeF_dtTrue_test_projection/true.npy',
        '/Users/ozgun/DataspellProjects/CS401-soffritto/results/test_iTransformer_custom_ftM_sl96_ll48_pl1_dm512_nh4_el2_dl2_df2048_fc1_ebtimeF_dtTrue_test_projection/pred.npy'
    )
    print(gm.all_metrics())
    gm.plot_heatmaps()