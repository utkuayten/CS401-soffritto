import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

class GenomicMetrics:
    def __init__(self, true_path, pred_path, eps=1e-8):
        # 1) Load and squeeze out the singleton “channel” dimension
        raw_true   = np.load(true_path)   # e.g. (2370, 1, 16)
        raw_pred   = np.load(pred_path)   # e.g. (2370, 1, 16)
        true       = np.squeeze(raw_true, axis=1)   # → (2370, 16)
        pred_logits= np.squeeze(raw_pred, axis=1)   # → (2370, 16)

        # 2) True profiles → ensure nonnegative and sum to 1
        self.true = np.clip(true, 0, None)
        self.true /= (self.true.sum(axis=1, keepdims=True) + eps)

        # 3) Pred logits → softmax → probability vectors
        shifted    = pred_logits - pred_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        self.pred  = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + eps)

        # prep for per‐bin metrics
        self.N, self.K    = self.true.shape
        self.positions   = np.arange(self.K)  # [0,1,…,15]
        self.eps         = eps

    def kl_divergence(self):
        p = self.true + self.eps
        q = self.pred + self.eps
        # per‐bin KL, then average
        return np.mean(np.sum(p * np.log(p / q), axis=1))

    def mse(self):
        return np.mean((self.true - self.pred) ** 2)

    def pearson_r(self):
        return stats.pearsonr(self.true.ravel(), self.pred.ravel())[0]

    def spearman_r(self):
        return stats.spearmanr(self.true.ravel(), self.pred.ravel())[0]

    def wasserstein(self):
        # per‐bin Earth Mover’s Distance, then average
        dists = [
            wasserstein_distance(self.positions, self.positions, t, p)
            for t, p in zip(self.true, self.pred)
        ]
        return np.mean(dists)

    def ks_statistic(self):
        # per‐bin max CDF difference, then average
        ks_vals = [
            np.max(np.abs(np.cumsum(t) - np.cumsum(p)))
            for t, p in zip(self.true, self.pred)
        ]
        return np.mean(ks_vals)

    def all_metrics(self):
        return {
            'KL Divergence':          self.kl_divergence(),
            'Mean Squared Error':     self.mse(),
            "Pearson's r":            self.pearson_r(),
            "Spearman's ρ":           self.spearman_r(),
            'Wasserstein Distance':   self.wasserstein(),
            'KS Statistic':           self.ks_statistic(),
        }

    def plot_heatmaps(self):
        """
        Displays heatmaps of the true vs. predicted 16-fraction profiles.
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        im0 = axes[0].imshow(self.true, aspect='auto', cmap='gray_r')
        axes[0].set_title('True 16-Fraction Profiles')
        axes[0].set_ylabel('Genomic Bin')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(self.pred, aspect='auto', cmap='gray_r')
        axes[1].set_title('Predicted 16-Fraction Profiles')
        axes[1].set_ylabel('Genomic Bin')
        axes[1].set_xlabel('Fraction Index')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

# ── Usage ──────────────────────────────────────────────────────────────────
gm = GenomicMetrics('/Users/ozgun/DataspellProjects/CS401-soffritto/results/test_iTransformer_custom_ftM_sl48_ll24_pl1_dm512_nh4_el2_dl2_df2048_fc1_ebtimeF_dtTrue_test_projection/true.npy', '/Users/ozgun/DataspellProjects/CS401-soffritto/results/test_iTransformer_custom_ftM_sl48_ll24_pl1_dm512_nh4_el2_dl2_df2048_fc1_ebtimeF_dtTrue_test_projection/true.npy')
print(gm.all_metrics())
gm.plot_heatmaps()

