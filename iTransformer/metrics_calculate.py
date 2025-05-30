import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance

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

# ── Usage ──────────────────────────────────────────────────────────────────
gm = GenomicMetrics('/Users/ozgun/DataspellProjects/CS401-soffritto/results/exp1_iTransformer_custom_M_ft96_sl48_ll1_pl512_dm8_nh2_el1_dl2048_df1_fctimeF_ebTrue_dttest_projection_0/true.npy', '/Users/ozgun/DataspellProjects/CS401-soffritto/results/exp1_iTransformer_custom_M_ft96_sl48_ll1_pl512_dm8_nh2_el1_dl2048_df1_fctimeF_ebTrue_dttest_projection_0/pred.npy')
results = gm.all_metrics()
for name, val in results.items():
    print(f"{name:25s}: {val:.6f}")

