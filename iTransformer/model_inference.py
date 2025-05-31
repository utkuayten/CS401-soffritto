import os
import random
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import stats, wasserstein_distance
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
# replace with your actual import paths
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast

# reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_checkpoint(exp, path, device):
    ckpt = os.path.join(path, "checkpoint.pth") if os.path.isdir(path) else path
    data = torch.load(ckpt, map_location=device)
    sd   = data.get("state_dict", data)
    clean= {k.replace("module.", ""): v for k, v in sd.items()}
    exp.model.load_state_dict(clean)
def run_iTransformer_inference():
    # --- settings ---
    checkpoint = "/Users/ozgun/DataspellProjects/CS401-soffritto/checkpoints/test_iTransformer_custom_ftM_sl96_ll48_pl1_dm512_nh4_el2_dl2_df2048_fc1_ebtimeF_dtTrue_test_projection/checkpoint.pth"
    batch_size = 128

    # ─── Hyperparameters (hp) ───────────────────────────────────────────────────
    hp = dict(
        # model & mode
        is_training       = 0,                # inference mode
        model_id          = 'best_iTransformer',
        model             = 'iTransformer',

        # forecasting task
        seq_len           = 96,
        label_len         = 48,
        pred_len          = 1,
        freq              = 'h',

        # data dimensions
        enc_in            = 9,
        dec_in            = 16,
        c_out             = 16,

        # transformer architecture
        d_model           = 512,
        n_heads           = 4,
        e_layers          = 2,
        d_layers          = 2,
        d_ff              = 2048,
        moving_avg        = 25,
        factor            = 1,
        distil            = True,             # default distilling on
        dropout           = 0.28817048937,
        embed             = 'timeF',
        activation        = 'gelu',
        output_attention  = False,
        do_predict        = True,
        use_multi_gpu     = False,
        checkpoint = checkpoint,

        # iTransformer-specific
        exp_name               = 'MTSF',
        channel_independence   = False,
        inverse                = False,
        class_strategy         = 'projection',
        use_norm               = 0,           # boolean flag as int
        efficient_training     = False,
        partial_start_index    = 0,
        setting                = 'best_params_run',
    )

    # ─── Constants (const) ───────────────────────────────────────────────────────
    const = dict(
        data            = 'custom',
        root_path       = './iTransformer/data/',
        data_path       = 'H1_genomic.csv',
        features        = 'M',
        target          = 'target_1',
        checkpoints     = './checkpoints/',
        train_chroms    = [1,2,3,4,5,6,7,10,11,12,13,14,15,16,17,18,19,20,21,22],
        val_chroms      = [8],
        # test_chroms   = [9],                # uncomment if you need it
        num_workers     = 10,
        use_gpu         = torch.backends.mps.is_available(),
        devices         = '0',
    )

    # ─── Building the args Namespace ─────────────────────────────────────────────
    args = type('A', (), {})()
    for d in (hp, const):
        for k, v in d.items():
            setattr(args, k, v)


    args.device = torch.device("mps" if args.use_gpu else "cpu")

    # init & load model
    exp = Exp_Long_Term_Forecast(args)
    exp.device = args.device
    exp.model.to(args.device)
    load_checkpoint(exp, checkpoint, args.device)
    exp.model.eval()

    # DataLoader for test split
    data_obj, _ = exp._get_data(flag="test")
    loader = DataLoader(data_obj,
                        batch_size=1,
                        shuffle=False,
                        drop_last=False,
                        num_workers=args.num_workers)

    # collect logits & truths
    infer_and_evaluate(args.setting, args.checkpoint, args)
class GenomicMetrics:
    def __init__(self, true_path, pred_path, eps=1e-8):
        # Load and squeeze out the singleton dimension
        raw_true   = np.load(true_path)   # shape: (N, 1, 16)
        raw_pred   = np.load(pred_path)   # shape: (N, 1, 16)
        # Force shape → (N,16)
        true = raw_true.reshape(raw_true.shape[0], -1)
        log_pred = raw_pred.reshape(raw_pred.shape[0], -1)

        # 1) Normalize true → probability vectors
        self.true = np.clip(true, 0, None)
        self.true /= (self.true.sum(axis=1, keepdims=True) + eps)

        # 2) Convert log‐probs → probabilities
        self.log_pred = log_pred
        self.pred     = np.exp(self.log_pred)

        self.N, self.K  = self.true.shape
        self.eps        = eps
        self.positions  = np.arange(self.K)

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

def infer_and_evaluate(setting, checkpoint, args):
    # 1) Prepare model
    exp = Exp_Long_Term_Forecast(args)
    exp.model.to(args.device)
    load_checkpoint(exp, checkpoint, args.device)
    exp.model.eval()

    # 2) Build test loader
    _, test_loader = exp._get_data(flag='test')

    # 3) Inference loop (mirrors Exp_Long_Term_Forecast.test)
    # ─── Inference + Collection ───────────────────────────────────────────
    all_preds, all_trues = [], []
    all_log, all_true = [],[]
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)

            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.float().to(args.device)
                batch_y_mark = batch_y_mark.float().to(args.device)

            # build decoder input exactly as in your test()
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(args.device)

            # forward
            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            # outputs: (batch_size, pred_len, c_out)

            all_preds.append(outputs.cpu().numpy())
            all_log.append(outputs.cpu())
            all_true.append(batch_y[:, -args.pred_len:, :].cpu())
            # ground truth for the same window
            all_trues.append(batch_y[:, -args.pred_len:, :].cpu().numpy())

    # Stack and reshape → (N, c_out)
    preds = np.concatenate(all_preds, axis=0).reshape(-1, args.c_out)
    trues = np.concatenate(all_trues, axis=0).reshape(-1, args.c_out)

    logp_inf = torch.cat(all_log, dim=0)
    true_inf = torch.cat(all_true, dim=0)
    # squeeze any singleton channel
    if logp_inf.ndim == 3 and logp_inf.shape[1] == 1:
        logp_inf = logp_inf.squeeze(1)
    if true_inf.ndim == 3 and true_inf.shape[1] == 1:
        true_inf = true_inf.squeeze(1)

    # to probabilities
    eps = 1e-8
    p_inf = logp_inf.exp().clamp(min=eps)
    q_inf = true_inf.clamp(min=eps)

    kl_none = nn.KLDivLoss(reduction="none")
    per_kl_inf  = kl_none(p_inf.log(), q_inf).sum(1).cpu().numpy()
    print(per_kl_inf.mean())
    # 5) (Optional) save to disk
    outdir = f'./results/{setting}/'
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'pred.npy'), preds)
    np.save(os.path.join(outdir, 'true.npy'), trues)
    print("Saved to", outdir)

    # 6) Compute & print metrics
    gm = GenomicMetrics(
        os.path.join(outdir, 'true.npy'),
        os.path.join(outdir, 'pred.npy')
    )
    print("\n=== Evaluation Metrics ===")
    for name, val in gm.all_metrics().items():
        print(f"{name:25s}: {val:.6f}")

    # save to disk
    np.save("iTransformer_pred.npy", preds)
    np.save("iTransformer_true.npy", trues)
    print("Saved iTransformer_pred.npy and iTransformer_true.npy")

    # optional: visualize a heatmap of the first sample
    sample_idx = 0
    true_vec = trues[sample_idx, 0, :]
    logit_vec = preds[sample_idx, 0, :]
    # softmax to get probabilities
    shifted = logit_vec - logit_vec.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].bar(np.arange(16), true_vec)
    ax[0].set_title("True 16-Fraction")
    ax[1].bar(np.arange(16), probs)
    ax[1].set_title("iTransformer Predicted")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_iTransformer_inference()