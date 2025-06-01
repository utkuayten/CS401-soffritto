#!/usr/bin/env python3
import os, sys, random

import torch, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from torch.utils.data import DataLoader
import torch.nn as nn
from scipy.stats import (
    spearmanr, pearsonr,
    wilcoxon, wasserstein_distance, ks_2samp
)

# reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark   = False

# project path
THIS_DIR    = os.path.dirname(__file__)
PROJECT_ROOT= os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
from transofritto.informer.exp.exp_informer import Exp_Informer

def load_checkpoint(exp, path, device):
    ckpt = os.path.join(path, "checkpoint.pth") if os.path.isdir(path) else path
    data = torch.load(ckpt, map_location=device)
    sd   = data.get("state_dict", data)
    clean= {k.replace("module.", ""): v for k, v in sd.items()}
    exp.model.load_state_dict(clean)

def run_inference():
    # --- settings ---
    checkpoint = "/Users/utkuayten/DataspellProjects/CS401-soffritto/transofritto/best_model/mESC_CV_informer/checkpoint.pth"
    batch_size = 128

    # hyperparams + constants
    hp = dict(seq_len=32, label_len=16, pred_len=1,
              d_model=1024, e_layers=3, d_layers=1,
              n_heads=8, d_ff=2048, dropout=0.16,
              factor=7, learning_rate= 0.00021769231169858132, mix=False,
              train_chroms = [  1,
                                2,
                                3,
                                4,
                                5,
                                7,
                                8,
                                10,
                                11,
                                12,
                                13,
                                14,
                                15,
                                16,
                                17,
                                18,
                                19,
                                20,
                                21,
                                22],
              val_chroms = [])

    const = dict(model="informer", data="custom", features="M", target="target_1", freq="w",
                 root_path="data", data_path="H1_genomic.csv", checkpoints="./checkpoints/",
                 enc_in=9, dec_in=16, c_out=16,
                 num_workers=0, use_multi_gpu=False, use_gpu=False,
                 devices="0", gpu=0, inverse=False, output_attention=False,
                 distil=True, attn="prob", embed="timeF", activation="gelu",
                 padding=0, use_amp=False)

    # build args
    args = type("A", (), {})()
    for k, v in {**const, **hp, "batch_size": batch_size}.items():
        setattr(args, k, v)
    args.device = torch.device("cpu")

    # init & load model
    exp = Exp_Informer(args)
    exp.device = args.device
    exp.model.to(args.device)
    load_checkpoint(exp, checkpoint, args.device)
    exp.model.eval()

    # DataLoader for test split
    data_obj, _ = exp._get_data(flag="test")
    loader = DataLoader(data_obj, batch_size=1,
                        shuffle=False, drop_last=False, num_workers=0)

    # collect model log‐probs & truths
    all_logp, all_true = [], []
    with torch.no_grad():
        for bx, by, bxm, bym in loader:
            logp, true = exp._process_one_batch(data_obj, bx, by, bxm, bym)
            all_logp.append(logp.cpu())
            all_true.append(true.cpu())

    logp_inf = torch.cat(all_logp, dim=0)
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

    # load Soffritto outputs
    soff_p = torch.from_numpy(
        np.load("/Users/utkuayten/DataspellProjects/CS401-soffritto/soffritto/predictions/mESC_chr9_pred_intra_cell_line.npy")
    ).float()
    soff_q = torch.from_numpy(
        np.load("/Users/utkuayten/DataspellProjects/CS401-soffritto/soffritto/predictions/mESC_chr9_pred_intra_cell_line.npy_true.npy")
    ).float()
    if soff_p.ndim == 3 and soff_p.shape[1] == 1:
        soff_p = soff_p.squeeze(1)
    if soff_q.ndim == 3 and soff_q.shape[1] == 1:
        soff_q = soff_q.squeeze(1)
    p_soff = soff_p.clamp(min=eps)
    q_soff = soff_q.clamp(min=eps)

    # ALIGN windows → bins
    seq_len, label_len = args.seq_len, args.label_len
    Nw = p_inf.shape[0]
    bin_idx = np.arange(Nw) + seq_len
    mask    = bin_idx < p_soff.shape[0]

    p_inf_a   = p_inf[mask]
    q_inf_a   = q_inf[mask]

    p_soff_a  = p_soff[bin_idx[mask]]
    q_soff_a  = q_soff[bin_idx[mask]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 2), facecolor='white')
    fig.patch.set_facecolor('white')

    # 1) Real (“true”) 16‐fraction from Informer’s labels (q_inf_a[0])
    real_vec = q_inf_a[0].cpu().numpy()[None, :]  # shape (1, 16)
    ax = axes[0]
    im0 = ax.imshow(real_vec, aspect='auto', cmap='Greys')
    ax.set_title("Real (q_inf[0])")
    ax.set_xticks(range(16))
    ax.set_yticks([])
    ax.set_xlabel("Fraction (1–16)")

    # 2) Soffritto’s prediction for that same window (p_soff_a[0])
    soff_vec = p_soff_a[0].cpu().numpy()[None, :]
    ax = axes[1]
    im1 = ax.imshow(soff_vec, aspect='auto', cmap='Greys')
    ax.set_title("Soffritto pred (p_soff[0])")
    ax.set_xticks(range(16))
    ax.set_yticks([])
    ax.set_xlabel("Fraction (1–16)")
    print(p_soff_a[0].sum())
    print(p_inf_a[0].sum())
    # 3) Informer’s prediction for that same window (p_inf_a[0])
    inf_vec = p_inf_a[0].cpu().numpy()[None, :]
    ax = axes[2]
    im2 = ax.imshow(inf_vec, aspect='auto', cmap='Greys')
    ax.set_title("Informer pred (p_inf[0])")
    ax.set_xticks(range(16))
    ax.set_yticks([])
    ax.set_xlabel("Fraction (1–16)")

    plt.tight_layout()
    out_dist = os.path.join(os.path.dirname(checkpoint), "first_window_distributions.png")
    plt.savefig(out_dist, dpi=300, facecolor='white')
    plt.close(fig)

    print("Saved first‐window 1×16 heatmaps to", out_dist)

    # plot Soffritto raw heatmaps (truth vs prediction)
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(q_soff_a, aspect='auto', cmap='Greys', vmin=0, vmax=q_soff_a.max())
    plt.title('Soffritto ground-truth 16-fraction\n(chrom 9, H1)')
    plt.xlabel('S-phase fraction (1…16)')
    plt.ylabel('Bin index')

    plt.subplot(1,2,2)
    plt.imshow(p_soff_a, aspect='auto', cmap='Greys', vmin=0, vmax=q_soff_a.max())
    plt.title('Soffritto predictions')
    plt.xlabel('S-phase fraction (1…16)')

    plt.tight_layout()
    plt.savefig('heatmaps_truth_vs_soffritto_gray.png', dpi=300)
    plt.close()

    # plot Informer raw heatmaps (truth vs prediction)
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(q_inf_a, aspect='auto', cmap='Greys', vmin=0, vmax=q_inf_a.max())
    plt.title('Observed 16-fraction profiles\n(chrom 9, H1)')
    plt.xlabel('S-phase fraction (1…16)')
    plt.ylabel('Bin index')
    plt.subplot(1,2,2)
    plt.imshow(p_inf_a, aspect='auto', cmap='Greys', vmin=0, vmax=q_inf_a.max())
    plt.title('Informer predictions')
    plt.xlabel('S-phase fraction (1…16)')
    plt.tight_layout()
    plt.savefig('heatmaps_truth_vs_informer_gray.png', dpi=300)

    # compute metrics
    # 1) KL
    kl_none = nn.KLDivLoss(reduction="none")
    per_kl_inf  = kl_none(p_inf_a.log(), q_inf_a).sum(1).cpu().numpy()
    per_kl_soff = kl_none(p_soff_a.log(), q_soff_a).sum(1).cpu().numpy()
    W_kl, p_kl   = wilcoxon(per_kl_inf, per_kl_soff, alternative="less")

    # 2) Spearman
    rho_inf  = [spearmanr(p_inf_a[i].numpy(),  q_inf_a[i].numpy()).correlation
                for i in range(len(p_inf_a))]
    rho_soff = [spearmanr(p_soff_a[i].numpy(), q_soff_a[i].numpy()).correlation
                for i in range(len(p_soff_a))]
    W_rho, p_rho = wilcoxon(rho_inf, rho_soff, alternative="greater")

    # 3) MSE
    per_mse_inf  = ((p_inf_a - q_inf_a)**2).sum(1).cpu().numpy()
    per_mse_soff = ((p_soff_a - q_soff_a)**2).sum(1).cpu().numpy()
    W_mse, p_mse = wilcoxon(per_mse_inf, per_mse_soff, alternative="less")

    # 4) Pearson
    pear_inf  = [pearsonr(p_inf_a[i].numpy(),  q_inf_a[i].numpy())[0]
                 for i in range(len(q_inf_a))]
    pear_soff = [pearsonr(p_soff_a[i].numpy(), q_soff_a[i].numpy())[0]
                 for i in range(len(q_soff_a))]
    W_pr, p_pr = wilcoxon(pear_inf, pear_soff, alternative="greater")

    # 5) Wasserstein
    classes = np.arange(p_inf_a.shape[1])
    w_inf  = [wasserstein_distance(classes, classes,
                                   q_inf_a[i].numpy(), p_inf_a[i].numpy())
              for i in range(len(q_inf_a))]
    w_soff = [wasserstein_distance(classes, classes,
                                   q_soff_a[i].numpy(), p_soff_a[i].numpy())
              for i in range(len(q_soff_a))]
    W_w, p_w = wilcoxon(w_inf, w_soff, alternative="less")

    # 6) KS‐statistic
    ks_inf  = [ks_2samp(q_inf_a[i].numpy(),  p_inf_a[i].numpy()).statistic
               for i in range(len(q_inf_a))]
    ks_soff = [ks_2samp(q_soff_a[i].numpy(), p_soff_a[i].numpy()).statistic
               for i in range(len(q_soff_a))]
    W_ks, p_ks = wilcoxon(ks_inf, ks_soff, alternative="less")

    # package for plotting
    titles = ["KL divergence", "Spearman’s ρ", "MSE",
              "Pearson’s r", "Wasserstein", "KS statistic"]
    metrics = [
        (per_kl_inf,  per_kl_soff,  p_kl),
        (rho_inf,     rho_soff,     p_rho),
        (per_mse_inf, per_mse_soff, p_mse),
        (pear_inf,    pear_soff,    p_pr),
        (w_inf,       w_soff,       p_w),
        (ks_inf,      ks_soff,      p_ks),
    ]

    # PLOT 2×3 grid of violins (smoothed via seaborn)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for ax, title, (A_inf, A_so, pval) in zip(axes, titles, metrics):
        sns.violinplot(data=[A_inf, A_so],
                       inner="box", cut=0, bw=0.2,
                       ax=ax, color="skyblue")
        ax.set_xticks([0,1])
        ax.set_xticklabels(["Informer","Soffritto"])
        ax.set_title(title)
        if title=="KL divergence":
            ax.set_ylabel("Per-bin value")
        if title=="Pearson’s r":
            ax.set_ylim(0,1.2)
        # significance star
        y = max(np.max(A_inf), np.max(A_so))
        star = "***" if pval<=0.001 else ("**" if pval<=0.01 else ("*" if pval<=0.05 else "ns"))
        ax.plot([0,1],[y*1.05]*2,'k-')
        ax.text(0.5, y*1.08, star, ha='center', fontsize=14)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(checkpoint), "six_metric_violins.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved all panels to", out)

    # — Save a single image with 3 gray‐scale heatmaps: ground truth / Soffritto / Informer —
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
    fig.patch.set_facecolor('white')

    # 1) True ground-truth (using Informer’s “true” windows: q_inf_a)
    ax = axes[0]
    im_true = ax.imshow(q_inf_a.cpu().numpy(), aspect='auto', cmap='Greys')
    ax.set_title('Ground truth (16-fraction, chr 9)')
    ax.set_xlabel('Fraction (1–16)')
    ax.set_ylabel('Window index')
    fig.colorbar(im_true, ax=ax, fraction=0.046, pad=0.04)

    # 2) Soffritto predictions (p_soff_a)
    ax = axes[1]
    im_soff = ax.imshow(p_soff_a.cpu().numpy(), aspect='auto', cmap='Greys')
    ax.set_title('Soffritto predictions')
    ax.set_xlabel('Fraction (1–16)')
    fig.colorbar(im_soff, ax=ax, fraction=0.046, pad=0.04)

    # 3) Informer predictions (p_inf_a)
    ax = axes[2]
    im_inf = ax.imshow(p_inf_a.cpu().numpy(), aspect='auto', cmap='Greys')
    ax.set_title('Informer predictions')
    ax.set_xlabel('Fraction (1–16)')
    fig.colorbar(im_inf, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(checkpoint),
                            "combined_truth_soffritto_informer.png")
    plt.savefig(out_path, dpi=300, facecolor='white')
    plt.close(fig)

    print("Saved combined heatmaps (ground-truth / Soffritto / Informer) to", out_path)

    print(f"Mean KL (Informer):      {per_kl_inf.mean():.6f}")
    print(f"Mean KL (Soffritto):     {per_kl_soff.mean():.6f}")
    print(f"Mean Spearman ρ (Inf):   {np.mean(rho_inf):.6f}")
    print(f"Mean Spearman ρ (Soff):  {np.mean(rho_soff):.6f}")
    print(f"Mean MSE (Informer):     {per_mse_inf.mean():.6f}")
    print(f"Mean MSE (Soffritto):    {per_mse_soff.mean():.6f}")
    print(f"Mean Pearson’s r (Inf):  {np.mean(pear_inf):.6f}")
    print(f"Mean Pearson’s r (Soff): {np.mean(pear_soff):.6f}")
    print(f"Mean Wasserstein (Inf):  {np.mean(w_inf):.6f}")
    print(f"Mean Wasserstein (Soff): {np.mean(w_soff):.6f}")
    print(f"Mean KS stat (Informer): {np.mean(ks_inf):.6f}")
    print(f"Mean KS stat (Soffritto):{np.mean(ks_soff):.6f}")

if __name__=="__main__":
    run_inference()
