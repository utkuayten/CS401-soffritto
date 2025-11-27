#!/usr/bin/env python3
import os, sys, random, json
import torch, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from torch.utils.data import DataLoader
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr, wilcoxon, wasserstein_distance, ks_2samp
import pandas as pd

# reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark   = False

# allow imports from project root
THIS_DIR     = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from transofritto.informer.exp.exp_informer import Exp_Informer

def load_hyperparams(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_checkpoint(exp, path, device):
    ckpt = os.path.join(path, "checkpoint.pth") if os.path.isdir(path) else path
    data = torch.load(ckpt, map_location=device)
    sd   = data.get("state_dict", data)
    clean= {k.replace("module.", ""): v for k, v in sd.items()}
    exp.model.load_state_dict(clean)

def run_inference_for_cell(cell, param_json):
    # 1) Load hyperparameters from JSON
    hp = load_hyperparams(param_json)

    # 2) Build an args object from hp + cell‐specific fields
    args = type("A", (), {})()
    for k, v in hp.items():
        setattr(args, k, v)
    args.model_id   = f"{cell}_CV_informer"
    args.device     = torch.device("mps")
    args.batch_size = 128

    # Path to the Informer checkpoint
    checkpoint = (
        f"/Users/ozgun/DataspellProjects/CS401-soffritto/"
        f"transofritto/best_model/{cell}_CV_informer/checkpoint.pth"
    )

    # 3) Instantiate the model, load weights, set eval mode
    exp = Exp_Informer(args)
    exp.device = args.device
    exp.model.to(args.device)
    load_checkpoint(exp, checkpoint, args.device)
    exp.model.eval()

    # 4) Build a DataLoader over the “test” split
    data_obj, _ = exp._get_data(flag="test")
    loader = DataLoader(
        data_obj, batch_size=1, shuffle=False, drop_last=False,
        num_workers=args.num_workers
    )

    # 5) Collect the model’s log‐probabilities and true labels for “test”
    all_logp, all_true = [], []
    with torch.no_grad():
        for bx, by, bxm, bym in loader:
            logp, true = exp._process_one_batch(data_obj, bx, by, bxm, bym)
            all_logp.append(logp.cpu())
            all_true.append(true.cpu())

    logp_inf = torch.cat(all_logp, dim=0)
    true_inf = torch.cat(all_true, dim=0)
    if logp_inf.dim() == 3 and logp_inf.shape[1] == 1:
        logp_inf = logp_inf.squeeze(1)
    if true_inf.dim() == 3 and true_inf.shape[1] == 1:
        true_inf = true_inf.squeeze(1)

    # 6) Convert log‐probs → probabilities (clamp to avoid zeros)
    eps = 1e-8
    p_inf = logp_inf.exp().clamp(min=eps)  # shape [#windows, 16]
    q_inf = true_inf.clamp(min=eps)        # shape [#windows, 16]

    # 7) Load pre‐computed Soffritto .npy for this cell (chr9)
    soff_p = torch.from_numpy(
        np.load(f"soffritto/predictions/{cell}_chr9_pred_intra_cell_line.npy")
    ).float()
    soff_q = torch.from_numpy(
        np.load(f"soffritto/predictions/{cell}_chr9_pred_intra_cell_line.npy_true.npy")
    ).float()
    if soff_p.dim() == 3 and soff_p.shape[1] == 1:
        soff_p = soff_p.squeeze(1)
    if soff_q.dim() == 3 and soff_q.shape[1] == 1:
        soff_q = soff_q.squeeze(1)
    p_soff = soff_p.clamp(min=eps)  # shape [#bins, 16]
    q_soff = soff_q.clamp(min=eps)

    # 8) Align “windows” → “bins” exactly as before
    seq_len     = args.seq_len
    Nw          = p_inf.shape[0]
    bin_idx     = np.arange(Nw) + seq_len
    mask        = bin_idx < p_soff.shape[0]

    p_inf_a     = p_inf[mask]     # [#selected_bins, 16]
    q_inf_a     = q_inf[mask]
    p_soff_a    = p_soff[bin_idx[mask]]
    q_soff_a    = q_soff[bin_idx[mask]]

    # 9) Compute per‐bin metrics
    kl_none     = nn.KLDivLoss(reduction="none")
    per_kl_inf  = kl_none(p_inf_a.log(), q_inf_a).sum(1).cpu().numpy()
    per_kl_soff = kl_none(p_soff_a.log(), q_soff_a).sum(1).cpu().numpy()

    rho_inf  = [
        spearmanr(p_inf_a[i].numpy(), q_inf_a[i].numpy()).correlation
        for i in range(p_inf_a.shape[0])
    ]
    rho_soff = [
        spearmanr(p_soff_a[i].numpy(), q_soff_a[i].numpy()).correlation
        for i in range(p_soff_a.shape[0])
    ]

    pear_inf  = [
        pearsonr(p_inf_a[i].numpy(), q_inf_a[i].numpy())[0]
        for i in range(q_inf_a.shape[0])
    ]
    pear_soff = [
        pearsonr(p_soff_a[i].numpy(), q_soff_a[i].numpy())[0]
        for i in range(q_soff_a.shape[0])
    ]

    classes = np.arange(p_inf_a.shape[1])
    w_inf  = [
        wasserstein_distance(
            classes, classes,
            q_inf_a[i].numpy(), p_inf_a[i].numpy()
        )
        for i in range(q_inf_a.shape[0])
    ]
    w_soff = [
        wasserstein_distance(
            classes, classes,
            q_soff_a[i].numpy(), p_soff_a[i].numpy()
        )
        for i in range(q_soff_a.shape[0])
    ]

    ks_inf  = [
        ks_2samp(q_inf_a[i].numpy(), p_inf_a[i].numpy()).statistic
        for i in range(q_inf_a.shape[0])
    ]
    ks_soff = [
        ks_2samp(q_soff_a[i].numpy(), p_soff_a[i].numpy()).statistic
        for i in range(q_soff_a.shape[0])
    ]

    # 10) Argmax RT Fraction error
    argmax_inf   = p_inf_a.argmax(dim=1).cpu().numpy()
    argmax_q_inf = q_inf_a.argmax(dim=1).cpu().numpy()
    arg_err_inf  = np.abs(argmax_inf - argmax_q_inf)

    argmax_soff   = p_soff_a.argmax(dim=1).cpu().numpy()
    argmax_q_soff = q_soff_a.argmax(dim=1).cpu().numpy()
    arg_err_soff  = np.abs(argmax_soff - argmax_q_soff)

    # 11) Wilcoxon tests (paired, one‐sided)
    W_kl,  p_kl  = wilcoxon(per_kl_inf,  per_kl_soff,  alternative="less")
    W_rho, p_rho = wilcoxon(rho_inf,     rho_soff,     alternative="greater")
    W_pr,  p_pr  = wilcoxon(pear_inf,    pear_soff,    alternative="greater")
    W_w,   p_w   = wilcoxon(w_inf,       w_soff,       alternative="less")
    W_ks,  p_ks  = wilcoxon(ks_inf,      ks_soff,      alternative="less")
    W_arg, p_arg = wilcoxon(arg_err_inf, arg_err_soff, alternative="less")

    # 12) Return a single dictionary containing all arrays + p‐values
    return {
        "cell":       cell,
        "per_kl_inf":   per_kl_inf,
        "per_kl_soff":  per_kl_soff,
        "rho_inf":      np.array(rho_inf),
        "rho_soff":     np.array(rho_soff),
        "pear_inf":     np.array(pear_inf),
        "pear_soff":    np.array(pear_soff),
        "w_inf":        np.array(w_inf),
        "w_soff":       np.array(w_soff),
        "ks_inf":       np.array(ks_inf),
        "ks_soff":      np.array(ks_soff),
        "arg_err_inf":  arg_err_inf,
        "arg_err_soff": arg_err_soff,

        # store p‐values for annotation
        "p_kl":  p_kl,
        "p_rho": p_rho,
        "p_pr":  p_pr,
        "p_w":   p_w,
        "p_ks":  p_ks,
        "p_arg": p_arg
    }

def plot_all_cells_per_metric(cell_metrics_list, out_dir):
    """
    Given a list of cell‐metrics dicts (one per cell),
    create six separate figures—one for each metric.
    Each figure has 1 row × len(cell_metrics_list) columns.
    In each subplot: draw Informer vs. Soffritto violins,
    and annotate with Wilcoxon p-value (“*” / “**” / “***” / “ns”).
    """
    # Make sure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # List of (metric_name, informer_key, soffritto_key, pval_key, y_label) tuples
    metrics_info = [
        ("KL divergence",   "per_kl_inf",   "per_kl_soff",  "p_kl",  "KL divergence"),
        ("Spearman’s ρ",    "rho_inf",      "rho_soff",     "p_rho", "Spearman’s ρ"),
        ("Pearson’s r",     "pear_inf",     "pear_soff",    "p_pr",  "Pearson’s r"),
        ("Wasserstein",     "w_inf",        "w_soff",       "p_w",   "Wasserstein"),
        ("KS statistic",    "ks_inf",       "ks_soff",      "p_ks",  "KS statistic"),
        ("Argmax RT Error", "arg_err_inf",  "arg_err_soff", "p_arg", "Argmax RT Error"),
    ]

    n_cells = len(cell_metrics_list)
    cell_names = [d["cell"] for d in cell_metrics_list]

    for (metric_title, key_inf, key_soff, key_p, y_label) in metrics_info:
        # Create a wide figure: one row, n_cells columns
        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_cells,
            figsize=(4 * n_cells, 4),
            constrained_layout=True
        )

        # If only one cell, axes might be an Axes object; force it into a list
        if n_cells == 1:
            axes = [axes]

        for idx, cell_dict in enumerate(cell_metrics_list):
            ax = axes[idx]
            vals_inf  = cell_dict[key_inf]
            vals_soff = cell_dict[key_soff]
            pval      = cell_dict[key_p]

            # Draw the two violins side by side
            sns.violinplot(
                data=[vals_inf, vals_soff],
                inner="box",
                cut=0,
                bw=0.2,
                ax=ax,
                palette=["#1f77b4", "#ff7f0e"]
            )
            ax.set_title(f"{cell_dict['cell']}")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Informer", "Soffritto"], rotation=0)

            # Determine significance star
            if pval <= 0.001:
                star = "***"
            elif pval <= 0.01:
                star = "**"
            elif pval <= 0.05:
                star = "*"
            else:
                star = "ns"

            # Compute a y‐position slightly above the taller of the two violins
            max_inf  = np.max(vals_inf)
            max_soff = np.max(vals_soff)
            y_max    = max(max_inf, max_soff)
            y_star   = y_max * 1.05  # 5% above the max

            # Draw a short horizontal line from x=0 to x=1 at height y_star
            ax.plot([0, 1], [y_star, y_star], color="black", linewidth=1.0)
            # Place the star text at (0.5, y_star * 1.02)
            ax.text(0.5, y_star * 1.02, star,
                    ha="center", va="bottom",
                    color="black", fontsize=12)

            # Only show y‐axis label on the first subplot
            if idx == 0:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel("")

        # Give the entire figure a super‐title (optional)
        fig.suptitle(metric_title, fontsize=16)

        # Save:
        out_path = os.path.join(out_dir, f"all_cells_{key_inf}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {metric_title} figure → {out_path}")

def build_long_df_for_metric(results, metric_inf_key, metric_soff_key):
    """
    Given a list of dicts (one per cell), each containing:
       - "cell": <str>
       - metric_inf_key: 1D numpy array
       - metric_soff_key: 1D numpy array
    Returns a pandas DataFrame with columns [ "cell", "model", "value" ].
    """
    rows = []
    for cell_dict in results:
        cell_name = cell_dict["cell"]
        # Informer values (list or 1D numpy)
        inf_vals = cell_dict[metric_inf_key]
        for v in inf_vals:
            rows.append({"cell": cell_name, "model": "Informer", "value": v})
        # Soffritto values
        soff_vals = cell_dict[metric_soff_key]
        for v in soff_vals:
            rows.append({"cell": cell_name, "model": "Soffritto", "value": v})

    return pd.DataFrame(rows)
def plot_one_metric_over_cells(results, metric_title, metric_inf_key,
                               metric_soff_key, pval_key, output_path):
    """
    - results: list of dicts, each dict has:
        "cell", metric_inf_key, metric_soff_key, pval_key
    - metric_title: string, e.g. "KL divergence"
    - metric_inf_key: e.g. "per_kl_inf"
    - metric_soff_key: e.g. "per_kl_soff"
    - pval_key: e.g. "p_kl"
    - output_path: where to save the single PNG file

    This function:
      a) builds a long‐form DataFrame,
      b) calls seaborn.violinplot with x=cell, y=value, hue=model,
      c) loops over each cell to annotate the Wilcoxon p-value.
    """

    # 1) Build the long DataFrame
    df = build_long_df_for_metric(results, metric_inf_key, metric_soff_key)

    # 2) Initialize figure
    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # 3) Draw grouped violins
    sns.violinplot(
        data=df,
        x="cell",
        y="value",
        hue="model",
        dodge=True,
        inner="box",
        cut=0,
        bw=0.2,
        palette=["#1f77b4", "#ff7f0e"],  # blue=Informer, orange=Soffritto
        ax=ax
    )

    # 4) Compute where to place the Wilcoxon stars for each cell
    #    We need to know the x‐coordinates of each category
    #    When dodge=True, seaborn will place “Informer” at x_i - width/2,
    #    and “Soffritto” at x_i + width/2, where x_i is the integer index of
    #    that cell name in the categories. The default width is 0.8, so half‐width
    #    is 0.4. We’ll peek at the tick locations:
    xticks = ax.get_xticks()   # e.g. [0,1,2,3,4] for 5 cells
    # The two violins for each cell will be centered at (x_i - 0.2) and (x_i + 0.2)
    dodge_offset = 0.2

    # 5) Loop through each cell and annotate
    #    We must find that cell’s subset of the DataFrame so we can get its max Y
    for i, cell_dict in enumerate(results):
        cell_name = cell_dict["cell"]
        pval      = cell_dict[pval_key]

        # Extract only that cell’s rows from df
        df_cell = df[df["cell"] == cell_name]
        inf_vals = df_cell[df_cell["model"] == "Informer"]["value"].values
        soff_vals= df_cell[df_cell["model"] == "Soffritto"]["value"].values

        # Determine significance star
        if pval <= 0.001:
            star = "***"
        elif pval <= 0.01:
            star = "**"
        elif pval <= 0.05:
            star = "*"
        else:
            star = "ns"

        # Compute y‐position just above the taller distribution
        y_max_inf = np.max(inf_vals)
        y_max_soff= np.max(soff_vals)
        y_star    = max(y_max_inf, y_max_soff) * 1.05

        # The Informer violin is at x = xticks[i] - dodge_offset
        # The Soffritto violin is at x = xticks[i] + dodge_offset
        x_left  = xticks[i] - dodge_offset
        x_right = xticks[i] + dodge_offset

        # Draw a short horizontal line between (x_left, y_star) and (x_right, y_star)
        ax.plot([x_left, x_right], [y_star, y_star], color="black", linewidth=1)

        # Place the star text at x = xticks[i], y = y_star * 1.02
        ax.text(xticks[i], y_star * 1.02, star,
                ha="center", va="bottom", color="black", fontsize=12)

    # 6) Final formatting
    ax.set_title(metric_title, fontsize=16)
    ax.set_xlabel("")  # “cell” is already clear from x‐ticks
    ax.set_ylabel(metric_title)
    ax.legend(title="", loc="upper right")

    # 7) Save figure, close
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    cell_lines = ["H9", "HCT116", "mESC", "mNPC"]
    results = []

    # 1) Run inference on each cell → collect 5 dicts
    for cell in cell_lines:
        param_json = (
            f"/Users/ozgun/DataspellProjects/CS401-soffritto/"
            f"transofritto/best_model/{cell}_CV_informer/"
            f"{cell}_val_6_hyperparameters.json"
        )
        cell_metrics = run_inference_for_cell(cell, param_json)
        results.append(cell_metrics)

    # 2) For each of the six metrics, plot a single figure:
    output_dir = "/Users/ozgun/DataspellProjects/CS401-soffritto/transofritto/best_model"
    os.makedirs(output_dir, exist_ok=True)

    # KL divergence
    plot_one_metric_over_cells(
        results,
        metric_title="KL divergence",
        metric_inf_key="per_kl_inf",
        metric_soff_key="per_kl_soff",
        pval_key="p_kl",
        output_path=os.path.join(output_dir, "all_cells_KL.png")
    )

    # Spearman’s ρ
    plot_one_metric_over_cells(
        results,
        metric_title="Spearman’s ρ",
        metric_inf_key="rho_inf",
        metric_soff_key="rho_soff",
        pval_key="p_rho",
        output_path=os.path.join(output_dir, "all_cells_Spearman.png")
    )

    # Pearson’s r
    plot_one_metric_over_cells(
        results,
        metric_title="Pearson’s r",
        metric_inf_key="pear_inf",
        metric_soff_key="pear_soff",
        pval_key="p_pr",
        output_path=os.path.join(output_dir, "all_cells_Pearson.png")
    )

    # Wasserstein distance
    plot_one_metric_over_cells(
        results,
        metric_title="Wasserstein distance",
        metric_inf_key="w_inf",
        metric_soff_key="w_soff",
        pval_key="p_w",
        output_path=os.path.join(output_dir, "all_cells_Wasserstein.png")
    )

    # KS statistic
    plot_one_metric_over_cells(
        results,
        metric_title="KS statistic",
        metric_inf_key="ks_inf",
        metric_soff_key="ks_soff",
        pval_key="p_ks",
        output_path=os.path.join(output_dir, "all_cells_KS.png")
    )

    # Argmax RT Error
    plot_one_metric_over_cells(
        results,
        metric_title="Argmax RT Error",
        metric_inf_key="arg_err_inf",
        metric_soff_key="arg_err_soff",
        pval_key="p_arg",
        output_path=os.path.join(output_dir, "all_cells_ArgmaxError.png")
    )