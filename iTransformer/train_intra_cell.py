import argparse
import os
import sys
import json
import shutil
from pathlib import Path

import numpy as np
import torch  # used only for availability checks/logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from run import main as run_model_main  # must run training + exp.test()


def parse_args():
    parser = argparse.ArgumentParser(description="iTransformer")

    # basic config
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--model_id", type=str, default="test", help="model id")
    parser.add_argument(
        "--model",
        type=str,
        default="iTransformer",
        help="model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]",
    )

    # data loader
    parser.add_argument("--data", type=str, default="custom", help="dataset type")
    parser.add_argument("--root_path", type=str, default="./iTransformer/data/", help="root path of the data file")
    parser.add_argument("--data_path", type=str, default="H1_genomic.csv", help="data csv file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, "
             "S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument("--target", type=str, default="target_1", help="target feature in S or MS task")
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s,t,h,d,b,w,m], or more detailed like 15min",
    )
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="location of model checkpoints")

    # genomic arguments
    parser.add_argument(
        "--train_chroms",
        nargs="+",
        type=int,
        help="List of chromosomes for training",
        default={1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
    )
    parser.add_argument("--val_chroms", nargs="+", type=int, help="List of chromosomes for validation", default=[9])
    # parser.add_argument('--test_chroms', nargs='+', type=int, help='List of chromosomes for testing', default=[9])

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=64, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=32, help="start token length")  # not needed in inverted
    parser.add_argument("--pred_len", type=int, default=1, help="prediction sequence length")

    # model define
    parser.add_argument("--enc_in", type=int, default=9, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=16, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=16, help="output size")
    parser.add_argument("--d_model", type=int, default=256, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=2, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=2, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--moving_avg", type=int, default=25, help="window size of moving average")
    parser.add_argument("--factor", type=int, default=2, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1572093622, help="dropout")
    parser.add_argument("--embed", type=str, default="timeF", help="time features encoding, options:[timeF,fixed,learned]")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument("--output_attention", action="store_true", help="whether to output attention in encoder")
    parser.add_argument("--do_predict", action="store_true", help="whether to predict unseen future data")

    # optimization
    parser.add_argument("--num_workers", type=int, default=10, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0004952468083, help="optimizer learning rate")
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="KL", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training", default=False)

    # --- Wavelet options ---
    parser.add_argument("--use_wavelet", action="store_true", help="Enable wavelet features (e.g., SWT) on inputs")
    parser.add_argument("--wavelet_name", type=str, default="db4", help="PyWavelets wavelet name (e.g., db4, coif1)")
    parser.add_argument("--wavelet_levels", type=int, default=1, help="Number of decomposition levels (>=1)")
    parser.add_argument("--keep_original", action="store_true", help="Concatenate original features with wavelet bands")
    parser.add_argument("--wavelet_where", type=str, default="dataset", choices=["dataset", "model"], help="Where to apply wavelet transform")

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")

    # iTransformer
    parser.add_argument("--exp_name", type=str, required=False, default="MTSF", help="experiemnt name, options:[MTSF, partial_train]")
    parser.add_argument("--channel_independence", type=bool, default=False, help="whether to use channel_independence mechanism")
    parser.add_argument("--inverse", action="store_true", help="inverse output data", default=False)
    parser.add_argument("--class_strategy", type=str, default="projection", help="projection/average/cls_token")
    parser.add_argument("--use_norm", type=int, default=False, help="use norm and denorm")
    parser.add_argument("--efficient_training", type=bool, default=False, help="whether to use efficient_training (exp_name should be partial train)")
    parser.add_argument("--partial_start_index", type=int, default=0, help="start index of variates for partial training")
    parser.add_argument("--setting", type=str, default="best_params_run", help="setting")

    return parser.parse_args()


def compute_setting(args, ii: int = 0) -> str:
    """
    Mirrors the common iTransformer/iInformer run.py setting format.
    Your uploaded run.py shows class_strategy and ii are included in the setting.
    """
    return (
        f"{args.model_id}_{args.model}"
        f"_ft{args.features}"
        f"_sl{args.seq_len}"
        f"_ll{args.label_len}"
        f"_pl{args.pred_len}"
        f"_dm{args.d_model}"
        f"_nh{args.n_heads}"
        f"_el{args.e_layers}"
        f"_dl{args.d_layers}"
        f"_df{args.d_ff}"
        f"_fc{args.factor}"
        f"_eb{args.embed}"
        f"_dt{args.distil}"
        f"_{args.des}"
        f"_{args.class_strategy}"
        f"_{ii}"
    )


def find_test_results_dir(base: Path, expected_setting: str) -> Path:
    """
    Prefer ./test_results/<expected_setting>/.
    If it doesn't exist (because upstream run.py may format setting slightly differently),
    fall back to the most recently modified subfolder of ./test_results/.
    """
    cand = base / expected_setting
    if cand.exists() and cand.is_dir():
        return cand

    if not base.exists():
        raise FileNotFoundError(f"test_results folder not found: {base}")

    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subfolders found in {base} (no test outputs were produced)")

    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs[0]


def save_args_json(dst: Path, args) -> None:
    # argparse Namespace -> dict
    d = vars(args).copy()
    # normalize sets to lists for JSON
    for k, v in list(d.items()):
        if isinstance(v, set):
            d[k] = sorted(list(v))
    dst.write_text(json.dumps(d, indent=2), encoding="utf-8")


def main(args=None):
    if args is None:
        args = parse_args()

    # Paths (do not change your tuned hyperparameters; only ensure folders exist)
    here = Path(__file__).resolve().parent
    checkpoints_dir = here / "checkpoints"
    results_dir = here / "results"
    test_results_dir = here / "test_results"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Ensure checkpoints path used by training code points to this folder
    args.checkpoints = str(checkpoints_dir)

    # Compute setting (used for locating artifacts)
    expected_setting = compute_setting(args, ii=0)
    print(f"[INFO] Expected setting: {expected_setting}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()} | args.use_gpu={args.use_gpu} | args.gpu={args.gpu}")

    print(f"Encoder input,Decoder input {args.enc_in}")

    # Run training + internal test() (this is what creates ./test_results/<setting>/pred.npy,true.npy,metrics.npy)
    ret = run_model_main(args)
    if ret is not None:
        print(f"[INFO] run.py returned: {ret}")

    # Locate produced folder
    produced_dir = find_test_results_dir(test_results_dir, expected_setting)
    produced_setting = produced_dir.name
    print(f"[INFO] Using produced test_results folder: {produced_dir}")

    # Destination folder
    out_dir = results_dir / produced_setting
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save args/config snapshot
    save_args_json(out_dir / "args.json", args)

    # Copy/rename outputs
    # Upstream saves: pred.npy, true.npy, metrics.npy
    src_pred = produced_dir / "pred.npy"
    src_true = produced_dir / "true.npy"
    src_metrics = produced_dir / "metrics.npy"

    if not src_pred.exists() or not src_true.exists():
        raise FileNotFoundError(
            f"Missing pred/true outputs in {produced_dir}. "
            f"Expected files: {src_pred.name}, {src_true.name}"
        )

    # Rename to requested names
    shutil.copy2(src_pred, out_dir / "preds.npy")
    shutil.copy2(src_true, out_dir / "trues.npy")

    metrics_json = {"setting": produced_setting}

    if src_metrics.exists():
        shutil.copy2(src_metrics, out_dir / "metrics.npy")
        m = np.load(src_metrics, allow_pickle=True)
        # In your exp.test(), metrics.npy is [mae, mse, rmse, mape, mspe]
        # We store them as-is in JSON too.
        if m.size >= 5:
            metrics_json.update(
                {
                    "mae": float(m[0]),
                    "mse": float(m[1]),
                    "rmse": float(m[2]),
                    "mape": float(m[3]),
                    "mspe": float(m[4]),
                }
            )
        else:
            metrics_json["metrics_raw"] = m.tolist()
    else:
        metrics_json["warning"] = "metrics.npy not found in test_results; only preds/trues were saved."

    (out_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    print("[INFO] Saved artifacts to:", out_dir)
    print("  -", out_dir / "preds.npy")
    print("  -", out_dir / "trues.npy")
    if (out_dir / "metrics.npy").exists():
        print("  -", out_dir / "metrics.npy")
    print("  -", out_dir / "metrics.json")
    print("  -", out_dir / "args.json")

    return metrics_json


if __name__ == "__main__":
    main()