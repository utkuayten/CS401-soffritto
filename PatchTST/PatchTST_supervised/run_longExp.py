import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import json
import shutil
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description='PatchTST / Formers training runner (genomic variant with chrom splits)'
    )

    # -------------------- ORIGINAL ARGS (UNCHANGED / ALL KEPT) --------------------
    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='PatchTST',
                        help='model name, options: [Autoformer, Informer, Transformer, PatchTST]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='H1_genomic.csv', help='data file (or will be set from --cell)')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='target_1', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=32, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=16, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # PatchTST (kept as in original)
    parser.add_argument('--fc_dropout', type=float, default=0.001834363147, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.006436888576, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=8, help='patch length')
    parser.add_argument('--stride', type=int, default=2, help='stride')
    parser.add_argument('--padding_patch', default='None', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=1, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers
    parser.add_argument('--embed_type', type=int, default=1,
                        help='0: default 1: value+temporal+positional 2: value+temporal 3: value+positional 4: value only')
    parser.add_argument('--enc_in', type=int, default=9, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=16, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=16, help='output size')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=5, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.2474417505, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0004894470598, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # -------------------- NEW ARGS (ADDED FROM INFORMER GENOMIC RUNNER) --------------------
    parser.add_argument('--setting', type=str, default=None, help='explicit run setting name')
    parser.add_argument('--cell', type=str, required=False, help='cell name to derive data path (e.g., H1)')
    parser.add_argument('--train_chroms', nargs='+', type=int, help='List of chromosomes for training',
                        default={1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22})
    parser.add_argument('--val_chroms', nargs='+', type=int,  help='List of chromosomes for validation',
                        default=[6])
    parser.add_argument('--attn', type=str, default='prob', help='attention type (if a Former uses it)')
    parser.add_argument('--weight_decay', type=float, default=0.00555477304, help='optimizer weight decay')

    # Wavelet options (safe no-ops if you don’t use them downstream)
    parser.add_argument('--use_wavelet', action='store_true',
                        help='Enable wavelet features (e.g., SWT) on inputs')
    parser.add_argument('--wavelet_name', type=str, default='db4',
                        help='PyWavelets wavelet name')
    parser.add_argument('--wavelet_levels', type=int, default=1,
                        help='Number of decomposition levels (>=1)')
    parser.add_argument('--keep_original', action='store_true',
                        help='Concatenate original features with wavelet bands')
    parser.add_argument('--wavelet_where', type=str, default='dataset',
                        choices=['dataset', 'model'],
                        help='Where to apply wavelet transform')

    # Feature selection
    parser.add_argument('--selected_cols', nargs='+', type=str,
                        default=['H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1',
                                 'H3K4me3', 'H3K9me3', 'GC_content', 'gene_density', '2-stage','date'],
                        help='Columns to use as inputs')

    return parser


# -------------------- NEW: robust exporting of preds/trues/metrics --------------------

SRC_PRED_NAMES = ["pred.npy", "preds.npy", "y_pred.npy"]
SRC_TRUE_NAMES = ["true.npy", "trues.npy", "y_true.npy"]
SRC_METRIC_NAMES = ["metrics.npy", "metric.npy"]

def _save_args_json(out_dir: Path, args) -> None:
    d = vars(args).copy()
    # JSON cannot serialize sets
    for k, v in list(d.items()):
        if isinstance(v, set):
            d[k] = sorted(list(v))
    (out_dir / "args.json").write_text(json.dumps(d, indent=2), encoding="utf-8")

def _find_first_existing(folder: Path, names) -> Path | None:
    for n in names:
        p = folder / n
        if p.exists():
            return p
    return None

def _find_run_output_folder(base_dir: Path, setting: str) -> Path | None:
    """
    Try common output locations used by these repos:
      - <base>/test_results/<setting>/
      - <base>/results/<setting>/
      - <cwd>/test_results/<setting>/
      - <cwd>/results/<setting>/
    If not found, return None.
    """
    candidates = [
        base_dir / "test_results" / setting,
        base_dir / "results" / setting,
        Path.cwd() / "test_results" / setting,
        Path.cwd() / "results" / setting,
        ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None

def _fallback_latest_subdir(base_dir: Path) -> Path | None:
    """
    If we cannot locate <setting>, pick the most recently modified subdir
    under base_dir/test_results or base_dir/results.
    """
    for parent in [base_dir / "test_results", base_dir / "results", Path.cwd() / "test_results", Path.cwd() / "results"]:
        if parent.exists() and parent.is_dir():
            subs = [p for p in parent.iterdir() if p.is_dir()]
            if subs:
                subs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return subs[0]
    return None

def export_outputs(setting: str, args, base_dir: Path) -> None:
    """
    After exp.test(setting), copy outputs into args.results_path/<setting>/ as:
      preds.npy, trues.npy, metrics.npy (+ metrics.json + args.json)
    """
    results_root = Path(getattr(args, "results_path", base_dir / "results"))
    out_dir = results_root / setting
    out_dir.mkdir(parents=True, exist_ok=True)

    src_dir = _find_run_output_folder(base_dir, setting)
    if src_dir is None:
        src_dir = _fallback_latest_subdir(base_dir)

    if src_dir is None:
        raise FileNotFoundError(
            "Could not find any output folder. Looked under test_results/ and results/."
        )

    src_pred = _find_first_existing(src_dir, SRC_PRED_NAMES)
    src_true = _find_first_existing(src_dir, SRC_TRUE_NAMES)
    src_met  = _find_first_existing(src_dir, SRC_METRIC_NAMES)

    if src_pred is None or src_true is None:
        raise FileNotFoundError(
            f"Could not find pred/true .npy in {src_dir}.\n"
            f"Expected one of {SRC_PRED_NAMES} and one of {SRC_TRUE_NAMES}."
        )

    shutil.copy2(src_pred, out_dir / "preds.npy")
    shutil.copy2(src_true, out_dir / "trues.npy")
    if src_met is not None:
        shutil.copy2(src_met, out_dir / "metrics.npy")

    # Write args snapshot
    _save_args_json(out_dir, args)

    # Write a small metrics.json (best effort)
    metrics_json = {
        "setting": setting,
        "source_dir": str(src_dir),
        "preds_file": "preds.npy",
        "trues_file": "trues.npy",
    }

    if src_met is not None:
        try:
            m = np.load(src_met, allow_pickle=True)
            # Typical format in these repos: [mae, mse, rmse, mape, mspe]
            if hasattr(m, "shape") and m.size >= 5:
                metrics_json.update(
                    {
                        "mae": float(m.flat[0]),
                        "mse": float(m.flat[1]),
                        "rmse": float(m.flat[2]),
                        "mape": float(m.flat[3]),
                        "mspe": float(m.flat[4]),
                    }
                )
            else:
                metrics_json["metrics_raw"] = m.tolist() if hasattr(m, "tolist") else str(m)
        except Exception as e:
            metrics_json["metrics_read_error"] = str(e)

    (out_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    print(f"[INFO] Exported outputs to: {out_dir}")
    print(f"       - {out_dir / 'preds.npy'}")
    print(f"       - {out_dir / 'trues.npy'}")
    if (out_dir / "metrics.npy").exists():
        print(f"       - {out_dir / 'metrics.npy'}")
    print(f"       - {out_dir / 'metrics.json'}")
    print(f"       - {out_dir / 'args.json'}")


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    # -------------------- Repro --------------------
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # -------------------- Device --------------------
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # -------------------- Derived paths & constants (genomic) --------------------
    base_dir = os.path.dirname(__file__)
    default_root = os.path.join(base_dir, "data")
    default_ckpt = os.path.join(base_dir, "checkpoints")
    default_results = os.path.join(base_dir, "results")

    # Normalize root/checkpoints
    if args.root_path in (None, './data/ETT/', './data'):
        args.root_path = default_root
    if args.checkpoints in (None, './checkpoints/'):
        args.checkpoints = default_ckpt
    args.results_path = default_results  # handy for saving metrics

    # If user provided --cell, prefer {root}/{cell}_genomic.csv
    if getattr(args, 'cell', None):
        args.data_path = os.path.join(args.root_path, f"{args.cell}_genomic.csv")
        if args.data == 'custom':
            args.freq = "w"
            args.embed = "timeF"
            args.output_attention = False
            args.distil = False

    # -------------------- Setting string --------------------
    if not args.setting:
        if getattr(args, 'cell', None) and args.val_chroms is not None:
            val_str = "-".join(str(c) for c in args.val_chroms) if len(args.val_chroms) else "none"
            args.setting = f"{args.cell}_val_{val_str}"
        else:
            args.setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed,
                args.distil, args.des
            )

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    base_dir_path = Path(base_dir).resolve()

    if args.is_training:
        for ii in range(args.itr):
            # If the user already provided a setting, reuse it; otherwise append the iteration idx
            if args.setting:
                setting = f"{args.setting}_{ii}"
            else:
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed,
                    args.distil, args.des, ii
                )

            exp = Exp(args)  # set experiments
            print(f'>>>>>>> start training : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)

            # ---- NEW: Export preds/trues/metrics into results/<setting>/ ----
            export_outputs(setting, args, base_dir_path)

            if args.do_predict:
                print(f'>>>>>>> predicting : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = args.setting or '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed,
            args.distil, args.des, ii
        )
        exp = Exp(args)  # set experiments
        print(f'>>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)

        # ---- NEW: Export preds/trues/metrics into results/<setting>/ ----
        export_outputs(setting, args, base_dir_path)

        torch.cuda.empty_cache()