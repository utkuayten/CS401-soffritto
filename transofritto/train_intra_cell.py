import os
import argparse
from run_model import run_model_main

def parse_args():
    parser = argparse.ArgumentParser(description="Train Informer on intra-cell chromosomes with configurable parameters.")

    parser.add_argument('--setting', type=str, default=None)
    parser.add_argument('--cell', type=str, required=True)
    parser.add_argument('--train_chroms', nargs='+', type=int, required=True)
    parser.add_argument('--val_chroms', nargs='*', default = [], type=int, required=True)

    # Sequence
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--label_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=1)

    # Architecture
    parser.add_argument('--enc_in', type=int, default=9)
    parser.add_argument('--dec_in', type=int, default=16)
    parser.add_argument('--c_out', type=int, default=16)
    parser.add_argument('--e_layers', type=int, default=1)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.03)
    parser.add_argument('--attn', type=str, default='prob')
    parser.add_argument('--factor', type=int, default=5)
    parser.add_argument('--activation', type=str, default='gelu')

    # Training
    parser.add_argument('--learning_rate', type=float, default=0.000045)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=5)

    # GPU
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--devices', type=str, default='0')

    # --- Wavelet options ---
    parser.add_argument('--use_wavelet', action='store_true',
                        help='Enable wavelet features (e.g., SWT) on inputs')
    parser.add_argument('--wavelet_name', type=str, default='db4',
                        help='PyWavelets wavelet name (e.g., db4, coif1, sym4)')
    parser.add_argument('--wavelet_levels', type=int, default=1,
                        help='Number of decomposition levels (>=1)')
    parser.add_argument('--keep_original', action='store_true',
                        help='Concatenate original features with wavelet bands')
    parser.add_argument('--wavelet_where', type=str, default='dataset',
                        choices=['dataset','model'],
                        help='Where to apply wavelet transform')

    return parser.parse_args()

def main(args=None):
    if args is None:
        args = parse_args()

    # Derived paths
    base_dir = os.path.dirname(__file__)
    args.root_path = os.path.join(base_dir, "data")
    args.data_path = os.path.join(args.root_path, f"{args.cell}_genomic.csv")
    args.checkpoints = os.path.join(base_dir, "checkpoints")
    args.results_path = os.path.join(base_dir, "results")

    # Constant config (can be changed if needed)
    args.model = "informer"
    args.target = "target_1"
    args.freq = "w"
    args.embed = "timeF"
    args.output_attention = False
    args.distil = True
    args.mix = False
    args.data = "custom"
    args.features = "M"
    args.inverse = False
    args.padding = 0

    if not args.setting:
        val_str = "-".join(str(c) for c in args.val_chroms)
        args.setting = f"{args.cell}_val_{val_str}"


        # Compute post-wavelet input sizes if applied in the dataset
    if args.use_wavelet and args.wavelet_where == 'dataset':
        # For SWT, per original feature you get 2*levels bands (cA_l and cD_l).
        # If keep_original=True, total multiplier = (1 + 2*levels), else = (2*levels).
        mult = (1 + 2 * args.wavelet_levels) if args.keep_original else (2 * args.wavelet_levels)

        # Only auto-bump if user kept defaults (so we don't override explicit values).
        default_enc_in, default_dec_in = 9, 16
        if args.enc_in == default_enc_in:
            args.enc_in = args.enc_in * mult
        if args.dec_in == default_dec_in:
            args.dec_in = args.dec_in * mult

        print(f"[WAVELET] Enabled ({args.wavelet_name}, L={args.wavelet_levels}, "
              f"keep_original={args.keep_original}) -> multiplier x{mult}")
        print(f"[WAVELET] enc_in={args.enc_in}, dec_in={args.dec_in} (adjusted for dataset wavelets)")


    metrics = run_model_main(args)
    print(f"[INFO] Training finished with metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    main()
