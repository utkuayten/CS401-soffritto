import os
import argparse
from run_model import run_model_main

def parse_args():
    parser = argparse.ArgumentParser(description="Train GAT+Informer on intra-cell chromosomes with configurable parameters.")

    parser.add_argument('--setting', type=str, default=None)
    parser.add_argument('--cell', type=str, default="H1")
    parser.add_argument('--train_chroms', nargs='+', type=int, default=[1,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22])
    parser.add_argument('--val_chroms', nargs='*', default=[6], type=int)
    parser.add_argument('--test_chroms', nargs='*', default=[9], type=int)

    # Sequence
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--label_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=1)

    # Architecture (Informer)
    parser.add_argument('--enc_in', type=int, default=9)
    parser.add_argument('--dec_in', type=int, default=16)
    parser.add_argument('--c_out', type=int, default=16)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.012087945956316543)
    parser.add_argument('--attn', type=str, default='full')
    parser.add_argument('--factor', type=int, default=7)
    parser.add_argument('--activation', type=str, default='relu')

    # -------- NEW: GAT front-end configs --------
    parser.add_argument('--model', type=str, default='gatinformer',
                        choices=['informer', 'informerstack', 'gatinformer'])
    parser.add_argument('--gat_layers', type=int, default=2)
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--gat_k', type=int, default=2, help="Local neighbor window for 1D GAT: attends to +/- k bins.")
    parser.add_argument('--gat_hidden', type=int, default=None, help="Hidden dim inside stacked GAT (defaults to enc_in).")
    parser.add_argument('--gat_dropout', type=float, default=0.0)
    parser.add_argument('--apply_gat_to_dec', action='store_true',
                        help="If set, also applies GAT to decoder inputs (usually leave off).")

    # Training
    parser.add_argument('--learning_rate', type=float, default=0.00025356419877530715)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lradj', type=str, default='type3')
    parser.add_argument('--weight_decay', type=float, default=0.005470219047192386)
    parser.add_argument('--num_workers', type=int, default=1)

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

    # Feature selection
    parser.add_argument(
        '--selected_cols',
        nargs='+',
        type=str,
        default=['H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1',
                 'H3K4me3', 'H3K9me3', 'GC_content', 'gene_density', '2-stage'],
        help='List of feature column names to use for training (default: all 9 features)'
    )

    # Decoder mode
    parser.add_argument(
        "--decoding_mode",
        type=str,
        default="cost_aware-1",
        choices=["teacher-forced", "cost-aware-1", "cost-aware-2"],
        help="Decoder input strategy."
    )

    parser.add_argument(
        "--rt2_col",
        type=str,
        default="2-stage",
        help="Column name to treat as rt2/2rt for cost-aware-2 (used only if Dataset_Custom can't auto-detect rt2_idx)."
    )

    return parser.parse_args()

def main(args=None):
    if args is None:
        args = parse_args()

    base_dir = os.path.dirname(__file__)
    args.root_path = os.path.join(base_dir, "data")
    args.data_path = os.path.join(args.root_path, f"{args.cell}_genomic.csv")
    args.checkpoints = os.path.join(base_dir, "checkpoints")
    args.results_path = os.path.join(base_dir, "results")

    # Constant config
    args.target = "target_1"
    args.freq = "w"
    args.embed = "timeF"
    args.output_attention = False
    args.distil = False
    args.mix = False
    args.data = "custom"
    args.features = "M"
    args.inverse = False
    args.padding = 0

    if not args.setting:
        val_str = "-".join(str(c) for c in args.val_chroms)
        args.setting = f"{args.cell}_val_{val_str}"

    print(args)
    metrics = run_model_main(args)
    print(f"[INFO] Training finished with metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    main()