import os
import argparse
from run_model import run_model_main

def parse_args():
    parser = argparse.ArgumentParser(description="Train Informer on intra-cell chromosomes with configurable parameters.")

    parser.add_argument('--setting', type=str, default=None)
    parser.add_argument('--cell', type=str, default="H1")
    parser.add_argument('--train_chroms', nargs='+', type=int, default=[1,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22])
    parser.add_argument('--val_chroms', nargs='*', default=[6], type=int)
    parser.add_argument('--test_chroms', nargs='*', default=[9], type=int)

    # Sequence
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--label_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=1)

    # Architecture
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

    # Feature selection for Informer
    parser.add_argument(
        '--selected_cols',
        nargs='+',
        type=str,
        default=['H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1',
                 'H3K4me3', 'H3K9me3', 'GC_content', 'gene_density', '2-stage'],
        help='Informer feature columns.'
    )

    # NEW: separate feature selection + scaling for Soffritto LSTM pipeline
    parser.add_argument(
        '--lstm_selected_cols',
        nargs='+',
        type=str,
        default=None,
        help='Feature columns for the LSTM teacher pipeline. If omitted, defaults to Informer selected_cols but scaled independently.'
    )
    parser.add_argument(
        '--lstm_scale',
        action='store_true',
        help='Enable scaling for LSTM pipeline (default: ON).'
    )
    parser.add_argument(
        '--no_lstm_scale',
        dest='lstm_scale',
        action='store_false',
        help='Disable scaling for LSTM pipeline.'
    )
    parser.set_defaults(lstm_scale=True)

    # Decoder mode
    parser.add_argument(
        "--decoding_mode",
        type=str,
        default="cost-aware-3",
        choices=["teacher-forced", "cost-aware-1", "cost-aware-2", "cost-aware-3"],
        help="Decoder input strategy. cost-aware-3 uses pretrained Soffritto BiLSTM predictions as decoder history (LSTM uses its own data pipeline)."
    )

    # rt2 column name for cost-aware-2 indexing (must exist in raw CSV input columns)
    parser.add_argument(
        "--rt2_col",
        type=str,
        default="2-stage",
        help="Column name to treat as rt2/2-stage for cost-aware-2."
    )

    # ---- cost-aware-3 (LSTM teacher) options ----
    parser.add_argument(
        "--lstm_model_path",
        type=str,
        default='data/trained_models/H1_intra_cell_line_model.pth',
        help="Path to pretrained Soffritto LSTM .pth checkpoint (required for cost-aware-3)."
    )
    parser.add_argument(
        "--lstm_hyperparameter_file",
        type=str,
        default='data/trained_models/H1_intra_cell_line_model_hyperparameters.json',
        help="Path to JSON file containing hidden_size and num_layers saved by train_intra_cell_line.py (optional if you pass lstm_hidden_size/num_layers)."
    )
    parser.add_argument(
        "--lstm_hidden_size",
        type=int,
        default=None,
        help="Hidden size of the pretrained LSTM teacher (used if lstm_hyperparameter_file not provided)."
    )
    parser.add_argument(
        "--lstm_num_layers",
        type=int,
        default=None,
        help="Number of layers of the pretrained LSTM teacher (used if lstm_hyperparameter_file not provided)."
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
    args.model = "informer"
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
        val_str = "-".join(str(c) for c in args.val_chroms) if args.val_chroms else "none"
        args.setting = f"{args.cell}_val_{val_str}"

    print(args)
    metrics = run_model_main(args)
    print(f"[INFO] Training finished with metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    main()