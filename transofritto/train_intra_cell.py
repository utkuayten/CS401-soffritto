import os
import argparse
from run_model import run_model_main

def parse_args():
    parser = argparse.ArgumentParser(description="Train Informer on intra-cell chromosomes with configurable parameters.")

    parser.add_argument('--setting', type=str, required=False)

    # Genomic args
    parser.add_argument('--cell', type=str, required=True)
    parser.add_argument('--train_chroms', nargs='+', type=int, help='List of chromosomes for training')
    parser.add_argument('--val_chroms', nargs='+', type=int,  help='List of chromosomes for validation')

    # Sequence lengths
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--label_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=1)

    # Model architecture
    parser.add_argument('--enc_in', type=int, default=9, help = "encoder input dimension")
    parser.add_argument('--dec_in', type=int, default=16, help = "decoder input dimension")
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

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.000045)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=5)

    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--devices', type=str, default='0')


    return parser.parse_args()

def main(args=None):
    if args is None:
        args = parse_args()

    # Add derived/default paths
    base_dir = os.path.dirname(__file__)
    args.root_path = os.path.join(base_dir, "data")
    args.data_path = f"{args.cell}_genomic.csv"
    args.checkpoints = os.path.join(base_dir, "checkpoints")
    args.results_path = os.path.join(base_dir, "results")

    # Constant parameters.
    args.model = "informer"
    args.setting = "multitarget"
    args.target = "target_1"
    args.freq = "w"
    args.embed = "timeF"
    args.output_attention = False
    args.distil = True
    args.mix = False
    args.data = "custom"
    args.features = "M"
    args.inverse = False

    # Run training
    metrics = run_model_main(args)
    print(f"[INFO] Training finished with metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    main()
