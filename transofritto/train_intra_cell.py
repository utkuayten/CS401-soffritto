from run_model import run_model_main  # assuming run_model_main(args) returns metrics
from argparse import Namespace
import os

def main(args=None):
    if args is None:
        args = parse_args()

    # Add missing paths
    base_dir = os.path.dirname(__file__)
    args.root_path = os.path.join(base_dir, "data")
    args.data_path = f"{args.cell}_genomic.csv"
    args.checkpoints = os.path.join(base_dir, "checkpoints")
    args.results_path = os.path.join(base_dir, "results")
    args.model = "informer"
    args.setting = "multitarget"
    args.target = "target_1"

    # Run training using refactored run_model.py
    metrics = run_model_main(args)
    print(f"[INFO] Training finished with metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    main()
