import sys
import os
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

def main():
    script_path = os.path.join(os.path.dirname(__file__), 'run_model.py')
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    checkpoints_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')

    train_chroms = ["1", "2", "3", "4", "5", "6"]
    val_chroms = ["7"]

    command = [
        "python", script_path,
        "--model", "informer",
        "--setting", "multitarget",
        "--root_path", data_dir,
        "--data_path", "H1_genomic.csv",
        "--target", "target_1",
        "--seq_len", "32",
        "--label_len", "16",
        "--pred_len", "1",
        "--train_epochs", "10",
        "--batch_size", "32",
        "--learning_rate", "0.0001",
        "--train_chroms", *train_chroms,
        "--val_chroms", *val_chroms,
        "--checkpoints", checkpoints_dir,
        "--weight_decay", "0.001",
        "--num_workers", str(5),
        "--train_epochs", str(1)
    ]

    result = subprocess.run(command)
    print(f"\n[INFO] Script exited with return code {result.returncode}")

if __name__ == "__main__":
    main()
