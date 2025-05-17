import subprocess
import os,sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
def main():
    # Command to run the script with arguments
    script_path = os.path.join(os.path.dirname(__file__), 'run.py')
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    checkpoints_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    train_chroms = ["1", "2", "3", "4", "5", "6"]
    val_chroms = ["7"]
    command = [
        "python", script_path,
        "--is_training", "1",                 # train mode
        "--model_id", "exp1",                 # experiment ID
        "--model", "iTransformer",            # model type
        "--data", "custom",                   # dataset type
        "--root_path", data_dir, # data path
        "--data_path", "H1_genomic.csv",
        "--features", "M",
        "--seq_len", "96",
        "--label_len", "48",
        "--pred_len", "1",
        "--enc_in", "9",
        "--dec_in", "16",
        "--c_out", "16",
        "--train_chroms", *train_chroms,
        "--val_chroms", *val_chroms,
        "--class_strategy", "projection",
        "--loss", "KL",                       # assuming you're using KL divergence
        "--exp_name", "MTSF",
        "--use_norm", "0",
        "--target", "target_1",
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print("Experiment completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Experiment failed with return code:", e.returncode)

if __name__ == "__main__":
    main()
