#!/usr/bin/env python
import argparse
import os
import sys
import subprocess
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="LOCO‐Soffritto: For a given outer test chromosome, load a pre‐trained Soffritto model "
                    "and compute per‐chromosome validation scores by invoking predict_intra_cell_line.py.")
    parser.add_argument(
        '--cell', type=str,
        help='Cell type (e.g., "mouseCellX" or "humanCellY")', default='H1')
    parser.add_argument(
        '--test_chrom', type=int,
        help='Outer test chromosome number (e.g., 1, 2, ..., 19 or 21)', default=9)
    parser.add_argument(
        '--model_path', type=str,
        help='Path to the trained model state_dict (e.g., best_model.pth)', default='/Users/ozgun/DataspellProjects/CS401-soffritto/soffritto/trained_models/H1_intra_cell_line_model.pth')
    parser.add_argument(
        '--hyperparameter_file', type=str,
        help='Path to the JSON file containing the best hyperparameters '
             '(e.g., {"hidden_size":..., "num_layers":...})', default='/Users/ozgun/DataspellProjects/CS401-soffritto/soffritto/trained_models/H1_intra_cell_line_model_hyperparameters.json')
    parser.add_argument(
        '--features_file', type=str,
        help='Path to the .npz (or .npz‐equivalent) features file used by predict_intra_cell_line.py '
             '(contains everything for all chromosomes)', default='/Users/ozgun/DataspellProjects/CS401-soffritto/transofritto/data/raw/H1_features.npz')
    parser.add_argument(
        '--labels_file', type=str,
        help='Path to the .npz (or .npz‐equivalent) labels file for 16‐fraction RT values', default='/Users/ozgun/DataspellProjects/CS401-soffritto/transofritto/data/raw/H1_labels.npz')
    parser.add_argument(
        '--pred_dir', type=str, default='./predictions',
        help='Directory where per‐chromosome predictions (and true values) will be saved. '
             'Defaults to ./predictions')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Make sure the output directory exists
    if not os.path.isdir(args.pred_dir):
        os.makedirs(args.pred_dir, exist_ok=True)

    # 2) Determine full list of chromosomes based on "cell"
    #    (Matches the logic in LOChromOut_CV_train.py: mouse has 1–19, human has 1–21)
    if args.cell.startswith("m"):
        all_chroms = list(range(1, 20))   # mouse: 1..19
    else:
        all_chroms = list(range(1, 22))   # human: 1..21

    outer_test = int(args.test_chrom)
    if outer_test not in all_chroms:
        raise ValueError(f"test_chrom={outer_test} not in expected list {all_chroms}")

    # 3) Build the “outer_train” list: all chromosomes except the outer test
    outer_train = [c for c in all_chroms if c != outer_test]
    if len(outer_train) == 0:
        raise ValueError("No chromosomes remain after removing the outer test. "
                         "Check --test_chrom and --cell inputs.")

    results = []

    # 4) For each chromosome in outer_train, treat it as a “validation/test” fold
    for val_ch in outer_train:
        #
        # We will call:
        #   python predict_intra_cell_line.py
        #     --train_features_file  <features_file>
        #     --test_features_file   <features_file>
        #     --test_labels_file     <labels_file>
        #     --model_path           <args.model_path>
        #     --pred_file            <pred_base>     (no “.npy” suffix)
        #     --train_chromosomes    <space‐sep list of chr in outer_train>
        #     --test_chromosomes     <val_ch>
        #     --hyperparameter_file  <args.hyperparameter_file>
        #
        # We always pass the same “outer_train” list for train_chromosomes,
        # since the scaler was fitted on all non‐outer_test chroms.  We then
        # isolate val_ch as the one‐chrom validation.
        #
        train_chroms_str = " ".join(str(c) for c in outer_train)
        test_chroms_str = str(val_ch)

        # basename for saved predictions (no extension).  predict_intra_cell_line.py
        # will produce:  <pred_base>.npy   and   <pred_base>_true.npy
        pred_base = os.path.join(
            args.pred_dir,
            f"{args.cell}_outer{outer_test}_val{val_ch}"
        )

        # 4a) Invoke predict_intra_cell_line.py as a subprocess
        cmd = [
                  sys.executable, "-u", "./soffritto/predict_intra_cell_line.py",
                  "--train_features_file", args.features_file,
                  "--test_features_file",  args.features_file,
                  "--test_labels_file",    args.labels_file,
                  "--model_path",          args.model_path,
                  "--pred_file",           pred_base,
                  "--train_chromosomes"
              ] + train_chroms_str.split() + [
                  "--test_chromosomes",    test_chroms_str,
                  "--hyperparameter_file", args.hyperparameter_file
              ]

        print(f"\n[LOCO‐Soffritto] Running prediction on val_chrom={val_ch} (outer_test={outer_test}) …")
        print("  Command:", " ".join(cmd))
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"[ERROR] predict_intra_cell_line.py failed for val_chrom={val_ch}.  Skipping.")
            continue

        # 4b) After a successful run, load the saved prediction & true arrays
        #     predict_intra_cell_line.py does:
        #       np.save(args.pred_file, pred)       → writes "<pred_base>.npy"
        #       np.save(f"{args.pred_file}_true", y_true)  → writes "<pred_base>_true.npy"
        pred_path = pred_base + ".npy"
        true_path = pred_base + "_true.npy"

        if not (os.path.isfile(pred_path) and os.path.isfile(true_path)):
            print(f"[WARNING] Missing output files for val_ch={val_ch}: "
                  f"{pred_path} or {true_path} not found.  Skipping.")
            continue

        pred_arr = np.load(pred_path)       # shape: (n_samples, 16)
        true_arr = np.load(true_path)       # shape: (n_samples, 16)

        # 4c) Compute KL‐Divergence per‐sample and average
        #     KL(p || q) = Σ_i p_i * log(p_i / q_i).  We’ll use a small eps to avoid log(0).
        eps = 1e-8
        p = true_arr + eps
        q = pred_arr + eps
        # sum over the 16 fractions, then mean over all samples
        kl_per_sample = np.sum(p * np.log(p / q), axis=1)
        kl_score = float(np.mean(kl_per_sample))

        results.append((val_ch, kl_score))
        print(f"  → Val Chrom {val_ch}: KL‐Divergence = {kl_score:.6f}")

    # 5) After looping over all val_chroms, print a summary
    print(f"\n[SUMMARY] Per‐Chromosome KL Scores  (Outer Test Chrom = {outer_test})")
    for (chrom, score) in results:
        print(f"  Chrom {chrom:2d}:  KL = {score:.6f}")

    if results:
        avg_score = float(np.mean([s for (_, s) in results]))
        print(f"\n[AVERAGE  KL SCORE (over {len(results)} folds)]: {avg_score:.6f}")
    else:
        print("\n[ERROR] No valid folds completed—no results to summarize.")


if __name__ == "__main__":
    main()