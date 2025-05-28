import argparse
from argparse import Namespace
import os
from train_intra_cell_line import main as train_intra_cell_line_main

def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested LOCO-CV for Soffritto model: Outer test chrom + inner cross-validation on remaining chroms")
    parser.add_argument(
        '--cell', type=str, required=True, help='Cell type to train on')
    parser.add_argument(
        '--test_chrom', type=int, required=True, help='Outer test chromosome number')


    parser.add_argument(
        '--learning_rate', type=float, help='Learning rate for optimizer', default=0.001)
    parser.add_argument(
        '--num_epochs', type=int, help='Number of training epochs', default=100)
    parser.add_argument(
        '--batch_size', type=int, help='Batch size for training', default=64)
    parser.add_argument(
        '--num_hiddens', type=int, help='Hidden size for LSTM', default=64)
    parser.add_argument(
        '--num_layers', type=int, help='Number of LSTM layers', default=4)
    parser.add_argument(
        '--weight_decay', type=float, help='L2 regularization coefficient', default=0.0001)
    return parser.parse_args()

def get_available_chroms(cell):
    return list(range(1, 20)) if cell.startswith("m") else list(range(1, 22))

def main():
    args = parse_args()
    outer_test = args.test_chrom
    chroms_all = get_available_chroms(args.cell)

    if outer_test not in chroms_all:
        raise ValueError(
            f"[ERROR] test_chrom {outer_test} not in available_chroms {chroms_all}")

    inner_chroms = [c for c in chroms_all if c != outer_test]
    results = []

    print(f"\n[INFO] Nested LOCO-CV for Soffritto: Outer test chrom = {outer_test}")
    print(f"[INFO] Inner validation on chromosomes: {inner_chroms}")

    for val_ch in inner_chroms:
        train_chs = [c for c in inner_chroms if c != val_ch]
        setting = f"outerTest{outer_test}_val{val_ch}"
        model_path = os.path.join('./checkpoints/soffritto', f"{setting}.pth")
        hyper_file = os.path.join('./checkpoints/soffritto/', f"{setting}_hyperparams.json")

        run_args = Namespace(
            features=f'./transofritto/data/raw/{args.cell}_features.npz',
            labels=f'./transofritto/data/raw/{args.cell}_labels.npz',
            model_path=model_path,
            train_chromosomes=[str(c) for c in train_chs],
            val_chromosomes=str(val_ch),
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            num_hiddens=args.num_hiddens,
            num_layers=args.num_layers,
            weight_decay=args.weight_decay,
            hyperparameter_file=hyper_file
        )

        print(f"\n[INFO] Fold: Train on {train_chs} | Validate on chromosome {val_ch}")
        result = train_intra_cell_line_main(run_args)

        if result is not None and 'val_score' in result:
            results.append((val_ch, result['val_score']))
        else:
            print(
                f"[WARNING] No valid val_score returned for fold val_chrom {val_ch}")

    # Summary
    print(f"\n[SUMMARY] Inner Validation Scores (Outer Test Chrom: {outer_test})")
    for chrom, score in results:
        print(f"  Val Chrom {chrom}: {score:.4f}")

    if results:
        avg_score = sum(score for _, score in results) / len(results)
        print(f"\n[AVERAGE INNER VAL SCORE]: {avg_score:.4f}")
    else:
        print("\n[ERROR] No validation results obtained.")

if __name__ == '__main__':
    main()