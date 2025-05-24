import argparse
from argparse import Namespace
from train_intra_cell import main as train_intra_cell_main

def parse_args():
    parser = argparse.ArgumentParser(description="Nested LOCO-CV: Outer test chrom + inner cross-validation on remaining chroms")
    parser.add_argument('--cell', type=str, required=True)
    parser.add_argument('--test_chrom', type=int, required=True)

    # Model & training hyperparameters
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--label_len', type=int, default=16)
    parser.add_argument('--pred_len', type=int, default=1)

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
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--factor', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.000045)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--devices', type=str, default='0')
    return parser.parse_args()

def get_available_chroms(cell):
    return list(range(1, 20)) if cell.startswith("m") else list(range(1, 22))

def main():
    args = parse_args()
    outer_test_chrom = args.test_chrom
    chroms_all = get_available_chroms(args.cell)

    if outer_test_chrom not in chroms_all:
        raise ValueError(f"[ERROR] test_chrom {outer_test_chrom} is not in {chroms_all}")

    inner_chroms = [c for c in chroms_all if c != outer_test_chrom]
    results = []

    print(f"\n[INFO] Nested LOCO: Outer test chrom = {outer_test_chrom}")
    print(f"[INFO] Inner LOCO folds on: {inner_chroms}")

    for inner_val_chrom in inner_chroms:
        inner_train_chroms = [c for c in inner_chroms if c != inner_val_chrom]

        run_args = Namespace(
            cell=args.cell,
            train_chroms=inner_train_chroms,
            val_chroms=[inner_val_chrom],
            setting=f"{args.cell}_outerTest{outer_test_chrom}_val{inner_val_chrom}",

            # Hyperparameters
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            enc_in=args.enc_in,
            dec_in=args.dec_in,
            c_out=args.c_out,
            e_layers=args.e_layers,
            d_layers=args.d_layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            attn=args.attn,
            activation=args.activation,
            factor=args.factor,
            learning_rate=args.learning_rate,
            train_epochs=args.train_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            lradj=args.lradj,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            use_multi_gpu=args.use_multi_gpu,
            gpu=args.gpu,
            devices=args.devices,
        )

        print(f"\n[INFO] Inner Fold: Train on {len(inner_train_chroms)} chroms | Validate on {inner_val_chrom}")
        result = train_intra_cell_main(run_args)

        if result and 'val_score' in result:
            results.append((inner_val_chrom, result['val_score']))
        else:
            print(f"[WARNING] No valid result returned for val_chrom {inner_val_chrom}")

    # Summary
    print("\n[SUMMARY] Inner Validation Scores (Outer Test Chrom: {})".format(outer_test_chrom))
    for chrom, score in results:
        print(f"  Val Chrom {chrom}: {score:.4f}")

    if results:
        avg = sum(score for _, score in results) / len(results)
        print(f"\n[AVERAGE INNER VAL SCORE]: {avg:.4f}")
    else:
        print("\n[ERROR] No results from any inner fold.")

if __name__ == "__main__":
    main()
