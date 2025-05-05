import numpy as np
import pandas as pd
import os

def process_cell_data(cell: str, input_path: str = "data") -> pd.DataFrame:
    """Processes genomic feature and label data for a given cell type.

    Args:
        cell (str): Cell type name (e.g., "HCT116", "H1").
        input_path (str): Path to save the final CSV file.

    Returns:
        pd.DataFrame: Combined DataFrame of features, targets, coordinates, and chromosome info.
    """

    # Load feature and label files
    features = np.load(f"{input_path}/raw//{cell}_features.npz")
    labels = np.load(f"{input_path}/raw/{cell}_labels.npz")
    coords_df = pd.read_csv(f"{input_path}/raw/{cell}_coordinates.bedgraph", sep="\t", header=None,
                            names=["chr", "start", "end", "value"])
    coords_df['date'] = ((coords_df['start'] + coords_df['end']) // 2).astype(int)

    chroms = ['22', '9', '17', '7', '13', '20', '8', '15', '19', '18',
              '5', '14', '3', '10', '21', '1', '12', '2', '11', '4', '16', '6']

    feat_list = []
    label_list = []
    coord_list = []
    chrom_list = []

    for chrom in chroms:
        if chrom in features and chrom in labels:
            X = features[chrom]
            Y = labels[chrom]
            chrom_coords = coords_df[coords_df["chr"] == f"chr{chrom}"]["date"].values

            min_len = min(len(X), len(Y), len(chrom_coords))
            feat_list.append(X[:min_len])
            label_list.append(Y[:min_len])
            coord_list.append(chrom_coords[:min_len])
            chrom_list.append(np.full(min_len, chrom))

    # Combine
    X_all = np.concatenate(feat_list)
    Y_all = np.concatenate(label_list)
    coords_all = np.concatenate(coord_list)
    chrom_all = np.concatenate(chrom_list)

    # Build dataframe
    df = pd.DataFrame(X_all, columns=[f'feat{i}' for i in range(X_all.shape[1])])
    df['date'] = coords_all
    df['chrom'] = chrom_all

    target_df = pd.DataFrame(Y_all, columns=[f'target_{i+1}' for i in range(Y_all.shape[1])])
    df = pd.concat([df[['chrom', 'date'] + [f'feat{i}' for i in range(X_all.shape[1])]], target_df], axis=1)

    # Save CSV
    output_file = os.path.join(input_path, f"{cell}_genomic.csv")
    df.to_csv(output_file, index=False)
    print(f"{cell}_genomic.csv created with shape:", df.shape)

    return df


def main():
    cells = ["H1", "H9", "HCT116", "mESC" ,"mNPC"]
    for cell in cells:
        process_cell_data(cell)


if __name__ == "__main__":
    main()
