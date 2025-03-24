import numpy as np
import pandas as pd

# Load files
features = np.load("data/H1_features.npz")
labels = np.load("data/H1_labels.npz")
coords_df = pd.read_csv("data/H1_coordinates.bedgraph", sep="\t", header=None,
                        names=["chr", "start", "end", "value"])
coords_df['date'] = ((coords_df['start'] + coords_df['end']) // 2).astype(int)

# Chromosome list
chroms = ['22', '9', '17', '7', '13', '20', '8', '15', '19', '18',
          '5', '14', '3', '10', '21', '1', '12', '2', '11', '4', '16', '6']

feat_list = []
label_list = []
coord_list = []

for chrom in chroms:
    if chrom in features and chrom in labels:
        X = features[chrom]
        Y = labels[chrom]

        # Get matching coordinates
        chrom_coords = coords_df[coords_df["chr"] == f"chr{chrom}"]["date"].values

        # Align lengths
        min_len = min(len(X), len(Y), len(chrom_coords))
        feat_list.append(X[:min_len])
        label_list.append(Y[:min_len])
        coord_list.append(chrom_coords[:min_len])

# Combine
X_all = np.concatenate(feat_list)
Y_all = np.concatenate(label_list)
coords_all = np.concatenate(coord_list)

# Build dataframe
df = pd.DataFrame(X_all, columns=[f'feat{i}' for i in range(X_all.shape[1])])
df['date'] = coords_all
target_df = pd.DataFrame(Y_all, columns=[f'target_{i+1}' for i in range(Y_all.shape[1])])
df = pd.concat([df[['date'] + [f'feat{i}' for i in range(X_all.shape[1])]], target_df], axis=1)

# Save
df.to_csv("data/H1_genomic.csv", index=False)
print("✅ H1_genomic.csv created with shape:", df.shape)
