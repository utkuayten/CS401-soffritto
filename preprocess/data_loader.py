import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class RepliSeqFeatureLabelWithCoordinates(Dataset):
    def __init__(self, x_path, y_path, bedgraph_path, chromosomes, size=(96, 48, 48), scale=True):
        self.x_path = x_path
        self.y_path = y_path
        self.bedgraph_path = bedgraph_path
        self.chromosomes = chromosomes
        self.seq_len, self.label_len, self.pred_len = size
        self.scale = scale
        self.__load_data__()

    def __load_data__(self):
        x_data = np.load(self.x_path)
        y_data = np.load(self.y_path)

        coord_df = pd.read_csv(
            self.bedgraph_path, sep='\t', header=None,
            names=['chr', 'start', 'end', 'value']
        )
        coord_df['mid'] = ((coord_df['start'] + coord_df['end']) / 2).astype(int)

        x_all, y_all, coord_all = [], [], []

        for chrom in self.chromosomes:
            if chrom in x_data and chrom in y_data:
                x = x_data[chrom]
                y = y_data[chrom]
                mids = coord_df[coord_df['chr'] == f'chr{chrom}']['mid'].values.reshape(-1, 1)

                min_len = min(len(x), len(y), len(mids))
                x_all.append(x[:min_len])
                y_all.append(y[:min_len])
                coord_all.append(mids[:min_len])
            else:
                print(f"[Warning] Missing chromosome {chrom}")

        self.x_all = np.concatenate(x_all, axis=0)
        self.y_all = np.concatenate(y_all, axis=0)
        self.coordinates = np.concatenate(coord_all, axis=0)

        if self.scale:
            self.x_all = (self.x_all - self.x_all.mean(0)) / (self.x_all.std(0) + 1e-8)
            self.y_all = (self.y_all - self.y_all.mean(0)) / (self.y_all.std(0) + 1e-8)

        self.coord_max = self.coordinates.max()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.x_all[s_begin:s_end]
        seq_y = self.y_all[r_begin:r_end]
        seq_x_mark = self.coordinates[s_begin:s_end] / self.coord_max
        seq_y_mark = self.coordinates[r_begin:r_end] / self.coord_max

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(seq_x_mark, dtype=torch.float32),
            torch.tensor(seq_y_mark, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.x_all) - self.seq_len - self.pred_len + 1

# ------------------ MAIN ------------------

if __name__ == '__main__':
    chroms = ['22', '9', '17', '7', '13', '20', '8', '15', '19', '18',
              '5', '14', '3', '10', '21', '1', '12', '2', '11', '4', '16', '6']

    dataset = RepliSeqFeatureLabelWithCoordinates(
        x_path='data/H1_features.npz',
        y_path='data/H1_labels.npz',
        bedgraph_path='data/H1_coordinates.bedgraph',
        chromosomes=chroms,
        size=(96, 48, 48),
        scale=True
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for seq_x, seq_y, seq_x_mark, seq_y_mark in loader:
        print("seq_x:", seq_x.shape)        # e.g., [32, 96, 9]
        print("seq_y:", seq_y.shape)        # e.g., [32, 96, 1 or 9 or whatever]
        # What should we set as prediction length.
        print("seq_x_mark:", seq_x_mark.shape)  # e.g., [32, 96, 1]
        print("seq_y_mark:", seq_y_mark.shape)  # e.g., [32, 96, 1]
        break
