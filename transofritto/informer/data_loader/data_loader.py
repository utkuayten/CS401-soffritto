import os, sys
# assume this file lives two levels under your project root,
# adjust the number of '..' if needed
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.insert(0, PROJECT_ROOT)
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
from informer.utils.timefeatures import genomic_features,time_features
from informer.utils.tools import StandardScaler

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path, train_chroms, val_chroms , flag='train', size=None,
                 features='MS', data_path='ETTh1.csv',
                 target='target_1', scale=True, inverse=False, timeenc=0, freq='h', selected_cols = None,
                 ):

        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 48
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.train_chroms = train_chroms
        self.val_chroms = val_chroms
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.selected_cols = selected_cols
        print(selected_cols)
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        mask_train = df_raw['chrom'].isin(self.train_chroms)
        if len(self.val_chroms)>0:
            mask_val = df_raw['chrom'].isin(self.val_chroms)
        else:
            mask_val = pd.Series(False, index=df_raw.index)
        mask_test  = ~(mask_train | mask_val)

        masks = [mask_train, mask_val, mask_test]
        mask  = masks[self.set_type]   # 0=train, 1=val, 2=test

        # 3) Normalize 'date' using train‐only range
        df_raw['date'] = df_raw['date'].astype(np.int64)
        date_min = df_raw.loc[mask_train, 'date'].min()
        date_max = df_raw.loc[mask_train, 'date'].max()
        df_raw['date'] = (df_raw['date'] - date_min) / (date_max - date_min) - 0.5

        # 4) Normalize 'chrom' into [-0.5, +0.5]
        df_raw['chrom'] = ((df_raw['chrom'] - 1) / 23) - 0.5

        all_cols = list(df_raw.columns)

        target_start_idx = all_cols.index(self.target)
        target_cols = all_cols[target_start_idx:]


        input_cols = [
            c for c in all_cols
            if c not in target_cols + ['date','chrom']
        ]
  
        train_cols = list(set(self.selected_cols) & set(input_cols))

        if not train_cols:
          raise ValueError(
              f"No matching columns found!\n"
              f"Selected: {self.selected_cols}\n"
              f"Available: {input_cols}"
          )
        
        # 5) Build time‐stamp features
        df_stamp = df_raw[['chrom','date']][mask]
        self.data_stamp = genomic_features(df_stamp)

        df_data   = df_raw[train_cols]
        df_target = df_raw[target_cols]

        # 7) Fit scaler on train, then transform all
        if self.scale:
            self.scaler.fit(df_data.loc[mask_train].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 8) Finally slice X and y for this split
        self.data_x = data[mask]
        self.data_y = df_target.values[mask]


    def __getitem__(self, index):
      
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)