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
    def __init__(
            self,
            root_path,
            train_chroms,
            test_chroms,
            val_chroms,
            flag='train',
            size=None,
            features='MS',
            data_path='ETTh1.csv',
            target='target_1',
            scale=True,
            inverse=False,
            timeenc=0,
            freq='h',
            selected_cols=None,
            # -------- NEW: separate LSTM "from scratch" pipeline --------
            lstm_selected_cols=None,
            lstm_scale=True,
            # -------- optional: rt2 feature name override --------
            rt2_col_name="2-stage",
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
        self.test_chroms = test_chroms

        self.features = features
        self.target = target

        # Informer pipeline
        self.scale = scale
        self.inverse = inverse

        # LSTM pipeline (independent scaler + independent feature selection)
        self.lstm_scale = lstm_scale

        self.timeenc = timeenc
        self.freq = freq

        self.selected_cols = selected_cols
        self.lstm_selected_cols = lstm_selected_cols

        self.root_path = root_path
        self.data_path = data_path

        self.rt2_col_name = rt2_col_name

        self.__read_data__()

    def __read_data__(self):
        # Two independent scalers
        self.scaler = StandardScaler()
        self.lstm_scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        mask_train = df_raw['chrom'].isin(self.train_chroms)
        if len(self.val_chroms) > 0:
            mask_val = df_raw['chrom'].isin(self.val_chroms)
        else:
            mask_val = pd.Series(False, index=df_raw.index)
        mask_test = df_raw['chrom'].isin(self.test_chroms)

        masks = [mask_train, mask_val, mask_test]
        mask = masks[self.set_type]  # 0=train, 1=val, 2=test

        # Normalize 'date' using train‐only range (kept for marks)
        df_raw['date'] = df_raw['date'].astype(np.int64)
        date_min = df_raw.loc[mask_train, 'date'].min()
        date_max = df_raw.loc[mask_train, 'date'].max()
        df_raw['date'] = (df_raw['date'] - date_min) / (date_max - date_min) - 0.5

        # Normalize 'chrom' into [-0.5, +0.5] (kept for marks)
        df_raw['chrom'] = ((df_raw['chrom'] - 1) / 23) - 0.5

        all_cols = list(df_raw.columns)

        target_start_idx = all_cols.index(self.target)
        target_cols = all_cols[target_start_idx:]

        # Input columns: everything except targets + date/chrom
        input_cols = [
            c for c in all_cols
            if c not in target_cols + ['date', 'chrom']
        ]
        self.input_cols = input_cols

        # rt2 index inside *input feature space* (used by cost-aware-2 in Informer)
        if self.rt2_col_name not in self.input_cols:
            raise ValueError(
                f"rt2_col_name={self.rt2_col_name!r} not found in input_cols. "
                f"Available input cols: {self.input_cols}"
            )
        self.rt2_idx = self.input_cols.index(self.rt2_col_name)

        # ---- Informer feature selection ----
        if self.selected_cols is None:
            # default: all input features
            informer_cols = list(input_cols)
        else:
            informer_cols = list(set(self.selected_cols) & set(input_cols))
            if not informer_cols:
                raise ValueError(
                    f"No matching Informer columns found!\n"
                    f"selected_cols={self.selected_cols}\n"
                    f"available input_cols={input_cols}"
                )

        # ---- LSTM feature selection (independent) ----
        if self.lstm_selected_cols is None:
            # By default, mirror Informer feature selection, but with its own scaling
            lstm_cols = list(informer_cols)
        else:
            lstm_cols = list(set(self.lstm_selected_cols) & set(input_cols))
            if not lstm_cols:
                raise ValueError(
                    f"No matching LSTM columns found!\n"
                    f"lstm_selected_cols={self.lstm_selected_cols}\n"
                    f"available input_cols={input_cols}"
                )

        self.informer_cols = informer_cols
        self.lstm_cols = lstm_cols

        # Time‐stamp features (marks) for Informer
        df_stamp = df_raw[['chrom', 'date']][mask]
        self.data_stamp = genomic_features(df_stamp)

        # Build X/y
        df_x_informer = df_raw[informer_cols]
        df_x_lstm = df_raw[lstm_cols]
        df_y = df_raw[target_cols]

        # Fit scalers on TRAIN split only, then transform entire series, then slice by split mask
        if self.scale:
            self.scaler.fit(df_x_informer.loc[mask_train].values)
            x_informer_all = self.scaler.transform(df_x_informer.values)
        else:
            x_informer_all = df_x_informer.values

        if self.lstm_scale:
            self.lstm_scaler.fit(df_x_lstm.loc[mask_train].values)
            x_lstm_all = self.lstm_scaler.transform(df_x_lstm.values)
        else:
            x_lstm_all = df_x_lstm.values

        # Slice for this split
        self.data_x = x_informer_all[mask]
        self.data_x_lstm = x_lstm_all[mask]
        self.data_y = df_y.values[mask]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]                 # Informer inputs
        seq_x_lstm = self.data_x_lstm[s_begin:s_end]       # LSTM inputs (independent pipeline)
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # Return BOTH X streams
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_lstm

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)