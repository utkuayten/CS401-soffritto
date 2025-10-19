import os, sys
# assume this file lives two levels under your project root,
# adjust the number of '..' if needed
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

# use your existing helpers for consistency with Informer/iTransformer
from utils.timefeatures import genomic_features, time_features
from utils.tools import StandardScaler

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    """
    PatchTST-ready dataset with chromosome-aware splits and column selection.

    Returns:
        seq_x:      [seq_len, C_in]  float32
        seq_y:      [label_len + pred_len, C_out] (usually your target columns)
        seq_x_mark: [seq_len, F_mark]  (genomic time features; optional for PatchTST)
        seq_y_mark: [label_len + pred_len, F_mark]
    """
    def __init__(
            self,
            root_path,
            train_chroms,
            val_chroms,
            flag='train',
            size=None,                 # [seq_len, label_len, pred_len]
            features='MS',
            data_path='ETTh1.csv',
            target='target_1',
            scale=True,
            inverse=False,
            timeenc=0,
            freq='h',
            selected_cols=None,
    ):
        # window sizes
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 48
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        # config
        self.train_chroms = list(train_chroms)
        self.val_chroms = list(val_chroms) if val_chroms is not None else []
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse      # kept for API compatibility
        self.timeenc = timeenc
        self.freq = freq
        self.selected_cols = selected_cols if selected_cols is not None else []
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    # ---------------- internal helpers ----------------
    @staticmethod
    def _normalize_date_and_chrom(df, mask_train):
        """
        Normalize 'date' into [-0.5, 0.5] using train range,
        and map 'chrom' into [-0.5, 0.5] with human autosomes 1..23 assumption.
        """
        # ensure integer-like date
        df = df.copy()
        # handle both int and datetime; if datetime, convert to int64
        if np.issubdtype(df['date'].dtype, np.datetime64):
            df['date'] = df['date'].astype('int64')
        else:
            # force numeric; if already numeric it's a no-op
            df['date'] = pd.to_numeric(df['date'], errors='coerce').fillna(0).astype(np.int64)

        # train-only min/max
        date_min = df.loc[mask_train, 'date'].min()
        date_max = df.loc[mask_train, 'date'].max()
        # guard against zero division if degenerate range
        if pd.isna(date_min) or pd.isna(date_max) or date_max == date_min:
            df['date'] = 0.0
        else:
            df['date'] = (df['date'] - date_min) / (date_max - date_min) - 0.5

        # chrom → [-0.5, 0.5]; assume values in 1..23
        # if your data already numeric 1..23 it's fine; otherwise map if needed.
        df['chrom'] = pd.to_numeric(df['chrom'], errors='coerce').fillna(1).astype(np.int64)
        df['chrom'] = ((df['chrom'] - 1) / 23.0) - 0.5
        return df

    # ---------------------------------------------------
    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # masks by chromosome
        mask_train = df_raw['chrom'].isin(self.train_chroms)
        if len(self.val_chroms) > 0:
            mask_val = df_raw['chrom'].isin(self.val_chroms)
        else:
            mask_val = pd.Series(False, index=df_raw.index)
        mask_test = ~(mask_train | mask_val)

        split_masks = [mask_train, mask_val, mask_test]
        mask = split_masks[self.set_type]

        # normalize date & chrom using *train* range
        df_raw = self._normalize_date_and_chrom(df_raw, mask_train)

        # column partitioning
        all_cols = list(df_raw.columns)
        if self.target not in all_cols:
            raise ValueError(f"Target column '{self.target}' not found in CSV columns: {all_cols}")

        target_start_idx = all_cols.index(self.target)
        target_cols = all_cols[target_start_idx:]  # everything from first target onward = outputs
        # inputs = everything except targets and the meta columns
        meta_cols = ['date', 'chrom']
        input_cols = [c for c in all_cols if c not in target_cols + meta_cols]

        # selected train columns (intersection), fallback to all inputs if selected_cols not provided
        if self.selected_cols:
            train_cols = sorted(set(self.selected_cols).intersection(input_cols))
            if not train_cols:
                raise ValueError(
                    "No matching columns between selected_cols and available inputs.\n"
                    f"selected_cols={self.selected_cols}\n"
                    f"available_inputs={input_cols}"
                )
        else:
            train_cols = input_cols  # use all input features by default

        # time/genomic marks for this split
        df_stamp = df_raw[['chrom', 'date']][mask].copy()
        # For PatchTST, marks are optional; we still provide them to keep interface parity.
        try:
            data_stamp = genomic_features(df_stamp)  # shape: [N, F_mark]
        except Exception:
            # fallback to generic time features if genomic_features not available
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values, unit='s', origin='unix'), freq=self.freq).T

        # build X (inputs) and y (targets)
        df_x_all = df_raw[train_cols]
        df_y_all = df_raw[target_cols]

        # fit scaler on train ONLY, then transform all inputs
        if self.scale:
            self.scaler.fit(df_x_all.loc[mask_train].values.astype(np.float32))
            data_x_all = self.scaler.transform(df_x_all.values).astype(np.float32)
        else:
            data_x_all = df_x_all.values.astype(np.float32)

        data_y_all = df_y_all.values.astype(np.float32)

        # slice to current split
        self.data_x = data_x_all[mask.values]
        self.data_y = data_y_all[mask.values]
        self.data_stamp = data_stamp.astype(np.float32)

        # basic sanity: ensure enough length for windows
        min_required = self.seq_len + self.pred_len
        if len(self.data_x) < min_required:
            raise ValueError(
                f"Not enough rows in split '{['train','val','test'][self.set_type]}' "
                f"for seq_len={self.seq_len}, pred_len={self.pred_len}. "
                f"Got {len(self.data_x)} rows; need ≥ {min_required}."
            )

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # encoder input (PatchTST uses this)
        seq_x = self.data_x[s_begin:s_end]                      # [seq_len, C_in]
        # target window (kept for API compatibility; PatchTST uses future segment)
        seq_y = self.data_y[r_begin:r_end]                      # [label_len+pred_len, C_out]

        # time/genomic marks (optional for PatchTST)
        seq_x_mark = self.data_stamp[s_begin:s_end]             # [seq_len, F_mark]
        seq_y_mark = self.data_stamp[r_begin:r_end]             # [label_len+pred_len, F_mark]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray):
        """
        Invert the StandardScaler on input-space tensors/arrays.
        Note: This inverts *inputs* (X) scaling, not target scaling.
        """
        return self.scaler.inverse_transform(data)