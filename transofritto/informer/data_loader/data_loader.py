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
    """Genomic time-series dataset.

    Key fixes vs the previous version:
      1) Preserve selected_cols ORDER (set() was scrambling feature order).
      2) Provide an unscaled, fixed-order 9-feature tensor for the Soffritto LSTM teacher.
      3) Prevent windows from crossing chromosome boundaries (critical for correctness).
    """

    # Fixed feature order used by Soffritto NPZ training (verified by correlation):
    # [H3K27ac,H3K27me3,H3K36me3,H3K4me1,H3K4me3,H3K9me3,GC_content,gene_density,2-stage]
    SOFFRITTO_TEACHER_COLS = [
        "H3K27ac","H3K27me3","H3K36me3","H3K4me1","H3K4me3","H3K9me3",
        "GC_content","gene_density","2-stage"
    ]

    def __init__(self, root_path, train_chroms, test_chroms, val_chroms , flag='train', size=None,
                 features='MS', data_path='ETTh1.csv',
                 target='target_1', scale=True, inverse=False, timeenc=0, freq='h', selected_cols=None):
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
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.selected_cols = selected_cols or []
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Keep original chromosome id for boundary-aware indexing
        chrom_orig = df_raw['chrom'].astype(int).values

        mask_train = df_raw['chrom'].isin(self.train_chroms)
        if len(self.val_chroms)>0:
            mask_val = df_raw['chrom'].isin(self.val_chroms)
        else:
            mask_val = pd.Series(False, index=df_raw.index)
        mask_test  = df_raw['chrom'].isin(self.test_chroms)

        masks = [mask_train, mask_val, mask_test]
        mask  = masks[self.set_type]   # 0=train, 1=val, 2=test

        # ---- Normalize timestamp-like columns for Informer stamp features ----
        df_raw['date'] = df_raw['date'].astype(np.int64)
        date_min = df_raw.loc[mask_train, 'date'].min()
        date_max = df_raw.loc[mask_train, 'date'].max()
        # avoid div-by-zero in degenerate splits
        denom = (date_max - date_min) if (date_max - date_min) != 0 else 1
        df_raw['date'] = (df_raw['date'] - date_min) / denom - 0.5

        # Normalize 'chrom' into [-0.5, +0.5] for stamps (NOT used in X)
        df_raw['chrom'] = ((df_raw['chrom'].astype(float) - 1) / 23) - 0.5

        all_cols = list(df_raw.columns)

        # Targets are assumed contiguous from target_1 onward
        target_start_idx = all_cols.index(self.target)
        target_cols = all_cols[target_start_idx:]

        # Input columns: everything except target_* plus date/chrom
        input_cols = [c for c in all_cols if c not in target_cols + ['date','chrom']]
        self.input_cols = input_cols

        # ---- Fix 1: preserve order of selected cols ----
        train_cols = [c for c in self.selected_cols if c in input_cols]
        if not train_cols:
            raise ValueError(
                f"No matching columns found!\nSelected: {self.selected_cols}\nAvailable: {input_cols}"
            )
        self.train_cols = train_cols

        # index of 2-stage within the Informer input feature tensor (for cost-aware-2)
        self.rt2_col_name = "2-stage"
        self.rt2_idx = self.train_cols.index(self.rt2_col_name) if self.rt2_col_name in self.train_cols else None


        # ---- Fix 2: prepare unscaled Soffritto-teacher features (always 9 dims) ----
        missing_teacher = [c for c in self.SOFFRITTO_TEACHER_COLS if c not in df_raw.columns]
        if missing_teacher:
            raise ValueError(f"Missing required Soffritto teacher columns in CSV: {missing_teacher}")

        # stamp features for the CURRENT split
        df_stamp = df_raw[['chrom','date']][mask]
        self.data_stamp = genomic_features(df_stamp)

        # X for Informer (selected subset), scaled by train-only stats
        df_data = df_raw[self.train_cols]
        if self.scale:
            self.scaler.fit(df_data.loc[mask_train].values)
            data_x_all = self.scaler.transform(df_data.values)
        else:
            data_x_all = df_data.values

        # X for LSTM teacher (fixed 9 cols), UN-SCALED (matches NPZ training)
        data_teacher_all = df_raw[self.SOFFRITTO_TEACHER_COLS].values.astype(np.float32)

        # y targets for the split (probabilities, no scaling)
        df_target = df_raw[target_cols]
        data_y_all = df_target.values.astype(np.float32)

        # slice by split mask (keeps original genomic ordering)
        self.data_x = data_x_all[mask].astype(np.float32)
        self.data_x_teacher = data_teacher_all[mask]
        self.data_y = data_y_all[mask]

        # store chrom ids for boundary-aware indexing
        self.chrom_ids = chrom_orig[mask].astype(int)

        # ---- Fix 3: build valid window start indices per chromosome ----
        # We assume rows are ordered by (chrom, genomic position) in the CSV (true for your file).
        self.valid_starts = []
        n = len(self.chrom_ids)
        if n == 0:
            return

        # run-length encode contiguous chromosome segments
        start = 0
        while start < n:
            chrom = self.chrom_ids[start]
            end = start + 1
            while end < n and self.chrom_ids[end] == chrom:
                end += 1
            L = end - start
            max_start = L - self.seq_len - self.pred_len + 1
            if max_start > 0:
                self.valid_starts.extend([start + s for s in range(max_start)])
            start = end

    def __getitem__(self, index):
        s_begin = self.valid_starts[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # extra tensor for LSTM teacher (unscaled, fixed 9 dims)
        seq_x_teacher = self.data_x_teacher[s_begin:s_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_teacher

    def __len__(self):
        return len(self.valid_starts)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
