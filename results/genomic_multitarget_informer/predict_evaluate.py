import os, sys
# assume this file lives two levels under your project root,
# adjust the number of '..' if needed
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.insert(0, PROJECT_ROOT)
import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Bring your project root onto PYTHONPATH if needed
# sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

from transofritto.informer.exp.exp_informer import Exp_Informer
from transofritto.informer.data.data_loader import Dataset_Custom
from torch.utils.data import DataLoader
from metrics_calculate import evaluate_all_metrics, compute_metrics
class InferenceModel:
    def __init__(self, args, checkpoint_path: str):
        self.args   = args
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # 1) build model
        from transofritto.informer.exp.exp_informer import Exp_Informer
        exp = Exp_Informer(self.args)
        self.model = exp.model

        # 2) load weights
        sd = torch.load(checkpoint_path, map_location="cpu")
        # if you saved {"model_state_dict": ..., ...}
        if isinstance(sd, dict) and "model_state_dict" in sd:
            self.model.load_state_dict(sd["model_state_dict"])
        else:
            # assume you just saved model.state_dict()
            self.model.load_state_dict(sd)

        self.model.to(self.device).eval()

    def predict(self, flag: str = 'test') -> (np.ndarray, np.ndarray):
        """
        Run inference over the Dataset_Custom and return (preds, trues).
        preds/trues arrays will have shape [N_windows, pred_len, c_out].
        """
        # 3) Build the dataset & loader just as in training
        ds = Dataset_Custom(
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            inverse=self.args.inverse,
            freq=self.args.freq,
            cols=self.args.cols,
        )
        loader = DataLoader(
            ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

        all_preds = []
        all_trues = []
        with torch.no_grad():
            for bx, by, bxm, bym in loader:
                bx   = bx.to(self.args.device, dtype=torch.float32)
                bxm  = bxm.to(self.args.device, dtype=torch.float32)
                bym  = bym.to(self.args.device, dtype=torch.float32)

                # create decoder input of zeros, then warm up
                dec_inp = torch.zeros(
                    bx.size(0),
                    self.args.label_len + self.args.pred_len,
                    self.args.c_out,
                    device=self.args.device,
                    )
                dec_inp[:, : self.args.label_len, :] = by[:, : self.args.label_len, :].to(self.args.device, dtype=torch.float32)

                # forward through model
                if self.args.output_attention:
                    out = self.model(bx, bxm, dec_inp, bym)[0]
                else:
                    out = self.model(bx, bxm, dec_inp, bym)

                # out shape: [B, pred_len, c_out]
                all_preds.append(out.cpu().numpy())
                # ground truth last pred_len steps along feature dim
                true_slice = by[:, -self.args.pred_len :, :].numpy()
                all_trues.append(true_slice)

        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)
        return preds, trues

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        m1 = evaluate_all_metrics(y_true, y_pred)
        m2 = compute_metrics(y_true, y_pred)

        # 2a) Option A: merge via unpacking (Python 3.5+)
        merged = {**m1, **m2}

        # 2b) Option B: update one dict in place
        # merged = m1.copy()
        # merged.update(m2)

        return merged

    def evaluate_from_files(self, true_file_path: str, pred_file_path: str) -> dict:
        y_true = np.load(true_file_path)
        y_pred = np.load(pred_file_path)
        return self.evaluate(y_true, y_pred)