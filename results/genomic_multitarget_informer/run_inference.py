import torch
#from torch.utils.data import DataLoader
# from trained_models.genomic_multitarget_informer.data.data_loader import GenomicDataLoader  # Example import
from predict_evaluate import InferenceModel
import sys
import os
import numpy as np

def main():
    # 1) Create the inference model from your checkpoint
    model_inference = InferenceModel(
        checkpoint_path="checkpoints/genomic_multitarget_informer/checkpoint.pth"
    )

    # 2) Option A: Evaluate directly from saved .npy files (pred & true)
    #    (This is if you already have pred.npy and true.npy)
    #print("Evaluating from existing pred.npy and true.npy ...")
    #model_inference.evaluate_from_files(
    #    true_file_path="./results/genomic_multitarget_informer/true.npy",
    #    pred_file_path="./results/genomic_multitarget_informer/pred.npy"
    #)

    # 2) Option B: Run actual inference again and compute metrics
    #    (Uncomment if you want to re-run predictions.)
    #
    data_loader = DataLoader(
     GenomicDataLoader(...),
     batch_size=32,
     shuffle=False
    )
    #
    # # Predict
     predictions = model_inference.predict(data_loader)
    # # Suppose you have the corresponding ground truth as an array or from the data loader
     ground_truth = np.load()
    #
    # # Evaluate
    # model_inference.evaluate(ground_truth, predictions)

if __name__ == "__main__":
    main()