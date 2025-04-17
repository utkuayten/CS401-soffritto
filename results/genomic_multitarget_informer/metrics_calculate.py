import numpy as np
from scipy.stats import spearmanr
# If you prefer to use scipy's wasserstein_distance, you can, but here we compute an L1 distance on CDFs.
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def compute_ARFE(y_true, y_pred):
    """
    Computes the Argmax RT Fraction Error (ARFE) for each bin.
    y_true and y_pred are assumed to be arrays of shape (bins, 16)
    """
    # Compute the argmax (index of highest probability) for each bin
    argmax_true = np.argmax(y_true, axis=1)
    argmax_pred = np.argmax(y_pred, axis=1)
    arfe = np.abs(argmax_true - argmax_pred)
    return arfe

def compute_KL_divergence(y_true, y_pred, epsilon=1e-9):
    # Clip values to avoid log(0)
    y_true_clipped = np.clip(y_true, epsilon, 1)
    y_pred_clipped = np.clip(y_pred, epsilon, 1)
    # Compute KL divergence per bin
    kl = np.sum(y_true_clipped * np.log(y_true_clipped / y_pred_clipped), axis=1)
    return kl

def compute_spearman(y_true, y_pred):
    """
    Computes Spearman correlation for each bin between observed and predicted 16-fraction vectors.
    Returns an array of correlation coefficients (one per bin).
    """
    correlations = []
    for i in range(y_true.shape[0]):
        # spearmanr returns a tuple (correlation, p-value)
        corr, _ = spearmanr(y_true[i, :], y_pred[i, :])
        correlations.append(corr)
    return np.array(correlations)

def compute_KS(y_true, y_pred):
    """
    Computes the Kolmogorov-Smirnov (KS) statistic for each bin.
    This is defined as the maximum absolute difference between the observed and predicted cumulative distributions.
    """
    # Compute cumulative distribution functions (CDFs)
    cdf_true = np.cumsum(y_true, axis=1)
    cdf_pred = np.cumsum(y_pred, axis=1)
    ks = np.max(np.abs(cdf_true - cdf_pred), axis=1)
    return ks

def compute_Wasserstein(y_true, y_pred):
    """
    Computes a simple Wasserstein distance (as an L1 distance on the CDFs) for each bin.
    """
    # Compute CDFs
    cdf_true = np.cumsum(y_true, axis=1)
    cdf_pred = np.cumsum(y_pred, axis=1)
    # Here we use the sum of absolute differences as an approximation
    wasserstein = np.sum(np.abs(cdf_true - cdf_pred), axis=1)
    return wasserstein

def evaluate_all_metrics(y_true, y_pred):
    """
    Expects y_true and y_pred to be numpy arrays of either shape:
        - (bins, 16), where each row is the probability distribution for a bin, OR
        - (samples, timesteps, 16). In this case, we reshape to combine samples and timesteps.
    Computes ARFE, KL divergence, Spearman correlation, KS statistic, and Wasserstein distance.
    Prints and returns the mean values for each metric.
    """
    # If data is 3D, reshape to (bins, 16)
    if y_true.ndim == 3:
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    # Reshape assuming y_true's length is a multiple of 5:
    #y_true_reshaped = y_true.reshape(-1, )
    # Aggregate along the second axis (choose an aggregation that makes sense, e.g., np.max):
    #y_true_aggregated = np.max(y_true_reshaped, axis=1)
    # Then compute the argmax:


    arfe = compute_ARFE(y_true, y_pred)
    kl = compute_KL_divergence(y_true, y_pred)
    spearman_corr = compute_spearman(y_true, y_pred)
    ks = compute_KS(y_true, y_pred)
    wasserstein = compute_Wasserstein(y_true, y_pred)

    print("Mean ARFE: {:.4f}".format(np.mean(arfe)))
    print("Mean KL divergence: {:.4f}".format(np.mean(kl)))
    print("Mean Spearman correlation: {:.4f}".format(np.mean(spearman_corr)))
    print("Mean KS statistic: {:.4f}".format(np.mean(ks)))
    print("Mean Wasserstein distance: {:.4f}".format(np.mean(wasserstein)))

    return {
        "ARFE": arfe,
        "Mean ARFE": np.mean(arfe),
        "KL": kl,
        "Mean KL": np.mean(kl),
        "Spearman": spearman_corr,
        "Mean Spearman": np.mean(spearman_corr),
        "KS": ks,
        "Mean KS": np.mean(ks),
        "Wasserstein": wasserstein,
        "Mean Wasserstein": np.mean(wasserstein)
    }
def compute_metrics(y_true, y_pred):
    """
    Computes metrics given 3D arrays (samples, timesteps, targets).
    We use a threshold for MAPE computation to avoid huge errors when y_true is near zero.
    """
    # Reshape to (samples * timesteps, targets)
    y_true_reshaped = y_true.reshape(-1, y_true.shape[-1])
    y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])

    # Standard epsilon to avoid division by zero
    epsilon = 1e-9

    # If many y_true values are close to zero, using them directly can blow up MAPE.
    # Here we use a threshold: if |y_true| < threshold, use threshold instead.
    threshold = 1e-3  # Adjust this based on the scale of your data
    denominator = np.where(np.abs(y_true_reshaped) < threshold, threshold, np.abs(y_true_reshaped))
    mape_per_target = np.mean(np.abs((y_true_reshaped - y_pred_reshaped) / denominator), axis=0) * 100
    mape_overall = np.mean(mape_per_target)

    mse = mean_squared_error(y_true_reshaped, y_pred_reshaped)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_reshaped, y_pred_reshaped)
    r2 = r2_score(y_true_reshaped, y_pred_reshaped)

    print("MAPE per target:", mape_per_target)
    print("Overall MAPE: {:.4f}".format(mape_overall))
    print("MSE: {:.4f}".format(mse))
    print("RMSE: {:.4f}".format(rmse))
    print("MAE: {:.4f}".format(mae))
    print("R2: {:.4f}".format(r2))

    return {
        "MAPE_per_target": mape_per_target,
        "MAPE_overall": mape_overall,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }
if __name__ == "__main__":
    # Load your predictions and true labels (adjust paths as needed)
    pred = np.load("/Users/ozgun/DataspellProjects/Soffritto/predictions/H1_chr9_pred_intra_cell_line.npy")
    true = np.load('/Users/ozgun/DataspellProjects/Soffritto/predictions/9.npy')

    # Convert numpy arrays to PyTorch tensors
    pred_tensor = torch.tensor(pred, dtype=torch.float32)
    true_tensor = torch.tensor(true, dtype=torch.float32)

    # Use torch.log to compute logarithm over the tensor
    log_pred_tensor = torch.log(pred_tensor)

    # Compute KL divergence using the tensor inputs with reduction 'batchmean'
    result = F.kl_div(log_pred_tensor, true_tensor, reduction='batchmean', log_target=False)

    # Evaluate other metrics (make sure evaluate_all_metrics is updated to handle tensors if needed)
    metrics = evaluate_all_metrics(true, pred)
    print(result)