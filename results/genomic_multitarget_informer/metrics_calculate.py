import numpy as np
from scipy.stats import spearmanr
# If you prefer to use scipy's wasserstein_distance, you can, but here we compute an L1 distance on CDFs.

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

# Example usage:
if __name__ == "__main__":
    # Load your predictions and true labels (adjust paths as needed)
    y_true = np.load("results/genomic_multitarget_informer/true.npy")
    y_pred = np.load("results/genomic_multitarget_informer/pred.npy")

    metrics = evaluate_all_metrics(y_true, y_pred)