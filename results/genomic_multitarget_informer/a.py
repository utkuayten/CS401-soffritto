import numpy as np

def MAPE(pred, true, epsilon=1e-10):
    denominator = np.where(np.abs(true) < epsilon, epsilon, true)
    return np.mean(np.abs((pred - true) / denominator))

def MSPE(pred, true, epsilon=1e-10):
    denominator = np.where(np.abs(true) < epsilon, epsilon, true)
    return np.mean(np.square((pred - true) / denominator))

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred)**2)) / np.sqrt(np.sum((true - true.mean())**2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0))**2 * (pred - pred.mean(0))**2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mae, mse, rmse, mape, mspe

# Example usage:
if __name__ == '__main__':
    # Example data where some true values are zero
    true = np.load("true.npy")
    pred = np.load("pred.npy")

    true = true.reshape(-1,16)
    pred = pred.reshape(-1,16)

    print(true.shape)
    print(pred.shape)
    print("MAPE:", MAPE(pred, true))
    print("MSPE:", MSPE(pred, true))

