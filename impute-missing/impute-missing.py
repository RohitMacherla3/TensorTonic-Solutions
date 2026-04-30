import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    X = np.array(X, dtype=float)
    is_1d = X.ndim == 1
    if is_1d:
        X = X.reshape(-1, 1)
    for col in range(X.shape[1]):
        column = X[:, col]
        mask = np.isnan(column)
        if mask.any():
            if mask.all():
                fill_value = 0.0
            elif strategy == 'mean':
                fill_value = np.nanmean(column)
            else:
                fill_value = np.nanmedian(column)
            column[mask] = fill_value
    return X.flatten() if is_1d else X