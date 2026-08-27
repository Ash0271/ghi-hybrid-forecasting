"""
Shared evaluation metrics used across all hybridization techniques below.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score


def evaluate(y_true, y_pred):
    """Return RMSE, MAPE, MAE, and R2 for a set of predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAPE": mape, "MAE": mae, "R2": r2}
