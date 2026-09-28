"""
Standardized evaluation metrics for BOF endpoint prediction.

Industrial quality thresholds (see TOLERANCE below, the values this module
actually enforces): T_final MAE <= 10 deg C [zhou2024bof], C_final MAE <=
0.020 wt% [bae2020bof], P_final MAE <= 0.003 wt% [bae2020bof].
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Industrial quality thresholds [zhou2024bof, bae2020bof]
# T: ±10°C (zhou2024bof Table 9 hit rate criterion)
# C: ±0.020 wt% (bae2020bof evaluation threshold)
# P: ±0.003 wt% (bae2020bof evaluation threshold)
TOLERANCE = {
    'T_final': 10.0,    # °C
    'C_final': 0.020,   # wt%
    'P_final': 0.003,   # wt%
}


def compute_metrics(y_true, y_pred, target):
    """
    Compute MAE, RMSE, R2 and within_tolerance fraction.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    target : str - one of 'T_final', 'C_final', 'P_final'

    Returns
    -------
    dict with keys: MAE, RMSE, R2, within_tolerance
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    tol = TOLERANCE.get(target, np.inf)
    within_tol = float(np.mean(np.abs(y_true - y_pred) <= tol))

    return {
        'MAE':              round(mae, 6),
        'RMSE':             round(rmse, 6),
        'R2':               round(r2, 6),
        'within_tolerance': round(within_tol, 4),
    }
