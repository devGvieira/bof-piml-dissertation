"""Quick check: GPU availability for XGBoost, LightGBM, CatBoost."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

rng = np.random.default_rng(0)
X = rng.standard_normal((50, 5))
y = rng.standard_normal(50)

# --- XGBoost ---
try:
    from xgboost import XGBRegressor
    import xgboost as xgb
    print(f"XGBoost {xgb.__version__}")
    m = XGBRegressor(device='cuda', tree_method='hist', n_estimators=10, verbosity=0)
    m.fit(X, y)
    print("  CUDA: OK")
except Exception as e:
    print(f"  CUDA error: {e}")

# --- LightGBM ---
try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb
    print(f"LightGBM {lgb.__version__}")
    m = LGBMRegressor(device='gpu', n_estimators=10, verbose=-1)
    m.fit(X, y)
    print("  GPU: OK")
except Exception as e:
    print(f"  GPU error: {e}")

# --- CatBoost ---
try:
    from catboost import CatBoostRegressor
    import catboost
    print(f"CatBoost {catboost.__version__}")
    m = CatBoostRegressor(task_type='GPU', iterations=10, verbose=False)
    m.fit(X, y)
    print("  GPU: OK")
except Exception as e:
    print(f"  GPU error: {e}")

# --- Train_models import ---
from scripts.train_models import get_optuna_params
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study()
trial = study.ask()
for name in ['rf', 'xgboost', 'lightgbm', 'catboost', 'knn']:
    p = get_optuna_params(trial, name)
    print(f"  {name}: {list(p.keys())}")

print("Import OK")