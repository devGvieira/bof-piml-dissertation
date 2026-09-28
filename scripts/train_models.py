"""
Script 3: Train all models across three experimental configurations.

Configurations [ghalati2023review, he2021automl, xia2024dmpinn]:
  Config A — raw features, full model set tuned via Optuna
  Config B — raw features, MLP deep architecture search via Optuna
  Config C — raw + 8 physics features, full model set + tuned MLP

Model set (all literature-justified):
  Ridge [hoerl1970ridge], Lasso [tibshirani1996regression],
  ElasticNet [zou2005regularization], Random Forest [brunton2020machine],
  Gradient Boosting, XGBoost, LightGBM, CatBoost [ghalati2023review],
  SVR [widodo2007svm], KNN [ding2011knnimpute], MLP [goodfellow2016deep]

Protocol [kohavi1995study, stone1974crossvalidation]:
  - 70/15/15 train/val/test (seed=42)
  - Optuna 20 trials, TPE sampler, MAE objective
  - k=3 cross-validation on train split

Output: results/metrics.csv, models/{config}_{model}_{target}.joblib
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from scripts.utils.metrics import compute_metrics, TOLERANCE
from scripts.utils.physics import RAW_FEATURES

TARGETS = ['T_final', 'C_final', 'P_final']

PHYSICS_FEATURES = [
    'theo_o2_demand', 'o2_efficiency', 'enthalpy_surplus',
    'total_oxidized_mass', 'decarb_rate', 'slag_basicity',
    'scrap_melt_capacity', 'fe_oxidation_loss',
]

N_OPTUNA_TRIALS = 20   # 20 trials: sufficient convergence with N=10000; balances speed and quality
N_CV_FOLDS = 3         # 3-fold to reduce wall time; train set = 7000 samples


# --- Model factory ---

def get_base_models():
    """
    Return dict of (name -> model_class, default_params) for Configs A and C.
    All models literature-justified [ghalati2023review, he2021automl].
    """
    return {
        'ridge':    Ridge,
        'lasso':    Lasso,
        'elasticnet': ElasticNet,
        'rf':       RandomForestRegressor,
        'gbm':      GradientBoostingRegressor,
        'xgboost':  XGBRegressor,
        'lightgbm': LGBMRegressor,
        'catboost': CatBoostRegressor,
        'svr':      SVR,
        'knn':      KNeighborsRegressor,
        'mlp':      MLPRegressor,
    }


def get_optuna_params(trial, model_name):
    """Hyperparameter search spaces, bounded by literature [he2021automl]."""
    if model_name == 'ridge':
        return {'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True)}
    elif model_name == 'lasso':
        return {'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True)}
    elif model_name == 'elasticnet':
        return {
            'alpha':   trial.suggest_float('alpha', 1e-4, 10.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
        }
    elif model_name == 'rf':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth':    trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'n_jobs': 22,
        }
    elif model_name == 'gbm':
        return {
            'n_estimators':  trial.suggest_int('n_estimators', 50, 300),
            'max_depth':     trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample':     trial.suggest_float('subsample', 0.5, 1.0),
        }
    elif model_name == 'xgboost':
        return {
            'n_estimators':  trial.suggest_int('n_estimators', 50, 400),
            'max_depth':     trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample':     trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'device': 'cuda',
            'tree_method': 'hist',
        }
    elif model_name == 'lightgbm':
        return {
            'n_estimators':  trial.suggest_int('n_estimators', 50, 400),
            'max_depth':     trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'num_leaves':    trial.suggest_int('num_leaves', 20, 100),
            'n_jobs': 22,  # GPU build not available; use all CPU threads
        }
    elif model_name == 'catboost':
        return {
            'iterations':    trial.suggest_int('iterations', 50, 400),
            'depth':         trial.suggest_int('depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'task_type': 'GPU',
        }
    elif model_name == 'svr':
        return {
            'C':     trial.suggest_float('C', 0.1, 100.0, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'epsilon': trial.suggest_float('epsilon', 0.001, 0.5, log=True),
        }
    elif model_name == 'knn':
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
            'weights':     trial.suggest_categorical('weights', ['uniform', 'distance']),
            'n_jobs': 22,
        }
    elif model_name == 'mlp':
        n_layers = trial.suggest_int('n_layers', 1, 3)
        units    = trial.suggest_int('units', 32, 256)
        return {
            'hidden_layer_sizes': tuple([units] * n_layers),
            'alpha':              trial.suggest_float('alpha', 1e-5, 0.1, log=True),
            'learning_rate_init': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'max_iter': 500,
            'early_stopping': True,
        }
    return {}


def get_deep_mlp_params(trial):
    """
    Config B: deep MLP architecture search [goodfellow2016deep, kingma2015adam].
    n_layers: (1,4), n_units: (32,512), alpha (L2 regularization as dropout proxy)
    """
    n_layers = trial.suggest_int('n_layers', 1, 4)
    units    = trial.suggest_int('units', 32, 512)
    return {
        'hidden_layer_sizes': tuple([units] * n_layers),
        'alpha':              trial.suggest_float('alpha', 1e-5, 0.5, log=True),
        'learning_rate_init': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'max_iter': 1000,
        'early_stopping': True,
        'n_iter_no_change': 20,
    }


def tune_model(model_name, model_class, param_fn, X_train, y_train, n_trials):
    """Run Optuna tuning, return best model [he2021automl]."""
    # All models use n_jobs=1 for CV folds — joblib process-spawn overhead in WSL
    # is large enough to dominate for fast models (Ridge, MLP, etc.).
    # Parallelism is handled at the model level: RF/KNN/LightGBM via n_jobs=22,
    # XGBoost/CatBoost via GPU.
    cv_n_jobs = 1

    def objective(trial):
        params = param_fn(trial)
        if model_name == 'catboost':
            params['verbose'] = False
        if model_name in ('xgboost',):
            params['verbosity'] = 0
        model = model_class(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=N_CV_FOLDS,
            scoring='neg_mean_absolute_error',
            n_jobs=cv_n_jobs,
        )
        return -scores.mean()

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Reconstruct full param dict via param_fn so fixed params (device='cuda',
    # n_jobs=22, etc.) are included — study.best_params only contains suggest_* keys.
    best_params = param_fn(study.best_trial)
    if 'lr' in best_params:
        best_params['learning_rate_init'] = best_params.pop('lr')
    if model_name == 'catboost':
        best_params['verbose'] = False
    if model_name == 'xgboost':
        best_params['verbosity'] = 0

    best_model = model_class(**best_params)
    return best_model, study.best_value


def train_and_evaluate(config_name, feature_cols, df, model_name, model_class,
                       param_fn, n_trials=N_OPTUNA_TRIALS):
    """Train + evaluate one model for all 3 targets, return metrics rows."""
    rows = []

    for target in TARGETS:
        X_tr = df[df['split'] == 'train'][feature_cols].values
        y_tr = df[df['split'] == 'train'][target].values
        X_te = df[df['split'] == 'test'][feature_cols].values
        y_te = df[df['split'] == 'test'][target].values

        # Scale features
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        print(f"  [{config_name}] {model_name} / {target} — tuning {n_trials} trials...")
        best_model, cv_mae = tune_model(
            model_name, model_class, param_fn, X_tr_s, y_tr, n_trials
        )

        best_model.fit(X_tr_s, y_tr)
        y_pred = best_model.predict(X_te_s)

        m = compute_metrics(y_te, y_pred, target)

        # CV std on best model (one extra k-fold pass for variance reporting)
        cv_scores = cross_val_score(
            best_model, X_tr_s, y_tr,
            cv=N_CV_FOLDS,
            scoring='neg_mean_absolute_error',
            n_jobs=1,
        )
        cv_mae_std = float(cv_scores.std())

        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = f'models/{config_name}_{model_name}_{target}.joblib'
        joblib.dump({'model': best_model, 'scaler': scaler}, model_path)

        rows.append({
            'config': config_name,
            'model':  model_name,
            'target': target,
            'MAE':    m['MAE'],
            'RMSE':   m['RMSE'],
            'R2':     m['R2'],
            'within_tolerance': m['within_tolerance'],
            'cv_mae':     round(cv_mae, 6),
            'cv_mae_std': round(cv_mae_std, 6),
        })
        print(f"    MAE={m['MAE']:.4f}, R2={m['R2']:.4f}, "
              f"within_tol={m['within_tolerance']:.2%}")

    return rows


def main():
    os.makedirs('results', exist_ok=True)

    raw_path  = 'data/features_raw.csv'
    phys_path = 'data/features_physics.csv'

    if not os.path.exists(raw_path) or not os.path.exists(phys_path):
        print("ERROR: Feature files not found. Run feature_engineering.py first.")
        sys.exit(1)

    df_raw  = pd.read_csv(raw_path)
    df_phys = pd.read_csv(phys_path)

    all_results = []
    models = get_base_models()

    # =======================================================================
    # Config A: raw features, full model set [he2021automl, ghalati2023review]
    # =======================================================================
    print("\n=== Config A: raw features, full model set ===")
    for model_name, model_class in models.items():
        rows = train_and_evaluate(
            'A_raw', RAW_FEATURES, df_raw,
            model_name, model_class,
            lambda trial, mn=model_name: get_optuna_params(trial, mn),
        )
        all_results.extend(rows)

    # =======================================================================
    # Config B: raw features, deep MLP search [goodfellow2016deep, kingma2015adam]
    # =======================================================================
    print("\n=== Config B: raw features, deep MLP architecture search ===")
    rows = train_and_evaluate(
        'B_deep', RAW_FEATURES, df_raw,
        'mlp_deep', MLPRegressor,
        get_deep_mlp_params,
        n_trials=N_OPTUNA_TRIALS,
    )
    all_results.extend(rows)

    # =======================================================================
    # Config C: physics features, full model set + tuned MLP
    # [xia2024dmpinn, willard2020integrating]
    # =======================================================================
    print("\n=== Config C: raw + physics features, full model set ===")
    phys_all_features = RAW_FEATURES + PHYSICS_FEATURES
    for model_name, model_class in models.items():
        rows = train_and_evaluate(
            'C_physics', phys_all_features, df_phys,
            model_name, model_class,
            lambda trial, mn=model_name: get_optuna_params(trial, mn),
        )
        all_results.extend(rows)

    # Deep MLP for Config C as well
    rows = train_and_evaluate(
        'C_physics', phys_all_features, df_phys,
        'mlp_deep', MLPRegressor,
        get_deep_mlp_params,
    )
    all_results.extend(rows)

    # Save metrics
    df_metrics = pd.DataFrame(all_results)
    df_metrics.to_csv('results/metrics.csv', index=False)
    print(f"\nSaved results/metrics.csv ({len(df_metrics)} rows)")

    # Quick summary
    print("\n=== Summary: within_tolerance by config ===")
    summary = df_metrics.groupby(['config', 'target'])['within_tolerance'].max()
    print(summary.to_string())


if __name__ == '__main__':
    main()
