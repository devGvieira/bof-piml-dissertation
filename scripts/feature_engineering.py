"""
Script 2: Feature engineering — compute 8 physics-derived features.

Prepares two datasets:
- data/features_raw.csv    : raw input variables + targets only
- data/features_physics.csv: raw + 8 physics-derived features + targets

Physics features based on:
- chattopadhyay2022hybrid, shao2024hybrid (O2 and enthalpy features)
- willard2020integrating (domain-derived feature rationale)
- bae2020bof (stoichiometric basis)

Preprocessing:
- StandardScaler fitted on training split to avoid data leakage
- Scalers saved for use during inference/evaluation
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from scripts.utils.physics import RAW_FEATURES, calc_physics_features

TARGETS = ['T_final', 'C_final', 'P_final']

PHYSICS_FEATURES = [
    'theo_o2_demand', 'o2_efficiency', 'enthalpy_surplus',
    'total_oxidized_mass', 'decarb_rate', 'slag_basicity',
    'scrap_melt_capacity', 'fe_oxidation_loss',
]


def add_physics_features(df):
    """Compute all 8 physics-derived features for every row."""
    physics_rows = []
    for _, row in df.iterrows():
        physics_rows.append(calc_physics_features(row.to_dict()))
    phys_df = pd.DataFrame(physics_rows, index=df.index)
    return pd.concat([df, phys_df], axis=1)


def fit_and_save_scaler(X_train, feature_names, scaler_path):
    scaler = StandardScaler()
    scaler.fit(X_train[feature_names])
    joblib.dump(scaler, scaler_path)
    return scaler


def main():
    src = 'data/bof_synthetic.csv'
    if not os.path.exists(src):
        print(f"ERROR: {src} not found. Run generate_bof_data.py first.")
        sys.exit(1)

    os.makedirs('models', exist_ok=True)

    df = pd.read_csv(src)
    print(f"Loaded {len(df)} rows from {src}")

    # Train/val/test split indices (70/15/15) [kohavi1995study]
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_train = int(0.70 * len(df))
    n_val   = int(0.15 * len(df))
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    df['split'] = 'train'
    df.loc[val_idx, 'split']  = 'val'
    df.loc[test_idx, 'split'] = 'test'

    # === Dataset A/B: raw features only ===
    df_raw = df[RAW_FEATURES + TARGETS + ['split']].copy()
    df_raw.to_csv('data/features_raw.csv', index=False)
    print(f"Saved data/features_raw.csv ({len(RAW_FEATURES)} input features)")

    # Fit StandardScaler on train split only
    X_train_raw = df_raw[df_raw['split'] == 'train']
    fit_and_save_scaler(X_train_raw, RAW_FEATURES, 'models/scaler_raw.joblib')

    # === Dataset C: raw + physics features ===
    df_phys = add_physics_features(df[RAW_FEATURES + TARGETS + ['split']].copy())

    # Check for NaN or Inf introduced by physics calculations
    all_features = RAW_FEATURES + PHYSICS_FEATURES
    n_inf = np.isinf(df_phys[all_features]).sum().sum()
    n_nan = df_phys[all_features].isnull().sum().sum()
    if n_inf > 0 or n_nan > 0:
        print(f"  Replacing {n_inf} Inf and {n_nan} NaN values in physics features")
        df_phys[all_features] = df_phys[all_features].replace(
            [np.inf, -np.inf], np.nan
        )
        for col in all_features:
            df_phys[col] = df_phys[col].fillna(df_phys[col].median())

    df_phys = df_phys[all_features + TARGETS + ['split']]
    df_phys.to_csv('data/features_physics.csv', index=False)
    print(f"Saved data/features_physics.csv ({len(all_features)} input features)")

    X_train_phys = df_phys[df_phys['split'] == 'train']
    fit_and_save_scaler(X_train_phys, all_features, 'models/scaler_physics.joblib')

    print("\nFeature statistics (physics features, train split):")
    print(X_train_phys[PHYSICS_FEATURES].describe().round(4))

    print(f"\nSplit sizes: train={n_train}, val={n_val}, test={len(test_idx)}")


if __name__ == '__main__':
    main()
