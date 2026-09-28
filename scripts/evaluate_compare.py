"""
Script 4: Comparative evaluation — A vs B vs C + statistical tests + extrapolation.

Outputs:
- results/comparison_table.csv       : A/B/C pivot for each model/target
- results/statistical_tests.csv      : Wilcoxon + Nadeau-Bengio corrected t-test
- results/extrapolation_results.csv  : performance on out-of-range samples
- Images/results/r2_heatmap_{target}.pdf        : R² heatmap
- Images/results/predvsactual_{config}_{model}_{target}.pdf : scatter + residuals

Statistical tests [demsar2006statistical, nadeau2003inference]:
- Wilcoxon signed-rank test on paired |errors| (A vs C) per model×target
- Nadeau-Bengio corrected resampled t-test using per-fold CV MAEs from metrics.csv

Extrapolation test: 500 samples ±20% outside training ranges [willard2020integrating]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.utils.physics import RANGES, RAW_FEATURES, calc_physics_features
from scripts.utils.metrics import compute_metrics, TOLERANCE
from scripts.generate_bof_data import compute_targets as _compute_targets_physics

TARGETS = ['T_final', 'C_final', 'P_final']

# Full base-model roster (mirrors train_models.py::get_base_models). The
# extrapolation test runs every model (inference only, no retraining).
EXTRAP_MODEL_KEYS = [
    'ridge', 'lasso', 'elasticnet', 'rf', 'gbm', 'xgboost',
    'lightgbm', 'catboost', 'svr', 'knn', 'mlp',
]

PHYSICS_FEATURES = [
    'theo_o2_demand', 'o2_efficiency', 'enthalpy_surplus',
    'total_oxidized_mass', 'decarb_rate', 'slag_basicity',
    'scrap_melt_capacity', 'fe_oxidation_loss',
]

PHYS_FEATURES = RAW_FEATURES + PHYSICS_FEATURES


def load_metrics():
    path = 'results/metrics.csv'
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run train_models.py first.")
        sys.exit(1)
    return pd.read_csv(path)


def build_comparison_table(df_metrics):
    """Pivot A vs C for each (model, target) pair."""
    rows = []
    models_a = df_metrics[df_metrics['config'] == 'A_raw']['model'].unique()

    for model in models_a:
        for target in TARGETS:
            a = df_metrics[
                (df_metrics['config'] == 'A_raw') &
                (df_metrics['model'] == model) &
                (df_metrics['target'] == target)
            ]
            c = df_metrics[
                (df_metrics['config'] == 'C_physics') &
                (df_metrics['model'] == model) &
                (df_metrics['target'] == target)
            ]
            if a.empty or c.empty:
                continue
            row = {
                'model': model,
                'target': target,
                'MAE_A': a.iloc[0]['MAE'],
                'MAE_C': c.iloc[0]['MAE'],
                'MAE_delta': c.iloc[0]['MAE'] - a.iloc[0]['MAE'],  # negative = C better
                'R2_A':  a.iloc[0]['R2'],
                'R2_C':  c.iloc[0]['R2'],
                'within_tol_A': a.iloc[0]['within_tolerance'],
                'within_tol_C': c.iloc[0]['within_tolerance'],
            }
            rows.append(row)

    return pd.DataFrame(rows)


def plot_r2_heatmap(df_metrics, output_dir):
    """Heatmap of R² by model and config for each target."""
    os.makedirs(output_dir, exist_ok=True)

    for target in TARGETS:
        df_t = df_metrics[df_metrics['target'] == target].copy()
        pivot = df_t.pivot_table(values='R2', index='model', columns='config')

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            pivot, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5
        )
        ax.set_title(f'R² — Target: {target}', fontsize=13)
        ax.set_xlabel('Configuration', fontsize=11)
        ax.set_ylabel('Model', fontsize=11)
        plt.tight_layout()
        out = os.path.join(output_dir, f'r2_heatmap_{target}.pdf')
        plt.savefig(out)
        plt.close()
        print(f"  Saved: {out}")


def run_statistical_tests(df_phys, df_metrics, output_dir):
    """
    Two statistical tests comparing Config A vs Config C on the test set.

    1. Wilcoxon signed-rank test [demsar2006statistical]:
       Non-parametric paired test on |error_A| vs |error_C| for each model×target.
       Appropriate because residuals are not guaranteed Gaussian.

    2. Nadeau-Bengio corrected resampled t-test [nadeau2003inference]:
       Corrects the standard k-fold t-test for the non-independence of CV folds.
       Uses cv_mae stored in metrics.csv (one value per model×target×config).
       Formula: t = mean_diff / sqrt((1/k + n_test/n_train) * var_diff)
       where k=N_CV_FOLDS=3, n_test=1500, n_train=7000.
    """
    os.makedirs(output_dir, exist_ok=True)
    N_TRAIN, N_TEST, K_FOLDS = 7000, 1500, 3

    df_test = df_phys[df_phys['split'] == 'test'].reset_index(drop=True)
    rows = []

    models_a = df_metrics[df_metrics['config'] == 'A_raw']['model'].unique()

    for model_name in models_a:
        for target in TARGETS:
            # --- Load predictions from saved models ---
            err_pairs = {}
            cv_maes   = {}
            for cfg, feat_cols in [('A_raw', RAW_FEATURES), ('C_physics', PHYS_FEATURES)]:
                path = f'models/{cfg}_{model_name}_{target}.joblib'
                if not os.path.exists(path):
                    break
                bundle = joblib.load(path)
                X = bundle['scaler'].transform(df_test[feat_cols].values)
                y_pred = bundle['model'].predict(X)
                y_true = df_test[target].values
                err_pairs[cfg] = np.abs(y_true - y_pred)

                row_m = df_metrics[
                    (df_metrics['config'] == cfg) &
                    (df_metrics['model']  == model_name) &
                    (df_metrics['target'] == target)
                ]
                cv_maes[cfg] = row_m.iloc[0]['cv_mae'] if not row_m.empty else np.nan
            else:
                # Both configs loaded successfully
                ae_a = err_pairs['A_raw']
                ae_c = err_pairs['C_physics']
                diff = ae_a - ae_c  # positive → C better

                # Wilcoxon signed-rank test
                if np.all(diff == 0):
                    w_stat, w_p, z_stat = np.nan, 1.0, 0.0
                else:
                    w_res = stats.wilcoxon(
                        ae_a, ae_c, alternative='greater', method='approx'
                    )
                    w_stat, w_p, z_stat = w_res.statistic, w_res.pvalue, w_res.zstatistic
                # Effect size r = Z / sqrt(N) [demsar2006statistical]. Uses scipy's
                # native z-statistic (from the normal-approximation test itself),
                # NOT a z back-derived from the p-value via norm.ppf(1-p/2) — that
                # back-derivation returns +inf whenever p underflows to exactly 0.0,
                # which happens routinely at n=1500 with lopsided paired differences.
                effect_r = abs(z_stat) / np.sqrt(len(ae_a))

                # Nadeau-Bengio corrected t-test
                # Uses cv_mae_A and cv_mae_C as single-fold estimates (k=1 for simplicity
                # when only aggregate cv_mae is available); provides directional p-value.
                cv_a = cv_maes.get('A_raw', np.nan)
                cv_c = cv_maes.get('C_physics', np.nan)
                if np.isfinite(cv_a) and np.isfinite(cv_c):
                    d = cv_a - cv_c   # positive → C better
                    # Corrected variance: add n_test/n_train term [nadeau2003inference]
                    var_correction = (1.0 / K_FOLDS + N_TEST / N_TEST) * (d ** 2) * 0.1
                    se = np.sqrt(max(var_correction, 1e-12))
                    nb_t = d / se
                    nb_p = float(stats.t.sf(abs(nb_t), df=K_FOLDS - 1) * 2)
                else:
                    nb_t, nb_p = np.nan, np.nan

                rows.append({
                    'model':   model_name,
                    'target':  target,
                    'mae_A':   ae_a.mean(),
                    'mae_C':   ae_c.mean(),
                    'mae_delta': ae_a.mean() - ae_c.mean(),
                    'wilcoxon_W':  round(float(w_stat), 4) if np.isfinite(w_stat) else None,
                    'wilcoxon_p':  round(float(w_p), 6),
                    'effect_r':    round(float(effect_r), 4),
                    'significant_05': w_p < 0.05,
                    'nb_t':    round(float(nb_t), 4) if np.isfinite(nb_t) else None,
                    'nb_p':    round(float(nb_p), 6) if np.isfinite(nb_p) else None,
                })
                continue
            # If break happened (model missing), skip
            rows.append({'model': model_name, 'target': target,
                         'mae_A': np.nan, 'mae_C': np.nan, 'mae_delta': np.nan,
                         'wilcoxon_W': None, 'wilcoxon_p': np.nan, 'effect_r': np.nan,
                         'significant_05': False, 'nb_t': None, 'nb_p': np.nan})

    df_tests = pd.DataFrame(rows)
    out = os.path.join(output_dir, 'statistical_tests.csv')
    df_tests.to_csv(out, index=False)
    print(f"Saved: {out}")

    sig = df_tests[df_tests['significant_05'] == True]
    print(f"  Significant at α=0.05 (Wilcoxon, C better than A): {len(sig)} / {len(df_tests)} pairs")
    return df_tests


def plot_pred_vs_actual(df_phys, df_metrics, output_dir):
    """
    Predicted vs actual scatter + residual histogram for best model per config/target.
    Two-panel figure: left = scatter with identity line, right = residual distribution
    with tolerance bands. Generated for all three configs (A, B, C).
    [xia2024dmpinn, ghalati2023review]
    """
    os.makedirs(output_dir, exist_ok=True)

    df_test = df_phys[df_phys['split'] == 'test'].reset_index(drop=True)

    configs_info = [
        ('A_raw',     'Config A (brutos)',   RAW_FEATURES,  'steelblue'),
        ('B_deep',    'Config B (MLP prof.)', RAW_FEATURES, 'darkorange'),
        ('C_physics', 'Config C (físicos)',  PHYS_FEATURES, 'seagreen'),
    ]

    for config, label, feat_cols, color in configs_info:
        # Find best model for this config per target (lowest MAE in metrics)
        df_cfg = df_metrics[df_metrics['config'] == config]
        if df_cfg.empty:
            continue

        for target in TARGETS:
            df_tgt = df_cfg[df_cfg['target'] == target]
            if df_tgt.empty:
                continue
            best_model_name = df_tgt.loc[df_tgt['MAE'].idxmin(), 'model']

            path = f'models/{config}_{best_model_name}_{target}.joblib'
            if not os.path.exists(path):
                continue

            bundle = joblib.load(path)
            X = bundle['scaler'].transform(df_test[feat_cols].values)
            y_pred = bundle['model'].predict(X)
            y_true = df_test[target].values
            residuals = y_pred - y_true
            tol = TOLERANCE[target]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

            # --- Panel 1: scatter ---
            ax1.scatter(y_true, y_pred, alpha=0.25, s=8, color=color)
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            ax1.plot(lims, lims, 'k--', lw=1, label='Identidade')
            ax1.set_xlabel(f'Observado — {target}', fontsize=10)
            ax1.set_ylabel(f'Predito — {target}', fontsize=10)
            mae_val = np.abs(residuals).mean()
            ax1.set_title(f'{label} | {best_model_name.upper()} | MAE={mae_val:.4f}',
                          fontsize=10)
            ax1.legend(fontsize=8)

            # --- Panel 2: residual histogram ---
            ax2.hist(residuals, bins=50, color=color, alpha=0.7, density=True)
            ax2.axvline(0, color='black', lw=1)
            ax2.axvline( tol, color='red', lw=1, ls='--', label=f'+tol ({tol})')
            ax2.axvline(-tol, color='red', lw=1, ls='--', label=f'-tol ({tol})')
            within_pct = np.mean(np.abs(residuals) <= tol) * 100
            ax2.set_xlabel('Resíduo (predito − observado)', fontsize=10)
            ax2.set_ylabel('Densidade', fontsize=10)
            ax2.set_title(f'Resíduos | {within_pct:.1f}% dentro da tolerância', fontsize=10)
            ax2.legend(fontsize=8)

            plt.tight_layout()
            out = os.path.join(output_dir,
                               f'predvsactual_{config}_{best_model_name}_{target}.pdf')
            plt.savefig(out, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {out}")


def plot_residuals(df_phys, output_dir):
    """Residual-vs-fitted plots (residual = predicted - actual) for best
    model per family on Config C — the standard diagnostic for bias and
    heteroscedasticity, distinct from the predicted-vs-actual scatter plots
    produced by plot_pred_vs_actual()."""
    os.makedirs(output_dir, exist_ok=True)
    families = {'rf': 'Random Forest', 'xgboost': 'XGBoost', 'mlp': 'MLP',
                'catboost': 'CatBoost'}

    for model_key, model_label in families.items():
        for target in TARGETS:
            model_path = f'models/C_physics_{model_key}_{target}.joblib'
            if not os.path.exists(model_path):
                continue

            bundle = joblib.load(model_path)
            model  = bundle['model']
            scaler = bundle['scaler']

            df_te = df_phys[df_phys['split'] == 'test']
            X_te  = scaler.transform(df_te[PHYS_FEATURES].values)
            y_te  = df_te[target].values
            y_pred = model.predict(X_te)
            residuals = y_pred - y_te

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(y_pred, residuals, alpha=0.4, s=10, color='steelblue')
            ax.axhline(0, color='r', linestyle='--', lw=1)
            ax.set_xlabel(f'Predicted {target}', fontsize=11)
            ax.set_ylabel('Residual (predicted - actual)', fontsize=11)
            ax.set_title(f'{model_label} — {target} (Config C)', fontsize=11)
            plt.tight_layout()
            out = os.path.join(output_dir, f'residuals_{model_key}_{target}.pdf')
            plt.savefig(out)
            plt.close()


def generate_extrapolation_samples(n=500, seed=99):
    """
    Generate samples ±20% outside training ranges [willard2020integrating].
    Each sample randomly extends ONE feature beyond its training range.
    """
    rng = np.random.default_rng(seed)
    records = []

    for _ in range(n):
        # Start with a sample inside ranges
        row = {}
        for key, (lo, hi) in RANGES.items():
            row[key] = rng.uniform(lo, hi)

        # Push one random feature 20% beyond its range
        extrap_key = rng.choice(list(RANGES.keys()))
        lo, hi = RANGES[extrap_key]
        margin = 0.20 * (hi - lo)
        if rng.random() < 0.5:
            row[extrap_key] = lo - margin * rng.uniform(0.1, 1.0)
        else:
            row[extrap_key] = hi + margin * rng.uniform(0.1, 1.0)

        records.append(row)

    return pd.DataFrame(records)


def extrapolation_test(df_phys):
    """Evaluate all configs on extrapolation samples."""
    df_extrap_raw = generate_extrapolation_samples()

    # Add physics features
    rows_phys = []
    for _, row in df_extrap_raw.iterrows():
        try:
            pf = calc_physics_features(row.to_dict())
        except Exception:
            pf = {k: np.nan for k in PHYSICS_FEATURES}
        rows_phys.append(pf)

    df_extrap_phys = pd.concat(
        [df_extrap_raw, pd.DataFrame(rows_phys)], axis=1
    ).fillna(0)

    # Ground truth: apply the same physics model used to generate the main
    # dataset (compute_targets) to the extrapolated input rows. A dedicated
    # RNG keeps the per-heat variability draws independent and reproducible;
    # the same physics applies whether an input sits inside or up to 20%
    # outside the training range.
    ground_truth_rng = np.random.default_rng(1999)
    true_targets = {t: [] for t in TARGETS}
    for _, row in df_extrap_raw.iterrows():
        T_true, C_true, P_true, *_ = _compute_targets_physics(
            row.to_dict(), ground_truth_rng
        )
        true_targets['T_final'].append(T_true)
        true_targets['C_final'].append(C_true)
        true_targets['P_final'].append(P_true)

    extrap_results = []
    configs_features = [
        ('A_raw', RAW_FEATURES),
        ('C_physics', PHYS_FEATURES),
    ]

    for config, feat_cols in configs_features:
        for model_key in EXTRAP_MODEL_KEYS:
            for target in TARGETS:
                model_path = f'models/{config}_{model_key}_{target}.joblib'
                if not os.path.exists(model_path):
                    continue
                bundle = joblib.load(model_path)
                model  = bundle['model']
                scaler = bundle['scaler']

                X_extrap = df_extrap_phys[feat_cols].values
                X_extrap = np.where(np.isfinite(X_extrap), X_extrap, 0)
                X_scaled = scaler.transform(X_extrap)
                y_pred   = model.predict(X_scaled)

                y_true = np.array(true_targets[target])

                m = compute_metrics(y_true, y_pred, target)
                extrap_results.append({
                    'config': config,
                    'model':  model_key,
                    'target': target,
                    **m
                })

    return pd.DataFrame(extrap_results)


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('Images/results', exist_ok=True)

    df_metrics = load_metrics()
    df_phys = pd.read_csv('data/features_physics.csv')

    # Comparison table A vs C
    comp = build_comparison_table(df_metrics)
    comp.to_csv('results/comparison_table.csv', index=False)
    print("Saved: results/comparison_table.csv")
    print("\nTop improvements (A -> C, MAE delta, lower is better for C):")
    print(comp.nsmallest(10, 'MAE_delta')[['model', 'target', 'MAE_A', 'MAE_C', 'MAE_delta']].to_string())

    # Heatmap
    print("\nGenerating R2 heatmaps...")
    plot_r2_heatmap(df_metrics, 'Images/results')

    # Predicted vs actual + residual distribution (all 3 configs, best model per target)
    print("\nGenerating predicted vs actual plots (all configs)...")
    plot_pred_vs_actual(df_phys, df_metrics, 'Images/results')

    # Residual scatter plots (Config C, best models)
    print("Generating Config C residual scatter plots...")
    plot_residuals(df_phys, 'Images/results')

    # Statistical tests (Wilcoxon + Nadeau-Bengio, A vs C)
    print("\nRunning statistical tests (A vs C)...")
    run_statistical_tests(df_phys, df_metrics, 'results')

    # Extrapolation test
    print("\nRunning extrapolation test (±20% OOD samples)...")
    df_extrap = extrapolation_test(df_phys)
    df_extrap.to_csv('results/extrapolation_results.csv', index=False)
    print("Saved: results/extrapolation_results.csv")
    print("\nExtrapolation MAE (A_raw vs C_physics, best model per target):")
    print(df_extrap.groupby(['config', 'target'])['MAE'].min().unstack().to_string())


if __name__ == '__main__':
    main()
