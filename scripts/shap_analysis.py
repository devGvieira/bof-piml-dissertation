"""
Script 5: SHAP interpretability analysis on Config C models.

Generates SHAP summary plots, beeswarm plots and dependence plots
for the top-performing models (RF, XGBoost, MLP) trained on Config C.

Expected physical coherence [manojlovic2022eaf, xia2024dmpinn]:
- theo_o2_demand    -> high importance for T_final, C_final
- slag_basicity     -> high importance for P_final
- decarb_rate       -> negative influence on C_final
- enthalpy_surplus  -> positive influence on T_final

Outputs:
  Images/shap/summary_{model}_{target}.pdf  — beeswarm summary plot
  Images/shap/dependence_{feature}_{target}.pdf — dependence plots
  results/shap_importance.csv              — mean |SHAP| by feature
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.utils.physics import RAW_FEATURES

TARGETS = ['T_final', 'C_final', 'P_final']

PHYSICS_FEATURES = [
    'theo_o2_demand', 'o2_efficiency', 'enthalpy_surplus',
    'total_oxidized_mass', 'decarb_rate', 'slag_basicity',
    'scrap_melt_capacity', 'fe_oxidation_loss',
]

PHYS_FEATURES = RAW_FEATURES + PHYSICS_FEATURES

# Key dependence plots to generate [manojlovic2022eaf, xia2024dmpinn]
DEPENDENCE_PAIRS = [
    ('theo_o2_demand',  'T_final'),
    ('theo_o2_demand',  'C_final'),
    ('slag_basicity',   'P_final'),
    ('decarb_rate',     'C_final'),
    ('enthalpy_surplus','T_final'),
]

N_BACKGROUND = 200  # for KernelExplainer background sample size


def load_model_and_data(model_key, target, df_phys):
    model_path = f'models/C_physics_{model_key}_{target}.joblib'
    if not os.path.exists(model_path):
        return None, None, None, None

    bundle = joblib.load(model_path)
    model  = bundle['model']
    scaler = bundle['scaler']

    df_te = df_phys[df_phys['split'] == 'test'].copy()
    X_te  = scaler.transform(df_te[PHYS_FEATURES].values)

    return model, scaler, X_te, df_te


def compute_shap_values(model, model_key, X_scaled, feature_names):
    """
    Select SHAP explainer based on model type:
    - RF, XGBoost: TreeExplainer (fast, exact) [manojlovic2022eaf]
    - MLP: KernelExplainer with 200-sample background
    """
    if model_key in ('rf', 'xgboost'):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_scaled)
    else:
        # KernelExplainer for sklearn MLP [goodfellow2016deep]
        rng = np.random.default_rng(42)
        background_idx = rng.choice(len(X_scaled),
                                     size=min(N_BACKGROUND, len(X_scaled)),
                                     replace=False)
        background = X_scaled[background_idx]
        explainer = shap.KernelExplainer(model.predict, background)
        shap_vals = explainer.shap_values(X_scaled[:200])  # subsample test
        X_scaled  = X_scaled[:200]

    return shap_vals, X_scaled


def plot_summary(shap_vals, X_scaled, feature_names, model_key, target, output_dir):
    """Beeswarm summary plot — top 10 features."""
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_vals, X_scaled,
        feature_names=feature_names,
        max_display=10,
        show=False,
        plot_type='dot',
    )
    plt.title(f'SHAP Summary — {model_key.upper()} / {target}', fontsize=12)
    plt.tight_layout()
    out = os.path.join(output_dir, f'summary_{model_key}_{target}.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_dependence(shap_vals, X_scaled, feature_names, feature, model_key,
                    target, output_dir):
    """Dependence plot for a single feature."""
    if feature not in feature_names:
        return
    feat_idx = list(feature_names).index(feature)

    fig, ax = plt.subplots(figsize=(6, 4))
    shap.dependence_plot(
        feat_idx, shap_vals, X_scaled,
        feature_names=feature_names,
        ax=ax, show=False,
    )
    ax.set_title(f'SHAP Dependence: {feature} → {target}', fontsize=11)
    plt.tight_layout()
    out = os.path.join(output_dir, f'dependence_{feature}_{target}.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def main():
    shap_dir = 'Images/shap'
    os.makedirs(shap_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    if not os.path.exists('data/features_physics.csv'):
        print("ERROR: data/features_physics.csv not found. Run feature_engineering.py first.")
        sys.exit(1)

    df_phys = pd.read_csv('data/features_physics.csv')

    importance_rows = []

    for model_key in ('rf', 'xgboost', 'mlp'):
        for target in TARGETS:
            print(f"\nSHAP: {model_key} / {target}")
            model, scaler, X_te, df_te = load_model_and_data(model_key, target, df_phys)

            if model is None:
                print(f"  Skipping — model file not found")
                continue

            shap_vals, X_used = compute_shap_values(
                model, model_key, X_te, PHYS_FEATURES
            )

            # Summary plot
            plot_summary(shap_vals, X_used, PHYS_FEATURES,
                         model_key, target, shap_dir)

            # Mean absolute SHAP importance (magnitude) + mean signed SHAP
            # (direction — needed for H4/H5, which are about sign, not magnitude)
            mean_abs = np.abs(shap_vals).mean(axis=0)
            mean_signed = shap_vals.mean(axis=0)
            for fname, imp, signed in zip(PHYS_FEATURES, mean_abs, mean_signed):
                importance_rows.append({
                    'model':   model_key,
                    'target':  target,
                    'feature': fname,
                    'mean_abs_shap': round(float(imp), 8),
                    'mean_signed_shap': round(float(signed), 8),
                })

            # Dependence plots for key physics features
            for feat, tgt in DEPENDENCE_PAIRS:
                if tgt == target:
                    plot_dependence(shap_vals, X_used, PHYS_FEATURES,
                                    feat, model_key, target, shap_dir)

    # Save importance table
    df_imp = pd.DataFrame(importance_rows)
    df_imp.to_csv('results/shap_importance.csv', index=False)
    print(f"\nSaved: results/shap_importance.csv")

    # Print top features per target
    print("\nTop 5 features by mean |SHAP| (averaged across models):")
    avg_imp = df_imp.groupby(['target', 'feature'])['mean_abs_shap'].mean()
    for target in TARGETS:
        top = avg_imp[target].nlargest(5)
        print(f"\n  {target}:")
        for feat, val in top.items():
            tag = ' ← physics' if feat not in RAW_FEATURES else ''
            print(f"    {feat}: {val:.6f}{tag}")

    # Physical coherence hypothesis tests
    test_shap_hypotheses(df_imp)


def test_shap_hypotheses(df_imp):
    """
    Test physical coherence hypotheses H1–H5 against SHAP results.

    H1: theo_o2_demand OR its normalized form o2_efficiency is top-3 importance
        for T_final (either counts as the O2-stoichiometric-demand signal)
    H2: theo_o2_demand OR o2_efficiency is top-3 importance for C_final
        (same OR criterion)
    H3: slag_basicity  is top-3 importance for P_final
    H4: decarb_rate SHAP direction is predominantly negative for C_final
        (higher initial C-load / blow-time → lower predicted C_final)
    H5: enthalpy_surplus SHAP direction is predominantly positive for T_final
        (higher heat input → higher predicted T_final)

    For H4/H5 we load the best available model's raw SHAP values from the
    importance table (sign information requires the raw values saved during
    summary plot generation — here we check the mean signed SHAP if available,
    otherwise flag as 'requires raw shap_vals').

    Saves results to: results/shap_hypothesis_tests.txt
    [manojlovic2022eaf, xia2024dmpinn, bae2020bof, chattopadhyay2022hybrid]
    """
    results = []
    avg_imp = df_imp.groupby(['target', 'feature'])['mean_abs_shap'].mean()

    def top_n_features(target, n=3):
        if target not in avg_imp:
            return []
        return list(avg_imp[target].nlargest(n).index)

    # H1
    top_T = top_n_features('T_final')
    h1 = ('theo_o2_demand' in top_T) or ('o2_efficiency' in top_T)
    results.append(('H1', h1,
                     f"theo_o2_demand or o2_efficiency in top-3 for T_final. "
                     f"Top-3: {top_T}"))

    # H2
    top_C = top_n_features('C_final')
    h2 = ('theo_o2_demand' in top_C) or ('o2_efficiency' in top_C)
    results.append(('H2', h2,
                     f"theo_o2_demand or o2_efficiency in top-3 for C_final. "
                     f"Top-3: {top_C}"))

    # H3
    top_P = top_n_features('P_final')
    h3 = 'slag_basicity' in top_P
    results.append(('H3', h3,
                     f"slag_basicity in top-3 for P_final. Top-3: {top_P}"))

    # H4 & H5: direction hypotheses, resolved from mean_signed_shap (averaged
    # across the rf/xgboost/mlp models tested here); mean_abs_shap loses the
    # sign, mean_signed_shap keeps it.
    avg_signed = df_imp.groupby(['target', 'feature'])['mean_signed_shap'].mean()

    def signed_value(feature, target):
        key = (target, feature)
        return float(avg_signed[key]) if key in avg_signed.index else None

    h4_val = signed_value('decarb_rate', 'C_final')
    h4 = (h4_val is not None) and (h4_val < 0)
    results.append(('H4', h4,
                     f"decarb_rate mean signed SHAP for C_final = "
                     f"{h4_val:.6f} (hypothesis: negative)"
                     if h4_val is not None else
                     "decarb_rate / C_final not found in importance table"))

    h5_val = signed_value('enthalpy_surplus', 'T_final')
    h5 = (h5_val is not None) and (h5_val > 0)
    results.append(('H5', h5,
                     f"enthalpy_surplus mean signed SHAP for T_final = "
                     f"{h5_val:.6f} (hypothesis: positive)"
                     if h5_val is not None else
                     "enthalpy_surplus / T_final not found in importance table"))

    # Print and save
    lines = ["=" * 60,
             "SHAP Physical Coherence Hypothesis Tests",
             "=" * 60]
    all_passed = True
    for hyp, result, description in results:
        if result is None:
            status = "MANUAL"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
            all_passed = False
        lines.append(f"\n{hyp}: [{status}]")
        lines.append(f"  {description}")

    lines.append("\n" + "=" * 60)
    if all_passed:
        lines.append("All testable hypotheses PASSED — model shows physical coherence.")
    else:
        lines.append("WARNING: one or more hypotheses FAILED — check for spurious correlations.")
    lines.append("=" * 60)

    output = "\n".join(lines)
    print(output)

    out_path = 'results/shap_hypothesis_tests.txt'
    with open(out_path, 'w') as f:
        f.write(output)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
