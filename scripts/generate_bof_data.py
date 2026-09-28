"""
Script 1: Generate synthetic BOF dataset via Monte Carlo sampling.

Methodology:
- Monte Carlo sampling within physical bounds [xia2024dmpinn, chattopadhyay2022hybrid, bae2020bof]
- Two-tier physics separation (academic robustness design):
    Tier 1 — Generator (this script): uses heat-specific physics constants drawn from
    distributions representing real heat-to-heat process variability (lance wear,
    refractory condition, O2 purity, slag viscosity). These constants are NOT observable
    from standard process inputs and are therefore NOT replicated in feature computation.
    Tier 2 — Feature computation (calc_physics_features): uses FIXED published mean
    constants (theoretical complete oxidation, no heat-specific efficiency).
    This structural gap prevents physics features from being algebraically equivalent
    to target generation equations. [chattopadhyay2022hybrid, logar2022eaf]
- Enthalpy balance + mass balance validation (threshold 10% / yield > 80%) [logar2022eaf]
- N = 10000 valid samples, seed = 42 for reproducibility [buggineni2024synthetic]
- Targets: T_final, C_final, P_final with calibrated Gaussian measurement noise [xia2024dmpinn]

Output: data/bof_synthetic.csv
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts.utils.physics import (
    RANGES, RAW_FEATURES,
    calc_Q_in, calc_Q_out_consistent, validate_enthalpy,
    calc_co2_fraction, calc_total_oxidized_mass,
)

# Target ranges (endpoint variables) [xia2024dmpinn]
TARGET_RANGES = {
    'T_final': (1620.0, 1700.0),  # final bath temperature (°C)
    'C_final': (0.02, 0.10),      # final carbon content (wt%)
    'P_final': (0.005, 0.025),    # final phosphorus content (wt%)
}

# Gaussian noise for targets (calibrated to represent real process uncertainty)
# Noise sigma from xia2024dmpinn (T: 5°C), ghalati2023review (C: 0.003, P: 0.001)
NOISE = {
    'T_final': 5.0,
    'C_final': 0.003,
    'P_final': 0.001,
}

N_TARGET = 10000
SEED = 42


def _uniform(rng, key, n):
    lo, hi = RANGES[key]
    return rng.uniform(lo, hi, n)


def compute_targets(row, rng):
    """
    Compute T_final, C_final, P_final via two-step physical model + process variability + noise.

    Design principle — two-tier physics separation [chattopadhyay2022hybrid, xia2024dmpinn]:
    The generator represents "what actually happened in this heat", including heat-to-heat
    variability in lance condition, refractory wear, and slag chemistry that are NOT
    observable at the process control level. Feature computation (calc_physics_features)
    uses FIXED published constants (literature means) — what an engineer computes from
    tables before the heat. This deliberate mismatch ensures that physics-derived features
    are informative but NOT algebraically equivalent to the target generation equations,
    preventing the trivial circularity that would arise if both used identical constants.

    Per-sample variability parameters and their calibration:
    - k_decarb:  CV=12% — lance position, nozzle wear, O2 purity variation
                 [chattopadhyay2022hybrid Table 2; typical ±10–15% heat-to-heat]
    - eff_heat:  CV=6%  — refractory condition, off-gas heat recovery variability
                 [madhavan2020heatbalance; Logar 2012 reports ±5–8% efficiency spread]
    - c_fr:      CV=4%  — carbon removal completeness (lance/bath mixing)
    - si_fr:     CV=3%  — silicon (near-complete; lower variability)
    - mn_fr:     CV=10% — manganese (partial removal; higher variability)
    - p_fr:      CV=5%  — phosphorus partition coefficient variability

    Measurement noise (additive, after physics model) represents sensor uncertainty:
    T: ±5°C, C: ±0.003 wt%, P: ±0.001 wt% [xia2024dmpinn].

    Returns
    -------
    tuple : (T_final, C_final, P_final, Q_in, eff_heat, Q_scrap, m_steel_hm)
        Q_in, eff_heat, Q_scrap and m_steel_hm are the exact quantities this
        function used to derive T_final — returned so the enthalpy validator
        (calc_Q_out_consistent) checks the post-noise/post-clip T_final
        against the same physics basis.
    """
    m_hm     = row['m_hm']
    T_hm     = row['T_hm']
    C_hm     = row['C_hm']
    Si_hm    = row['Si_hm']
    Mn_hm    = row['Mn_hm']
    P_hm     = row['P_hm']
    f_scrap  = row['f_scrap']
    CaO_kg_t = row['CaO_kg_t']
    MgO_kg_t = row['MgO_kg_t']
    C_scrap  = row['C_scrap']
    O2_flow  = row['O2_flow']
    t_blow   = row['t_blow']

    m_hm_kg    = m_hm * 1000.0
    m_scrap_kg = m_hm_kg * f_scrap
    # HM-only steel mass for heat balance denominator (scrap accounted via Q_scrap term)
    m_steel_hm = m_hm_kg * 0.92

    # O2 blown (Nm3 -> kg, density 1.43 kg/Nm3)
    O2_total_kg = O2_flow * t_blow * 1.43
    r_o2 = O2_total_kg / m_hm_kg

    # === Heat-to-heat physics variability ===
    # Each BOF heat has slightly different effective constants due to lance wear,
    # refractory age, O2 purity, and slag viscosity. These are NOT observable from
    # the standard process control inputs and are therefore NOT replicated in
    # calc_physics_features(), which uses fixed published mean values.
    # This gap is the key structural separation between generator and features.
    k_decarb = float(np.clip(rng.normal(78.0,  78.0 * 0.12), 40.0, 120.0))  # CV=12%
    eff_heat  = float(np.clip(rng.normal(0.64,  0.64 * 0.06),  0.50,  0.76))  # CV=6%
    c_fr  = float(np.clip(rng.normal(0.90, 0.04),  0.72, 0.98))               # CV≈4%
    si_fr = float(np.clip(rng.normal(0.92, 0.03),  0.80, 0.99))               # CV≈3%
    mn_fr = float(np.clip(rng.normal(0.50, 0.05),  0.35, 0.65))               # CV=10%
    p_fr  = float(np.clip(rng.normal(0.86, 0.04),  0.70, 0.96))               # CV≈5%

    # === Step 1: C_final — exponential decarburization with heat-specific k_decarb ===
    # Published mean K_DECARB=78 [chattopadhyay2022hybrid]; varies per heat.
    # Feature decarb_rate = C_hm/t_blow uses no decarburization constant at all.
    C_final_prelim = float(np.clip(
        C_hm * np.exp(-k_decarb * r_o2), *TARGET_RANGES['C_final']
    ))

    # === Step 2: CO/CO2 fraction from preliminary C_final [chattopadhyay2022hybrid] ===
    f_co2 = calc_co2_fraction(C_hm, C_final_prelim)

    # === T_final — enthalpy balance with heat-specific efficiency and removal fractions ===
    # Features use calc_Q_in with theoretical complete-oxidation fractions (1.0 each);
    # generator uses heat-specific partial fractions (c_fr, si_fr, mn_fr, p_fr).
    Q_in = calc_Q_in(
        m_hm, T_hm, C_hm, Si_hm, Mn_hm, P_hm,
        C_fraction_removed=c_fr,
        Si_fraction_removed=si_fr,
        Mn_fraction_removed=mn_fr,
        P_fraction_removed=p_fr,
        co2_fraction=f_co2,
        m_scrap_kg=m_scrap_kg,
        C_scrap=C_scrap,
    )
    Q_scrap = m_scrap_kg * 0.68 * (1500 - 25)
    # eff_heat is heat-specific; feature enthalpy_surplus omits efficiency (published tables
    # report gross heat release, not net available heat after furnace losses).
    Q_available = Q_in * eff_heat - Q_scrap
    T_final_phys = 25.0 + Q_available / (m_steel_hm * 0.75)
    T_final_phys = float(np.clip(T_final_phys, *TARGET_RANGES['T_final']))

    # === C_final model (use preliminary result — already clipped) ===
    C_final_phys = C_final_prelim

    # === P_final — basicity-driven removal with heat-specific partition efficiency ===
    # Slag basicity formula identical to feature slag_basicity; the difference is that
    # here p_fr introduces heat-specific variability in the basicity-to-removal mapping.
    SiO2 = m_hm_kg * (Si_hm / 100) * (60.085 / 28.085)
    CaO_total = CaO_kg_t * m_hm_kg / 1000.0
    MgO_total = MgO_kg_t * m_hm_kg / 1000.0
    basicity = (CaO_total + MgO_total) / max(SiO2, 1.0)

    # Base removal from basicity (feature uses same formula, fixed cap 0.95)
    p_removal_base = float(np.clip(0.80 + 0.15 * (basicity - 1.5) / 3.5, 0.80, 0.95))
    T_factor = (1680.0 - T_final_phys) / 60.0
    p_removal_nominal = float(np.clip(p_removal_base + 0.02 * T_factor, 0.80, 0.95))
    # Heat-specific partition efficiency multiplier (p_fr mean=0.86 → divides out mean)
    p_removal_eff = float(np.clip(p_removal_nominal * (p_fr / 0.86), 0.70, 0.96))

    P_final_phys = P_hm * (1.0 - p_removal_eff)
    P_final_phys = float(np.clip(P_final_phys, *TARGET_RANGES['P_final']))

    # Add calibrated noise [xia2024dmpinn]
    T_final = float(np.clip(
        T_final_phys + rng.normal(0, NOISE['T_final']), *TARGET_RANGES['T_final']
    ))
    C_final = float(np.clip(
        C_final_phys + rng.normal(0, NOISE['C_final']), *TARGET_RANGES['C_final']
    ))
    P_final = float(np.clip(
        P_final_phys + rng.normal(0, NOISE['P_final']), *TARGET_RANGES['P_final']
    ))

    return T_final, C_final, P_final, Q_in, eff_heat, Q_scrap, m_steel_hm


def generate_dataset(n_target=N_TARGET, seed=SEED):
    rng = np.random.default_rng(seed)
    records = []
    n_processed = 0
    n_rejected = 0

    print(f"Generating {n_target} valid samples (seed={seed})...")

    while len(records) < n_target:
        n_batch = max(1000, (n_target - len(records)) * 2)

        # Sample all input features from canonical RANGES
        batch = {key: _uniform(rng, key, n_batch) for key in RANGES}

        for i in range(n_batch):
            if len(records) >= n_target:
                break

            n_processed += 1
            row = {key: float(batch[key][i]) for key in RANGES}

            # Compute targets (Q_in/eff_heat/Q_scrap/m_steel_hm are reused for
            # validation, on the same physics basis used to derive T_final)
            T_final, C_final, P_final, Q_in, eff_heat, Q_scrap, m_steel_hm = \
                compute_targets(row, rng)
            row['T_final'] = T_final
            row['C_final'] = C_final
            row['P_final'] = P_final

            # Enthalpy balance validation [logar2022eaf, xia2024dmpinn] — checks
            # whether the post-noise/post-clip T_final still respects the
            # generator's own energy balance within tolerance (see
            # calc_Q_out_consistent's docstring).
            Q_out = calc_Q_out_consistent(m_steel_hm, T_final, Q_in, eff_heat, Q_scrap)
            enthalpy_ok = validate_enthalpy(Q_in, Q_out)

            # Mass balance validation: steel yield must exceed 80% of total charge
            m_total = row['m_hm'] * 1000.0 * (1.0 + row['f_scrap'])
            m_oxidized = calc_total_oxidized_mass(
                row['m_hm'] * 1000.0, row['C_hm'],
                row['Si_hm'], row['Mn_hm'], row['P_hm']
            )
            yield_frac = (m_total - m_oxidized) / max(m_total, 1.0)
            mass_ok = yield_frac > 0.80

            # Reject unless both checks pass: samples violating either
            # criterion are discarded.
            if not enthalpy_ok or not mass_ok:
                n_rejected += 1
                continue

            records.append(row)

    df = pd.DataFrame(records[:n_target])
    # Reorder: RAW_FEATURES first, then targets
    df = df[RAW_FEATURES + ['T_final', 'C_final', 'P_final']]

    print(f"Done. Evaluated: {n_processed}, Rejected: {n_rejected}, "
          f"Rejection rate: {n_rejected / n_processed:.1%}")
    return df


def main():
    os.makedirs('data', exist_ok=True)
    df = generate_dataset()

    out_path = 'data/bof_synthetic.csv'
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path} ({len(df)} rows x {len(df.columns)} cols)")
    print("\nTarget statistics:")
    print(df[['T_final', 'C_final', 'P_final']].describe().round(4))
    print(f"\nNull values: {df.isnull().sum().sum()}")


if __name__ == '__main__':
    main()