"""
BOF Thermodynamic and Stoichiometric Constants and Functions.

All constants derived from:
- Cai et al. (2021) [cai2021heat] for heat capacities and enthalpies
- Logar et al. (2022) [logar2022eaf] for heat loss estimates
- Bae (2020) [bae2020bof] for stoichiometric ratios
"""

import numpy as np

# --- Thermodynamic constants ---
Cp_hm = 0.82      # kJ/(kg·°C) - hot metal heat capacity [cai2021heat]
Cp_steel = 0.75   # kJ/(kg·°C) - steel heat capacity [cai2021heat]
Cp_scrap = 0.68   # kJ/(kg·°C) - scrap heat capacity (approximation)
T_ref = 25.0      # °C - reference temperature
T_scrap = 25.0    # °C - scrap initial temperature
T_melt = 1500.0   # °C - scrap melting reference point

# --- Oxidation enthalpies (kJ/mol) [bae2020bof, cai2021heat] ---
DH = {
    'C_CO':   -110.5,   # C + 1/2 O2 -> CO
    'C_CO2':  -393.5,   # C + O2 -> CO2
    'Si':     -877.8,   # Si + O2 -> SiO2
    'Mn':     -384.7,   # Mn + 1/2 O2 -> MnO
    'P':     -1492.0,   # 2P + 5/2 O2 -> P2O5
    'Fe':     -264.0,   # Fe + 1/2 O2 -> FeO
}

# --- Molar masses (g/mol) ---
M = {
    'C':  12.011,
    'Si': 28.085,
    'Mn': 54.938,
    'P':  30.974,
    'Fe': 55.845,
    'O':  15.999,
}

# --- O2 stoichiometric ratios (kg O2 / kg element) [bae2020bof] ---
# Based on dominant oxidation reactions in BOF
STOICH_O2 = {
    'C':  (16 / 12.011),          # C -> CO (dominant at high T)
    'Si': (2 * 15.999 / 28.085),  # Si -> SiO2
    'Mn': (15.999 / 54.938),      # Mn -> MnO
    'P':  (2.5 * 15.999 / 30.974),  # 2P -> P2O5
}

# Fraction of heat losses relative to Q_in [chattopadhyay2022hybrid, madhavan2020heatbalance]
# Literature range: 1.3-5.9%; Chattopadhyay uses 5% as best-fit value
HEAT_LOSS_FRAC = 0.05

# --- Canonical input variable ranges (literature-based) [xia2024dmpinn, bae2020bof] ---
RANGES = {
    'm_hm':     (150.0, 320.0),   # hot metal mass (tonnes)
    'T_hm':     (1250.0, 1450.0), # hot metal temperature (°C)
    'C_hm':     (3.8, 4.8),       # carbon in hot metal (wt%)
    'Si_hm':    (0.3, 1.2),       # silicon (wt%)
    'Mn_hm':    (0.2, 0.8),       # manganese (wt%)
    'P_hm':     (0.06, 0.15),     # phosphorus (wt%)
    'S_hm':     (0.01, 0.05),     # sulfur (wt%) — sampled, not in physics model
    'f_scrap':  (0.10, 0.30),     # scrap fraction of charge
    'CaO_kg_t': (15.0, 60.0),     # CaO addition (kg/t steel)
    'MgO_kg_t': (5.0, 20.0),      # MgO addition (kg/t steel) [bof_physics.md]
    'C_scrap':  (0.05, 0.25),     # carbon in scrap (wt%) [bof_physics.md]
    'O2_flow':  (300.0, 800.0),   # oxygen flow rate (Nm3/min)
    't_blow':   (12.0, 22.0),     # blow time (minutes)
    'lance_h':  (1.2, 2.5),       # lance height (m) — sampled, not in physics model
}

# Canonical ordered raw feature list (all sampled inputs) [xia2024dmpinn]
RAW_FEATURES = [
    'm_hm', 'T_hm', 'C_hm', 'Si_hm', 'Mn_hm', 'P_hm', 'S_hm',
    'f_scrap', 'CaO_kg_t', 'MgO_kg_t', 'C_scrap',
    'O2_flow', 't_blow', 'lance_h',
]


def calc_co2_fraction(C_hm, C_final):
    """
    Estimate CO2 fraction of carbon oxidation product at blow endpoint.

    At the start of the blow (C_final ≈ C_hm) almost all C forms CO;
    as carbon is depleted toward the end of the blow, CO2 fraction increases.

    Formula based on CO/CO2 split model from [chattopadhyay2022hybrid]:
        f_CO2 = clip(0.05 + 0.35 × (1 - C_final / C_hm), 0.05, 0.40)

    Parameters
    ----------
    C_hm   : float  Initial carbon in hot metal (wt%)
    C_final: float  Final carbon at endpoint (wt%)

    Returns
    -------
    float : fraction of carbon that forms CO2 (range 0.05–0.40)
    """
    if C_hm <= 0:
        return 0.05
    ratio = np.clip(1.0 - C_final / C_hm, 0.0, 1.0)
    return float(np.clip(0.05 + 0.35 * ratio, 0.05, 0.40))


def calc_Q_in(m_hm, T_hm, C_hm, Si_hm, Mn_hm, P_hm,
              C_fraction_removed=0.9, Si_fraction_removed=0.9,
              Mn_fraction_removed=0.5, P_fraction_removed=0.8,
              co2_fraction=0.0,
              m_scrap_kg=0.0, C_scrap=0.0, f_C_removed_scrap=0.9):
    """
    Calculate total heat input: sensible heat of hot metal + oxidation heat.

    Parameters
    ----------
    m_hm : float
        Hot metal mass (tonnes)
    T_hm : float
        Hot metal temperature (°C)
    C_hm, Si_hm, Mn_hm, P_hm : float
        Element mass fractions in hot metal (wt%)
    *_fraction_removed : float
        Fraction of each element oxidized during blow
    co2_fraction : float
        Fraction of carbon that forms CO2 instead of CO (0.0 = all CO, default).
        Use calc_co2_fraction() to compute this from C_hm and C_final.
    m_scrap_kg : float
        Scrap mass (kg). If > 0 and C_scrap > 0, scrap-C oxidation heat is added.
    C_scrap : float
        Carbon content of scrap (wt%). Contributes oxidation heat during blow.
    f_C_removed_scrap : float
        Fraction of scrap carbon oxidized (default 0.9, same as hot metal C).

    Returns
    -------
    float : Q_in in kJ
    """
    m_hm_kg = m_hm * 1000.0

    # Sensible heat of hot metal
    Q_sensible = m_hm_kg * Cp_hm * (T_hm - T_ref)

    # Effective carbon oxidation enthalpy (blended CO/CO2) [chattopadhyay2022hybrid]
    dh_C_eff = co2_fraction * abs(DH['C_CO2']) + (1.0 - co2_fraction) * abs(DH['C_CO'])

    # Oxidation heat contributions
    # mass_oxidized in kg; molar_mass in g/mol → convert kg to g (×1000) before dividing
    def q_oxidation(frac_elem, fraction_removed, dh_kj_mol, molar_mass):
        mass_oxidized = m_hm_kg * frac_elem * fraction_removed  # kg
        moles_oxidized = (mass_oxidized * 1000.0) / molar_mass  # kg→g, then /（g/mol）= mol
        return abs(dh_kj_mol) * moles_oxidized

    Q_C   = q_oxidation(C_hm / 100,  C_fraction_removed,  dh_C_eff,    M['C'])
    Q_Si  = q_oxidation(Si_hm / 100, Si_fraction_removed, DH['Si'],    M['Si'])
    Q_Mn  = q_oxidation(Mn_hm / 100, Mn_fraction_removed, DH['Mn'],    M['Mn'])
    Q_P   = q_oxidation(P_hm / 100,  P_fraction_removed,  DH['P'],     M['P'])

    # Scrap carbon oxidation heat [bof_physics.md]
    if m_scrap_kg > 0 and C_scrap > 0:
        mass_C_scrap = m_scrap_kg * (C_scrap / 100) * f_C_removed_scrap  # kg
        Q_C_scrap = dh_C_eff * (mass_C_scrap * 1000.0 / M['C'])  # kg->g, then /(g/mol) = mol
    else:
        Q_C_scrap = 0.0

    return Q_sensible + Q_C + Q_Si + Q_Mn + Q_P + Q_C_scrap


def calc_Q_out(m_steel_kg, T_final, Q_in, heat_loss_frac=HEAT_LOSS_FRAC):
    """
    Calculate total heat output: sensible heat of steel + thermal losses.

    Returns
    -------
    float : Q_out in kJ
    """
    Q_steel = m_steel_kg * Cp_steel * (T_final - T_ref)
    Q_losses = heat_loss_frac * Q_in
    return Q_steel + Q_losses


def calc_Q_out_consistent(m_steel_hm, T_final, Q_in, eff_heat, Q_scrap):
    """
    Enthalpy-balance validator consistent with how the generator
    computes T_final (see generate_bof_data.py::compute_targets).

    The generator derives T_final from
        m_steel_hm * Cp_steel * (T_final_phys - T_ref) = Q_in * eff_heat - Q_scrap
    using the same m_steel_hm (hot-metal-only mass) and per-heat eff_heat that
    are passed in here. Substituting this relation back in gives Q_out == Q_in
    (energy is conserved by construction): the residual comes from the
    measurement noise and target-range clipping applied to T_final after the
    physics step, so the 10% criterion checks whether that perturbation stayed
    physically plausible.

    Parameters
    ----------
    m_steel_hm : float  Hot-metal-only steel mass (kg) — same basis the
                         generator used, NOT calc_steel_mass()'s scrap-inclusive mass.
    T_final    : float  Final temperature actually reported (post-noise, post-clip).
    Q_in       : float  Heat input (kJ), as computed by calc_Q_in() with the
                         same heat-specific removal fractions/CO2 blend used at generation.
    eff_heat   : float  The per-heat thermal efficiency drawn for this sample.
    Q_scrap    : float  Scrap-melting heat demand (kJ), as used at generation.

    Returns
    -------
    float : Q_out in kJ
    """
    Q_steel = m_steel_hm * Cp_steel * (T_final - T_ref)
    Q_losses = (1.0 - eff_heat) * Q_in
    return Q_steel + Q_scrap + Q_losses


def validate_enthalpy(Q_in, Q_out, threshold=0.10):
    """
    Check enthalpy balance criterion: |Q_in - Q_out| / Q_in < threshold.

    Returns
    -------
    bool : True if balance is within tolerance
    """
    if Q_in <= 0:
        return False
    return abs(Q_in - Q_out) / Q_in < threshold


def calc_steel_mass(m_hm_kg, f_scrap):
    """
    Canonical steel mass formula: ~92% of total charge (HM + scrap) ends as steel.

    m_steel = m_hm_kg × (1 + f_scrap) × 0.92

    The 8% loss accounts for elements oxidized to slag and carried off as off-gas.
    Based on mass balance from [bae2020bof, chattopadhyay2022hybrid].

    Parameters
    ----------
    m_hm_kg : float  Hot metal mass in kg
    f_scrap  : float  Scrap fraction of hot metal mass

    Returns
    -------
    float : estimated steel mass in kg
    """
    return m_hm_kg * (1.0 + f_scrap) * 0.92


def calc_total_oxidized_mass(m_hm_kg, C_hm, Si_hm, Mn_hm, P_hm,
                              C_fr=0.9, Si_fr=0.9, Mn_fr=0.5, P_fr=0.8):
    """
    Calculate total mass of elements oxidized from hot metal (kg).

    Used for mass balance validation in generate_dataset.

    Returns
    -------
    float : mass of oxidized elements in kg
    """
    return m_hm_kg * (
        (C_hm / 100) * C_fr +
        (Si_hm / 100) * Si_fr +
        (Mn_hm / 100) * Mn_fr +
        (P_hm / 100) * P_fr
    )


def calc_theo_o2_demand(m_hm, C_hm, Si_hm, Mn_hm, P_hm):
    """
    Calculate theoretical O2 demand (kg) to oxidize all impurities.

    Returns
    -------
    float : theoretical O2 demand in kg
    """
    m_hm_kg = m_hm * 1000.0
    theo_o2 = 0.0
    for elem, frac in [('C', C_hm / 100), ('Si', Si_hm / 100),
                       ('Mn', Mn_hm / 100), ('P', P_hm / 100)]:
        theo_o2 += m_hm_kg * frac * STOICH_O2[elem]
    return theo_o2


def calc_physics_features(row):
    """
    Compute all 8 physics-derived features for a single data row (dict-like).

    Returns
    -------
    dict : feature_name -> value
    """
    m_hm      = row['m_hm']
    T_hm      = row['T_hm']
    C_hm      = row['C_hm']
    Si_hm     = row['Si_hm']
    Mn_hm     = row['Mn_hm']
    P_hm      = row['P_hm']
    f_scrap   = row['f_scrap']
    CaO_kg_t  = row['CaO_kg_t']
    MgO_kg_t  = row.get('MgO_kg_t', 0.0)   # defensive default for legacy rows
    O2_flow   = row['O2_flow']    # Nm3/min
    t_blow    = row['t_blow']     # min
    # C_final is NOT read here — all 8 features are computed from input variables only.
    # Avoids target leakage: C_final is a prediction target, unknown at inference time.

    m_hm_kg = m_hm * 1000.0

    # Total O2 actually blown (Nm3 -> kg: density ~1.43 kg/Nm3)
    O2_blown_kg = O2_flow * t_blow * 1.43

    # 1. Theoretical O2 demand
    theo_o2 = calc_theo_o2_demand(m_hm, C_hm, Si_hm, Mn_hm, P_hm)

    # 2. O2 efficiency
    o2_efficiency = theo_o2 / max(O2_blown_kg, 1.0)

    # 3. Enthalpy surplus — theoretical gross heat release minus scrap melting demand.
    # Uses complete oxidation fractions (1.0 each) and no CO2 correction: this is
    # what published thermodynamic tables report as gross available heat.
    # DELIBERATE MISMATCH with generator: generator uses heat-specific partial fractions
    # (c_fr~0.90, si_fr~0.92, mn_fr~0.50, p_fr~0.86) and a heat-specific efficiency
    # factor (eff_heat~0.64 ±6%). The feature gives "published theoretical maximum";
    # the target reflects actual heat-specific energy balance.
    # This gap prevents the algebraic identity that would arise if both used identical
    # constants. [chattopadhyay2022hybrid; bae2020bof — theoretical O2 demand basis]
    Q_in_theoretical = calc_Q_in(
        m_hm, T_hm, C_hm, Si_hm, Mn_hm, P_hm,
        C_fraction_removed=1.0,
        Si_fraction_removed=1.0,
        Mn_fraction_removed=1.0,
        P_fraction_removed=1.0,
        co2_fraction=0.0,   # published tables assume CO product; no CO2 correction
    )
    Q_scrap_needed = m_hm_kg * f_scrap * Cp_scrap * (T_melt - T_scrap)
    enthalpy_surplus = Q_in_theoretical - Q_scrap_needed

    # 4. Theoretical total oxidizable mass (complete removal — published maximum).
    # Uses fraction=1.0 for all elements, matching the theoretical basis of
    # enthalpy_surplus above. Generator uses heat-specific partial fractions.
    total_oxidized = m_hm_kg * (
        (C_hm / 100) * 1.0 + (Si_hm / 100) * 1.0 +
        (Mn_hm / 100) * 1.0 + (P_hm / 100) * 1.0
    )

    # 5. Theoretical decarburization rate (wt%/min)
    # C_hm / t_blow: initial carbon loading per unit blow time (input variables only).
    # Avoids leakage — C_final is a target, unknown at inference time.
    # Captures the C-loading/time relationship: high initial C with short blow = higher demand.
    # [chattopadhyay2022hybrid: typical removal efficiency 85–95%; bae2020bof]
    decarb_rate = C_hm / max(t_blow, 1.0)

    # 6. Slag basicity (CaO + MgO) / SiO2 [bae2020bof]
    SiO2_factor = 60.085 / 28.085  # molar mass ratio SiO2/Si
    SiO2_produced = m_hm_kg * (Si_hm / 100) * SiO2_factor
    CaO_total = CaO_kg_t * m_hm_kg / 1000.0
    MgO_total = MgO_kg_t * m_hm_kg / 1000.0
    slag_basicity = (CaO_total + MgO_total) / max(SiO2_produced, 1.0)

    # 7. Scrap melt capacity (tonnes of scrap meltable)
    scrap_melt_capacity = enthalpy_surplus / max(
        Cp_scrap * 1000 * (T_melt - T_scrap), 1.0
    )

    # 8. Fe oxidation loss (kg)
    excess_o2 = max(O2_blown_kg - theo_o2, 0.0)
    fe_oxidation_loss = excess_o2 / STOICH_O2['C'] * (M['Fe'] / M['C'])

    return {
        'theo_o2_demand':      theo_o2,
        'o2_efficiency':       o2_efficiency,
        'enthalpy_surplus':    enthalpy_surplus,
        'total_oxidized_mass': total_oxidized,
        'decarb_rate':         decarb_rate,
        'slag_basicity':       slag_basicity,
        'scrap_melt_capacity': scrap_melt_capacity,
        'fe_oxidation_loss':   fe_oxidation_loss,
    }


def calc_physics_features_perturbed(row, noise_frac=0.15, seed=None):
    """
    Compute physics-derived features with ±noise_frac perturbation on all
    thermodynamic constants (ΔH values, Cp values, stoichiometric ratios).

    Used for Config D ablation: tests whether Config C gains are robust to
    systematic mismatch between the engineering physics model and the process.
    A noise_frac=0.15 (±15%) represents conservative model uncertainty —
    published BOF constants vary ±5–20% across sources [chattopadhyay2022hybrid,
    cai2021heat, bae2020bof].

    Each call with the same seed produces the same perturbation (reproducible),
    but the perturbation is applied at the CONSTANT level (not per-sample), to
    simulate a systematically mis-calibrated engineering model rather than
    random per-sample noise.

    Parameters
    ----------
    row        : dict-like  Input feature row (same as calc_physics_features)
    noise_frac : float      Fractional noise on constants (default 0.15 = ±15%)
    seed       : int or None  RNG seed for reproducibility (default None)

    Returns
    -------
    dict : same keys as calc_physics_features, computed with perturbed constants
    """
    rng = np.random.default_rng(seed if seed is not None else 17)

    def perturb(val):
        return val * (1.0 + rng.uniform(-noise_frac, noise_frac))

    m_hm      = row['m_hm']
    T_hm      = row['T_hm']
    C_hm      = row['C_hm']
    Si_hm     = row['Si_hm']
    Mn_hm     = row['Mn_hm']
    P_hm      = row['P_hm']
    f_scrap   = row['f_scrap']
    CaO_kg_t  = row['CaO_kg_t']
    MgO_kg_t  = row.get('MgO_kg_t', 0.0)
    O2_flow   = row['O2_flow']
    t_blow    = row['t_blow']

    m_hm_kg = m_hm * 1000.0
    O2_blown_kg = O2_flow * t_blow * perturb(1.43)  # perturb O2 density

    # Perturbed stoichiometric O2 ratios
    stoich_p = {k: perturb(v) for k, v in STOICH_O2.items()}
    theo_o2 = sum(m_hm_kg * (frac / 100) * stoich_p[elem]
                  for elem, frac in [('C', C_hm), ('Si', Si_hm),
                                     ('Mn', Mn_hm), ('P', P_hm)])
    o2_efficiency = theo_o2 / max(O2_blown_kg, 1.0)

    # Perturbed Cp for enthalpy surplus
    cp_hm_p   = perturb(Cp_hm)
    cp_scrap_p = perturb(Cp_scrap)
    dh_p = {k: perturb(abs(v)) for k, v in DH.items()}

    def q_ox(frac_pct, dh_kj_mol, molar_mass):
        return m_hm_kg * (frac_pct / 100) * 1.0 * (dh_kj_mol * 1000.0 / molar_mass)

    Q_sensible = m_hm_kg * cp_hm_p * (T_hm - T_ref)
    Q_C   = q_ox(C_hm,  dh_p['C_CO'],  M['C'])
    Q_Si  = q_ox(Si_hm, dh_p['Si'],    M['Si'])
    Q_Mn  = q_ox(Mn_hm, dh_p['Mn'],    M['Mn'])
    Q_P   = q_ox(P_hm,  dh_p['P'],     M['P'])
    Q_in_p = Q_sensible + Q_C + Q_Si + Q_Mn + Q_P

    Q_scrap_needed = m_hm_kg * f_scrap * cp_scrap_p * (T_melt - T_scrap)
    enthalpy_surplus = Q_in_p - Q_scrap_needed

    total_oxidized = m_hm_kg * (
        (C_hm / 100) + (Si_hm / 100) + (Mn_hm / 100) + (P_hm / 100)
    )

    decarb_rate = C_hm / max(t_blow, 1.0)

    SiO2_factor_p = perturb(60.085 / 28.085)
    SiO2_produced = m_hm_kg * (Si_hm / 100) * SiO2_factor_p
    CaO_total = CaO_kg_t * m_hm_kg / 1000.0
    MgO_total = MgO_kg_t * m_hm_kg / 1000.0
    slag_basicity = (CaO_total + MgO_total) / max(SiO2_produced, 1.0)

    scrap_melt_capacity = enthalpy_surplus / max(
        cp_scrap_p * 1000 * (T_melt - T_scrap), 1.0
    )

    stoich_C_p = stoich_p['C']
    excess_o2 = max(O2_blown_kg - theo_o2, 0.0)
    fe_oxidation_loss = excess_o2 / max(stoich_C_p, 1e-6) * (M['Fe'] / M['C'])

    return {
        'theo_o2_demand':      theo_o2,
        'o2_efficiency':       o2_efficiency,
        'enthalpy_surplus':    enthalpy_surplus,
        'total_oxidized_mass': total_oxidized,
        'decarb_rate':         decarb_rate,
        'slag_basicity':       slag_basicity,
        'scrap_melt_capacity': scrap_melt_capacity,
        'fe_oxidation_loss':   fe_oxidation_loss,
    }
