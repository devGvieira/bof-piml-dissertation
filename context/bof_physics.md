# BOF Process — Physics Reference for Synthetic Data & Feature Engineering

## 1. Input Variable Ranges (Literature-Based)

### Hot Metal (Ferro-Gusa)
| Variable | Symbol | Min | Max | Unit |
|----------|--------|-----|-----|------|
| Mass | m_hm | 150 | 320 | tonnes |
| Temperature | T_hm | 1250 | 1450 | °C |
| Carbon | C_hm | 3.8 | 4.8 | wt% |
| Silicon | Si_hm | 0.3 | 1.2 | wt% |
| Manganese | Mn_hm | 0.2 | 0.8 | wt% |
| Phosphorus | P_hm | 0.06 | 0.15 | wt% |
| Sulfur | S_hm | 0.01 | 0.05 | wt% |

### Scrap
| Variable | Min | Max | Unit |
|----------|-----|-----|------|
| Scrap fraction | 0.10 | 0.30 | fraction of charge |
| Scrap C content | 0.05 | 0.25 | wt% |

### Fluxes & Oxygen
| Variable | Min | Max | Unit |
|----------|-----|-----|------|
| CaO addition | 15 | 60 | kg/t steel |
| MgO addition | 5 | 20 | kg/t steel |
| O₂ flow rate | 300 | 800 | Nm³/min |
| Blow time | 12 | 22 | minutes |
| Lance height | 1.2 | 2.5 | meters |

## 2. Target Variable Ranges (Endpoint)

| Variable | Symbol | Min | Max | Unit |
|----------|--------|-----|-----|------|
| Final temperature | T_end | 1620 | 1700 | °C |
| Final carbon | C_end | 0.02 | 0.10 | wt% |
| Final phosphorus | P_end | 0.005 | 0.025 | wt% |

## 3. Oxidation Stoichiometry

| Reaction | M_element | Stoich O₂/element | ΔH (kJ/mol) |
|----------|-----------|-------------------|-------------|
| C + ½O₂ → CO | 12.011 | 16/12 = 1.333 | -110.5 |
| C + O₂ → CO₂ | 12.011 | 32/12 = 2.667 | -393.5 |
| Si + O₂ → SiO₂ | 28.085 | 32/28.085 = 1.139 | -877.8 |
| Mn + ½O₂ → MnO | 54.938 | 16/54.938 = 0.291 | -384.7 |
| 2P + 5/2 O₂ → P₂O₅ | 30.974 | 5×16/(2×30.974) = 1.291 | -1492.0 |
| Fe + ½O₂ → FeO | 55.845 | 16/55.845 = 0.286 | -264.0 |

### Theoretical O₂ Demand
```
O2_demand = m_hm × Σ (wt%_element × stoich_ratio_element)
```

## 4. Enthalpy Balance

### Heat Input
```
Q_in = Q_sensible_hm + Q_oxidation
Q_sensible_hm = m_hm × Cp_hm × (T_hm - T_ref)    [Cp_hm ≈ 0.82 kJ/(kg·°C), T_ref = 25°C]
Q_oxidation = Σ (mass_element_oxidized × |ΔH| / molar_mass_element)
```

### Heat Output
```
Q_out = Q_sensible_steel + Q_sensible_slag + Q_gas + Q_losses
Q_sensible_steel = m_steel × Cp_steel × (T_end - T_ref)   [Cp_steel ≈ 0.75 kJ/(kg·°C)]
Q_losses ≈ 5-12% of Q_in
```

### Validation Constraint
```
|Q_in - Q_out| / Q_in < 0.10   (any synthetic data point violating this is implausible)
```

## 5. Physics-Derived Features for ML

| Feature Name | Formula | Physical Meaning |
|-------------|---------|-----------------|
| `theo_o2_demand` | Σ(element_mass × stoich_ratio) | Minimum O₂ for all impurities |
| `o2_efficiency` | theo_o2_demand / actual_o2_blown | O₂ utilization rate |
| `enthalpy_surplus` | Q_in - Q_out | Energy for heating/melting scrap |
| `total_oxidized_mass` | Σ(Δelement × m_hm) | Total mass removed from bath |
| `decarb_rate` | (C_hm - C_end) / blow_time | Carbon removal speed |
| `slag_basicity` | CaO / (Si_hm × m_hm × SiO2_factor) | P removal capacity |
| `scrap_melt_capacity` | enthalpy_surplus / (Cp_scrap × ΔT) | Max scrap meltable |
| `fe_oxidation_loss` | (actual_o2 - theo_o2) × fe_stoich | Iron lost to slag |
