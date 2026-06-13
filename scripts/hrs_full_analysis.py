# HRS Complete Analysis — SAI/BAI/BoAI Indices + GMM Clustering + Mediation
# Author: Generated 2026-06-11
# Purpose: Full analysis of HRS (Health and Retirement Study, USA) for the Sensory-Brain-Body Aging Triad project
# Data: RAND HRS Longitudinal File 2020 + Gateway Harmonized HRS + Cross-Wave Cognitive Imputation

import os, sys, warnings, json
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats as scipy_stats

# ============================================================
# 0. SETUP
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HRS_BASE = os.path.join(PROJECT_ROOT, 'data_raw', 'HRS', 'HRS')
OUT_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(os.path.join(OUT_DIR, 'tables'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'figures'), exist_ok=True)

print("=" * 70)
print("HRS Complete Analysis — SAI/BAI/BoAI + GMM + Mediation")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n=== 1. Loading HRS data ===")

# --- 1a. RAND HRS Longitudinal File ---
rand_file = os.path.join(HRS_BASE, 'RAND HRS Longitudinal File 2020', 'randhrs1992_2020v2.dta')
print(f"Loading RAND: {rand_file}")

# Key variables from RAND (wave 3 = first wave with ser7, or use the baseline wave for each respondent)
rand_vars = [
    'hhidpn',                                    # ID
    'ragender', 'rabyear', 'raeduc',             # Demographics (cross-wave)
    'r1agey_b',                                  # Age at baseline (wave 1)
    # Wave 3: first wave with full cognitive battery (imrc + dlrc + ser7)
    'r3agey_b', 'r3jsight', 'r3imrc', 'r3dlrc', 'r3ser7', 'r3cogtot',
    'r3adl5a', 'r3adl6a', 'r3iadl5a',           # ADL/IADL summaries
    'r3height', 'r3weight', 'r3bmi',             # Anthropometrics
    'r3smokev', 'r3drink',                       # Lifestyle
    'r3hibp', 'r3diab', 'r3hearte', 'r3stroke',  # Chronic conditions
    'r3cesd',                                     # Depression
]

df_rand = pd.read_stata(rand_file, columns=rand_vars)
print(f"  RAND: {len(df_rand):,} rows, {len(df_rand.columns)} columns")

# --- 1b. Gateway Harmonized HRS ---
gw_file = os.path.join(HRS_BASE, 'Gateway Harmonized HRS', 'H_HRS_d.dta')
print(f"Loading Gateway: {gw_file}")

# Gateway sensory variables (wave 3) — hhidpn already exists
# Naming: r3sight = self-rated eyesight (1=exc,2=vgood,3=good,4=fair,5=poor,6=blind)
#         r3hearing = self-rated hearing (1=exc...5=poor)
#         r3dsight = distance vision, r3nsight = near vision
gw_vars = ['hhidpn',
           'r3sight',    # Self-rated eyesight (vision)
           'r3hearing',  # Self-rated hearing
           'r3dsight',   # Distance vision
           'r3nsight',   # Near vision
           'r3hearaid']  # Hearing aid use

df_gw = pd.read_stata(gw_file, columns=gw_vars)

# Convert categorical to numeric codes (Stata labeled values)
for c in ['r3sight', 'r3hearing', 'r3dsight', 'r3nsight']:
    if hasattr(df_gw[c].dtype, 'categories'):
        # Extract numeric code from category labels like '4.fair'
        df_gw[c + '_num'] = df_gw[c].cat.codes + 1  # cat.codes are 0-based
        # Handle missing (NaN) -> 0
        df_gw[c + '_num'] = df_gw[c + '_num'].fillna(0).astype(int)
    else:
        df_gw[c + '_num'] = pd.to_numeric(df_gw[c], errors='coerce').fillna(0)

print(f"  Gateway: {len(df_gw):,} rows")

# --- 1c. CogImp NOT needed — RAND already has r3imrc, r3dlrc, r3ser7 ---
# The RAND Longitudinal File includes the cognitive measures directly.
# CogImp adds imputed values for missing waves but wave 3 has good coverage.
print(f"  Using RAND cognitive measures directly (r3imrc, r3dlrc, r3ser7)")

# ============================================================
# 2. MERGE: RAND + Gateway only
# ============================================================
print("\n=== 2. Merging data ===")

# Merge RAND + Gateway
df = df_rand.merge(df_gw[['hhidpn', 'r3sight_num', 'r3hearing_num', 'r3dsight_num', 'r3nsight_num', 'r3hearaid']],
                    on='hhidpn', how='left', suffixes=('', '_gw'))
print(f"  After RAND+Gateway merge: {len(df):,} rows")

# ============================================================
# 3. FILTER: Age >= 50 at baseline
# ============================================================
print("\n=== 3. Filtering (age >= 50) ===")

# Use wave 1 age (r1agey_b) as the baseline age
df['age'] = pd.to_numeric(df['r1agey_b'], errors='coerce')
print(f"  Before filter: {len(df):,} rows, age range [{df['age'].min():.0f}-{df['age'].max():.0f}]")
df = df[df['age'] >= 50].copy()
print(f"  After filter: {len(df):,} rows, {df['hhidpn'].nunique():,} unique IDs")

# ============================================================
# 4. BUILD INDICES
# ============================================================
print("\n=== 4. Building SAI / BAI / BoAI ===")

# --- 4a. SAI (Sensory Aging Index) ---
# Vision impairment: RAND r3jsight >= 4 (fair/poor/blind) OR Gateway r3sight_num >= 4
df['r3jsight'] = pd.to_numeric(df['r3jsight'], errors='coerce').fillna(0)
df['vi_rand'] = (df['r3jsight'] >= 4).astype(int)

# Gateway vision: r3sight_num (eyesight) >= 4
df['vi_gw'] = (df['r3sight_num'] >= 4).astype(int)

# Combined vision impairment: either source
df['vi'] = ((df['vi_rand'] == 1) | (df['vi_gw'] == 1)).astype(int)

# Hearing impairment: Gateway r3hearing_num >= 4 (fair/poor)
# Also check RAND r3hearing if available
df['hi'] = (df['r3hearing_num'] >= 4).astype(int)

# DSI: Dual Sensory Impairment
df['dsi'] = ((df['vi'] == 1) & (df['hi'] == 1)).astype(int)

# SAI: 0, 50, 100
df['SAI'] = (df['vi'] + df['hi']) / 2 * 100

# Report impairment rates
vi_rate = df['vi'].mean() * 100
hi_rate = df['hi'].mean() * 100
dsi_rate = df['dsi'].mean() * 100
print(f"  Vision impairment (r3jsight>=4): {vi_rate:.1f}%")
print(f"  Hearing impairment (R3SIGHT>=4): {hi_rate:.1f}%")
print(f"  DSI (both): {dsi_rate:.1f}%")

# --- 4b. BAI (Brain Aging Index) ---
# Use RAND wave 3 cognitive measures: r3imrc (0-10), r3dlrc (0-10), r3ser7 (0-5)
for cog_var, max_val in [('imrc', 10), ('dlrc', 10), ('ser7', 5)]:
    rand_col = f'r3{cog_var}'
    if rand_col in df.columns:
        df[f'{cog_var}_val'] = pd.to_numeric(df[rand_col], errors='coerce')
    else:
        df[f'{cog_var}_val'] = np.nan

# Normalize each component to 0-1 scale
for cog_var, max_val in [('imrc', 10), ('dlrc', 10), ('ser7', 5)]:
    col = f'{cog_var}_val'
    df[f'{cog_var}_norm'] = (df[col].clip(0, max_val) / max_val).fillna(0)

# BAI: average of normalized scores * 100
df['BAI'] = (df['imrc_norm'] + df['dlrc_norm'] + df['ser7_norm']) / 3 * 100

# Flag: how many cognitive components are available (not NaN originally)
cog_available = df['imrc_val'].notna().astype(int) + df['dlrc_val'].notna().astype(int) + df['ser7_val'].notna().astype(int)
df['cog_available'] = cog_available
# Only compute valid BAI if at least 1 component available, else NaN
df.loc[cog_available == 0, 'BAI'] = np.nan

print(f"  BAI: min={df['BAI'].min():.0f}, max={df['BAI'].max():.0f}, mean={df['BAI'].mean():.0f}, std={df['BAI'].std():.0f}")

# Report component availability
for var in ['imrc', 'dlrc', 'ser7']:
    avail = df[f'{var}_val'].notna().mean() * 100
    print(f"    {var}: {avail:.0f}% available")

# --- 4c. BoAI (Body Aging Index) ---
# Components: ADL (r3adl5a, 0-5), IADL (r3iadl5a, 0-5), BMI, chronic conditions
boai_components = []

# ADL: higher = more difficulty (reverse for BoAI)
df['r3adl5a'] = pd.to_numeric(df['r3adl5a'], errors='coerce')
adl_max = df['r3adl5a'].max()
if adl_max > 0:
    adl_norm = (df['r3adl5a'].fillna(0) / adl_max)
    boai_components.append(1 - adl_norm)  # Reverse: lower ADL difficulty = better
    print(f"  ADL: mean={df['r3adl5a'].mean():.2f}, max={adl_max:.0f}")

# IADL: higher = more difficulty (reverse)
df['r3iadl5a'] = pd.to_numeric(df['r3iadl5a'], errors='coerce')
iadl_max = df['r3iadl5a'].max()
if iadl_max > 0:
    iadl_norm = (df['r3iadl5a'].fillna(0) / iadl_max)
    boai_components.append(1 - iadl_norm)
    print(f"  IADL: mean={df['r3iadl5a'].mean():.2f}, max={iadl_max:.0f}")

# BMI: quadratic penalty (both underweight and obese are unhealthy)
df['r3bmi'] = pd.to_numeric(df['r3bmi'], errors='coerce')
bmi_ideal = 23  # Optimal BMI for older adults
bmi_dev = np.abs(df['r3bmi'].fillna(bmi_ideal) - bmi_ideal)
bmi_dev_max = bmi_dev.quantile(0.99)
if bmi_dev_max > 0:
    bmi_norm = 1 - (bmi_dev.clip(0, bmi_dev_max) / bmi_dev_max)
    boai_components.append(bmi_norm)
    print(f"  BMI: mean={df['r3bmi'].mean():.1f}, optimal=23")

# Chronic conditions: need numeric extraction BEFORE BoAI construction
for cond in ['r3hibp', 'r3diab', 'r3hearte', 'r3stroke']:
    df[cond + '_num'] = df[cond].astype(str).str.startswith('1').astype(int)

chronic_sum = df['r3hibp_num'] + df['r3diab_num'] + df['r3hearte_num'] + df['r3stroke_num']
chronic_max = chronic_sum.max()
if chronic_max > 0:
    chronic_norm = 1 - (chronic_sum / chronic_max)
    boai_components.append(chronic_norm)
    print(f"  Chronic: mean={chronic_sum.mean():.2f}, max={chronic_max:.0f}")

# Composite BoAI
if boai_components:
    df['BoAI_raw'] = sum(boai_components) / len(boai_components)
    mnb, mxb = df['BoAI_raw'].min(), df['BoAI_raw'].max()
    df['BoAI'] = ((df['BoAI_raw'] - mnb) / (mxb - mnb) * 100).clip(0, 100)
    print(f"  BoAI: min={df['BoAI'].min():.0f}, max={df['BoAI'].max():.0f}, mean={df['BoAI'].mean():.0f}, std={df['BoAI'].std():.0f}")

# ============================================================
# 5. PREPARE ANALYSIS DATASET
# ============================================================
print("\n=== 5. Preparing analysis dataset ===")

# Helper: extract numeric from categorical (Stata labeled) columns
def cat_to_num(series):
    """Convert Stata categorical to numeric, handling labeled values like '1.male', '0.no'"""
    if hasattr(series.dtype, 'categories'):
        return series.cat.codes.astype(float)
    return pd.to_numeric(series, errors='coerce')

# Demographics
df['female'] = (cat_to_num(df['ragender']) == 1).astype(int)  # cat.codes: 0=male, 1=female
df['age'] = df['age'].astype(float)
df['educ_yrs'] = cat_to_num(df['raeduc'])  # 0=ltHS, 1=GED, 2=HS, 3=some college, 4=college+

# CESD for mediation
df['r3cesd'] = pd.to_numeric(df['r3cesd'], errors='coerce')
df['cesd'] = df['r3cesd']

# Drop rows missing key indices
df_clean = df.dropna(subset=['SAI', 'BAI', 'BoAI']).copy()
print(f"  Complete cases: {len(df_clean):,} (from {len(df):,})")

# ============================================================
# 6. DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 70)
print("=== 6. DESCRIPTIVE STATISTICS ===")

print(f"\n{'Metric':<35s} {'Value':>15s}")
print("-" * 52)
for label, val, fmt in [
    ('N (complete cases)', len(df_clean), 'd'),
    ('Age, mean (SD)', f"{df_clean['age'].mean():.1f} ({df_clean['age'].std():.1f})", 's'),
    ('Female, %', f"{df_clean['female'].mean()*100:.1f}", 's'),
    ('Education (0=ltHS..4=college+)', f"{df_clean['educ_yrs'].mean():.1f}", 's'),
    ('SAI, mean (SD)', f"{df_clean['SAI'].mean():.1f} ({df_clean['SAI'].std():.1f})", 's'),
    ('BAI, mean (SD)', f"{df_clean['BAI'].mean():.1f} ({df_clean['BAI'].std():.1f})", 's'),
    ('BoAI, mean (SD)', f"{df_clean['BoAI'].mean():.1f} ({df_clean['BoAI'].std():.1f})", 's'),
    ('Vision impairment, %', f"{df_clean['vi'].mean()*100:.1f}", 's'),
    ('Hearing impairment, %', f"{df_clean['hi'].mean()*100:.1f}", 's'),
    ('DSI, %', f"{df_clean['dsi'].mean()*100:.1f}", 's'),
    ('CESD, mean (SD)', f"{df_clean['cesd'].mean():.1f} ({df_clean['cesd'].std():.1f})", 's'),
    ('Hypertension, %', f"{df_clean['r3hibp_num'].mean()*100:.1f}", 's'),
    ('Diabetes, %', f"{df_clean['r3diab_num'].mean()*100:.1f}", 's'),
    ('Heart disease, %', f"{df_clean['r3hearte_num'].mean()*100:.1f}", 's'),
    ('Stroke, %', f"{df_clean['r3stroke_num'].mean()*100:.1f}", 's'),
]:
    if fmt == 'd':
        print(f"  {label:<33s} {int(val):>15,}")
    else:
        print(f"  {label:<33s} {str(val):>15s}")

# ============================================================
# 7. GMM CLUSTERING (5 phenotypes)
# ============================================================
print("\n" + "=" * 70)
print("=== 7. GMM Clustering — 5 Phenotypes ===")

X = df_clean[['SAI', 'BAI', 'BoAI']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(n_components=5, random_state=42, n_init=20, max_iter=500)
df_clean['phenotype'] = gmm.fit_predict(X_scaled)

# Order phenotypes by mean BAI (lowest to highest)
pheno_order = df_clean.groupby('phenotype')['BAI'].mean().sort_values().index
pheno_map = {old: new for new, old in enumerate(pheno_order)}
df_clean['pheno_label'] = df_clean['phenotype'].map(pheno_map)

# Assign meaningful labels based on actual profiles
pheno_labels = {
    0: 'Type A: Multi-Domain Moderate Decline',
    1: 'Type B: Sensory-Dominant DSI (all DSI)',
    2: 'Type C: Cognitive-Predominant Decline',
    3: 'Type D: Sensory-First (mild sensory only)',
    4: 'Type E: Successful Aging (best)'
}

# Compute phenotype profiles
print(f"\n{'Type':<45s} {'n':>7s} {'%':>6s} {'SAI':>6s} {'BAI':>6s} {'BoAI':>6s} {'DSI%':>6s} {'Age':>6s} {'Fem%':>6s}")
print("-" * 100)

pheno_profiles = {}
for p in range(5):
    sub = df_clean[df_clean['pheno_label'] == p]
    pct = len(sub) / len(df_clean) * 100
    dsi_pct = sub['dsi'].mean() * 100
    sai_m = sub['SAI'].mean()
    bai_m = sub['BAI'].mean()
    boai_m = sub['BoAI'].mean()
    age_m = sub['age'].mean()
    fem_pct = sub['female'].mean() * 100

    pheno_profiles[p] = {
        'n': len(sub), 'pct': pct, 'sai': sai_m, 'bai': bai_m, 'boai': boai_m,
        'dsi': dsi_pct, 'age': age_m, 'female': fem_pct
    }

    print(f"  {pheno_labels.get(p, f'Type {p+1}'):<43s} {len(sub):>7,} {pct:>5.1f}% {sai_m:>5.0f} {bai_m:>5.0f} {boai_m:>5.0f} {dsi_pct:>5.1f}% {age_m:>5.1f} {fem_pct:>5.0f}%")

# BIC score
print(f"\n  GMM BIC: {gmm.bic(X_scaled):.0f}")
print(f"  GMM AIC: {gmm.aic(X_scaled):.0f}")

# ============================================================
# 8. MEDIATION ANALYSIS: DSI → CESD → BAI
# ============================================================
print("\n" + "=" * 70)
print("=== 8. Mediation: DSI → CESD → BAI ===")

df_med = df_clean.dropna(subset=['dsi', 'BAI', 'cesd', 'age']).copy()
print(f"  Mediation sample: N={len(df_med):,}")

# Baron & Kenny approach with covariates
# Step 1: DSI → BAI (total effect)
m1 = LinearRegression()
X1 = df_med[['dsi', 'age', 'female']].fillna(0)
m1.fit(X1, df_med['BAI'])
c_total = m1.coef_[0]  # DSI coefficient
print(f"  Step 1 (DSI→BAI): DSI coeff={c_total:.3f}, R2={m1.score(X1, df_med['BAI']):.3f}")

# Step 2: DSI → CESD (a path)
m2 = LinearRegression()
X2 = df_med[['dsi', 'age', 'female']].fillna(0)
m2.fit(X2, df_med['cesd'])
a_path = m2.coef_[0]
print(f"  Step 2 (DSI→CESD): DSI coeff={a_path:.3f}, R2={m2.score(X2, df_med['cesd']):.3f}")

# Step 3: DSI + CESD → BAI (b path + direct)
m3 = LinearRegression()
X3 = df_med[['dsi', 'cesd', 'age', 'female']].fillna(0)
m3.fit(X3, df_med['BAI'])
b_path = m3.coef_[1]  # CESD coefficient
c_direct = m3.coef_[0]  # DSI direct effect
print(f"  Step 3 (DSI+CESD→BAI): DSI={c_direct:.3f}, CESD={b_path:.3f}, R2={m3.score(X3, df_med['BAI']):.3f}")

# Indirect effect and proportion mediated
indirect = a_path * b_path
total = c_direct + indirect
prop_mediated = indirect / total * 100 if total != 0 else 0

print(f"\n  Indirect effect (a*b): {indirect:.3f}")
print(f"  Total effect: {total:.3f}")
print(f"  Direct effect (c'): {c_direct:.3f}")
print(f"  Proportion mediated: {prop_mediated:.1f}%")

# Bootstrap CI for indirect effect
n_boot = 1000
boot_indirect = []
for i in range(n_boot):
    idx = np.random.choice(len(df_med), len(df_med), replace=True)
    boot_df = df_med.iloc[idx]

    bm2 = LinearRegression()
    bm2.fit(boot_df[['dsi', 'age', 'female']].fillna(0), boot_df['cesd'])
    ba = bm2.coef_[0]

    bm3 = LinearRegression()
    bm3.fit(boot_df[['dsi', 'cesd', 'age', 'female']].fillna(0), boot_df['BAI'])
    bb = bm3.coef_[1]

    boot_indirect.append(ba * bb)

ci_low = np.percentile(boot_indirect, 2.5)
ci_high = np.percentile(boot_indirect, 97.5)
print(f"  Bootstrap 95% CI for indirect: [{ci_low:.3f}, {ci_high:.3f}]")

# Sobel test approximation
se_a = np.sqrt(np.sum((m2.predict(X2) - df_med['cesd'])**2) / (len(df_med)-3)) / np.sqrt(np.sum((X2['dsi'] - X2['dsi'].mean())**2))
se_b = np.sqrt(np.sum((m3.predict(X3) - df_med['BAI'])**2) / (len(df_med)-4)) / np.sqrt(np.sum((X3['cesd'] - X3['cesd'].mean())**2))
se_indirect = np.sqrt(a_path**2 * se_b**2 + b_path**2 * se_a**2)
sobel_z = indirect / se_indirect if se_indirect > 0 else 0
sobel_p = 2 * (1 - scipy_stats.norm.cdf(abs(sobel_z)))
print(f"  Sobel test: z={sobel_z:.3f}, p={sobel_p:.4f}")

# ============================================================
# 9. SAVE RESULTS
# ============================================================
print("\n=== 10. Saving results ===")

# Save processed dataset
df_out = df_clean[['hhidpn', 'age', 'female', 'educ_yrs', 'SAI', 'BAI', 'BoAI',
                    'vi', 'hi', 'dsi', 'cesd', 'r3bmi', 'r3hibp_num', 'r3diab_num',
                    'r3hearte_num', 'r3stroke_num', 'pheno_label']].copy()
df_out.columns = ['hhidpn', 'age', 'female', 'educ', 'SAI', 'BAI', 'BoAI',
                   'vi', 'hi', 'dsi', 'cesd', 'bmi', 'hibp', 'diab',
                   'hearte', 'stroke', 'phenotype']
out_csv = os.path.join(OUT_DIR, 'tables', 'hrs_full_analysis.csv')
df_out.to_csv(out_csv, index=False)
print(f"  Saved: {out_csv} ({len(df_out):,} rows)")

# Save phenotype profiles
import json
pheno_out = os.path.join(OUT_DIR, 'tables', 'hrs_phenotype_profiles.json')
with open(pheno_out, 'w') as f:
    json.dump({str(k): {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else int(vv) if isinstance(vv, np.integer) else vv
                        for kk, vv in v.items()}
               for k, v in pheno_profiles.items()}, f, indent=2)
print(f"  Saved: {pheno_out}")

# ============================================================
# 9. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL HRS SUMMARY")
print("=" * 70)

print(f"""
  Cohort: HRS (Health and Retirement Study, USA)
  N (baseline, complete): {len(df_clean):,}
  Age: {df_clean['age'].mean():.1f} ± {df_clean['age'].std():.1f}
  Female: {df_clean['female'].mean()*100:.0f}%
  Education: {df_clean['educ_yrs'].mean():.1f} (0=ltHS..4=college+)

  SAI: {df_clean['SAI'].mean():.1f} ± {df_clean['SAI'].std():.1f}
  BAI: {df_clean['BAI'].mean():.1f} ± {df_clean['BAI'].std():.1f}
  BoAI: {df_clean['BoAI'].mean():.1f} ± {df_clean['BoAI'].std():.1f}

  Vision impairment: {vi_rate:.1f}%
  Hearing impairment: {hi_rate:.1f}%
  DSI (Dual Sensory Impairment): {dsi_rate:.1f}%

  Mediation (DSI→CESD→BAI):
    Total effect: {total:.3f}
    Indirect (via CESD): {indirect:.3f}
    Proportion mediated: {prop_mediated:.1f}%
    Sobel z: {sobel_z:.3f} (p={sobel_p:.4f})

  5 Phenotypes identified via GMM
  BIC: {gmm.bic(X_scaled):.0f}
""")

print("Done! HRS analysis complete.")
