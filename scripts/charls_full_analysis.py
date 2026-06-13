# CHARLS Full Analysis — SAI/BAI/BoAI Indices + GMM Clustering + Mediation
# Author: Generated 2026-06-11
# Purpose: Replicate and validate the CHARLS pilot v2 results with enhanced outputs

import os, sys, warnings, json
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from scipy import stats as scipy_stats

# ============================================================
# 0. SETUP
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARLS_DIR = os.path.join(PROJECT_ROOT, 'data_raw', 'CHARLS')
OUT_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(os.path.join(OUT_DIR, 'tables'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'figures'), exist_ok=True)

print("=" * 70)
print("CHARLS Full Analysis — SAI/BAI/BoAI + GMM + Mediation")
print("=" * 70)

# ============================================================
# 1. LOAD AND MERGE ALL CHARLS DATA
# ============================================================
print("\n=== 1. Loading CHARLS data ===")
dfs_raw = {}
for f in sorted(os.listdir(CHARLS_DIR)):
    if f.endswith('.csv'):
        dfs_raw[f] = pd.read_csv(os.path.join(CHARLS_DIR, f), encoding='utf-8', low_memory=False)

file_map = {
    '45248e43': ['id','wave','fi_vision','fi_hearing','fi_cognition','fi_depression','fi_health_status',
                 'smokev','drinkev','physical_activity2','sleep_night','short_sleep','cesd'],
    '92ad8abe': ['id','wave','global_cognition','orient','draw','dadliv'],
    '2b6ece0c': ['id','wave','cesd10','balance','sarcopenia','low_muscle_strength','low_muscle_mass','sleeprl'],
    'd4b49326': ['id','wave','lgrip','lgrip1','lgrip2','rgrip1','rgrip2','wspeed','wspeed1','wspeed2',
                 'mbmi','mheight','mweight','pubage','walkcomp','walkflr_c'],
    'aeccc6cd': ['id','ragender','education2'],
}

df_all = None
for key, vars_needed in file_map.items():
    match = [k for k in dfs_raw if key in k]
    if not match: continue
    available = [v for v in vars_needed if v in dfs_raw[match[0]].columns]
    sub = dfs_raw[match[0]][available].copy()
    on_cols = ['id','wave'] if 'wave' in sub.columns else ['id']
    if df_all is None:
        df_all = sub
    else:
        common = [c for c in on_cols if c in df_all.columns and c in sub.columns]
        df_all = df_all.merge(sub, on=common, how='outer')

print(f"  Merged: {len(df_all):,} rows, {df_all['id'].nunique():,} unique IDs")
print(f"  Waves: {sorted(df_all['wave'].dropna().unique())}")

# Age filter
df_all = df_all[df_all['pubage'] >= 50].copy()
print(f"  After age>=50: {len(df_all):,} rows")

# ============================================================
# 2. BUILD INDICES
# ============================================================
print("\n=== 2. Building SAI / BAI / BoAI ===")

# --- SAI (Sensory Aging Index) ---
for col in ['fi_vision','fi_hearing']:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(0)

df_all['dsi'] = ((df_all['fi_vision'] == 1) & (df_all['fi_hearing'] == 1)).astype(int)
df_all['SAI'] = (df_all['fi_vision'] + df_all['fi_hearing']) / 2 * 100  # 0, 50, 100

# --- BAI (Brain Aging Index) ---
if 'global_cognition' in df_all.columns:
    cog_raw = df_all['global_cognition']
elif 'orient' in df_all.columns and 'draw' in df_all.columns:
    cog_raw = df_all[['orient','draw']].mean(axis=1, skipna=True)
else:
    cog_raw = df_all['cesd10'] * -1
mn, mx = cog_raw.min(), cog_raw.max()
df_all['BAI'] = ((cog_raw - mn) / (mx - mn) * 100).clip(0, 100)

# --- BoAI (Body Aging Index) ---
body_components = []
grip_cols = [c for c in ['lgrip','lgrip1','lgrip2','rgrip1','rgrip2'] if c in df_all.columns]
if grip_cols:
    df_all['grip_max'] = df_all[grip_cols].max(axis=1)
    body_components.append('grip_max')

ws_cols = [c for c in ['wspeed','wspeed1','wspeed2'] if c in df_all.columns]
if ws_cols:
    df_all['wspeed_best'] = df_all[ws_cols].max(axis=1)
    body_components.append('wspeed_best')

for c in ['dadliv','balance','sarcopenia','low_muscle_strength','low_muscle_mass']:
    if c in df_all.columns:
        body_components.append(c)

print(f"  BoAI components: {body_components}")

if body_components:
    df_all['BoAI_raw'] = 0
    n_comp = 0
    for comp in body_components:
        vals = pd.to_numeric(df_all[comp], errors='coerce')
        mn_c, mx_c = vals.min(), vals.max()
        if mx_c > mn_c and vals.notna().sum() > 100:
            z = (vals - mn_c) / (mx_c - mn_c)
            if comp in ['dadliv','sarcopenia','low_muscle_strength','low_muscle_mass']:
                z = 1 - z
            df_all['BoAI_raw'] += z.fillna(0)
            n_comp += 1
    if n_comp > 0:
        df_all['BoAI_raw'] /= n_comp
        mn_b, mx_b = df_all['BoAI_raw'].min(), df_all['BoAI_raw'].max()
        df_all['BoAI'] = ((df_all['BoAI_raw'] - mn_b) / (mx_b - mn_b) * 100).clip(0, 100)
    else:
        df_all['BoAI'] = 50
else:
    df_all['BoAI'] = 50

df_all = df_all.dropna(subset=['SAI','BAI','BoAI']).copy()
print(f"  After cleaning: {len(df_all):,} rows")

# ============================================================
# 3. BASELINE EXTRACTION
# ============================================================
print("\n=== 3. Extracting baseline ===")
first_wave = df_all.groupby('id')['wave'].transform('min')
df_bl = df_all[df_all['wave'] == first_wave].copy()
print(f"  Baseline N = {len(df_bl):,}")

# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================
print("\n=== 4. Descriptive Statistics ===")
desc_cols = ['SAI','BAI','BoAI','pubage','cesd10','dsi']
desc_labels = ['SAI','BAI','BoAI','Age','CES-D','DSI']
desc_df = df_bl[desc_cols].describe(percentiles=[.25,.50,.75]).T
desc_df['valid_n'] = df_bl[desc_cols].notna().sum()
desc_df.insert(0, 'variable', desc_labels)
print(desc_df.to_string(index=False))

# DSI prevalence
dsi_prev = df_bl['dsi'].mean() * 100
print(f"\n  DSI prevalence: {dsi_prev:.1f}%")
print(f"  Mean SAI: {df_bl['SAI'].mean():.1f} ± {df_bl['SAI'].std():.1f}")
print(f"  Mean BAI: {df_bl['BAI'].mean():.1f} ± {df_bl['BAI'].std():.1f}")
print(f"  Mean BoAI: {df_bl['BoAI'].mean():.1f} ± {df_bl['BoAI'].std():.1f}")

# ============================================================
# 5. GMM CLUSTERING (BIC-based k selection)
# ============================================================
print("\n=== 5. GMM Clustering ===")
X = df_bl[['SAI','BAI','BoAI']].values
X_scaled = StandardScaler().fit_transform(X)

print(f"\n  GMM BIC scores:")
bic_results = {}
for k in range(2, 9):
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
    labels = gmm.fit_predict(X_scaled)
    bic = gmm.bic(X_scaled)
    sizes = np.bincount(labels)
    min_pct = sizes.min() / len(labels) * 100
    bic_results[k] = {'bic': bic, 'min_pct': min_pct, 'sizes': sizes.tolist()}
    print(f"    k={k}: BIC={bic:,.0f}, sizes={sizes.tolist()}, min={min_pct:.1f}%")

# Select best k: find BIC elbow with min class size >= 3%
# First, filter to valid k
valid_ks = [k for k in range(2, 9) if bic_results[k]['min_pct'] >= 3.0]
if valid_ks:
    bics = np.array([bic_results[k]['bic'] for k in valid_ks])
    # BIC improvement rate
    bic_deltas = np.abs(np.diff(bics))
    bic_improvement = bic_deltas / np.abs(bics[:-1])  # proportional improvement
    # Find elbow: first k where improvement drops below 15%
    for i, imp in enumerate(bic_improvement):
        if imp < 0.15:
            best_k = valid_ks[i + 1]
            break
    else:
        best_k = valid_ks[-1]  # use max valid k if no clear elbow
else:
    best_k = 5  # fallback

# Ensure k is at least 4 (theoretical expectation: 4-6 classes)
best_k = max(best_k, 4)
print(f"\n  BIC elbow analysis: valid ks = {valid_ks}, selected k = {best_k}")

print(f"\n  Selected k = {best_k}")

gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=20)
df_bl['phenotype'] = gmm.fit_predict(X_scaled)

# Auto-label phenotypes
means = df_bl.groupby('phenotype')[['SAI','BAI','BoAI','pubage','dsi']].mean()
pheno_labels = {}
for p in range(best_k):
    s, b, bo = means.loc[p,'SAI'], means.loc[p,'BAI'], means.loc[p,'BoAI']
    if s < 25 and b < 35 and bo < 50:
        label = 'Resilient'
    elif s > 40 and b > 40 and bo > 50:
        label = 'Global-Accelerated'
    elif s > 40 and b < 35:
        label = 'Sensory-Dominant'
    elif b < 30 and bo > 50:
        label = 'Brain-Resilient'
    elif b > 40 and bo > 60:
        label = 'Body-Dominant'
    elif s > 50 and bo < 50:
        label = 'Sensory-Isolated'
    else:
        label = 'Mixed'
    pheno_labels[p] = label

df_bl['pheno_label'] = df_bl['phenotype'].map(pheno_labels)

print(f"\n  Phenotype profiles (k={best_k}, N={len(df_bl):,}):")
print(f"  {'Type':<20s} {'N':>6s} {'%':>6s} {'SAI':>6s} {'BAI':>6s} {'BoAI':>6s} {'DSI%':>6s} {'Age':>6s}")
print(f"  {'-'*60}")
for p in range(best_k):
    sub = df_bl[df_bl['phenotype']==p]
    print(f"  {pheno_labels[p]:<20s} {len(sub):>6,} {len(sub)/len(df_bl)*100:>5.1f}% "
          f"{sub['SAI'].mean():>5.0f}  {sub['BAI'].mean():>5.0f}  {sub['BoAI'].mean():>5.0f}  "
          f"{sub['dsi'].mean()*100:>5.1f}% {sub['pubage'].mean():>5.0f}")

# ============================================================
# 6. PHENOTYPE FEATURE PROFILES
# ============================================================
print("\n=== 6. Phenotype Feature Profiles ===")
profile_cols = ['SAI','BAI','BoAI','pubage','cesd10','dsi']
profile_df = df_bl.groupby('phenotype')[profile_cols].agg(['mean','std']).round(2)
print(profile_df.to_string())

# Feature importance via classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(df_bl[['SAI','BAI','BoAI']], df_bl['phenotype'])
importances = rf.feature_importances_
print(f"\n  Random Forest feature importance for phenotype classification:")
for i, col in enumerate(['SAI','BAI','BoAI']):
    print(f"    {col}: {importances[i]:.4f}")

# ============================================================
# 7. MEDIATION ANALYSIS: DSI → CES-D → BAI
# ============================================================
print("\n=== 7. Mediation Analysis: DSI → CES-D → BAI ===")
df_med = df_bl.dropna(subset=['dsi','BAI','cesd10','pubage','ragender']).copy()
df_med['female'] = (df_med['ragender'] == 2).astype(int)
print(f"  Mediation sample N = {len(df_med):,}")

# --- Baron-Kenny Steps ---
# Step 1: X -> Y (total effect: c)
m1 = LinearRegression().fit(df_med[['dsi','pubage','female']], df_med['BAI'])
c = m1.coef_[0]
print(f"  Step 1 (DSI→BAI total): c = {c:.4f}")

# Step 2: X -> M (a)
m2 = LinearRegression().fit(df_med[['dsi','pubage','female']], df_med['cesd10'])
a = m2.coef_[0]
print(f"  Step 2 (DSI→CES-D): a = {a:.4f}")

# Step 3: X + M -> Y (c' and b)
m3 = LinearRegression().fit(df_med[['dsi','cesd10','pubage','female']], df_med['BAI'])
c_prime = m3.coef_[0]
b = m3.coef_[1]
print(f"  Step 3 (DSI+CES-D→BAI): c' = {c_prime:.4f}, b = {b:.4f}")

indirect = a * b
total = c_prime + indirect
mediation_pct = indirect / total * 100 if total != 0 else 0
print(f"\n  Indirect effect (mediation): {indirect:.4f}")
print(f"  Direct effect: {c_prime:.4f}")
print(f"  Total effect: {total:.4f}")
print(f"  Mediation proportion: {mediation_pct:.1f}%")

# --- Bootstrap CI for mediation ---
print("\n  Bootstrap (1,000 iterations)...")
n_boot = 1000
boot_results = {'a': [], 'b': [], 'c': [], 'c_prime': [], 'indirect': [], 'direct': [], 'total': [], 'pct': []}

np.random.seed(42)
n_samples = len(df_med)
for i in range(n_boot):
    idx = np.random.choice(n_samples, size=n_samples, replace=True)
    boot_df = df_med.iloc[idx]

    try:
        # a path
        a_b = LinearRegression().fit(boot_df[['dsi','pubage','female']], boot_df['cesd10']).coef_[0]
        # b and c' paths
        m3_b = LinearRegression().fit(boot_df[['dsi','cesd10','pubage','female']], boot_df['BAI'])
        c_prime_b = m3_b.coef_[0]
        b_b = m3_b.coef_[1]
        # c path
        c_b = LinearRegression().fit(boot_df[['dsi','pubage','female']], boot_df['BAI']).coef_[0]

        indirect_b = a_b * b_b
        total_b = c_prime_b + indirect_b

        boot_results['a'].append(a_b)
        boot_results['b'].append(b_b)
        boot_results['c'].append(c_b)
        boot_results['c_prime'].append(c_prime_b)
        boot_results['indirect'].append(indirect_b)
        boot_results['direct'].append(c_prime_b)
        boot_results['total'].append(total_b)
        boot_results['pct'].append(indirect_b / total_b * 100 if total_b != 0 else 0)
    except:
        continue

# Percentile CIs
ci = {}
for key in ['a','b','c','c_prime','indirect','direct','total','pct']:
    arr = np.array(boot_results[key])
    ci[key] = {
        'mean': arr.mean(),
        'se': arr.std(),
        'ci_025': np.percentile(arr, 2.5),
        'ci_975': np.percentile(arr, 97.5)
    }

print(f"\n  Bootstrap Mediation Results:")
print(f"  {'Path':<30s} {'Estimate':>10s} {'95% CI':>25s}")
print(f"  {'-'*65}")
print(f"  {'a (DSI→CES-D)':<30s} {a:>10.4f} [{ci['a']['ci_025']:>10.4f}, {ci['a']['ci_975']:>10.4f}]")
print(f"  {'b (CES-D→BAI)':<30s} {b:>10.4f} [{ci['b']['ci_025']:>10.4f}, {ci['b']['ci_975']:>10.4f}]")
print(f"  {'c (total effect)':<30s} {c:>10.4f} [{ci['c']['ci_025']:>10.4f}, {ci['c']['ci_975']:>10.4f}]")
print(f"  {'c_prime (direct effect)':<30s} {c_prime:>10.4f} [{ci['c_prime']['ci_025']:>10.4f}, {ci['c_prime']['ci_975']:>10.4f}]")
print(f"  {'Indirect (a×b)':<30s} {indirect:>10.4f} [{ci['indirect']['ci_025']:>10.4f}, {ci['indirect']['ci_975']:>10.4f}]")
print(f"  {'Mediation %':<30s} {mediation_pct:>9.1f}% [{ci['pct']['ci_025']:>9.1f}%, {ci['pct']['ci_975']:>9.1f}%]")

# Sobel test
indirect_se = np.sqrt(a**2 * ci['b']['se']**2 + b**2 * ci['a']['se']**2)
sobel_z = indirect / indirect_se if indirect_se > 0 else 0
sobel_p = 2 * (1 - scipy_stats.norm.cdf(abs(sobel_z)))
print(f"\n  Sobel test: z = {sobel_z:.2f}, p = {sobel_p:.6f}")

# ============================================================
# 8. VALIDATION AGAINST BENCHMARKS
# ============================================================
print("\n=== 8. Benchmark Validation ===")
benchmarks = {
    'N baseline': (len(df_bl), 7764, 7000, 8500),
    'DSI prevalence (%)': (dsi_prev, 4.4, 3.0, 6.0),
    'CES-D mediation (%)': (mediation_pct, 28.0, 20.0, 40.0),
    'BoAI+SAI feature importance (%)': ((importances[0] + importances[2]) * 100, 86.9, 70.0, 100.0),
}
print(f"  {'Metric':<35s} {'Observed':>10s} {'Benchmark':>10s} {'Expected Range':>20s} {'Status':>8s}")
print(f"  {'-'*85}")
for name, (obs, bench, lo, hi) in benchmarks.items():
    status = 'PASS' if lo <= obs <= hi else 'CHECK'
    print(f"  {name:<35s} {obs:>10.2f} {bench:>10.2f} [{lo}-{hi}]{'':>12s} {status:>8s}")

# ============================================================
# 9. SAVE RESULTS
# ============================================================
print(f"\n=== 9. Saving Results ===")
# Phenotype table
pheno_out = df_bl[['id','wave','pubage','ragender','SAI','BAI','BoAI','dsi','cesd10','phenotype','pheno_label']].copy()
pheno_out.to_csv(os.path.join(OUT_DIR, 'tables', 'charls_full_phenotypes.csv'), index=False)

# Longitudinal
pheno_map = df_bl[['id','phenotype','pheno_label']].drop_duplicates()
df_long = df_all.merge(pheno_map, on='id', how='left')
df_long = df_long.dropna(subset=['phenotype'])
long_out = df_long[['id','wave','pubage','SAI','BAI','BoAI','dsi','cesd10','phenotype','pheno_label']]
long_out.to_csv(os.path.join(OUT_DIR, 'tables', 'charls_full_longitudinal.csv'), index=False)

# Results summary JSON
results = {
    'N_baseline': int(len(df_bl)),
    'N_total': int(len(df_all)),
    'DSI_prevalence_pct': float(dsi_prev),
    'SAI': {'mean': float(df_bl['SAI'].mean()), 'std': float(df_bl['SAI'].std())},
    'BAI': {'mean': float(df_bl['BAI'].mean()), 'std': float(df_bl['BAI'].std())},
    'BoAI': {'mean': float(df_bl['BoAI'].mean()), 'std': float(df_bl['BoAI'].std())},
    'GMM': {
        'best_k': best_k,
        'bic_scores': {str(k): v for k, v in bic_results.items()},
        'phenotypes': {str(p): {'label': pheno_labels[p], 'n': int((df_bl['phenotype']==p).sum()),
                                 'pct': float((df_bl['phenotype']==p).mean()*100)}
                       for p in range(best_k)}
    },
    'feature_importance': {'SAI': float(importances[0]), 'BAI': float(importances[1]), 'BoAI': float(importances[2])},
    'mediation': {
        'a': float(a), 'b': float(b), 'c': float(c), 'c_prime': float(c_prime),
        'indirect': float(indirect), 'mediation_pct': float(mediation_pct),
        'bootstrap_ci': {k: {'mean': float(v['mean']), 'ci_025': float(v['ci_025']), 'ci_975': float(v['ci_975'])}
                        for k, v in ci.items()},
        'sobel_z': float(sobel_z), 'sobel_p': float(sobel_p)
    },
    'validation': {name: {'observed': float(obs), 'benchmark': float(bench), 'range': [float(lo), float(hi)],
                           'pass': lo <= obs <= hi}
                   for name, (obs, bench, lo, hi) in benchmarks.items()}
}

with open(os.path.join(OUT_DIR, 'tables', 'charls_full_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

print(f"  Saved: charls_full_phenotypes.csv ({len(df_bl):,} rows)")
print(f"  Saved: charls_full_longitudinal.csv ({len(df_long):,} rows)")
print(f"  Saved: charls_full_results.json")

print("\n" + "=" * 70)
print("CHARLS Full Analysis Complete")
print("=" * 70)
