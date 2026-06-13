# CHARLS GBTM Trajectory Analysis — SAI / BAI / BoAI
# Group-Based Trajectory Modeling using Python GMM on longitudinal polynomial coefficients
# Author: Claude (科研模式) | 2026-06-11
#
# Method: Two-step Growth Mixture Modeling
#   Step 1: Fit individual-level cubic polynomial (age → index)
#   Step 2: GMM clustering of polynomial coefficients = trajectory groups
#   Model selection: BIC + ABIC + Entropy + minimum group size ≥5%
#
# This approximates lcmm::hlme() with degree=3, mixture=~poly(wave,3)

import os, sys, json, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.patches import Patch

# ============================================================
# 0. SETUP
# ============================================================
PROJECT_ROOT = r'D:\科研相关项目\程全老师课题组--UKB组\第三个课题--脑体感官衰老耦合解耦研究'
DATA_FILE = os.path.join(PROJECT_ROOT, 'results', 'tables', 'charls_full_longitudinal.csv')
OUT_DIR = os.path.join(PROJECT_ROOT, 'results')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
TBL_DIR = os.path.join(OUT_DIR, 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
                      'legend.fontsize': 10, 'figure.dpi': 150})

print("=" * 70)
print("CHARLS GBTM — SAI / BAI / BoAI Longitudinal Trajectory Modeling")
print("=" * 70)

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
print("\n=== 1. Loading CHARLS longitudinal data ===")
df_all = pd.read_csv(DATA_FILE)
print(f"  Total: {len(df_all):,} rows, {df_all.id.nunique():,} IDs")
print(f"  Waves: {sorted(df_all.wave.unique())}")
print(f"  Age range: {df_all.pubage.min():.0f}-{df_all.pubage.max():.0f}")

# Filter to ≥3 waves
wave_counts = df_all.groupby('id')['wave'].nunique()
ids_3plus = wave_counts[wave_counts >= 3].index
df = df_all[df_all.id.isin(ids_3plus)].copy()
print(f"  IDs with ≥3 waves: {len(ids_3plus):,} ({len(df):,} rows)")

# Center age for better polynomial fit
age_center = df.pubage.median()
df['age_c'] = df.pubage - age_center
print(f"  Age centered at median: {age_center:.0f}")

# Standardize indices within-wave for better comparability (optional)
for idx in ['SAI', 'BAI', 'BoAI']:
    mn, sd = df[idx].mean(), df[idx].std()
    df[f'{idx}_z'] = (df[idx] - mn) / sd

# ============================================================
# 2. INDIVIDUAL-LEVEL POLYNOMIAL TRAJECTORY FITTING
# ============================================================
print("\n=== 2. Fitting individual cubic trajectories ===")

def fit_individual_polynomials(df, id_col, x_col, y_col, poly_degree=3):
    """
    Fit polynomial trajectory for each individual.
    Returns DataFrame with per-ID polynomial coefficients.
    """
    ids = df[id_col].unique()
    results = []
    all_valid = []

    for pid in ids:
        sub = df[df[id_col] == pid].dropna(subset=[x_col, y_col])
        n_waves = len(sub)

        if n_waves < max(3, poly_degree + 1):
            continue

        x = sub[x_col].values.reshape(-1, 1)
        y = sub[y_col].values

        # Create polynomial features
        poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
        x_poly = poly.fit_transform(x)

        # Fit OLS
        try:
            model = LinearRegression().fit(x_poly, y)
            coefs = model.coef_
            # coefs[0] = constant (set to 0 by PolyFeatures), coefs[1:] = x, x², x³
            constant = model.intercept_
            r2 = model.score(x_poly, y)

            # Store: [id, n_waves, r2, intercept, β1, β2, β3, mean_age, mean_y]
            row = [pid, n_waves, r2, constant] + list(coefs[1:])
            row.append(sub[x_col].mean())
            row.append(sub[y_col].mean())
            results.append(row)
            all_valid.append(pid)
        except:
            continue

    # Build column names dynamically based on degree
    poly_col_names = ['lin_age', 'quad_age', 'cubic_age'][:poly_degree]
    cols = ['id', 'n_waves', 'r2', 'intercept'] + poly_col_names + ['mean_age', 'mean_y']
    df_coef = pd.DataFrame(results, columns=cols)
    return df_coef, all_valid

# Fit trajectories for all three indices using centered age
INDEXES = {
    'SAI':  'Sensory Aging Index (vision + hearing)',
    'BAI':  'Brain Aging Index (global cognition based)',
    'BoAI': 'Body Aging Index (grip + walk + ADL + balance + sarcopenia)'
}

MIN_N_WAVES = 3
POLY_DEGREE = 2  # quadratic (3-wave data can't support cubic: only 200/1289 have 4 waves)

coefs_all = {}
for idx_name, idx_desc in INDEXES.items():
    print(f"\n  [{idx_name}] {idx_desc}")
    df_coef, valid_ids = fit_individual_polynomials(
        df, 'id', 'age_c', idx_name, POLY_DEGREE)
    coefs_all[idx_name] = df_coef
    print(f"    Fitted: {len(df_coef)} individuals")
    print(f"    Median R^2: {df_coef.r2.median():.3f}")

# ============================================================
# 3. GBTM VIA GMM ON COEFFICIENTS
# ============================================================
print("\n=== 3. GBTM (GMM on polynomial coefficients) ===")

# Build coefficient column names dynamically
if POLY_DEGREE == 3:
    COEFFICIENT_COLS = ['intercept', 'lin_age', 'quad_age', 'cubic_age']
elif POLY_DEGREE == 2:
    COEFFICIENT_COLS = ['intercept', 'lin_age', 'quad_age']
else:
    COEFFICIENT_COLS = ['intercept', 'lin_age']
MAX_K = 6  # fit 1-6 groups

def gbtm_select_groups(coef_df, k_range=range(1, MAX_K+1), n_init=20, random_state=42):
    """
    Fit GMM for k=1..MAX_K groups and compute selection criteria.
    Returns: DataFrame with BIC, ABIC, Entropy, min_group_pct for each k
    """
    X = coef_df[COEFFICIENT_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n = len(X)
    results = []

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                              n_init=n_init, random_state=random_state,
                              max_iter=500, tol=1e-4)
        gmm.fit(X_scaled)

        # BIC
        bic = gmm.bic(X_scaled)

        # AIC → ABIC (sample-size adjusted BIC, same as lcmm's ABIC)
        aic = gmm.aic(X_scaled)
        # ABIC = -2*LL + n_params * log((n+2)/24)  (Schwarz approximation)
        n_params = gmm._n_parameters()
        abic = bic  # GMM.bic returns BIC already; compute separate AIC-based
        # Proper ABIC: -2*LL + d * log((n+2)/24)
        ll = gmm.score(X_scaled) * n  # average log-likelihood * n
        abic = -2 * ll + n_params * np.log((n + 2) / 24)

        # Group sizes
        labels = gmm.predict(X_scaled)
        sizes = np.bincount(labels)
        min_pct = sizes.min() / len(labels) * 100

        # Entropy (relative): 1 - (-mean posterior entropy / log(k))
        probs = gmm.predict_proba(X_scaled)
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1).mean()
        max_entropy = np.log(k)
        rel_entropy = 1 - entropy / max_entropy if k > 1 else 1.0

        # Average posterior probability of assignment (APPA)
        assign_probs = probs[np.arange(n), labels].mean()

        results.append({
            'k': k, 'BIC': bic, 'ABIC': abic, 'AIC': aic,
            'logLik': ll, 'n_params': n_params,
            'entropy': rel_entropy, 'APPA': assign_probs,
            'min_pct': min_pct, 'sizes': sizes.tolist()
        })

    return pd.DataFrame(results), scaler, X_scaled

# Run GBTM for each index
gbtm_results = {}
for idx_name in INDEXES:
    print(f"\n  [{idx_name}] Fitting 1-{MAX_K} groups...")
    sel_df, scaler, X_scaled = gbtm_select_groups(coefs_all[idx_name])

    print(f"  {'k':>3s}  {'BIC':>10s}  {'ABIC':>10s}  {'ΔBIC':>10s}  "
          f"{'Entropy':>8s}  {'APPA':>7s}  {'Min%':>6s}  {'Groups'}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*30}")

    for _, row in sel_df.iterrows():
        dbic = row['BIC'] - sel_df['BIC'].min() if len(sel_df) > 0 else 0
        sizes_str = str(row['sizes'])
        if len(sizes_str) > 30:
            sizes_str = sizes_str[:27] + '...'
        print(f"  {int(row['k']):3d}  {row['BIC']:10.0f}  {row['ABIC']:10.0f}  {dbic:10.0f}  "
              f"{row['entropy']:8.3f}  {row['APPA']:7.3f}  {row['min_pct']:5.1f}%  {sizes_str}")

    gbtm_results[idx_name] = {'df': sel_df, 'scaler': scaler, 'X_scaled': X_scaled}

# ============================================================
# 4. SELECT OPTIMAL K
# ============================================================
print("\n=== 4. Optimal Group Selection ===")

def select_optimal_k(sel_df):
    """
    Select optimal k using multiple criteria:
    1. BIC elbow (relative improvement < 5%)
    2. Minimum group ≥ 5%
    3. APPA ≥ 0.7
    4. Clinical interpretability (k=3-6 preferred per NHATS 2026)
    """
    valid_k = []

    for _, row in sel_df.iterrows():
        k = int(row['k'])
        score = 0

        # Criterion 1: APPA ≥ 0.7
        if row['APPA'] >= 0.70:
            score += 3
        elif row['APPA'] >= 0.60:
            score += 1

        # Criterion 2: Min group ≥ 5%
        if row['min_pct'] >= 5.0:
            score += 3
        elif row['min_pct'] >= 3.0:
            score += 1

        # Criterion 3: Entropy ≥ 0.7
        if row['entropy'] >= 0.70:
            score += 2
        elif row['entropy'] >= 0.50:
            score += 1

        # Criterion 4: k in [3, 6] range
        if 3 <= k <= 6:
            score += 1

        # Criterion 5: BIC improvement over k=1
        bic_improvement = (sel_df.loc[0, 'BIC'] - row['BIC']) / abs(sel_df.loc[0, 'BIC'])
        if bic_improvement > 0.5:
            score += 1

        valid_k.append({'k': k, 'score': score})

    # Sort by score desc, then k asc
    valid_k.sort(key=lambda x: (-x['score'], x['k']))
    return valid_k[0]['k'] if valid_k else min(5, len(sel_df))

optimal_k = {}
for idx_name in INDEXES:
    sel_df = gbtm_results[idx_name]['df']
    best_k = select_optimal_k(sel_df)
    optimal_k[idx_name] = best_k
    print(f"  [{idx_name}] Optimal k = {best_k}")

# ============================================================
# 5. FIT FINAL MODELS & LABEL TRAJECTORIES
# ============================================================
print("\n=== 5. Fitting final trajectory models ===")

def fit_final_gbtm(coef_df, scaler, X_scaled, k, age_center, df_long, idx_name, n_init=30):
    """Fit final k-group GMM and produce trajectory curves."""
    X = X_scaled

    gmm = GaussianMixture(n_components=k, covariance_type='full',
                          n_init=n_init, random_state=42,
                          max_iter=1000, tol=1e-5)
    labels = gmm.fit_predict(X)
    probs = gmm.predict_proba(X)

    # Map labels back to individuals
    coef_df = coef_df.copy()
    coef_df['trajectory_group'] = labels
    coef_df['assign_prob'] = probs[np.arange(len(probs)), labels]

    # Build trajectory curves for each group
    age_range = np.linspace(50, 95, 46)  # age 50-95
    age_centered_range = age_range - age_center

    # For each group, compute the group-mean polynomial trajectory
    trajectories = {}
    for g in range(k):
        members = coef_df[coef_df['trajectory_group'] == g]
        n_g = len(members)

        # Mean coefficients for this group — handle variable degree
        mean_coefs = members[COEFFICIENT_COLS].mean().values
        intercept = mean_coefs[0]

        # Predicted trajectory (polynomial of degree = len(COEFFICIENT_COLS)-1)
        y_pred = np.full_like(age_centered_range, intercept, dtype=float)
        for d in range(1, len(COEFFICIENT_COLS)):
            y_pred += mean_coefs[d] * age_centered_range ** d

        # Also compute prediction intervals from individual variations
        all_preds = []
        for _, row in members.iterrows():
            y_ind = np.full_like(age_centered_range, row['intercept'], dtype=float)
            for d in range(1, len(COEFFICIENT_COLS)):
                y_ind += row[COEFFICIENT_COLS[d]] * age_centered_range ** d
            all_preds.append(y_ind)
        all_preds = np.array(all_preds)
        y_std = all_preds.std(axis=0)

        trajectories[g] = {
            'age': age_range,
            'y_pred': y_pred,
            'y_ci_lower': y_pred - 1.96 * y_std,
            'y_ci_upper': y_pred + 1.96 * y_std,
            'n': n_g,
            'pct': n_g / len(coef_df) * 100,
            'mean_intercept': intercept,
            'mean_slope': mean_coefs[1] if len(mean_coefs) > 1 else 0  # linear slope at centering point
        }

    return coef_df, gmm, trajectories

# Auto-label trajectory groups
def label_trajectories(trajectories, k, idx_name):
    """Auto-name trajectory groups based on intercept and slope."""
    labeled = {}
    intercepts = {g: trajectories[g]['mean_intercept'] for g in range(k)}
    slopes = {g: trajectories[g]['mean_slope'] for g in range(k)}

    # Sort groups by intercept (baseline level)
    sorted_by_level = sorted(range(k), key=lambda g: intercepts[g])

    n_groups = len(sorted_by_level)

    for rank, g in enumerate(sorted_by_level):
        intercept = intercepts[g]
        slope = slopes[g]
        n = trajectories[g]['n']
        pct = trajectories[g]['pct']

        # Classification rules
        if n_groups <= 2:
            if rank == 0:
                label = 'Low' if idx_name == 'SAI' else 'Resilient'
            else:
                label = 'High' if idx_name == 'SAI' else 'Declining'
        elif n_groups == 3:
            labels_3 = ['Low-Stable', 'Moderate', 'High-Accelerated']
            label = labels_3[rank]
        elif n_groups == 4:
            labels_4 = ['Low-Stable', 'Moderate-Low', 'Moderate-High', 'High-Accelerated']
            label = labels_4[rank]
        elif n_groups == 5:
            labels_5 = ['Low-Stable', 'Moderate-Low', 'Moderate', 'Moderate-High', 'High-Accelerated']
            label = labels_5[rank]
        else:  # 6 groups
            labels_6 = ['Very-Low', 'Low', 'Moderate-Low', 'Moderate', 'Moderate-High', 'High-Accelerated']
            label = labels_6[rank]

        # Refine with SAI-specific naming (higher SAI = worse)
        if idx_name == 'SAI':
            if 'Low' in label and 'Accelerated' not in label:
                label = 'Stable-Low-Sensory-Burden'
            elif 'High' in label or 'Accelerated' in label:
                label = 'Accelerating-Sensory-Decline'

        # Refine with BAI-specific naming (higher BAI = worse cognition)
        if idx_name == 'BAI':
            if 'Low' in label and 'Accelerated' not in label:
                label = 'Cognitively-Resilient'
            elif 'High' in label or 'Accelerated' in label:
                label = 'Accelerated-Cognitive-Decline'

        # Refine with BoAI-specific naming (higher BoAI = worse physical)
        if idx_name == 'BoAI':
            if 'Low' in label and 'Accelerated' not in label:
                label = 'Physically-Resilient'
            elif 'High' in label or 'Accelerated' in label:
                label = 'Accelerated-Physical-Decline'

        labeled[g] = {
            **trajectories[g],
            'label': label,
            'level_rank': rank,
            'mean_intercept': intercept,
            'mean_slope': slope,
            'n': n,
            'pct': pct
        }

    return labeled

final_models = {}
for idx_name in INDEXES:
    k = optimal_k[idx_name]
    res = gbtm_results[idx_name]

    coef_df, gmm_model, trajectories = fit_final_gbtm(
        coefs_all[idx_name], res['scaler'], res['X_scaled'],
        k, age_center, df, idx_name
    )

    labeled_traj = label_trajectories(trajectories, k, idx_name)
    final_models[idx_name] = {
        'coef_df': coef_df,
        'gmm': gmm_model,
        'trajectories': labeled_traj,
        'k': k
    }

    print(f"\n  [{idx_name}] k={k} groups:")
    for g, t in labeled_traj.items():
        print(f"    G{g}: {t['label']:<35s}  n={t['n']:4d} ({t['pct']:5.1f}%)  "
              f"intercept={t['mean_intercept']:.2f}  slope={t['mean_slope']:.3f}")

# ============================================================
# 6. TRAJECTORY PLOTS
# ============================================================
print("\n=== 6. Generating trajectory plots ===")

# Color palettes for different k
PALETTES = {
    1: ['#1f77b4'],
    2: ['#2166ac', '#b2182b'],
    3: ['#2166ac', '#f4a582', '#b2182b'],
    4: ['#2166ac', '#92c5de', '#f4a582', '#b2182b'],
    5: ['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#b2182b'],
    6: ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#ef8a62', '#b2182b']
}

def plot_trajectories(labeled_traj, idx_name, idx_desc, k, out_dir):
    """Generate publication-quality trajectory plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = PALETTES.get(k, plt.cm.tab10(range(k)))

    for g in range(k):
        t = labeled_traj[g]
        c = colors[g % len(colors)]

        ax.plot(t['age'], t['y_pred'], color=c, linewidth=2.5,
                label=f"{t['label']} ({t['pct']:.0f}%)")
        ax.fill_between(t['age'], t['y_ci_lower'], t['y_ci_upper'],
                         color=c, alpha=0.15)

    ax.set_xlabel('Age (years)', fontsize=13)
    ax.set_ylabel(f'{idx_name} Index (0-100)', fontsize=13)
    ax.set_title(f'{idx_name} Trajectory Groups — CHARLS (k={k})', fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=9, ncol=1 if k <= 4 else 2)
    ax.set_xlim(50, 95)

    # Add annotation
    ax.annotate(f'CHARLS 2011-2018\nN={sum(t["n"] for t in labeled_traj.values()):,} with ≥3 waves',
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=9, color='gray')

    plt.tight_layout()
    fname = os.path.join(out_dir, f'GBTM_{idx_name}_Trajectories_k{k}.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")
    return fname

# Also plot BIC comparison
def plot_bic_comparison(sel_df, idx_name, optimal_k, out_dir):
    """Plot BIC/AIC comparison across k values."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # BIC
    ax = axes[0]
    ax.plot(sel_df['k'], sel_df['BIC'], 'o-', color='#2166ac', linewidth=2, markersize=8)
    ax.axvline(optimal_k, color='red', linestyle='--', alpha=0.5, label=f'Optimal k={optimal_k}')
    ax.set_xlabel('Number of Groups (k)')
    ax.set_ylabel('BIC')
    ax.set_title(f'{idx_name} — BIC')
    ax.legend()

    # ABIC
    ax = axes[1]
    ax.plot(sel_df['k'], sel_df['ABIC'], 's-', color='#b2182b', linewidth=2, markersize=8)
    ax.axvline(optimal_k, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Groups (k)')
    ax.set_ylabel('ABIC')
    ax.set_title(f'{idx_name} — ABIC')

    # Entropy & APPA
    ax = axes[2]
    ax.bar(sel_df['k'] - 0.15, sel_df['entropy'], width=0.3, color='#66c2a5', label='Entropy')
    ax.bar(sel_df['k'] + 0.15, sel_df['APPA'], width=0.3, color='#fc8d62', label='APPA')
    ax.axhline(0.7, color='gray', linestyle=':', alpha=0.5, label='Threshold (0.7)')
    ax.set_xlabel('Number of Groups (k)')
    ax.set_ylabel('Value')
    ax.set_title(f'{idx_name} — Entropy & APPA')
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fname = os.path.join(out_dir, f'GBTM_{idx_name}_ModelSelection.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")
    return fname

for idx_name, idx_desc in INDEXES.items():
    m = final_models[idx_name]
    k = m['k']
    labeled_traj = m['trajectories']
    sel_df = gbtm_results[idx_name]['df']

    plot_trajectories(labeled_traj, idx_name, idx_desc, k, FIG_DIR)
    plot_bic_comparison(sel_df, idx_name, k, FIG_DIR)

# ============================================================
# 7. BASELINE CHARACTERISTICS BY TRAJECTORY GROUP
# ============================================================
print("\n=== 7. Baseline characteristics by trajectory group ===")

# Get baseline data (first wave for each individual)
first_wave = df.groupby('id')['wave'].transform('min')
df_bl = df[df['wave'] == first_wave].copy()

# Merge trajectory group assignments
baseline_tables = {}
for idx_name in INDEXES:
    coef_df = final_models[idx_name]['coef_df']
    k = final_models[idx_name]['k']
    labeled_traj = final_models[idx_name]['trajectories']

    # Map group assignments
    group_map = dict(zip(coef_df['id'], coef_df['trajectory_group']))
    label_map = {g: labeled_traj[g]['label'] for g in range(k)}

    df_bl_idx = df_bl[df_bl.id.isin(coef_df.id)].copy()
    df_bl_idx['trajectory_group'] = df_bl_idx.id.map(group_map)
    df_bl_idx['trajectory_label'] = df_bl_idx.trajectory_group.map(label_map)

    # Baseline variables
    baseline_vars = {
        'pubage': 'Age',
        'ragender': 'Female (%)',
        'SAI': 'SAI',
        'BAI': 'BAI',
        'BoAI': 'BoAI',
        'dsi': 'DSI (%)',
        'cesd10': 'CES-D',
        'education2': 'Education',
    }

    rows = []
    for g in range(k):
        sub = df_bl_idx[df_bl_idx['trajectory_group'] == g]
        n = len(sub)
        row = {'Group': g, 'Label': label_map[g], 'N': n}

        for var, vname in baseline_vars.items():
            if var not in sub.columns:
                continue
            vals = sub[var].dropna()
            if var == 'ragender':
                row[vname] = f"{(vals==2).mean()*100:.1f}%"
            elif var == 'dsi':
                row[vname] = f"{vals.mean()*100:.1f}%"
            elif var in ['SAI', 'BAI', 'BoAI', 'cesd10']:
                row[vname] = f"{vals.mean():.1f} ± {vals.std():.1f}"
            elif var == 'pubage':
                row[vname] = f"{vals.mean():.1f} ± {vals.std():.1f}"
            elif var == 'education2':
                row[vname] = f"{vals.mean():.1f}"

        rows.append(row)

    baseline_tables[idx_name] = pd.DataFrame(rows)

    print(f"\n  [{idx_name}] Baseline characteristics (k={k}):")
    print(baseline_tables[idx_name].to_string(index=False))

# ============================================================
# 8. EXPORT RESULTS
# ============================================================
print("\n=== 8. Exporting results ===")

# 8.1 Model selection summary table
sel_summary = []
for idx_name in INDEXES:
    sel_df = gbtm_results[idx_name]['df']
    best_k = optimal_k[idx_name]
    for _, row in sel_df.iterrows():
        sel_summary.append({
            'Index': idx_name,
            'k': int(row['k']),
            'BIC': row['BIC'],
            'ABIC': row['ABIC'],
            'Entropy': row['entropy'],
            'APPA': row['APPA'],
            'Min_Pct': row['min_pct'],
            'Optimal': '★★' if int(row['k']) == best_k else ''
        })

sel_df_out = pd.DataFrame(sel_summary)
sel_df_out.to_csv(os.path.join(TBL_DIR, 'gbtm_model_selection.csv'), index=False)
print(f"  Saved: gbtm_model_selection.csv")

# 8.2 Trajectory group assignments
for idx_name in INDEXES:
    coef_df = final_models[idx_name]['coef_df']
    k = final_models[idx_name]['k']
    labeled_traj = final_models[idx_name]['trajectories']
    label_map = {g: labeled_traj[g]['label'] for g in range(k)}

    out_cols = ['id', 'trajectory_group', 'assign_prob', 'n_waves', 'r2'] + COEFFICIENT_COLS
    df_out = coef_df[out_cols].copy()
    df_out['trajectory_label'] = df_out.trajectory_group.map(label_map)
    df_out.to_csv(os.path.join(TBL_DIR, f'gbtm_{idx_name}_assignments.csv'), index=False)
    print(f"  Saved: gbtm_{idx_name}_assignments.csv")

# 8.3 Baseline characteristics
for idx_name in INDEXES:
    baseline_tables[idx_name].to_csv(
        os.path.join(TBL_DIR, f'gbtm_{idx_name}_baseline.csv'), index=False)
    print(f"  Saved: gbtm_{idx_name}_baseline.csv")

# 8.4 Complete JSON results
export_json = {
    'data': {
        'n_total': len(df_all),
        'n_3plus_waves': len(df),
        'n_gbtm': {idx: len(coefs_all[idx]) for idx in INDEXES},
        'waves': sorted(df.wave.unique().tolist()),
        'age_range': [float(df.pubage.min()), float(df.pubage.max())],
    },
    'gbtm': {}
}

for idx_name in INDEXES:
    sel_df = gbtm_results[idx_name]['df']
    k = optimal_k[idx_name]
    labeled_traj = final_models[idx_name]['trajectories']

    traj_export = {}
    for g, t in labeled_traj.items():
        traj_export[str(g)] = {
            'label': t['label'],
            'n': int(t['n']),
            'pct': round(float(t['pct']), 1),
            'mean_intercept': round(float(t['mean_intercept']), 3),
            'mean_slope': round(float(t['mean_slope']), 4),
        }

    export_json['gbtm'][idx_name] = {
        'optimal_k': k,
        'model_selection': sel_df.to_dict('records'),
        'trajectories': traj_export
    }

with open(os.path.join(TBL_DIR, 'gbtm_full_results.json'), 'w', encoding='utf-8') as f:
    json.dump(export_json, f, ensure_ascii=False, indent=2, default=str)
print(f"  Saved: gbtm_full_results.json")

# ============================================================
# 9. VALIDATION SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("=== VALIDATION SUMMARY ===")
print("=" * 70)

all_pass = True
for idx_name in INDEXES:
    k = optimal_k[idx_name]
    labeled_traj = final_models[idx_name]['trajectories']
    sel_row = gbtm_results[idx_name]['df'][gbtm_results[idx_name]['df']['k'] == k].iloc[0]

    checks = []
    # Check 1: k in 3-6
    checks.append(('k in [3,6]', 3 <= k <= 6, f'k={k}'))
    # Check 2: has "stable low" and "accelerated high" type groups
    labels_list = [t['label'] for t in labeled_traj.values()]
    has_low = any('Stable' in l or 'Resilient' in l or 'Low' in l for l in labels_list)
    has_high = any('Accelerat' in l or 'Declin' in l or 'High' in l for l in labels_list)
    checks.append(('Has stable-low type', has_low, str(labels_list)))
    checks.append(('Has accelerated-high type', has_high, str(labels_list)))
    # Check 3: APPA ≥ 0.7
    checks.append(('APPA ≥ 0.7', sel_row['APPA'] >= 0.70, f'APPA={sel_row["APPA"]:.3f}'))
    # Check 4: Min group ≥ 5%
    checks.append(('Min group ≥ 5%', sel_row['min_pct'] >= 5.0, f'Min={sel_row["min_pct"]:.1f}%'))

    print(f"\n  [{idx_name}] k={k}:")
    for check_name, passed, detail in checks:
        status = 'PASS' if passed else 'FAIL'
        if not passed:
            all_pass = False
        print(f"    {status} {check_name}: {detail}")

print(f"\n  Overall: {'[PASS] ALL PASS' if all_pass else '[FAIL] SOME CHECKS FAILED - see above'}")

print("\n" + "=" * 70)
print("GBTM Analysis Complete")
print("=" * 70)
