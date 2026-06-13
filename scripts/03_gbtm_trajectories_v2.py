# CHARLS GBTM Trajectory Analysis v2 — SAI / BAI / BoAI
# Group-Based Trajectory Modeling using Python GMM
# Author: Claude (科研模式) | 2026-06-12
#
# v2 Improvements over v1:
#   1. SAI: raw-timeseries clustering (prevents polynomial overfit on 3 discrete values)
#   2. Expanded sample: >=2 waves for BAI/BoAI (3,549 IDs); >=3 for SAI
#   3. Raw age: use pubage directly (not centered) for interpretable intercepts
#   4. Slope-aware labeling: incorporate trajectory direction into group names
#
# Method variants:
#   SAI:  Raw value + slope feature clustering (SAI is 0/50/100 discrete)
#   BAI/BoAI: Polynomial coefficient clustering (continuous indices)

import os, sys, json, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

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
print("CHARLS GBTM v2 — SAI / BAI / BoAI Longitudinal Trajectory Modeling")
print("=" * 70)

# ============================================================
# 1. LOAD & MERGE WITH RAW DATA FOR SAI
# ============================================================
print("\n=== 1. Loading CHARLS longitudinal data ===")
df_all = pd.read_csv(DATA_FILE)
print(f"  Total: {len(df_all):,} rows, {df_all.id.nunique():,} IDs")
print(f"  Waves: {sorted(df_all.wave.unique())}")
print(f"  Age range: {df_all.pubage.min():.0f}-{df_all.pubage.max():.0f}")

# Compute wave counts per ID
wave_counts = df_all.groupby('id')['wave'].nunique()

# ── SAI: keep >=3 waves (need enough points for trajectory) ──
ids_3plus = wave_counts[wave_counts >= 3].index
df_sai = df_all[df_all.id.isin(ids_3plus)].copy()
print(f"  SAI (>=3 waves): {len(ids_3plus):,} IDs ({len(df_sai):,} rows)")

# ── BAI/BoAI: >=2 waves (continuous, fewer points OK) ──
ids_2plus = wave_counts[wave_counts >= 2].index
df_cont = df_all[df_all.id.isin(ids_2plus)].copy()
print(f"  BAI/BoAI (>=2 waves): {len(ids_2plus):,} IDs ({len(df_cont):,} rows)")

# ============================================================
# 2. SAI — RAW TIME-SERIES CLUSTERING (discrete 0/50/100)
# ============================================================
print("\n=== 2. SAI: Raw timeseries clustering (discrete-aware) ===")

def extract_sai_features(df, min_waves=3):
    """
    For SAI (0/50/100 discrete), extract interpretable features
    instead of polynomial coefficients.
    """
    records = []
    for pid, grp in df.groupby('id'):
        grp = grp.sort_values('pubage')
        n_waves = len(grp)

        if n_waves < min_waves:
            continue

        sai_vals = grp['SAI'].values
        ages = grp['pubage'].values

        # Features capturing the trajectory shape
        record = {
            'id': pid,
            'n_waves': n_waves,
            'sai_first': sai_vals[0],            # baseline level
            'sai_last': sai_vals[-1],            # endpoint level
            'sai_max': sai_vals.max(),           # worst sensory status
            'sai_min': sai_vals.min(),           # best sensory status
            'sai_mean': sai_vals.mean(),         # average burden
            'age_first': ages[0],
            'age_last': ages[-1],
            'age_span': ages[-1] - ages[0],
        }

        # Slope: (sai_last - sai_first) / age_span * 10 ≈ per-decade change
        age_span = ages[-1] - ages[0]
        if age_span > 0:
            record['sai_slope_per_decade'] = (sai_vals[-1] - sai_vals[0]) / age_span * 10
        else:
            record['sai_slope_per_decade'] = 0

        # Whether SAI ever reached 50 or 100
        record['ever_impaired'] = int((sai_vals >= 50).any())
        record['ever_dsi'] = int((sai_vals == 100).any())

        # Whether SAI EVER INCREASED during follow-up ( = sensory decline)
        record['ever_worsened'] = int((np.diff(sai_vals) > 0).any())

        # Pattern: stable-0, stable-50, stable-100, increasing, decreasing, fluctuating
        if sai_vals.max() == sai_vals.min():
            if sai_vals[0] == 0:
                record['pattern'] = 'always-0'
            elif sai_vals[0] == 50:
                record['pattern'] = 'always-50'
            else:
                record['pattern'] = 'always-100'
        elif np.all(np.diff(sai_vals) >= 0):
            record['pattern'] = 'worsening'
        elif np.all(np.diff(sai_vals) <= 0):
            record['pattern'] = 'improving'
        else:
            record['pattern'] = 'fluctuating'

        records.append(record)

    return pd.DataFrame(records)

df_sai_feat = extract_sai_features(df_sai, min_waves=3)
print(f"  Extracted features for {len(df_sai_feat)} individuals")

# Clustering features for SAI
SAI_FEATURE_COLS = ['sai_first', 'sai_last', 'sai_max', 'sai_mean',
                     'sai_slope_per_decade', 'ever_impaired', 'ever_dsi', 'ever_worsened']

def gbtm_select_groups(X_raw, k_range, n_init=20, random_state=42):
    """Fit GMM for k=1..max_k and compute selection criteria."""
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    n = len(X)
    results = []

    for k in k_range:
        if k > n / 5:  # avoid over-parameterization
            continue

        gmm = GaussianMixture(n_components=k, covariance_type='full',
                              n_init=n_init, random_state=random_state,
                              max_iter=500, tol=1e-4)
        gmm.fit(X)

        bic = gmm.bic(X)
        ll = gmm.score(X) * n
        n_params = gmm._n_parameters()
        abic = -2 * ll + n_params * np.log((n + 2) / 24)

        labels = gmm.predict(X)
        sizes = np.bincount(labels)
        min_pct = sizes.min() / n * 100

        probs = gmm.predict_proba(X)
        entropy_raw = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1).mean()
        max_entropy = np.log(k)
        rel_entropy = 1 - entropy_raw / max_entropy if k > 1 else 1.0
        appa = probs[np.arange(n), labels].mean()

        results.append({
            'k': k, 'BIC': bic, 'ABIC': abic, 'AIC': gmm.aic(X),
            'logLik': ll, 'n_params': n_params,
            'entropy': rel_entropy, 'APPA': appa,
            'min_pct': min_pct, 'sizes': sizes.tolist()
        })

    return pd.DataFrame(results), scaler, X

print("\n  [SAI] Fitting 1-6 groups (raw timeseries features)...")
X_sai = df_sai_feat[SAI_FEATURE_COLS].values
sel_sai, scaler_sai, X_sai_scaled = gbtm_select_groups(X_sai, range(1, 7))

for _, row in sel_sai.iterrows():
    dbic = row['BIC'] - sel_sai['BIC'].min()
    sizes_str = str(row['sizes'])
    if len(sizes_str) > 40:
        sizes_str = sizes_str[:37] + '...'
    print(f"    k={int(row['k'])}  BIC={row['BIC']:10.0f}  ABIC={row['ABIC']:10.0f}  "
          f"dBIC={dbic:8.0f}  Ent={row['entropy']:.3f}  APPA={row['APPA']:.3f}  "
          f"Min={row['min_pct']:.1f}%  {sizes_str}")

# Select optimal k for SAI
def select_optimal_k(sel_df, min_pct=5.0, min_appa=0.70, prefer_3to6=True):
    """Score each k on multiple criteria."""
    scored = []
    for _, row in sel_df.iterrows():
        k = int(row['k'])
        score = 0
        if row['APPA'] >= min_appa: score += 3
        elif row['APPA'] >= 0.60: score += 1
        if row['min_pct'] >= min_pct: score += 3
        elif row['min_pct'] >= 3.0: score += 1
        if row['entropy'] >= 0.70: score += 2
        elif row['entropy'] >= 0.50: score += 1
        if prefer_3to6 and 3 <= k <= 6: score += 1
        bic_imp = (sel_df.iloc[0]['BIC'] - row['BIC']) / abs(sel_df.iloc[0]['BIC'])
        if bic_imp > 0.3: score += 1
        scored.append((k, score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[0][0]

k_sai = select_optimal_k(sel_sai)
print(f"\n  [SAI] Optimal k = {k_sai}")

# Fit final SAI model
gmm_sai = GaussianMixture(n_components=k_sai, covariance_type='full',
                           n_init=30, random_state=42, max_iter=1000)
labels_sai = gmm_sai.fit_predict(X_sai_scaled)
probs_sai = gmm_sai.predict_proba(X_sai_scaled)
df_sai_feat['trajectory_group'] = labels_sai
df_sai_feat['assign_prob'] = probs_sai[np.arange(len(probs_sai)), labels_sai]

# Build SAI trajectory curves from raw data
def build_sai_trajectories(df_long, df_assign, k):
    """Build group-mean SAI trajectories from raw longitudinal data."""
    group_map = dict(zip(df_assign['id'], df_assign['trajectory_group']))
    df_merged = df_long[df_long.id.isin(df_assign.id)].copy()
    df_merged['group'] = df_merged.id.map(group_map)

    age_bins = np.arange(50, 96, 3)  # 3-year bins

    trajectories = {}
    for g in range(k):
        members = df_assign[df_assign['trajectory_group'] == g]
        n_g = len(members)
        sub = df_merged[df_merged['group'] == g]

        # Group mean by age bin
        sub['age_bin'] = pd.cut(sub['pubage'], bins=age_bins, labels=age_bins[:-1] + 1.5)
        grp_mean = sub.groupby('age_bin', observed=False)['SAI'].agg(['mean', 'std', 'count'])
        grp_mean = grp_mean[grp_mean['count'] >= 5]  # at least 5 obs per bin
        ages = grp_mean.index.astype(float).values
        means = grp_mean['mean'].values
        stds = grp_mean['std'].values

        # Overall group stats
        sai_slope = members['sai_slope_per_decade'].mean()
        sai_start = members['sai_first'].mean()
        ever_worsened = members['ever_worsened'].mean() * 100

        trajectories[g] = {
            'age': ages,
            'y_pred': means,
            'y_ci_lower': means - 1.96 * stds / np.sqrt(grp_mean['count'].values),
            'y_ci_upper': means + 1.96 * stds / np.sqrt(grp_mean['count'].values),
            'n': n_g,
            'pct': n_g / len(df_assign) * 100,
            'mean_sai_start': sai_start,
            'mean_slope_per_decade': sai_slope,
            'ever_worsened_pct': ever_worsened,
        }
    return trajectories, df_merged

traj_sai, df_sai_merged = build_sai_trajectories(df_sai, df_sai_feat, k_sai)

# Label SAI groups based on level + slope
def label_sai_groups(trajectories, k):
    """Label SAI groups using both starting level and slope direction."""
    labeled = {}
    for g in range(k):
        t = trajectories[g]
        start = t['mean_sai_start']
        slope = t['mean_slope_per_decade']
        worsened = t['ever_worsened_pct']

        # SAI: 0=no impairment, 50=single impairment, 100=DSI
        if start < 10 and worsened < 20:
            label = 'Stable-No-Sensory-Impairment'
        elif start < 10 and worsened >= 20:
            label = 'Late-Onset-Sensory-Decline'
        elif start >= 40 and worsened >= 30:
            label = 'Progressive-Sensory-Decline'
        elif start >= 40 and slope < -5:
            label = 'Improving-Sensory-Status'
        elif start >= 80:
            label = 'Persistent-DSI-Burden'
        elif start >= 40:
            label = 'Stable-Moderate-Sensory-Burden'
        elif worsened >= 80:
            label = 'Late-Onset-Sensory-Decline'
        elif start >= 15:
            label = 'Emerging-Sensory-Decline'
        else:
            label = f'Trajectory-Group-{g}'

        labeled[g] = {**t, 'label': label}
    return labeled

labeled_sai = label_sai_groups(traj_sai, k_sai)
print(f"\n  [SAI] k={k_sai} groups:")
for g, t in labeled_sai.items():
    print(f"    G{g}: {t['label']:<40s}  n={t['n']:4d} ({t['pct']:5.1f}%)  "
          f"SAI0={t['mean_sai_start']:.0f}  slope/dec={t['mean_slope_per_decade']:+.1f}  "
          f"worsened={t['ever_worsened_pct']:.0f}%")

# ============================================================
# 3. BAI & BoAI — POLYNOMIAL COEFFICIENT CLUSTERING (v2: >=2 waves, raw age)
# ============================================================
print("\n=== 3. BAI/BoAI: Polynomial coefficient clustering ===")

POLY_DEGREE = 1  # linear only (>=2 waves insufficient for quadratic)
COEFFICIENT_COLS = ['intercept', 'lin_age']  # intercept=predicted at age 50, lin_age=change per decade

INDEXES_CONT = {
    'BAI':  'Brain Aging Index',
    'BoAI': 'Body Aging Index',
}

def fit_individual_polynomials_v2(df, y_col, min_waves=2, poly_degree=2):
    """Fit polynomial using SCALED age (age-50)/10 for numerical stability.
    Intercept = predicted value at age 50 (interpretable baseline).
    """
    results = []
    for pid, grp in df.groupby('id'):
        grp = grp.dropna(subset=['pubage', y_col])
        n_waves = len(grp)

        if n_waves < max(min_waves, poly_degree):
            continue

        # Scale age: (age-50)/10 → age 50→0, 60→1, 70→2, 80→3, 90→4
        # Prevents x^2 explosion while keeping interpretable intercept (age 50)
        x = ((grp['pubage'] - 50) / 10).values.reshape(-1, 1)
        y = grp[y_col].values

        poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
        x_poly = poly.fit_transform(x)

        try:
            model = LinearRegression().fit(x_poly, y)
            coefs = model.coef_
            intercept = model.intercept_
            r2 = model.score(x_poly, y)

            row = [pid, n_waves, r2, intercept] + list(coefs[1:])
            row.append(grp['pubage'].mean())
            row.append(grp[y_col].mean())
            results.append(row)
        except:
            continue

    poly_col_names = ['lin_age', 'quad_age', 'cubic_age'][:poly_degree]
    cols = ['id', 'n_waves', 'r2', 'intercept'] + poly_col_names + ['mean_age', 'mean_y']
    return pd.DataFrame(results, columns=cols)

# Build cont-coefficient feature matrix for BAI
coefs_cont = {}
for idx_name, idx_desc in INDEXES_CONT.items():
    print(f"\n  [{idx_name}] {idx_desc}")
    # Use >=2 waves for continuous indices
    df_use = df_cont if idx_name in INDEXES_CONT else df_sai
    df_coef = fit_individual_polynomials_v2(df_use, idx_name, min_waves=2, poly_degree=POLY_DEGREE)
    coefs_cont[idx_name] = df_coef
    print(f"    Fitted: {len(df_coef)} individuals (>=2 waves)")
    print(f"    Median R^2: {df_coef.r2.median():.3f}")
    print(f"    Intercept range: {df_coef.intercept.min():.1f} - {df_coef.intercept.max():.1f}")
    print(f"    Age range: {df_coef.mean_age.min():.0f} - {df_coef.mean_age.max():.0f}")

# GBTM for continuous indices
gbtm_cont = {}
for idx_name in INDEXES_CONT:
    print(f"\n  [{idx_name}] Fitting 1-6 groups...")
    df_coef = coefs_cont[idx_name]
    X = df_coef[COEFFICIENT_COLS].values
    sel_df, scaler, X_scaled = gbtm_select_groups(X, range(1, 7))

    for _, row in sel_df.iterrows():
        dbic = row['BIC'] - sel_df['BIC'].min()
        sizes_str = str(row['sizes'])
        if len(sizes_str) > 40:
            sizes_str = sizes_str[:37] + '...'
        print(f"    k={int(row['k'])}  BIC={row['BIC']:10.0f}  ABIC={row['ABIC']:10.0f}  "
              f"dBIC={dbic:8.0f}  Ent={row['entropy']:.3f}  APPA={row['APPA']:.3f}  "
              f"Min={row['min_pct']:.1f}%  {sizes_str}")

    gbtm_cont[idx_name] = {'df': sel_df, 'scaler': scaler, 'X_scaled': X_scaled}

# Select & fit
optimal_k_cont = {}
final_cont = {}

for idx_name in INDEXES_CONT:
    sel_df = gbtm_cont[idx_name]['df']
    scaler = gbtm_cont[idx_name]['scaler']
    X_scaled = gbtm_cont[idx_name]['X_scaled']
    df_coef = coefs_cont[idx_name]
    k = select_optimal_k(sel_df)
    optimal_k_cont[idx_name] = k

    # Fit final
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                          n_init=30, random_state=42, max_iter=1000)
    labels = gmm.fit_predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    df_coef = df_coef.copy()
    df_coef['trajectory_group'] = labels
    df_coef['assign_prob'] = probs[np.arange(len(probs)), labels]

    # Build trajectories using scaled age for prediction
    age_range = np.linspace(50, 95, 46)
    age_scaled_range = (age_range - 50) / 10  # same scaling as fitting

    trajectories = {}
    for g in range(k):
        members = df_coef[df_coef['trajectory_group'] == g]
        n_g = len(members)
        mean_coefs = members[COEFFICIENT_COLS].mean().values
        intercept = mean_coefs[0]  # predicted at age 50

        y_pred = np.full_like(age_scaled_range, intercept, dtype=float)
        for d in range(1, len(COEFFICIENT_COLS)):
            y_pred += mean_coefs[d] * age_scaled_range ** d

        # Individual variation for CI
        all_preds = []
        for _, row in members.iterrows():
            y_ind = np.full_like(age_scaled_range, row['intercept'], dtype=float)
            for d in range(1, len(COEFFICIENT_COLS)):
                y_ind += row[COEFFICIENT_COLS[d]] * age_scaled_range ** d
            all_preds.append(y_ind)
        all_preds = np.array(all_preds)
        y_std = all_preds.std(axis=0)

        # Slope per year: lin_age is change per decade of age → /10 for per-year
        slope_per_year = mean_coefs[1] / 10 if len(mean_coefs) > 1 else 0

        trajectories[g] = {
            'age': age_range,
            'y_pred': y_pred,
            'y_ci_lower': y_pred - 1.96 * y_std,
            'y_ci_upper': y_pred + 1.96 * y_std,
            'n': n_g,
            'pct': n_g / len(df_coef) * 100,
            'mean_intercept': intercept,   # predicted at age 50
            'slope_per_year': slope_per_year,
        }

    # Label based on intercept level + slope direction
    labeled = {}
    for g in range(k):
        t = trajectories[g]
        level = t['mean_intercept']  # at age 50
        slope = t['slope_per_year']  # per year of chronological age

        # Level categorization (based on index at age 50)
        if idx_name == 'BAI':
            if level < 45: level_str = 'Low-BAI'
            elif level < 55: level_str = 'Mid-BAI'
            else: level_str = 'High-BAI'
        else:  # BoAI
            if level < 50: level_str = 'Low-BoAI'
            elif level < 60: level_str = 'Mid-BoAI'
            else: level_str = 'High-BoAI'

        # Slope categorization (per year of chronological age)
        if abs(slope) < 0.08:
            slope_str = 'Stable'
        elif slope > 0.20:
            slope_str = 'Fast-Accelerating'
        elif slope > 0:
            slope_str = 'Slow-Accelerating'
        elif slope < -0.20:
            slope_str = 'Fast-Improving'
        else:
            slope_str = 'Slow-Improving'

        label = f'{level_str}-{slope_str}'

        labeled[g] = {**t, 'label': label,
                      'mean_intercept': level,
                      'slope_per_year': slope,
                      'n': t['n'], 'pct': t['pct']}

    final_cont[idx_name] = {
        'coef_df': df_coef,
        'gmm': gmm,
        'trajectories': labeled,
        'k': k
    }

    print(f"\n  [{idx_name}] k={k} groups:")
    for g, t in labeled.items():
        print(f"    G{g}: {t['label']:<30s}  n={t['n']:4d} ({t['pct']:5.1f}%)  "
              f"int@50={t['mean_intercept']:.1f}  slope={t['slope_per_year']:+7.3f}/yr")

# ============================================================
# 4. TRAJECTORY PLOTS
# ============================================================
print("\n=== 4. Generating trajectory plots ===")

PALETTES = {
    2: ['#2166ac', '#b2182b'],
    3: ['#2166ac', '#f4a582', '#b2182b'],
    4: ['#2166ac', '#92c5de', '#f4a582', '#b2182b'],
    5: ['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#b2182b'],
    6: ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#ef8a62', '#b2182b']
}

def plot_trajectories(labeled_traj, k, idx_name, out_dir, ylabel=None):
    """Publication-quality trajectory plot."""
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
    ax.set_ylabel(ylabel or f'{idx_name} Index', fontsize=13)
    ax.set_title(f'{idx_name} Trajectory Groups — CHARLS (k={k})', fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=9, ncol=1 if k <= 4 else 2)
    ax.set_xlim(50, 95)

    plt.tight_layout()
    fname = os.path.join(out_dir, f'GBTM_v2_{idx_name}_Trajectories_k{k}.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(fname)}")
    return fname

def plot_bic_comparison(sel_df, idx_name, optimal_k, out_dir):
    """BIC/AIC/Entropy comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(sel_df['k'], sel_df['BIC'], 'o-', color='#2166ac', linewidth=2, markersize=8)
    ax.axvline(optimal_k, color='red', linestyle='--', alpha=0.5, label=f'Optimal k={optimal_k}')
    ax.set_xlabel('Number of Groups (k)'); ax.set_ylabel('BIC')
    ax.set_title(f'{idx_name} — BIC'); ax.legend()

    ax = axes[1]
    ax.plot(sel_df['k'], sel_df['ABIC'], 's-', color='#b2182b', linewidth=2, markersize=8)
    ax.axvline(optimal_k, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Groups (k)'); ax.set_ylabel('ABIC')
    ax.set_title(f'{idx_name} — ABIC')

    ax = axes[2]
    ax.bar(sel_df['k'] - 0.15, sel_df['entropy'], width=0.3, color='#66c2a5', label='Entropy')
    ax.bar(sel_df['k'] + 0.15, sel_df['APPA'], width=0.3, color='#fc8d62', label='APPA')
    ax.axhline(0.7, color='gray', linestyle=':', alpha=0.5, label='Threshold')
    ax.set_xlabel('Number of Groups (k)'); ax.set_ylabel('Value')
    ax.set_title(f'{idx_name} — Entropy & APPA'); ax.legend(); ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fname = os.path.join(out_dir, f'GBTM_v2_{idx_name}_ModelSelection.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(fname)}")

# SAI plot
plot_trajectories(labeled_sai, k_sai, 'SAI', FIG_DIR,
                  ylabel='SAI (0-100, higher = worse)')
plot_bic_comparison(sel_sai, 'SAI', k_sai, FIG_DIR)

# BAI and BoAI plots
for idx_name in INDEXES_CONT:
    m = final_cont[idx_name]
    plot_trajectories(m['trajectories'], m['k'], idx_name, FIG_DIR,
                      ylabel=f'{idx_name} Index')
    plot_bic_comparison(gbtm_cont[idx_name]['df'], idx_name, m['k'], FIG_DIR)

# ============================================================
# 5. BASELINE CHARACTERISTICS
# ============================================================
print("\n=== 5. Baseline characteristics by trajectory group ===")

first_wave_all = df_all.groupby('id')['wave'].transform('min')
df_bl_all = df_all[df_all['wave'] == first_wave_all].copy()

def build_baseline_table(df_bl, df_assign, k, trajectories, idx_name):
    """Build baseline characteristics table."""
    group_map = dict(zip(df_assign['id'], df_assign['trajectory_group']))
    label_map = {g: trajectories[g]['label'] for g in range(k)}

    df_bl_idx = df_bl[df_bl.id.isin(df_assign.id)].copy()
    df_bl_idx['group'] = df_bl_idx.id.map(group_map)
    df_bl_idx['label'] = df_bl_idx.group.map(label_map)

    rows = []
    for g in range(k):
        sub = df_bl_idx[df_bl_idx['group'] == g]
        n = len(sub)
        row = {'Group': g, 'Label': trajectories[g]['label'], 'N': n}

        for var in ['pubage', 'SAI', 'BAI', 'BoAI', 'cesd10']:
            if var not in sub.columns: continue
            vals = sub[var].dropna()
            row[var] = f"{vals.mean():.1f} ({vals.std():.1f})"

        if 'dsi' in sub.columns:
            row['DSI%'] = f"{sub['dsi'].mean()*100:.1f}"
        if 'ragender' in sub.columns:
            row['Female%'] = f"{(sub['ragender']==2).mean()*100:.1f}"

        rows.append(row)

    return pd.DataFrame(rows)

baseline_tables = {}

# SAI baseline
baseline_tables['SAI'] = build_baseline_table(df_bl_all, df_sai_feat, k_sai, labeled_sai, 'SAI')
print(f"\n  [SAI] Baseline (k={k_sai}):")
print(baseline_tables['SAI'].to_string(index=False))

for idx_name in INDEXES_CONT:
    m = final_cont[idx_name]
    baseline_tables[idx_name] = build_baseline_table(df_bl_all, m['coef_df'], m['k'],
                                                      m['trajectories'], idx_name)
    print(f"\n  [{idx_name}] Baseline (k={m['k']}):")
    print(baseline_tables[idx_name].to_string(index=False))

# ============================================================
# 6. EXPORT
# ============================================================
print("\n=== 6. Exporting results ===")

# Model selection summary
sel_summary = []
# SAI
for _, row in sel_sai.iterrows():
    sel_summary.append({'Index': 'SAI', 'k': int(row['k']), 'BIC': row['BIC'],
                        'ABIC': row['ABIC'], 'Entropy': row['entropy'],
                        'APPA': row['APPA'], 'Min_Pct': row['min_pct'],
                        'Optimal': '★★' if int(row['k']) == k_sai else ''})
# BAI/BoAI
for idx_name in INDEXES_CONT:
    sel_df = gbtm_cont[idx_name]['df']
    bk = optimal_k_cont[idx_name]
    for _, row in sel_df.iterrows():
        sel_summary.append({'Index': idx_name, 'k': int(row['k']), 'BIC': row['BIC'],
                            'ABIC': row['ABIC'], 'Entropy': row['entropy'],
                            'APPA': row['APPA'], 'Min_Pct': row['min_pct'],
                            'Optimal': '★★' if int(row['k']) == bk else ''})

pd.DataFrame(sel_summary).to_csv(os.path.join(TBL_DIR, 'gbtm_v2_model_selection.csv'), index=False)

# Assignments
df_sai_feat[['id','trajectory_group','assign_prob','sai_first','sai_last',
              'sai_slope_per_decade','ever_worsened']].to_csv(
    os.path.join(TBL_DIR, 'gbtm_v2_SAI_assignments.csv'), index=False)

for idx_name in INDEXES_CONT:
    m = final_cont[idx_name]
    out_cols = ['id','trajectory_group','assign_prob','n_waves','r2'] + COEFFICIENT_COLS
    m['coef_df'][out_cols].to_csv(
        os.path.join(TBL_DIR, f'gbtm_v2_{idx_name}_assignments.csv'), index=False)

# Baseline
for idx_name, tbl in baseline_tables.items():
    tbl.to_csv(os.path.join(TBL_DIR, f'gbtm_v2_{idx_name}_baseline.csv'), index=False)

# JSON
export = {
    'version': 'v2',
    'date': '2026-06-12',
    'improvements': [
        'SAI: raw-timeseries features instead of polynomial (discrete-aware)',
        'BAI/BoAI: expanded to >=2 waves for larger sample',
        'Raw age used for interpretable intercepts',
        'Slope-aware labeling (intercept level + slope direction)',
    ],
    'gbtm': {
        'SAI': {'optimal_k': k_sai, 'model_selection': sel_sai.to_dict('records'),
                'trajectories': {str(g): {'label': t['label'], 'n': int(t['n']),
                    'pct': round(t['pct'], 1), 'sai_start': round(t['mean_sai_start'], 1),
                    'slope_per_decade': round(t['mean_slope_per_decade'], 1)}
                    for g, t in labeled_sai.items()}},
    }
}
for idx_name in INDEXES_CONT:
    m = final_cont[idx_name]
    sel_df = gbtm_cont[idx_name]['df']
    export['gbtm'][idx_name] = {
        'optimal_k': m['k'],
        'model_selection': sel_df.to_dict('records'),
        'trajectories': {str(g): {'label': t['label'], 'n': int(t['n']),
            'pct': round(t['pct'], 1), 'intercept_at_50': round(t['mean_intercept'], 1),
            'slope_per_year': round(t['slope_per_year'], 4)}
            for g, t in m['trajectories'].items()},
    }

with open(os.path.join(TBL_DIR, 'gbtm_v2_full_results.json'), 'w', encoding='utf-8') as f:
    json.dump(export, f, ensure_ascii=False, indent=2, default=str)

# ============================================================
# 7. VALIDATION
# ============================================================
print(f"\n{'='*70}")
print("=== VALIDATION SUMMARY ===")
print(f"{'='*70}")

# SAI validation
print(f"\n  [SAI] k={k_sai}:")
sai_checks = [
    ('k in [3,6]', 3 <= k_sai <= 6, f'k={k_sai}'),
    ('Has stable type',
     any('Stable' in t['label'] for t in labeled_sai.values()),
     [t['label'] for t in labeled_sai.values()]),
    ('Has decline/worsening type',
     any(w in str([t['label'] for t in labeled_sai.values()]) for w in ['Decline','Worsening','Progressive','Late-Onset']),
     [t['label'] for t in labeled_sai.values()]),
    ('APPA >= 0.7',
     sel_sai[sel_sai['k']==k_sai]['APPA'].iloc[0] >= 0.70,
     f"APPA={sel_sai[sel_sai['k']==k_sai]['APPA'].iloc[0]:.3f}"),
    ('Min group >= 5%',
     sel_sai[sel_sai['k']==k_sai]['min_pct'].iloc[0] >= 5.0,
     f"Min={sel_sai[sel_sai['k']==k_sai]['min_pct'].iloc[0]:.1f}%"),
]
for name, passed, detail in sai_checks:
    print(f"    {'[PASS]' if passed else '[FAIL]'} {name}: {detail}")

for idx_name in INDEXES_CONT:
    m = final_cont[idx_name]
    k = m['k']
    sel_row = gbtm_cont[idx_name]['df'][gbtm_cont[idx_name]['df']['k']==k].iloc[0]
    labels_list = [t['label'] for t in m['trajectories'].values()]

    print(f"\n  [{idx_name}] k={k}:")
    checks = [
        ('k in [3,6]', 3 <= k <= 6, f'k={k}'),
        ('Has stable type', any('Stable' in l for l in labels_list), str(labels_list)),
        ('Has accelerating/declining type',
         any(w in str(labels_list) for w in ['Accelerat','Declin','High','Improv']),
         str(labels_list)),
        ('APPA >= 0.7', sel_row['APPA'] >= 0.70, f"APPA={sel_row['APPA']:.3f}"),
        ('Min group >= 5%', sel_row['min_pct'] >= 5.0, f"Min={sel_row['min_pct']:.1f}%"),
    ]
    for name, passed, detail in checks:
        print(f"    {'[PASS]' if passed else '[FAIL]'} {name}: {detail}")

print(f"\n{'='*70}")
print("GBTM v2 Analysis Complete")
print(f"{'='*70}")
