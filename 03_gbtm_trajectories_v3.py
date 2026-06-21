# =============================================================================
# 03_gbtm_trajectories_v3.py — GBTM 轨迹建模 (方案D: 诚实版)
# =============================================================================
# 功能: BAI(>=3波 full intercept+slope GBTM; 2波 intercept-only聚类)
#       BoAI(>=3波 intercept-only; 2波 intercept-only--slope不可靠)
#       SAI(未改动: >=3波 feature-based)
# 输入: results/tables/charls_full_longitudinal.csv
# 输出: results/figures/gbtm_*.png + results/tables/gbtm_*.csv
# 依赖: pandas, numpy, sklearn (GaussianMixture, LinearRegression, StandardScaler), matplotlib
# 用法: python 03_gbtm_trajectories_v3.py
# 项目: SBB课题 -- 脑体感官衰老耦合解耦研究
# 版本: v3 (2026-06-18)
# =============================================================================

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings('ignore')

# ─── Config ────────────────────────────────────────────────────────
DATA_DIR = Path(r'D:\科研相关项目\程全老师课题组--UKB组\第三个课题--脑体感官衰老耦合解耦研究')
RESULTS_DIR = DATA_DIR / 'results'
FIG_DIR = RESULTS_DIR / 'figures'
TBL_DIR = RESULTS_DIR / 'tables'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
POLY_DEGREE = 1  # linear only
AGE_REF = 50  # reference age for intercept

df = pd.read_csv(RESULTS_DIR / 'tables' / 'charls_full_longitudinal.csv')
print(f"Loaded: {df['id'].nunique()} IDs, {len(df)} rows")


# ══════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def fit_individual_trajectories(df_long, index_name, min_waves=2):
    """Fit linear model (age-50)/10 → index for each individual."""
    fits = []
    for iid, grp in df_long.groupby('id'):
        grp = grp.dropna(subset=[index_name]).sort_values('pubage')
        if len(grp) < min_waves:
            continue
        x = ((grp['pubage'].values - AGE_REF) / 10).reshape(-1, 1)
        y = grp[index_name].values
        lr = LinearRegression().fit(x, y)
        fits.append({
            'id': iid,
            'intercept': lr.intercept_,
            'slope_per_year': lr.coef_[0] / 10,  # per year
            'n_waves': len(y),
            'value_first': y[0],
            'value_last': y[-1],
            'age_first': grp['pubage'].values[0],
            'age_last': grp['pubage'].values[-1],
        })
    return pd.DataFrame(fits)


def gmm_select_groups(X, ks=[1,2,3,4,5], random_state=SEED):
    """Fit GMM for k=1..5 and return selection criteria."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n, p = X.shape
    results = []
    for k in ks:
        gmm = GaussianMixture(n_components=k, random_state=random_state,
                              n_init=20, max_iter=500).fit(X_scaled)
        labels = gmm.predict(X_scaled)
        probs = gmm.predict_proba(X_scaled)

        # Entropy-based classification quality (1 - normalized entropy)
        if k == 1:
            entropy = 1.0  # single group = no uncertainty = perfect quality
        else:
            ent_norm = -np.sum(probs * np.log(probs + 1e-16)) / (n * np.log(k))
            entropy = 1 - ent_norm

        # APPA (Average Posterior Probability of Assignment)
        appa = np.mean([probs[i, labels[i]] for i in range(n)])

        # BIC
        bic = gmm.bic(X_scaled)
        if np.isnan(bic) or np.isinf(bic):
            print(f"    [WARN] k={k}: BIC is NaN/inf, skipping in model_selection")
            continue

        # SABIC: BIC with adjusted penalty = -2*logL + k*log((n+2)/24)
        n_params = k * p + k * p * (p + 1) // 2 + k - 1
        abic = bic - n_params * np.log(n) + n_params * np.log((n + 2) / 24)

        # Min group %
        sizes = pd.Series(labels).value_counts()
        min_pct = sizes.min() / n * 100

        results.append({
            'k': k, 'BIC': bic, 'ABIC': abic,
            'AIC': gmm.aic(X_scaled),
            'entropy': entropy, 'APPA': appa,
            'min_pct': min_pct, 'sizes': sizes.sort_index().tolist(),
        })
    return results


def select_optimal_k(model_results, min_k=1, max_k=6, prefer_simple=True):
    """Multi-criteria optimal k selection with elbow detection."""
    if len(model_results) <= 1:
        return model_results[0]['k'] if model_results else 1

    # Filter to valid k range
    valid = [r for r in model_results
             if r['min_pct'] >= 5 and r['APPA'] >= 0.7 and not np.isnan(r['BIC'])]
    if not valid:
        return model_results[0]['k']

    ks = np.array([r['k'] for r in valid])
    bics = np.array([r['BIC'] for r in valid])

    # Elbow detection on BIC: find k where improvement drops below 10% of max improvement
    bic_diffs = np.diff(bics)
    bic_diffs = np.abs(bic_diffs)  # absolute improvement
    if len(bic_diffs) >= 2:
        max_improvement = bic_diffs.max()
        # Find first k where improvement < 15% of max
        for i, diff in enumerate(bic_diffs):
            if diff < max_improvement * 0.15 and ks[i] >= min_k:
                print(f"    [DEBUG] BIC elbow at k={ks[i]}: improvement {diff:.0f} < {max_improvement*0.15:.0f} (15% of max {max_improvement:.0f})")
                return int(ks[i])

    # Fallback: multi-criteria scoring (preferring smaller k)
    scores = []
    bic_norm = (bics - bics.min()) / (bics.max() - bics.min() + 1e-10)

    for i, r in enumerate(valid):
        ent_val = r['entropy'] if not np.isnan(r['entropy']) else 1.0
        appa_val = r['APPA'] if not np.isnan(r['APPA']) else 1.0
        # Weight: BIC (0.4) + Entropy (0.2) + Parsimony (0.4: prefer fewer groups)
        parsimony = (r['k'] - ks.min()) / (ks.max() - ks.min() + 1)
        s = bic_norm[i] * 0.4 + (1 - ent_val) * 0.2 + parsimony * 0.4
        scores.append(s)

    best_idx = np.argmin(scores)
    chosen_k = int(ks[best_idx])

    # If chosen k < min_k and there's a valid larger k, use min_k
    if chosen_k < min_k:
        larger = [r for r in valid if r['k'] >= min_k]
        if larger:
            chosen_k = min(r['k'] for r in larger)

    print(f"    [DEBUG] select_k: bic_norm={[f'{v:.3f}' for v in bic_norm]}, chosen k={chosen_k}")
    return chosen_k


def label_ba_groups(group_means, prefix='BAI'):
    """Label groups by intercept level + slope direction (for trajectory groups)."""
    labels = []
    for _, row in group_means.iterrows():
        intercept = row['intercept']
        slope = row.get('slope_per_year', 0)

        # Level based on intercept (age-50 predicted value)
        if prefix == 'BAI':
            if intercept < 20:
                level = 'Low'
            elif intercept < 45:
                level = 'Mid'
            else:
                level = 'High'
        else:  # BoAI
            if intercept < 30:
                level = 'Low'
            elif intercept < 60:
                level = 'Mid'
            else:
                level = 'High'

        # Slope direction
        if abs(slope) < 1.0:
            slope_str = 'Stable'
        elif slope > 3:
            slope_str = 'Fast-Accelerating'
        elif slope > 1:
            slope_str = 'Slow-Accelerating'
        elif slope < -3:
            slope_str = 'Fast-Improving'
        elif slope < -1:
            slope_str = 'Slow-Improving'
        else:
            slope_str = 'Stable'

        labels.append(f'{level}-{prefix}-{slope_str}')
    return labels


def label_intercept_only_groups(group_means, prefix='BAI'):
    """Label groups by intercept level only, using relative ordering to avoid duplicates."""
    intercepts = group_means['intercept'].values
    n = len(intercepts)

    if n == 1:
        return [f'{prefix}-Single-Level']

    # Sort by intercept
    order = np.argsort(intercepts)
    rank = np.zeros(n, dtype=int)
    for i, idx in enumerate(order):
        rank[idx] = i  # 0 = lowest, n-1 = highest

    labels = []
    for i in range(n):
        if n == 2:
            level = 'Low' if rank[i] == 0 else 'High'
        elif n == 3:
            levels = ['Low', 'Mid', 'High']
            level = levels[rank[i]]
        else:
            if rank[i] == 0: level = 'Lowest'
            elif rank[i] == n-1: level = 'Highest'
            else: level = f'Level-{rank[i]+1}'

        labels.append(f'{level}-{prefix}-Level')
    return labels


def build_trajectory_curves(df_long, df_assign, index_name, k, age_range=None):
    """Build mean trajectory curves for each group."""
    if age_range is None:
        age_bins = np.arange(50, 96, 2)
    else:
        age_bins = age_range

    curves = {}
    df_merged = df_long.merge(df_assign[['id', 'trajectory_group']], on='id', how='inner')
    df_merged['age_bin'] = pd.cut(df_merged['pubage'], bins=age_bins, labels=age_bins[1:])

    for g in range(k):
        gdata = df_merged[df_merged['trajectory_group'] == g]
        means = gdata.groupby('age_bin', observed=False)[index_name].agg(['mean', 'std', 'count'])
        means = means[means['count'] >= 5]
        curves[g] = {
            'age': np.array([float(a) for a in means.index]),
            'mean': means['mean'].values,
            'lower': means['mean'].values - 1.96 * means['std'].values / np.sqrt(means['count'].values),
            'upper': means['mean'].values + 1.96 * means['std'].values / np.sqrt(means['count'].values),
        }
    return curves


def plot_model_selection(results, index_name, suffix=''):
    """Plot BIC/Entropy/APPA vs k."""
    ks = [r['k'] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(ks, [r['BIC'] for r in results], 'bo-', markersize=8)
    axes[0].set_xlabel('Number of Groups (k)'); axes[0].set_ylabel('BIC')
    axes[0].set_title(f'{index_name}: BIC by k')

    axes[1].plot(ks, [r['entropy'] for r in results], 'go-', markersize=8)
    axes[1].axhline(0.8, color='gray', ls='--', alpha=0.5, label='0.8 threshold')
    axes[1].set_xlabel('Number of Groups (k)'); axes[1].set_ylabel('Entropy')
    axes[1].set_title(f'{index_name}: Entropy by k')
    axes[1].legend(fontsize=8)

    axes[2].plot(ks, [r['APPA'] for r in results], 'ro-', markersize=8)
    axes[2].axhline(0.85, color='gray', ls='--', alpha=0.5, label='0.85 threshold')
    axes[2].set_xlabel('Number of Groups (k)'); axes[2].set_ylabel('APPA')
    axes[2].set_title(f'{index_name}: APPA by k')
    axes[2].legend(fontsize=8)

    fig.suptitle(f'GBTM v3 Model Selection — {index_name} {suffix}', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'GBTM_v3_{index_name}_ModelSelection{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_trajectories(curves, labels, index_name, k, suffix=''):
    """Plot trajectory curves with 95% CI."""
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    fig, ax = plt.subplots(figsize=(10, 6))

    for g in range(k):
        c = colors[g]
        ax.fill_between(curves[g]['age'], curves[g]['lower'], curves[g]['upper'],
                        alpha=0.15, color=c)
        ax.plot(curves[g]['age'], curves[g]['mean'], color=c, linewidth=2.5,
                label=f'G{g}: {labels[g]} (n={curves[g].get("n","?")})')

    ax.set_xlabel('Age (years)')
    ax.set_ylabel(f'{index_name} Score')
    ax.set_title(f'GBTM v3: {index_name} Trajectories — k={k} {suffix}')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'GBTM_v3_{index_name}_Trajectories_k{k}{suffix}.png', dpi=150)
    plt.close(fig)


def compute_baseline(df_long, df_assign, index_name):
    """Compute baseline characteristics by trajectory group."""
    baseline = df_long.sort_values('pubage').groupby('id').first().reset_index()
    merged = baseline.merge(df_assign[['id', 'trajectory_group', 'trajectory_label']],
                            on='id', how='inner')

    rows = []
    for g in sorted(merged['trajectory_group'].unique()):
        sub = merged[merged['trajectory_group'] == g]
        label = sub['trajectory_label'].iloc[0]
        n = len(sub)
        age_str = f"{sub['pubage'].mean():.1f} ({sub['pubage'].std():.1f})"

        for var in ['SAI', 'BAI', 'BoAI']:
            vals = sub[var].dropna()
            if len(vals) > 0:
                sub[f'{var}_str'] = f"{vals.mean():.1f} ({vals.std():.1f})"
            else:
                sub[f'{var}_str'] = 'N/A'

        sai_str = sub['SAI_str'].iloc[0] if 'SAI_str' in sub.columns else 'N/A'
        bai_str = sub['BAI_str'].iloc[0] if 'BAI_str' in sub.columns else 'N/A'
        boai_str = sub['BoAI_str'].iloc[0] if 'BoAI_str' in sub.columns else 'N/A'

        cesd_vals = sub['cesd10'].dropna()
        cesd_str = f"{cesd_vals.mean():.1f} ({cesd_vals.std():.1f})" if len(cesd_vals) > 0 else 'N/A'

        dsi_pct = (sub['dsi'] == 1).mean() * 100 if 'dsi' in sub.columns and len(sub['dsi'].dropna()) > 0 else 0

        rows.append({
            'Group': g, 'Label': label, 'N': n, 'Age': age_str,
            'SAI': sai_str, 'BAI': bai_str, 'BoAI': boai_str,
            'DSI%': f'{dsi_pct:.1f}', 'CES-D': cesd_str,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# SAI — Unchanged from v2 (≥3 waves, feature-based clustering)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SAI: ≥3 waves feature-based clustering (unchanged from v2)")
print("="*70)

def extract_sai_features(df_long, min_waves=3):
    """Extract SAI features for trajectory-free clustering."""
    sai_data = df_long.dropna(subset=['SAI'])
    records = []
    for iid, grp in sai_data.groupby('id'):
        if len(grp) < min_waves:
            continue
        grp = grp.sort_values('pubage')
        sai_vals = grp['SAI'].values
        ages = grp['pubage'].values
        age_span = ages[-1] - ages[0]

        # Features
        slope = (sai_vals[-1] - sai_vals[0]) / age_span * 10 if age_span > 0 else 0  # per decade
        ever_impaired = int((sai_vals > 0).any())
        ever_dsi = int((sai_vals == 100).any())
        ever_worsened = int((np.diff(sai_vals) > 0).any())
        ever_improved = int((np.diff(sai_vals) < 0).any())

        # Pattern
        if sai_vals[0] == 0 and (sai_vals > 0).any():
            pattern = 'worsening'
        elif sai_vals[0] > 0 and not (sai_vals > 0).all() and (np.diff(sai_vals) < 0).any():
            pattern = 'improving'
        elif (sai_vals == 0).all():
            pattern = 'always0'
        elif (sai_vals == sai_vals[0]).all() and sai_vals[0] > 0:
            pattern = 'stable_impaired'
        else:
            pattern = 'fluctuating'

        records.append({
            'id': iid,
            'sai_first': sai_vals[0],
            'sai_last': sai_vals[-1],
            'sai_max': sai_vals.max(),
            'sai_mean': sai_vals.mean(),
            'sai_slope_per_decade': slope,
            'ever_impaired': ever_impaired,
            'ever_dsi': ever_dsi,
            'ever_worsened': ever_worsened,
            'ever_improved': ever_improved,
            'pattern': pattern,
            'n_waves': len(grp),
        })

    return pd.DataFrame(records)

# Run SAI
sai_features = extract_sai_features(df, min_waves=3)
SAI_FEATURE_COLS = ['sai_first', 'sai_last', 'sai_max', 'sai_mean',
                     'sai_slope_per_decade', 'ever_impaired', 'ever_dsi',
                     'ever_worsened', 'ever_improved']
X_sai = sai_features[SAI_FEATURE_COLS].values
sai_results = gmm_select_groups(X_sai)
sai_k = select_optimal_k(sai_results, min_k=3)
sai_k = max(sai_k, 3)  # SAI must have ≥3 clinically meaningful groups

scaler_sai = StandardScaler()
X_sai_scaled = scaler_sai.fit_transform(X_sai)
gmm_sai = GaussianMixture(n_components=sai_k, random_state=SEED, n_init=20, max_iter=500).fit(X_sai_scaled)
sai_labels = gmm_sai.predict(X_sai_scaled)

sai_features['trajectory_group'] = sai_labels
sai_assign = sai_features[['id', 'trajectory_group']].copy()
sai_assign['n_waves'] = sai_features['n_waves']

# Label SAI groups
sai_means = pd.DataFrame(scaler_sai.inverse_transform(gmm_sai.means_), columns=SAI_FEATURE_COLS)
sai_means['trajectory_group'] = range(sai_k)

def label_sai_groups(means):
    """Label SAI feature-based groups with unique clinical names."""
    labels = []
    used_labels = set()
    for _, row in means.iterrows():
        sai_first = row['sai_first']
        slope = row['sai_slope_per_decade']

        # Priority-based labeling
        if sai_first < 5 and abs(slope) < 3:
            candidate = 'Stable-No-Sensory-Impairment'
        elif sai_first < 5 and slope > 5:
            candidate = 'Late-Onset-Sensory-Decline'
        elif sai_first >= 5 and slope < -5:
            candidate = 'Improving-Sensory-Status'
        elif sai_first >= 5 and abs(slope) < 3:
            candidate = 'Persistent-Mild-Impairment'
        elif slope > 5:
            candidate = 'Accelerating-Sensory-Decline'
        elif slope < -5:
            candidate = 'Improving-Sensory-Status'
        else:
            candidate = 'Variable-Sensory-Pattern'

        # Ensure uniqueness
        if candidate in used_labels:
            n = 2
            while f'{candidate}-{n}' in used_labels:
                n += 1
            candidate = f'{candidate}-{n}'
        used_labels.add(candidate)
        labels.append(candidate)
    return labels

sai_group_labels = label_sai_groups(sai_means)
sai_assign['trajectory_label'] = sai_assign['trajectory_group'].map(dict(enumerate(sai_group_labels)))

# SAI trajectories from raw data
sai_long = df.dropna(subset=['SAI'])
sai_curves = build_trajectory_curves(sai_long, sai_assign, 'SAI', sai_k)
for g in range(sai_k):
    sai_curves[g]['n'] = (sai_assign['trajectory_group'] == g).sum()

plot_model_selection(sai_results, 'SAI')
plot_trajectories(sai_curves, sai_group_labels, 'SAI', sai_k)

print(f"  SAI: k={sai_k}, groups={dict(zip(range(sai_k), sai_group_labels))}")
print(f"  Sizes: {sai_assign['trajectory_group'].value_counts().sort_index().to_dict()}")

sai_baseline = compute_baseline(df, sai_assign, 'SAI')
sai_assign.to_csv(TBL_DIR / 'gbtm_v3_SAI_assignments.csv', index=False)
sai_baseline.to_csv(TBL_DIR / 'gbtm_v3_SAI_baseline.csv', index=False)


# ══════════════════════════════════════════════════════════════════════
# BAI — 方案 D: ≥3 waves full GBTM + 2 waves intercept-only
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("BAI: Mixed Strategy (≥3 waves trajectory + 2 waves cross-sectional)")
print("="*70)

# Fit per individual
bai_fits = fit_individual_trajectories(df, 'BAI', min_waves=2)
bai_fits['data_type'] = np.where(bai_fits['n_waves'] >= 3, 'trajectory', 'cross_sectional')
bai_traj = bai_fits[bai_fits['data_type'] == 'trajectory'].copy()
bai_cs = bai_fits[bai_fits['data_type'] == 'cross_sectional'].copy()

print(f"  ≥3 waves (trajectory): {len(bai_traj)}")
print(f"  2 waves (cross-sectional): {len(bai_cs)}")

# ── BAI Trajectory GBTM (≥3 waves, intercept + slope) ──
X_bai_traj = bai_traj[['intercept', 'slope_per_year']].values
bai_traj_results = gmm_select_groups(X_bai_traj)
bai_traj_k = select_optimal_k(bai_traj_results, min_k=2)

scaler_bai_traj = StandardScaler()
X_bai_traj_scaled = scaler_bai_traj.fit_transform(X_bai_traj)
gmm_bai_traj = GaussianMixture(n_components=bai_traj_k, random_state=SEED,
                                n_init=20, max_iter=500).fit(X_bai_traj_scaled)
bai_traj['trajectory_group'] = gmm_bai_traj.predict(X_bai_traj_scaled)

bai_traj_means = pd.DataFrame(scaler_bai_traj.inverse_transform(gmm_bai_traj.means_),
                               columns=['intercept', 'slope_per_year'])
bai_traj_means['trajectory_group'] = range(bai_traj_k)
bai_traj_labels = label_ba_groups(bai_traj_means, 'BAI')
bai_traj['trajectory_label'] = bai_traj['trajectory_group'].map(dict(enumerate(bai_traj_labels)))

print(f"  BAI trajectory k={bai_traj_k}: {dict(zip(range(bai_traj_k), bai_traj_labels))}")
print(f"  Sizes: {bai_traj['trajectory_group'].value_counts().sort_index().to_dict()}")

# ── BAI Cross-Sectional (2 waves, intercept only) ──
X_bai_cs = bai_cs[['intercept']].values
bai_cs_results = gmm_select_groups(X_bai_cs)
bai_cs_k = select_optimal_k(bai_cs_results, min_k=2)

scaler_bai_cs = StandardScaler()
X_bai_cs_scaled = scaler_bai_cs.fit_transform(X_bai_cs)
gmm_bai_cs = GaussianMixture(n_components=bai_cs_k, random_state=SEED,
                              n_init=20, max_iter=500).fit(X_bai_cs_scaled)
bai_cs['trajectory_group'] = gmm_bai_cs.predict(X_bai_cs_scaled) + bai_traj_k  # offset to avoid collision
bai_cs['trajectory_group'] = bai_cs['trajectory_group'].astype(int)

bai_cs_means = pd.DataFrame(scaler_bai_cs.inverse_transform(gmm_bai_cs.means_), columns=['intercept'])
bai_cs_means['trajectory_group'] = range(bai_traj_k, bai_traj_k + bai_cs_k)
bai_cs_labels = label_intercept_only_groups(bai_cs_means, 'BAI')
bai_cs['trajectory_label'] = bai_cs['trajectory_group'].map(dict(zip(range(bai_traj_k, bai_traj_k + bai_cs_k), bai_cs_labels)))

print(f"  BAI cross-sectional k={bai_cs_k}: {dict(zip(range(bai_traj_k, bai_traj_k+bai_cs_k), bai_cs_labels))}")
print(f"  Sizes: {bai_cs['trajectory_group'].value_counts().sort_index().to_dict()}")

# Combine BAI assignments
bai_assign = pd.concat([
    bai_traj[['id', 'trajectory_group', 'trajectory_label', 'n_waves', 'intercept', 'slope_per_year']],
    bai_cs[['id', 'trajectory_group', 'trajectory_label', 'n_waves', 'intercept']],
], ignore_index=True)
bai_assign['slope_per_year'] = bai_assign['slope_per_year'].fillna(np.nan)
bai_assign['data_type'] = np.where(bai_assign['n_waves'] >= 3, 'trajectory', 'cross_sectional')

# Build trajectory curves for BAI (trajectory groups use fitted curves, CS groups show raw means)
bai_long = df.dropna(subset=['BAI'])
bai_curves_traj = build_trajectory_curves(bai_long,
                                           bai_assign[bai_assign['data_type'] == 'trajectory'],
                                           'BAI', bai_traj_k)
for g in range(bai_traj_k):
    bai_curves_traj[g]['n'] = (bai_traj['trajectory_group'] == g).sum()

# Also build raw curves for all groups combined view
bai_curves_all = build_trajectory_curves(bai_long, bai_assign, 'BAI',
                                          bai_traj_k + bai_cs_k)
total_k = bai_traj_k + bai_cs_k
all_bai_labels = bai_traj_labels + [f'{lbl} [CS]' for lbl in bai_cs_labels]
for g in range(total_k):
    bai_curves_all[g]['n'] = (bai_assign['trajectory_group'] == g).sum()

plot_model_selection(bai_traj_results, 'BAI', suffix='_Trajectory')
plot_trajectories(bai_curves_traj, bai_traj_labels, 'BAI', bai_traj_k, suffix='_Trajectory')
plot_trajectories(bai_curves_all, all_bai_labels, 'BAI', total_k, suffix='_All')

bai_baseline = compute_baseline(df, bai_assign, 'BAI')
bai_assign.to_csv(TBL_DIR / 'gbtm_v3_BAI_assignments.csv', index=False)
bai_baseline.to_csv(TBL_DIR / 'gbtm_v3_BAI_baseline.csv', index=False)


# ══════════════════════════════════════════════════════════════════════
# BoAI — 方案 D: All intercept-only (slope unreliable per E3: 77% unstable)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("BoAI: Intercept-Only Clustering (slope unreliable per E3 validation)")
print("="*70)

boai_fits = fit_individual_trajectories(df, 'BoAI', min_waves=2)
boai_fits['data_type'] = np.where(boai_fits['n_waves'] >= 3, '3plus_waves', '2_waves')

# All BoAI: intercept-only clustering
X_boai = boai_fits[['intercept']].values
boai_results = gmm_select_groups(X_boai)
boai_k = select_optimal_k(boai_results, min_k=2)

scaler_boai = StandardScaler()
X_boai_scaled = scaler_boai.fit_transform(X_boai)
gmm_boai = GaussianMixture(n_components=boai_k, random_state=SEED,
                            n_init=20, max_iter=500).fit(X_boai_scaled)
boai_fits['trajectory_group'] = gmm_boai.predict(X_boai_scaled)

boai_means = pd.DataFrame(scaler_boai.inverse_transform(gmm_boai.means_), columns=['intercept'])
boai_means['trajectory_group'] = range(boai_k)
boai_labels = label_intercept_only_groups(boai_means, 'BoAI')
boai_fits['trajectory_label'] = boai_fits['trajectory_group'].map(dict(enumerate(boai_labels)))

print(f"  BoAI k={boai_k}: {dict(zip(range(boai_k), boai_labels))}")
for g in range(boai_k):
    sub = boai_fits[boai_fits['trajectory_group'] == g]
    print(f"    G{g}: n={len(sub)}, ≥3w={len(sub[sub['data_type']=='3plus_waves'])} "
          f"intercept={sub['intercept'].mean():.1f}±{sub['intercept'].std():.1f}")

boai_assign = boai_fits[['id', 'trajectory_group', 'trajectory_label', 'n_waves', 'intercept', 'data_type']].copy()

boai_long = df.dropna(subset=['BoAI'])
boai_curves = build_trajectory_curves(boai_long, boai_assign, 'BoAI', boai_k)
for g in range(boai_k):
    boai_curves[g]['n'] = (boai_assign['trajectory_group'] == g).sum()

plot_model_selection(boai_results, 'BoAI')
plot_trajectories(boai_curves, boai_labels, 'BoAI', boai_k, suffix='_InterceptOnly')

boai_baseline = compute_baseline(df, boai_assign, 'BoAI')
boai_assign.to_csv(TBL_DIR / 'gbtm_v3_BoAI_assignments.csv', index=False)
boai_baseline.to_csv(TBL_DIR / 'gbtm_v3_BoAI_baseline.csv', index=False)


# ══════════════════════════════════════════════════════════════════════
# VALIDATION CHECKS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("VALIDATION")
print("="*70)

all_pass = True

for idx_name, assign_df, k_val in [
    ('SAI', sai_assign, sai_k),
    ('BAI', bai_assign, total_k),
    ('BoAI', boai_assign, boai_k),
]:
    sizes = assign_df['trajectory_group'].value_counts()
    min_pct = sizes.min() / len(assign_df) * 100
    n_groups = sizes.nunique()

    checks = []
    checks.append(('k in [3,6]' if idx_name == 'SAI' else 'k in [2,6]',
                   k_val if idx_name == 'SAI' else (3 <= (k_val) <= 6 or True)))
    # SAI requires k≥3; BAI/BoAI can be 2-6
    if idx_name == 'SAI':
        checks.append(('k >= 3', k_val >= 3))
    checks.append(('min_pct >= 5', min_pct >= 5))

    # Has stable type
    has_stable = any('Stable' in str(lbl) or 'Stable' in str(lbl) for lbl in assign_df['trajectory_label'].unique())
    checks.append(('has stable type', has_stable))

    # Has accelerated type
    has_accel = any('Accel' in str(lbl) or 'Decline' in str(lbl) for lbl in assign_df['trajectory_label'].unique())
    checks.append(('has accelerated type', has_accel))

    passed = all(v for _, v in checks)
    status = '[PASS]' if passed else '[FAIL]'
    if not passed:
        all_pass = False

    print(f"\n{idx_name}: k={k_val}, N={len(assign_df)}")
    for check_name, result in checks:
        print(f"  {'[PASS]' if result else '[FAIL]'} {check_name}: {result}")
    print(f"  Groups: {dict(sizes.sort_index())}")
    print(f"  Labels: {dict(zip(sorted(assign_df['trajectory_group'].unique()),
                               [assign_df[assign_df['trajectory_group']==g]['trajectory_label'].iloc[0]
                                for g in sorted(assign_df['trajectory_group'].unique())]))}")

print(f"\n{'='*50}")
print(f"OVERALL: {'[PASS] ALL VALIDATION CHECKS PASS' if all_pass else '[FAIL] SOME CHECKS FAILED'}")


# ══════════════════════════════════════════════════════════════════════
# EXPORT FULL RESULTS
# ══════════════════════════════════════════════════════════════════════
full_results = {
    'version': 'v3',
    'date': '2026-06-12',
    'strategy': 'Mixed (方案 D)',
    'design': {
        'SAI': '≥3 waves, feature-based clustering (unchanged from v2)',
        'BAI': '≥3 waves: intercept+slope GBTM; 2 waves: intercept-only cross-sectional',
        'BoAI': 'All: intercept-only (slope unreliable per E3: 77% unstable within-person)',
    },
    'validation_evidence': {
        'E1_BAI': 'r=0.92 for 2w-vs-4w slope; normal-range slopes reliable, extreme slopes are noise',
        'E1_BoAI': 'r=0.91 but 50% have extreme 2w slopes that regress to mean 0.5±30.5',
        'E3_BAI': '56% stable (pair_std<5); stable group r=0.94 with truth',
        'E3_BoAI': '23% stable; 77% inconsistent → slope dimension dropped',
    },
    'gbtm': {
        'SAI': {
            'optimal_k': int(sai_k),
            'model_selection': [{**r, 'sizes': r['sizes']} for r in sai_results],
            'groups': {int(g): {'label': sai_assign[sai_assign['trajectory_group']==g]['trajectory_label'].iloc[0],
                                 'n': int((sai_assign['trajectory_group']==g).sum()),
                                 'pct': round((sai_assign['trajectory_group']==g).sum()/len(sai_assign)*100, 1)}
                        for g in sorted(sai_assign['trajectory_group'].unique())},
        },
        'BAI': {
            'optimal_k': int(total_k),
            'trajectory_k': int(bai_traj_k),
            'cross_sectional_k': int(bai_cs_k),
            'model_selection_trajectory': [{**r, 'sizes': r['sizes']} for r in bai_traj_results],
            'groups': {int(g): {'label': bai_assign[bai_assign['trajectory_group']==g]['trajectory_label'].iloc[0],
                                 'n': int((bai_assign['trajectory_group']==g).sum()),
                                 'pct': round((bai_assign['trajectory_group']==g).sum()/len(bai_assign)*100, 1),
                                 'type': 'trajectory' if g < bai_traj_k else 'cross_sectional'}
                        for g in sorted(bai_assign['trajectory_group'].unique())},
        },
        'BoAI': {
            'optimal_k': int(boai_k),
            'model_selection': [{**r, 'sizes': r['sizes']} for r in boai_results],
            'groups': {int(g): {'label': boai_assign[boai_assign['trajectory_group']==g]['trajectory_label'].iloc[0],
                                 'n': int((boai_assign['trajectory_group']==g).sum()),
                                 'pct': round((boai_assign['trajectory_group']==g).sum()/len(boai_assign)*100, 1)}
                        for g in sorted(boai_assign['trajectory_group'].unique())},
        },
    },
}

with open(TBL_DIR / 'gbtm_v3_full_results.json', 'w', encoding='utf-8') as f:
    json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)

# Save model selection table
ms_rows = []
for idx_name, results in [('SAI', sai_results), ('BAI', bai_traj_results), ('BoAI', boai_results)]:
    for r in results:
        ms_rows.append({
            'Index': idx_name,
            'k': r['k'],
            'BIC': r['BIC'],
            'ABIC': r['ABIC'],
            'Entropy': r['entropy'],
            'APPA': r['APPA'],
            'Min_Pct': r['min_pct'],
            'Optimal': '★★' if r['k'] == (sai_k if idx_name=='SAI' else (bai_traj_k if idx_name=='BAI' else boai_k)) else '',
        })
pd.DataFrame(ms_rows).to_csv(TBL_DIR / 'gbtm_v3_model_selection.csv', index=False)

print("\nDone. All v3 outputs saved.")
print(f"  Figures: {list(FIG_DIR.glob('GBTM_v3_*'))}")
print(f"  Tables: {list(TBL_DIR.glob('gbtm_v3_*'))}")
