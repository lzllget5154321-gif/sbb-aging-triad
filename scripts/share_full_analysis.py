# SHARE Full Analysis — Sensory-Aging-Index (SAI), Brain-Aging-Index (BAI), Body-Aging-Index (BoAI)
# Gateway to Global Aging harmonized SHARE data
# v1.0 — 2026-06-11
#
# Data constraints documented inline:
#   - NO current hearing variable available → SAI is vision-only (hwlvnear)
#   - NO gender (ragender) available → descriptive stats by gender omitted
#   - NO BMI variable in extract → BoAI uses ADL/IADL/grip/walk/mobility composite
#   - childhood retrospective vars (rachseeingdif/rachearpr) are in EMPTY file → unusable
#
# Output: CHARLS-compatible JSON format

import pandas as pd, numpy as np, os, json, warnings, sys, io
warnings.filterwarnings('ignore')
# Fix Windows GBK encoding for emoji
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# ===================== PATHS =====================
PROJECT = r'D:\科研相关项目\程全老师课题组--UKB组\第三个课题--脑体感官衰老耦合解耦研究'
DATA_DIR = os.path.join(PROJECT, 'data_raw', 'SHARE')
OUT_DIR = os.path.join(PROJECT, 'results', 'tables')
os.makedirs(OUT_DIR, exist_ok=True)

# ===================== DATA LOADING =====================
print('=' * 70)
print('SHARE (Survey of Health, Ageing and Retirement in Europe)')
print('Gateway to Global Aging — Harmonized Data')
print('=' * 70)

# File mapping (from actual directory listing)
FILE_MAP = {
    'cog_age': ('d0ada6e9', ['mergeid','wave','agey','orient','imrc','dlrc','ser7','hwlvnear','slfmem','verbf','numer_s','cogimp']),
    'body':    ('b2d774d5', ['mergeid','wave','cesd','cesdm','depress','eurod','lgrip1','lgrip2','rgrip1','rgrip2','lgrip','rgrip','gripsum','wspeed1','wspeed2','wspeed','walkcomp','walkflr','walkaid','sleepr','casp12']),
    'adl':     ('354a60bb', ['mergeid','wave','adlwa','adlwam','adlwaa','adla','adlam','adlaa','adlfive','adltot_s','iadla','iadlam','iadlaa','iadlfour','iadltot1_s','iadltot2_s','walkra','walk100a','chaira','mobilsev','hlthlm','shlt']),
    'grip':    ('4c567c9c', ['mergeid','wave','gripref']),
    'demo':    ('9d8f51fc', ['mergeid','wave','pubage']),
}

dfs = {}
for label, (key, cols) in FILE_MAP.items():
    match = [f for f in os.listdir(DATA_DIR) if key in f and f.endswith('.csv')]
    if not match:
        print(f'  ⚠️ {label}: file not found')
        continue
    fp = os.path.join(DATA_DIR, match[0])
    try:
        available = [c for c in cols if c in pd.read_csv(fp, nrows=0, encoding='utf-8', low_memory=False).columns]
        if available:
            dfs[label] = pd.read_csv(fp, usecols=available, encoding='utf-8', low_memory=False)
            print(f'  ✅ {label}: {len(dfs[label]):,} rows, {len(available)}/{len(cols)} vars')
        else:
            print(f'  ⚠️ {label}: 0 of {len(cols)} requested vars available')
    except Exception as e:
        print(f'  ❌ {label}: {e}')

# ===================== MERGE =====================
print('\n--- Merging ---')
df = None
for label in ['cog_age', 'body', 'adl', 'grip', 'demo']:
    if label not in dfs:
        continue
    sub = dfs[label].copy()
    on = ['mergeid','wave'] if 'wave' in sub.columns and (df is None or 'wave' in df.columns) else ['mergeid']
    common = [c for c in on if df is None or c in df.columns]
    if df is None:
        df = sub
    else:
        df = df.merge(sub, on=common, how='outer')
    print(f'  After {label}: {len(df):,} rows')

# ===================== AGE FILTER =====================
# Use agey (459K non-null) primary; pubage (161K non-null) as fallback
df['age'] = df['agey'].fillna(df.get('pubage', np.nan))
print(f'\nAge variable: agey={df["agey"].notna().sum():,}, pubage={df.get("pubage", pd.Series()).notna().sum():,}')
print(f'Combined age non-null: {df["age"].notna().sum():,}')

df = df[df['age'] >= 50].copy()
print(f'Age >= 50: N={len(df):,}')

# ===================== SAI (Sensory Aging Index) =====================
print('\n--- SAI Construction ---')
# ⚠️ LIMITATION: Only near-vision difficulty available. No hearing variable.
# hwlvnear: binary (0/1). Empirical evidence shows hwlvnear=1 associated with:
#   - worse cognition (orient/imrc/dlrc/ser7 all lower)
#   - slightly younger age (possible reporting bias)
# We code hwlvnear=1 as vision impairment for SAI computation.
# SAI is VISION-ONLY due to data limitation — documented.

df['hwlvnear'] = pd.to_numeric(df['hwlvnear'], errors='coerce')
df['vi_imp'] = (df['hwlvnear'] == 1).astype(int)  # 1 = near vision difficulty
df['hi_imp'] = np.nan  # NO HEARING DATA — documented limitation
df['dsi'] = df['vi_imp'].copy()  # DSI = vision impairment only (no hearing)
# SAI: single-sensory (vision) scaled to 0-100
df['SAI'] = df['vi_imp'] * 100  # 0=no impairment, 100=vision impairment

print(f'  hwlvnear=1 (vision difficulty): {df["vi_imp"].sum():,} / {df["vi_imp"].notna().sum():,} ({df["vi_imp"].mean()*100:.1f}%)')
print(f'  ⚠️ SAI is VISION-ONLY — no hearing variable in SHARE Gateway extract')
print(f'  ⚠️ DSI = vision impairment proxy (not true dual sensory impairment)')

# ===================== BAI (Brain Aging Index) =====================
print('\n--- BAI Construction ---')
# Cognitive battery: orient + imrc + dlrc + ser7
# orient: orientation (0-4, higher=better)
# imrc: immediate recall (0-10, higher=better)
# dlrc: delayed recall (0-10, higher=better)
# ser7: serial 7s (0-5, higher=better)

cog_vars = []
for c in ['orient','imrc','dlrc','ser7','verbf','numer_s']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        cog_vars.append(c)

if len(cog_vars) >= 2:
    # Reverse code so higher = worse (consistent with SAI/BoAI direction)
    # Then z-score composite, normalize to 0-100
    cog_data = df[cog_vars].copy()
    for c in cog_vars:
        mx = cog_data[c].max()
        # Reverse: max - value (higher raw = better → higher reversed = worse)
        cog_data[c] = mx - cog_data[c]

    # Z-score each, then average
    scaler = StandardScaler()
    cog_z = pd.DataFrame(scaler.fit_transform(cog_data.fillna(cog_data.mean())),
                         index=cog_data.index, columns=cog_vars)
    df['BAI_raw'] = cog_z.mean(axis=1)

    # Normalize to 0-100
    mn, mx = df['BAI_raw'].min(), df['BAI_raw'].max()
    if mx > mn:
        df['BAI'] = ((df['BAI_raw'] - mn) / (mx - mn) * 100).clip(0, 100)
    else:
        df['BAI'] = 50

    print(f'  Cognitive vars: {cog_vars}')
    print(f'  BAI derived from {len(cog_vars)}-item battery (vs CHARLS single global_cognition)')
else:
    print('  ❌ Insufficient cognitive variables')
    df['BAI'] = np.nan

# ===================== BoAI (Body Aging Index) =====================
print('\n--- BoAI Construction ---')

body_components = []

# 1. ADL/IADL composite
for c in ['adlwa','adla','adlfive','adltot_s']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

iadl_vars = [c for c in ['iadla','iadlfour','iadltot1_s'] if c in df.columns]

# ADL: higher = more difficulty → keep as-is
adl_available = [c for c in ['adlwa','adla','adlfive','adltot_s'] if c in df.columns]
if adl_available:
    adl_data = df[adl_available].copy()
    # Normalize each to 0-1
    for c in adl_available:
        mnv, mxv = adl_data[c].min(), adl_data[c].max()
        if mxv > mnv:
            adl_data[c] = (adl_data[c] - mnv) / (mxv - mnv)
    df['ADL_score'] = adl_data.mean(axis=1, skipna=True)
    body_components.append('ADL_score')
    print(f'  ADL from: {adl_available}')

# 2. IADL composite
if iadl_vars:
    iadl_data = df[iadl_vars].copy()
    for c in iadl_vars:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        mnv, mxv = iadl_data[c].min(), iadl_data[c].max()
        if mxv > mnv:
            iadl_data[c] = (iadl_data[c] - mnv) / (mxv - mnv)
    df['IADL_score'] = iadl_data.mean(axis=1, skipna=True)
    body_components.append('IADL_score')
    print(f'  IADL from: {iadl_vars}')

# 3. Grip strength (reverse: higher grip = better → lower score)
grip_vars = [c for c in ['lgrip','rgrip','gripsum','gripref','lgrip1'] if c in df.columns]
if grip_vars:
    best_grip = grip_vars[0]
    df[best_grip] = pd.to_numeric(df[best_grip], errors='coerce')
    mnv, mxv = df[best_grip].min(), df[best_grip].max()
    if mxv > mnv:
        # Higher grip = better → reverse: 1 - normalized
        df['Grip_score'] = 1 - (df[best_grip] - mnv) / (mxv - mnv)
        body_components.append('Grip_score')
    print(f'  Grip from: {best_grip}')

# 4. Walking speed (reverse: faster = better)
walk_vars = [c for c in ['wspeed','wspeed1'] if c in df.columns]
if walk_vars:
    best_walk = walk_vars[0]
    df[best_walk] = pd.to_numeric(df[best_walk], errors='coerce')
    mnv, mxv = df[best_walk].min(), df[best_walk].max()
    if mxv > mnv:
        df['Walk_score'] = 1 - (df[best_walk] - mnv) / (mxv - mnv)
        body_components.append('Walk_score')
    print(f'  Walk speed from: {best_walk}')

# 5. Mobility difficulty (walkra/walk100a: 0=no difficulty, 1=has difficulty)
mobil_vars = [c for c in ['walkra','walk100a','chaira','mobilsev'] if c in df.columns]
if mobil_vars:
    mobil_data = df[mobil_vars].copy()
    for c in mobil_vars:
        mobil_data[c] = pd.to_numeric(mobil_data[c], errors='coerce')
        mnv, mxv = mobil_data[c].min(), mobil_data[c].max()
        if mxv > mnv:
            mobil_data[c] = (mobil_data[c] - mnv) / (mxv - mnv)
    df['Mobil_score'] = mobil_data.mean(axis=1, skipna=True)
    body_components.append('Mobil_score')
    print(f'  Mobility from: {mobil_vars}')

# 6. Self-rated health (shlt: 1=excellent→5=poor, higher=worse → keep as-is)
if 'shlt' in df.columns:
    df['shlt'] = pd.to_numeric(df['shlt'], errors='coerce')
    mnv, mxv = df['shlt'].min(), df['shlt'].max()
    if mxv > mnv:
        df['SRH_score'] = (df['shlt'] - mnv) / (mxv - mnv)
        body_components.append('SRH_score')
    print(f'  Self-rated health from: shlt')

print(f'  Body components: {body_components}')

if body_components:
    df['BoAI_raw'] = df[body_components].mean(axis=1, skipna=True)
    mnb, mxb = df['BoAI_raw'].min(), df['BoAI_raw'].max()
    if mxb > mnb:
        df['BoAI'] = ((df['BoAI_raw'] - mnb) / (mxb - mnb) * 100).clip(0, 100)
    else:
        df['BoAI'] = 50
    print(f'  BoAI from {len(body_components)} domains')

# ===================== CES-D =====================
if 'cesd' in df.columns:
    df['cesd'] = pd.to_numeric(df['cesd'], errors='coerce')
    print(f'\nCES-D available: {df["cesd"].notna().sum():,} / {len(df):,}')

# ===================== COMPLETE CASES =====================
required = ['SAI','BAI','BoAI']
df_valid = df.dropna(subset=required).copy()
print(f'\n=== After dropna on SAI/BAI/BoAI ===')
print(f'Complete cases: N={len(df_valid):,}')
print(f'DSI (vision-only): {df_valid["dsi"].mean()*100:.1f}%')
print(f'SAI mean±sd: {df_valid["SAI"].mean():.1f}±{df_valid["SAI"].std():.1f}')
print(f'BAI mean±sd: {df_valid["BAI"].mean():.1f}±{df_valid["BAI"].std():.1f}')
print(f'BoAI mean±sd: {df_valid["BoAI"].mean():.1f}±{df_valid["BoAI"].std():.1f}')

# ===================== GMM CLUSTERING =====================
print('\n--- GMM Clustering ---')
X = df_valid[['SAI','BAI','BoAI']].values
gmm_results = {'best_k': None, 'bic_scores': {}}

for k in range(2, 9):
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
    gmm.fit(X)
    labels = gmm.predict(X)
    sizes = [int((labels == i).sum()) for i in range(k)]
    min_pct = min(sizes) / len(X) * 100
    bic = gmm.bic(X)
    gmm_results['bic_scores'][str(k)] = {
        'bic': float(bic), 'min_pct': float(min_pct), 'sizes': sizes
    }
    print(f'  k={k}: BIC={bic:.0f}, sizes={sizes}, min_pct={min_pct:.1f}%')

# Best k by BIC
best_k = min(gmm_results['bic_scores'].items(), key=lambda x: x[1]['bic'])
gmm_results['best_k'] = int(best_k[0])
print(f'  Best k={gmm_results["best_k"]} (BIC={best_k[1]["bic"]:.0f})')

# Final GMM with best k
gmm_final = GaussianMixture(n_components=gmm_results['best_k'], random_state=42, n_init=10)
labels = gmm_final.fit_predict(X)
df_valid['cluster'] = labels

# Phenotype labels based on cluster centroids
centroids = pd.DataFrame(gmm_final.means_, columns=['SAI','BAI','BoAI'])
phenotypes = {}
for i in range(gmm_results['best_k']):
    row = centroids.iloc[i]
    n = int((labels == i).sum())
    pct = float(n / len(df_valid) * 100)

    # Label logic
    sai_hi = row['SAI'] > centroids['SAI'].median()
    bai_hi = row['BAI'] > centroids['BAI'].median()
    boai_hi = row['BoAI'] > centroids['BoAI'].median()

    if sai_hi and bai_hi and boai_hi:
        label = 'Global-Accelerated'
    elif not sai_hi and not bai_hi and not boai_hi:
        label = 'Successful-Aging'
    elif sai_hi and not bai_hi and not boai_hi:
        label = 'Sensory-First'
    elif not sai_hi and bai_hi and not boai_hi:
        label = 'Brain-First'
    elif not sai_hi and not bai_hi and boai_hi:
        label = 'Body-First'
    elif bai_hi and boai_hi:
        label = 'Brain-Body'
    elif sai_hi and bai_hi:
        label = 'Sensory-Brain'
    elif sai_hi and boai_hi:
        label = 'Sensory-Body'
    else:
        label = 'Mixed'

    phenotypes[str(i)] = {'label': label, 'n': n, 'pct': pct}

gmm_results['phenotypes'] = phenotypes

for i, p in phenotypes.items():
    print(f'  Cluster {i}: {p["label"]:22s} n={p["n"]:>6,} ({p["pct"]:.1f}%)')

# ===================== MEDIATION =====================
print('\n--- CES-D Mediation: DSI → CES-D → BAI ---')
med_results = None
if 'cesd' in df_valid.columns:
    df_med = df_valid.dropna(subset=['dsi','BAI','cesd']).copy()
    print(f'Mediation N={len(df_med):,}')

    if len(df_med) > 100:
        from sklearn.linear_model import LinearRegression

        # Path a: DSI → CES-D
        m_a = LinearRegression().fit(df_med[['dsi']], df_med['cesd'])
        a = float(m_a.coef_[0])

        # Path b + c': CES-D + DSI → BAI
        m_bc = LinearRegression().fit(df_med[['dsi','cesd']], df_med['BAI'])
        b = float(m_bc.coef_[1])
        c_prime = float(m_bc.coef_[0])

        # Path c (total): DSI → BAI
        m_c = LinearRegression().fit(df_med[['dsi']], df_med['BAI'])
        c = float(m_c.coef_[0])

        indirect = a * b
        total = c
        med_pct = indirect / total * 100 if total != 0 else 0

        print(f'  a (DSI→CES-D): {a:.4f}')
        print(f'  b (CES-D→BAI): {b:.4f}')
        print(f'  c (total DSI→BAI): {c:.4f}')
        print(f'  c\' (direct): {c_prime:.4f}')
        print(f'  indirect (a×b): {indirect:.4f}')
        print(f'  Mediation: {med_pct:.1f}%')

        # Bootstrap
        n_boot = 1000
        np.random.seed(42)
        boot_indirect = []
        for _ in range(n_boot):
            idx = np.random.choice(len(df_med), len(df_med), replace=True)
            sb = df_med.iloc[idx]
            ba = LinearRegression().fit(sb[['dsi']], sb['cesd']).coef_[0]
            bb = LinearRegression().fit(sb[['dsi','cesd']], sb['BAI']).coef_[1]
            boot_indirect.append(ba * bb)

        boot_indirect = np.array(boot_indirect)
        ci_lo, ci_hi = np.percentile(boot_indirect, [2.5, 97.5])

        med_results = {
            'a': a, 'b': b, 'c': c, 'c_prime': c_prime,
            'indirect': indirect, 'total': total,
            'mediation_pct': med_pct,
            'bootstrap_ci': {
                'indirect': {'mean': float(boot_indirect.mean()),
                            'ci_025': float(ci_lo), 'ci_975': float(ci_hi)}
            },
            'n_mediation': int(len(df_med))
        }

        print(f'  Bootstrap 95%CI indirect: [{ci_lo:.4f}, {ci_hi:.4f}]')

# ===================== FEATURE IMPORTANCE (XGBoost-style) =====================
print('\n--- Feature Importance ---')
# Simple variance-based: how much does each index contribute to total variance?
var_sai = df_valid['SAI'].var()
var_bai = df_valid['BAI'].var()
var_boai = df_valid['BoAI'].var()
total_var = var_sai + var_bai + var_boai
feat_imp = {
    'SAI': float(var_sai / total_var),
    'BAI': float(var_bai / total_var),
    'BoAI': float(var_boai / total_var)
}
print(f'  SAI: {feat_imp["SAI"]:.1%}, BAI: {feat_imp["BAI"]:.1%}, BoAI: {feat_imp["BoAI"]:.1%}')

# ===================== RESULTS =====================
results = {
    'cohort': 'SHARE',
    'description': 'Survey of Health, Ageing and Retirement in Europe (Gateway harmonized)',
    'data_notes': {
        'sai_note': 'SAI is VISION-ONLY — no hearing variable available in this Gateway extract',
        'dsi_note': 'DSI = vision impairment proxy (hwlvnear=1). hwlvnear: near vision difficulty, binary',
        'bai_note': f'BAI from {len(cog_vars)}-item cognitive battery: {cog_vars}',
        'boai_note': f'BoAI from {len(body_components)} domains: {body_components}',
        'missing': ['hearing', 'ragender', 'bmi'],
        'age_var': 'agey (primary) / pubage (fallback)',
    },
    'N_baseline': int(len(df_valid)),
    'age_mean': float(df_valid['age'].mean()),
    'age_sd': float(df_valid['age'].std()),
    'DSI_prevalence_pct': float(df_valid['dsi'].mean() * 100),
    'SAI': {'mean': float(df_valid['SAI'].mean()), 'std': float(df_valid['SAI'].std())},
    'BAI': {'mean': float(df_valid['BAI'].mean()), 'std': float(df_valid['BAI'].std())},
    'BoAI': {'mean': float(df_valid['BoAI'].mean()), 'std': float(df_valid['BoAI'].std())},
    'correlations': {
        'SAI_BAI': float(df_valid['SAI'].corr(df_valid['BAI'])),
        'SAI_BoAI': float(df_valid['SAI'].corr(df_valid['BoAI'])),
        'BAI_BoAI': float(df_valid['BAI'].corr(df_valid['BoAI'])),
    },
    'GMM': gmm_results,
    'feature_importance': feat_imp,
    'mediation': med_results,
}

# ===================== EXPORT =====================
out_json = os.path.join(OUT_DIR, 'share_full_results.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'\n✅ Results → {out_json}')

# CSV export
df_out = df_valid[['mergeid','wave','age','SAI','BAI','BoAI','dsi','cesd']].copy()
df_out.columns = ['mergeid','wave','agey','SAI','BAI','BoAI','dsi','cesd']
out_csv = os.path.join(OUT_DIR, 'share_full_analysis.csv')
df_out.to_csv(out_csv, index=False)
print(f'✅ CSV → {out_csv} (N={len(df_out):,})')

# ===================== SUMMARY =====================
print('\n' + '=' * 70)
print('SHARE ANALYSIS SUMMARY')
print('=' * 70)
print(f'Baseline N (age≥50, complete SAI/BAI/BoAI): {len(df_valid):,}')
print(f'Age: {df_valid["age"].mean():.1f}±{df_valid["age"].std():.1f}')
print(f'DSI (vision proxy): {df_valid["dsi"].mean()*100:.1f}%')
print(f'SAI: {df_valid["SAI"].mean():.0f}±{df_valid["SAI"].std():.0f}')
print(f'BAI: {df_valid["BAI"].mean():.0f}±{df_valid["BAI"].std():.0f}')
print(f'BoAI: {df_valid["BoAI"].mean():.0f}±{df_valid["BoAI"].std():.0f}')
if med_results:
    print(f'CES-D mediation: {med_results["mediation_pct"]:.1f}%')
print(f'Best GMM clusters: k={gmm_results["best_k"]}')
for i, p in phenotypes.items():
    print(f'  Cluster {i}: {p["label"]} — {p["n"]:,} ({p["pct"]:.1f}%)')
print(f'\n⚠️ Limitations: vision-only SAI, no hearing, no gender, no BMI')
print('Done.')
