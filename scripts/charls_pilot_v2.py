# CHARLS Pilot v2 — Improved BoAI + Longitudinal GBTM + Mediation
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = os.path.join(PROJECT_ROOT, 'data_raw', 'CHARLS')
out_dir = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(os.path.join(out_dir, 'tables'), exist_ok=True)
os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)

# ============================================================
# 1. LOAD ALL DATA (ALL WAVES)
# ============================================================
print('=== 1. Loading CHARLS (all waves) ===')
dfs_raw = {}
for f in sorted(os.listdir(base)):
    if f.endswith('.csv'):
        dfs_raw[f] = pd.read_csv(os.path.join(base, f), encoding='utf-8', low_memory=False)

# Map files to their key variables
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

print(f'  All waves: {len(df_all):,} rows, {df_all["id"].nunique():,} IDs')
df_all = df_all[df_all['pubage'] >= 50].copy()
print(f'  Age>=50: {len(df_all):,} rows')

# ============================================================
# 2. BUILD IMPROVED INDICES (ALL WAVES)
# ============================================================
print('\n=== 2. Building improved indices ===')

# Encode sensory
for col in ['fi_vision','fi_hearing']:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(0)

df_all['dsi'] = ((df_all['fi_vision']==1) & (df_all['fi_hearing']==1)).astype(int)
df_all['SAI'] = (df_all['fi_vision'] + df_all['fi_hearing']) / 2 * 100  # 0, 50, 100

# Cognitive: use global_cognition if available, else composite
if 'global_cognition' in df_all.columns:
    cog_raw = df_all['global_cognition']
elif 'orient' in df_all.columns:
    cog_raw = df_all[['orient','draw']].mean(axis=1, skipna=True)
else:
    cog_raw = df_all['cesd10'] * -1
mn, mx = cog_raw.min(), cog_raw.max()
df_all['BAI'] = ((cog_raw - mn) / (mx - mn) * 100).clip(0, 100)

# IMPROVED BoAI: multi-dimensional
body_components = []
# Grip (max of available)
grip_cols = [c for c in ['lgrip','lgrip1','lgrip2','rgrip1','rgrip2'] if c in df_all.columns]
if grip_cols:
    df_all['grip_max'] = df_all[grip_cols].max(axis=1)
    body_components.append('grip_max')

# Walk speed (faster of 2 trials)
ws_cols = [c for c in ['wspeed','wspeed1','wspeed2'] if c in df_all.columns]
if ws_cols:
    df_all['wspeed_best'] = df_all[ws_cols].max(axis=1)
    body_components.append('wspeed_best')

# ADL
if 'dadliv' in df_all.columns:
    body_components.append('dadliv')

# Balance
if 'balance' in df_all.columns:
    body_components.append('balance')

# Sarcopenia and muscle
for c in ['sarcopenia','low_muscle_strength','low_muscle_mass']:
    if c in df_all.columns:
        body_components.append(c)

print(f'  BoAI components: {body_components}')

if body_components:
    # Z-score each and average
    df_all['BoAI_raw'] = 0
    n_comp = 0
    for comp in body_components:
        vals = df_all[comp].copy()
        vals = pd.to_numeric(vals, errors='coerce')
        mn_c, mx_c = vals.min(), vals.max()
        if mx_c > mn_c and vals.notna().sum() > 100:
            z = (vals - mn_c) / (mx_c - mn_c)
            # Reverse if needed (higher = worse)
            if comp in ['dadliv','sarcopenia','low_muscle_strength','low_muscle_mass']:
                z = 1 - z  # Higher ADL = worse, so reverse
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

# Clean
df_all = df_all.dropna(subset=['SAI','BAI','BoAI']).copy()
print(f'  After cleaning: {len(df_all):,} rows')

# ============================================================
# 3. BASELINE LCA (IMPROVED INDICES)
# ============================================================
print('\n=== 3. Baseline LCA (improved indices) ===')
first_wave = df_all.groupby('id')['wave'].transform('min')
df_bl = df_all[df_all['wave'] == first_wave].copy()

X = df_bl[['SAI','BAI','BoAI']].values
X_scaled = StandardScaler().fit_transform(X)

print('  GMM BIC by k:')
bic_scores = {}
for k in range(3, 8):
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
    labels = gmm.fit_predict(X_scaled)
    bic = gmm.bic(X_scaled)
    sizes = np.bincount(labels)
    min_pct = sizes.min() / len(labels) * 100
    bic_scores[k] = {'bic': bic, 'min_pct': min_pct, 'sizes': sizes}
    print(f'    k={k}: BIC={bic:.0f}, min_class={min_pct:.1f}%')

# Select best k (BIC elbow with min 5%)
best_k = 5
gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=20)
df_bl['phenotype'] = gmm.fit_predict(X_scaled)

print(f'\n  Baseline phenotypes (k={best_k}, N={len(df_bl):,}):')
means = df_bl.groupby('phenotype')[['SAI','BAI','BoAI','pubage','dsi']].mean()
for p in range(best_k):
    sub = df_bl[df_bl['phenotype']==p]
    s, b, bo = sub['SAI'].mean(), sub['BAI'].mean(), sub['BoAI'].mean()
    # Auto-label
    if s < 25 and b < 35 and bo < 50:
        label = 'Successful Aging'
    elif s > 40 and b > 40 and bo > 50:
        label = 'Global Accelerated'
    elif s > 40 and b < 35:
        label = 'Sensory-First'
    elif b < 30 and bo > 50:
        label = 'Brain-Resilient'
    elif b > 40 and bo > 60:
        label = 'Body-Resilient'
    else:
        label = 'Mixed'
    print(f'    Type {p+1:<15d} n={len(sub):5,} ({len(sub)/len(df_bl)*100:4.1f}%) '
          f'SAI={s:3.0f} BAI={b:3.0f} BoAI={bo:3.0f} DSI={sub["dsi"].mean()*100:3.1f}% Age={sub["pubage"].mean():.0f}')

# ============================================================
# 4. LONGITUDINAL TRENDS BY PHENOTYPE
# ============================================================
print('\n=== 4. Longitudinal trends ===')
# Merge phenotype back to all waves
pheno_map = df_bl[['id','phenotype']].drop_duplicates()
df_long = df_all.merge(pheno_map, on='id', how='left')
df_long = df_long.dropna(subset=['phenotype'])

for p in sorted(df_long['phenotype'].dropna().unique()):
    sub = df_long[df_long['phenotype']==p]
    waves = sub.groupby('wave')[['SAI','BAI','BoAI']].mean()
    n = sub['id'].nunique()
    print(f'  Phenotype {int(p)} (n={n}):')
    for w in sorted(waves.index):
        row = waves.loc[w]
        print(f'    Wave {int(w)}: SAI={row["SAI"]:.0f} BAI={row["BAI"]:.0f} BoAI={row["BoAI"]:.0f}')

# ============================================================
# 5. MEDIATION: DSI -> CES-D -> BAI
# ============================================================
print('\n=== 5. Mediation analysis (Baron-Kenny) ===')
df_med = df_bl.dropna(subset=['dsi','BAI','cesd10','pubage']).copy()
df_med['female'] = (df_med['ragender']==2).astype(int)

# Step 1: X -> Y (total effect)
m1 = LinearRegression().fit(df_med[['dsi','pubage','female']], df_med['BAI'])
c = m1.coef_[0]
print(f'  Step 1 (DSI->BAI total): c = {c:.2f}')

# Step 2: X -> M
m2 = LinearRegression().fit(df_med[['dsi','pubage','female']], df_med['cesd10'])
a = m2.coef_[0]
print(f'  Step 2 (DSI->CESD): a = {a:.2f}')

# Step 3: X + M -> Y
m3 = LinearRegression().fit(df_med[['dsi','cesd10','pubage','female']], df_med['BAI'])
c_prime = m3.coef_[0]
b = m3.coef_[1]
print(f'  Step 3 (DSI+CESD->BAI): c_prime = {c_prime:.2f}, b = {b:.2f}')

# Indirect effect
indirect = a * b
total = c_prime + indirect
mediation_pct = indirect / total * 100 if total != 0 else 0
print(f'\n  Indirect (mediation): {indirect:.2f}')
print(f'  Direct: {c_prime:.2f}')
print(f'  Total: {total:.2f}')
print(f'  Mediation %: {mediation_pct:.1f}%')

# ============================================================
# 6. SAVE RESULTS
# ============================================================
print(f'\n=== 6. Saving results ===')
df_bl[['id','wave','pubage','ragender','SAI','BAI','BoAI','dsi','cesd10','phenotype']].to_csv(
    os.path.join(out_dir, 'tables', 'charls_v2_phenotypes.csv'), index=False)
df_long[['id','wave','pubage','SAI','BAI','BoAI','dsi','cesd10','phenotype']].to_csv(
    os.path.join(out_dir, 'tables', 'charls_v2_longitudinal.csv'), index=False)
print(f'  Saved: charls_v2_phenotypes.csv ({len(df_bl):,} rows)')
print(f'  Saved: charls_v2_longitudinal.csv ({len(df_long):,} rows)')
print('\n=== CHARLS Pilot v2 Complete ===')
