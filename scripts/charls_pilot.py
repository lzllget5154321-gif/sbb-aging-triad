import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = os.path.join(PROJECT_ROOT, 'data_raw', 'CHARLS')

# Load
dfs_raw = {}
for f in sorted(os.listdir(base)):
    if f.endswith('.csv'):
        dfs_raw[f] = pd.read_csv(os.path.join(base, f), encoding='utf-8', low_memory=False)

# Selective merge
file_vars = {
    'charls_data_export_45248e43-794d-4135-a700-d18ee5a3bbea.csv':
        ['id','wave','fi_vision','fi_hearing','fi_cognition','fi_depression','fi_health_status',
         'smokev','drinkev','physical_activity2','sleep_night','short_sleep'],
    'charls_data_export_92ad8abe-8030-4339-a374-12b749171417.csv':
        ['id','wave','global_cognition','orient','draw','dadliv'],
    'charls_data_export_2b6ece0c-2883-404c-acce-9fbb936911a7.csv':
        ['id','wave','cesd10','balance','sarcopenia','low_muscle_strength','low_muscle_mass','sleeprl'],
    'charls_data_export_d4b49326-14a3-427f-8d9d-97ea648111af.csv':
        ['id','wave','lgrip','wspeed','mbmi','mheight','mweight','pubage'],
    'charls_data_export_aeccc6cd-3308-4866-aa5e-6a8343d2f6c1.csv':
        ['id','ragender','education2'],
}

df_main = None
for fname, vars_needed in file_vars.items():
    if fname in dfs_raw:
        available = [v for v in vars_needed if v in dfs_raw[fname].columns]
        sub = dfs_raw[fname][available].copy()
        if df_main is None:
            df_main = sub
        else:
            on_cols = ['id'] if 'wave' not in sub.columns or fname.endswith('aeccc6cd*.csv') else ['id','wave']
            df_main = df_main.merge(sub, on=on_cols, how='outer')

# Age filter
df_main = df_main[df_main['pubage'] >= 50].copy()

# Baseline
first_wave = df_main.groupby('id')['wave'].transform('min')
df_bl = df_main[df_main['wave'] == first_wave].copy()

# Build indices
for col in ['fi_vision','fi_hearing']:
    df_bl[col] = pd.to_numeric(df_bl[col], errors='coerce').fillna(0)

df_bl['dsi'] = ((df_bl['fi_vision']==1) & (df_bl['fi_hearing']==1)).astype(int)
df_bl['SAI_raw'] = (df_bl['fi_vision'] + df_bl['fi_hearing']) / 2

cog_avail = [v for v in ['global_cognition','orient','draw'] if v in df_bl.columns]
df_bl['cog_comp'] = df_bl[cog_avail].mean(axis=1, skipna=True)

for raw, name in [('SAI_raw','SAI'),('cog_comp','BAI')]:
    mn, mx = df_bl[raw].min(), df_bl[raw].max()
    if mx > mn:
        df_bl[name] = ((df_bl[raw] - mn) / (mx - mn) * 100).clip(0,100)

if 'dadliv' in df_bl.columns:
    mn, mx = df_bl['dadliv'].min(), df_bl['dadliv'].max()
    df_bl['BoAI'] = ((df_bl['dadliv'] - mn) / (mx - mn) * 100).clip(0,100)

df_clean = df_bl.dropna(subset=['SAI','BAI','BoAI']).copy()

# === DESCRIPTIVES ===
print(f'=== CHARLS PILOT: Baseline Descriptives ===')
print(f'N = {len(df_clean):,} (age>=50, first wave, all 3 indices)')
print(f'Age: {df_clean["pubage"].mean():.1f} +- {df_clean["pubage"].std():.1f}')
female = (df_clean['ragender']==2).sum()
print(f'Female: {female} ({female/len(df_clean)*100:.1f}%)')
print(f'DSI prevalence: {df_clean["dsi"].mean()*100:.1f}%')
print(f'Vision impairment: {(df_clean["fi_vision"]==1).mean()*100:.1f}%')
print(f'Hearing impairment: {(df_clean["fi_hearing"]==1).mean()*100:.1f}%')
print(f'SAI: {df_clean["SAI"].mean():.0f}+-{df_clean["SAI"].std():.0f} | BAI: {df_clean["BAI"].mean():.0f}+-{df_clean["BAI"].std():.0f} | BoAI: {df_clean["BoAI"].mean():.0f}+-{df_clean["BoAI"].std():.0f}')
print(f'SAI-BAI r={df_clean[["SAI","BAI"]].corr().iloc[0,1]:.3f} | SAI-BoAI r={df_clean[["SAI","BoAI"]].corr().iloc[0,1]:.3f} | BAI-BoAI r={df_clean[["BAI","BoAI"]].corr().iloc[0,1]:.3f}')

# === LCA CLUSTERING ===
print(f'\n=== LCA (GMM) Clustering ===')
X = df_clean[['SAI','BAI','BoAI']].values
X_scaled = StandardScaler().fit_transform(X)

for k in range(3, 8):
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
    labels = gmm.fit_predict(X_scaled)
    bic = gmm.bic(X_scaled)
    sizes = np.bincount(labels)
    min_pct = sizes.min() / len(labels) * 100
    print(f'  k={k}: BIC={bic:.0f}, sizes={sorted(sizes)}, min_class={min_pct:.1f}%')

# Best k (BIC elbow + min 5%)
best_k = 5
gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=20)
df_clean['phenotype'] = gmm.fit_predict(X_scaled)

# === PHENOTYPE PROFILES ===
print(f'\n=== PHENOTYPE PROFILES (k={best_k}) ===')
means = df_clean.groupby('phenotype')[['SAI','BAI','BoAI','pubage','dsi']].agg(['mean','std','count'])

for p in range(best_k):
    sub = df_clean[df_clean['phenotype']==p]
    n = len(sub)
    s, b, bo = sub['SAI'].mean(), sub['BAI'].mean(), sub['BoAI'].mean()

    # Classify
    if s > 30 and b < 50:
        label = 'Sensory-First (感官前哨)'
    elif b < 40 and bo > 30:
        label = 'Brain-Resilient (脑韧型)'
    elif b > 50 and bo < 25:
        label = 'Body-Resilient (体韧型)'
    elif s > 30 and b > 50:
        label = 'Global Accelerated (全面衰老)'
    else:
        label = 'Successful Aging (全面年轻)'

    print(f'\n  Type {p+1}: {label} (n={n:,}, {n/len(df_clean)*100:.1f}%)')
    print(f'    Age: {sub["pubage"].mean():.0f}yr | Female: {(sub["ragender"]==2).mean()*100:.0f}% | DSI: {sub["dsi"].mean()*100:.1f}%')
    print(f'    SAI={s:.0f} | BAI={b:.0f} | BoAI={bo:.0f}')
    print(f'    CES-D: {sub["cesd10"].mean():.1f}')

# === DSI -> COGNITION basic model ===
print(f'\n=== BASIC REGRESSION: DSI -> Cognition ===')
import statsmodels.api as sm
# DSI effect
X_reg = sm.add_constant(df_clean[['dsi','pubage','ragender']].fillna(0))
y = df_clean['BAI']
try:
    model = sm.OLS(y, X_reg).fit()
    print(model.summary().tables[1])
except:
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(X_reg.fillna(0), y.fillna(0))
    print(f'  DSI beta = {lr.coef_[1]:.3f} (sklearn LinearRegression)')
    print(f'  R2 = {lr.score(X_reg.fillna(0), y.fillna(0)):.3f}')
