# Multi-cohort replication: KLoSA + MHAS
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(PROJECT_ROOT, 'data_raw')
out_dir = os.path.join(PROJECT_ROOT, 'results', 'tables')

# ============================================================
# KLoSA ANALYSIS
# ============================================================
print('='*60)
print('KLoSA (Korea) Analysis')
print('='*60)

klosa_dir = os.path.join(data_root, 'KLoSA')
klosa = {}
for f in os.listdir(klosa_dir):
    if f.endswith('.csv'):
        try:
            klosa[f] = pd.read_csv(os.path.join(klosa_dir, f), encoding='utf-8', low_memory=False)
        except:
            pass

# Extract key variables
def get_sub(df_dict, key, vars_list):
    for k, df in df_dict.items():
        if key in k:
            avail = [v for v in vars_list if v in df.columns]
            return df[avail].copy()
    return None

df_k = None
merges = [
    ('health_1', ['pid','wave','sighta','dsighta','nsighta','hearinga','glasses',
                   'adlwb','iadlb','bmi','weight','height','hearte','stroke','diabetes','hypertens']),
    ('cognition', ['pid','wave','orient','orientp_k','draw']),
    ('physical', ['pid','wave','lgrip1','lgrip2','rgrip1','rgrip2']),
    ('psychosocia', ['pid','wave','cesd10a','cesd10am','cesd10b','cesd10bm','fsadl','sleeprl']),
    ('demographic', ['pid','wave','agey']),
    ('global_info', ['pid','ragender']),
    ('health_2', ['pid','wave','smokev','drinkev']),
    ('pension', ['pid','wave','pubage']),
]

for key, vars_list in merges:
    sub = get_sub(klosa, key, vars_list)
    if sub is None: continue
    on_cols = ['pid','wave'] if 'wave' in sub.columns else ['pid']
    common = [c for c in on_cols if c in (df_k.columns if df_k is not None else sub.columns)]
    if df_k is None:
        df_k = sub
    else:
        df_k = df_k.merge(sub, on=common, how='outer')

# Age filter (use agey or pubage)
age_col = 'agey' if 'agey' in df_k.columns else 'pubage'
df_k = df_k[df_k[age_col] >= 50].copy()
print(f'KLoSA age>=50: {len(df_k):,} rows, {df_k["pid"].nunique():,} IDs')

# Sensory
for col in ['sighta','dsighta','nsighta']:
    if col in df_k.columns:
        df_k[col] = pd.to_numeric(df_k[col], errors='coerce')
df_k['vision_imp'] = 0
if 'sighta' in df_k.columns:
    df_k['vision_imp'] = ((df_k['sighta']==1) | (df_k.get('dsighta',0)==1) | (df_k.get('nsighta',0)==1)).astype(int)
if 'hearinga' in df_k.columns:
    df_k['hearinga'] = pd.to_numeric(df_k['hearinga'], errors='coerce')
    df_k['hearing_imp'] = (df_k['hearinga']==1).astype(int)
else:
    df_k['hearing_imp'] = 0

df_k['dsi'] = ((df_k['vision_imp']==1) & (df_k['hearing_imp']==1)).astype(int)
df_k['SAI'] = (df_k['vision_imp'] + df_k['hearing_imp']) / 2 * 100

# Cognitive
cog_cols = [c for c in ['orient','orientp_k','draw'] if c in df_k.columns]
if cog_cols:
    df_k['cog_raw'] = df_k[cog_cols].mean(axis=1, skipna=True)
    mn, mx = df_k['cog_raw'].min(), df_k['cog_raw'].max()
    df_k['BAI'] = ((df_k['cog_raw'] - mn) / (mx - mn) * 100).clip(0, 100)

# Body
body_comp = []
for c in ['adlwb','iadlb','bmi','hearte','stroke','diabetes','hypertens']:
    if c in df_k.columns:
        body_comp.append(c)
if body_comp:
    df_k['BoAI_raw'] = 0
    n = 0
    for c in body_comp:
        vals = pd.to_numeric(df_k[c], errors='coerce')
        mn_v, mx_v = vals.min(), vals.max()
        if mx_v > mn_v and vals.notna().sum() > 100:
            z = (vals - mn_v) / (mx_v - mn_v)
            if c in ['adlwb','iadlb','hearte','stroke','diabetes','hypertens']:
                z = 1 - z
            df_k['BoAI_raw'] += z.fillna(0)
            n += 1
    if n > 0:
        df_k['BoAI_raw'] /= n
        mn_b, mx_b = df_k['BoAI_raw'].min(), df_k['BoAI_raw'].max()
        df_k['BoAI'] = ((df_k['BoAI_raw'] - mn_b) / (mx_b - mn_b) * 100).clip(0, 100)

# Baseline only
df_k = df_k.dropna(subset=['SAI','BAI','BoAI']).copy()
first_wave = df_k.groupby('pid')['wave'].transform('min')
df_k_bl = df_k[df_k['wave'] == first_wave].copy()
print(f'KLoSA baseline (all indices): N={len(df_k_bl):,}')

# LCA
X_k = df_k_bl[['SAI','BAI','BoAI']].values
X_k_s = StandardScaler().fit_transform(X_k)
gmm_k = GaussianMixture(n_components=5, random_state=42, n_init=10)
df_k_bl['phenotype'] = gmm_k.fit_predict(X_k_s)

print('\nKLoSA 5 phenotypes:')
for p in range(5):
    sub = df_k_bl[df_k_bl['phenotype']==p]
    print(f'  Type {p+1}: n={len(sub):5,} ({len(sub)/len(df_k_bl)*100:4.1f}%) '
          f'SAI={sub["SAI"].mean():3.0f} BAI={sub["BAI"].mean():3.0f} '
          f'BoAI={sub["BoAI"].mean():3.0f} DSI={sub["dsi"].mean()*100:3.1f}% '
          f'Age={sub.get(age_col, pd.Series([0])).mean():.0f}')

# ============================================================
# CROSS-NATIONAL COMPARISON
# ============================================================
print('\n' + '='*60)
print('CROSS-NATIONAL: CHARLS vs KLoSA')
print('='*60)

# Load CHARLS results
charls_path = os.path.join(out_dir, 'charls_v2_phenotypes.csv')
df_c = pd.read_csv(charls_path) if os.path.exists(charls_path) else None

if df_c is not None:
    print(f'\n{"Metric":<30s} {"CHARLS":>12s} {"KLoSA":>12s}')
    print('-'*55)
    for metric, c_col, k_col, fmt in [
        ('N (baseline)', 'id', 'pid', 'd'),
        ('DSI prevalence (%)', 'dsi', 'dsi', '.1f'),
        ('Mean SAI', 'SAI', 'SAI', '.0f'),
        ('Mean BAI', 'BAI', 'BAI', '.0f'),
        ('Mean BoAI', 'BoAI', 'BoAI', '.0f'),
    ]:
        if c_col in df_c.columns and k_col in df_k_bl.columns:
            c_val = df_c[c_col].mean() if fmt == '.1f' else (len(df_c) if fmt == 'd' else df_c[c_col].mean())
            k_val = df_k_bl[k_col].mean() if fmt == '.1f' else (len(df_k_bl) if fmt == 'd' else df_k_bl[k_col].mean())
            if fmt == 'd':
                print(f'{metric:<30s} {int(c_val):>12,} {int(k_val):>12,}')
            elif fmt == '.0f':
                print(f'{metric:<30s} {c_val:>12.0f} {k_val:>12.0f}')
            else:
                print(f'{metric:<30s} {c_val*100:>11.1f}% {k_val*100:>10.1f}%')

# Save KLoSA
df_k_bl[['pid','wave','SAI','BAI','BoAI','dsi','phenotype']].to_csv(
    os.path.join(out_dir, 'klosa_phenotypes.csv'), index=False)
print(f'\nKLoSA results saved')
print('Multi-cohort replication complete')
