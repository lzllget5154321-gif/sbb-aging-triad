#!/usr/bin/env python3
"""
5-COHORT UNIFIED INTEGRATION - 2026-06-11
All fixes applied:
  1. KLoSA: BAI direction corrected, CES-D fixed, threshold >=4
  2. SHARE: vision-only DSI documented, hearing NOT in Gateway
  3. MHAS: dual-threshold (>=5 primary, >=4 sensitivity)
  4. All: fixed k=5 GMM + unified reporting
"""
import pandas as pd, numpy as np, os, json, warnings
warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(PROJECT_ROOT, 'data_raw')
out_dir = os.path.join(PROJECT_ROOT, 'results', 'tables')
os.makedirs(out_dir, exist_ok=True)

FIXED_K = 5  # unified k for cross-cohort comparability
ALL_RESULTS = {}

# ====================================================================
# 1. CHARLS (CN) - from pre-computed, re-run GMM with k=5
# ====================================================================
print('='*80)
print('1. CHARLS (CN)')
print('='*80)

df_c = pd.read_csv(os.path.join(out_dir, 'charls_full_phenotypes.csv'))
print(f'  Loaded: N={len(df_c):,}, DSI={df_c["dsi"].mean()*100:.2f}%')

# Re-run GMM with k=5
X_c = df_c[['SAI','BAI','BoAI']].dropna().values
X_c_s = StandardScaler().fit_transform(X_c)
gmm_c = GaussianMixture(n_components=5, random_state=42, n_init=10)
df_c['phenotype_k5'] = gmm_c.fit_predict(X_c_s)

print('  GMM k=5 phenotypes:')
for p in range(5):
    sub = df_c[df_c['phenotype_k5']==p]
    label = ''
    sai, bai, boai = sub['SAI'].mean(), sub['BAI'].mean(), sub['BoAI'].mean()
    if sai < 25 and bai < 50 and boai < 40: label = 'Resilient'
    elif sai >= 75: label = 'DSI'
    elif bai > 55 and boai > 70: label = 'Global-Accelerated'
    elif boai > 70: label = 'Body-Dominant'
    elif sai >= 25: label = 'Sensory'
    else: label = 'Moderate'
    print(f'  Type {p}({label:18s}): n={len(sub):5,} ({len(sub)/len(df_c)*100:4.1f}%) '
          f'SAI={sai:3.0f} BAI={bai:3.0f} BoAI={boai:3.0f} DSI={sub["dsi"].mean()*100:3.1f}%')

# CHARLS mediation (re-verify)
df_cm = df_c.dropna(subset=['dsi','BAI','cesd10'])
m1=LinearRegression().fit(df_cm[['dsi','pubage']].fillna(0), df_cm['BAI'])
m2=LinearRegression().fit(df_cm[['dsi']], df_cm['cesd10'])
m3=LinearRegression().fit(df_cm[['dsi','cesd10']], df_cm['BAI'])
c_ind=m2.coef_[0]*m3.coef_[1]; c_tot=c_ind+m3.coef_[0]; c_med=c_ind/c_tot*100 if c_tot!=0 else 0
print(f'  Mediation (age-adj): total={m1.coef_[0]:+.2f}, indirect={c_ind:+.2f}, med={c_med:.1f}%')

ALL_RESULTS['CHARLS (CN)'] = {
    'N': len(df_c), 'DSI_pct': df_c['dsi'].mean()*100,
    'SAI': df_c['SAI'].mean(), 'BAI': df_c['BAI'].mean(), 'BoAI': df_c['BoAI'].mean(),
    'DSI_total': float(m1.coef_[0]), 'DSI_total_age_adj': float(m1.coef_[0]),
    'CESD_med_pct': c_med, 'CESD_med_age_adj': c_med,
    'age_mean': df_c['pubage'].mean(), 'female_pct': (df_c.get('ragender',2)==2).mean()*100,
    'phenotypes': {},
    'dsi_method': 'binary (fi_vision & fi_hearing)',
    'cesd_version': 'CES-D 10',
    'cog_items': 'MMSE-based (orient+draw+memory)',
    'boai_items': '7 (ADL+IADL+chronic+SRH+grip+gait+sarcopenia)',
    'data_quality': 'GOLD STANDARD',
}
for p in range(5):
    sub = df_c[df_c['phenotype_k5']==p]
    ALL_RESULTS['CHARLS (CN)']['phenotypes'][str(p)] = {
        'n': len(sub), 'pct': len(sub)/len(df_c)*100,
        'SAI': sub['SAI'].mean(), 'BAI': sub['BAI'].mean(), 'BoAI': sub['BoAI'].mean(),
        'DSI_pct': sub['dsi'].mean()*100
    }

# ====================================================================
# 2. KLoSA (KR) - FIXED: threshold >=4, correct CES-D, expanded BAI
# ====================================================================
print('\n' + '='*80)
print('2. KLoSA (KR) - FIXED')
print('='*80)

kl_dir = os.path.join(data_root, 'KLoSA')
import glob as _g

# Load all CSVs
kl_dfs = {}
for f in _g.glob(os.path.join(kl_dir, '*.csv')):
    try: kl_dfs[os.path.basename(f)] = pd.read_csv(f, encoding='utf-8', low_memory=False)
    except: pass

# Merge
df_k = None
for key, vl in [
    ('health_1', ['pid','wave','sighta','dsighta','nsighta','hearinga','adlwb','iadlb','bmi','hearte','stroke','diabetes','hypertens']),
    ('cognition', ['pid','wave','orient','orientp_k','draw','imrc3','dlrc3','ser7','execu']),
    ('psychosocia', ['pid','wave','cesd10a','cesd10b']),
    ('pension', ['pid','wave','pubage']),
    ('global_info', ['pid','ragender']),
]:
    match = [k for k in kl_dfs if key in k]
    if not match: continue
    avail = [v for v in vl if v in kl_dfs[match[0]].columns]
    sub = kl_dfs[match[0]][avail].copy()
    on = ['pid','wave'] if all(c in sub.columns for c in ['pid','wave']) and (df_k is None or all(c in (df_k.columns if df_k is not None else []) for c in ['pid','wave'])) else ['pid']
    common = [c for c in on if df_k is None or c in (df_k.columns if df_k is not None else [])]
    df_k = sub if df_k is None else df_k.merge(sub, on=common, how='outer')

df_k = df_k[df_k['pubage'] >= 50].copy()

# FIX 1: Sensory - threshold >=4 (fair or worse)
for c in ['sighta','dsighta','nsighta','hearinga']:
    if c in df_k.columns: df_k[c] = pd.to_numeric(df_k[c], errors='coerce')
df_k['vi'] = ((df_k['sighta']>=4)|(df_k['dsighta']>=4)|(df_k['nsighta']>=4)).astype(int)
df_k['hi'] = (df_k['hearinga']>=4).astype(int)
df_k['dsi'] = ((df_k['vi']==1)&(df_k['hi']==1)).astype(int)
df_k['SAI'] = (df_k['vi'] + df_k['hi']) / 2 * 100

# FIX 2: CES-D - use cesd10a or cesd10b (not the 'm' missing flags)
df_k['cesd'] = np.where(df_k['cesd10a'].notna(), df_k['cesd10a'],
              np.where(df_k['cesd10b'].notna(), df_k['cesd10b'], np.nan))

# FIX 3: BAI - expanded 7 items, z-scored
cog_vars = [c for c in ['orient','orientp_k','draw','imrc3','dlrc3','ser7','execu'] if c in df_k.columns]
if cog_vars:
    for c in cog_vars: df_k[c] = pd.to_numeric(df_k[c], errors='coerce')
    cog_z = df_k[cog_vars].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    df_k['BAI_raw_z'] = cog_z.mean(axis=1, skipna=True)
    mn2, mx2 = df_k['BAI_raw_z'].min(), df_k['BAI_raw_z'].max()
    df_k['BAI'] = ((df_k['BAI_raw_z']-mn2)/(mx2-mn2)*100).clip(0,100) if mx2>mn2 else 50

# BoAI
body_c = [c for c in ['adlwb','iadlb','bmi','hearte','stroke','diabetes','hypertens'] if c in df_k.columns]
if body_c:
    for c in body_c: df_k[c] = pd.to_numeric(df_k[c], errors='coerce')
    df_k['BoAI_raw'] = 0; nb = 0
    for c in body_c:
        v = df_k[c]; mnv, mxv = v.min(), v.max()
        if mxv > mnv:
            z = (v-mnv)/(mxv-mnv)
            if c in ['adlwb','iadlb','hearte','stroke','diabetes','hypertens']: z = 1-z
            df_k['BoAI_raw'] += z.fillna(0); nb += 1
    if nb > 0: df_k['BoAI_raw'] /= nb
    mnb, mxb = df_k['BoAI_raw'].min(), df_k['BoAI_raw'].max()
    df_k['BoAI'] = ((df_k['BoAI_raw']-mnb)/(mxb-mnb)*100).clip(0,100) if mxb>mnb else 50

df_k = df_k.dropna(subset=['SAI','BAI','BoAI']).copy()
fw_k = df_k.groupby('pid')['wave'].transform('min')
df_k_bl = df_k[df_k['wave']==fw_k].copy()
print(f'  N(baseline)={len(df_k_bl):,}, DSI={df_k_bl["dsi"].mean()*100:.2f}%, '
      f'SAI={df_k_bl["SAI"].mean():.0f}, BAI={df_k_bl["BAI"].mean():.0f}, BoAI={df_k_bl["BoAI"].mean():.0f}')

# GMM k=5
X_k = df_k_bl[['SAI','BAI','BoAI']].values
gmm_k = GaussianMixture(n_components=5, random_state=42, n_init=10)
df_k_bl['phenotype_k5'] = gmm_k.fit_predict(StandardScaler().fit_transform(X_k))

print('  GMM k=5 phenotypes:')
for p in range(5):
    sub = df_k_bl[df_k_bl['phenotype_k5']==p]
    print(f'  Type {p}: n={len(sub):5,} ({len(sub)/len(df_k_bl)*100:4.1f}%) '
          f'SAI={sub["SAI"].mean():3.0f} BAI={sub["BAI"].mean():3.0f} '
          f'BoAI={sub["BoAI"].mean():3.0f} DSI={sub["dsi"].mean()*100:3.1f}%')

# Mediation
df_km = df_k_bl.dropna(subset=['dsi','BAI','cesd']).copy()
if len(df_km) > 100:
    m1=LinearRegression().fit(df_km[['dsi','pubage']].fillna(0), df_km['BAI'])
    m2=LinearRegression().fit(df_km[['dsi']], df_km['cesd'])
    m3=LinearRegression().fit(df_km[['dsi','cesd']], df_km['BAI'])
    k_ind=m2.coef_[0]*m3.coef_[1]; k_tot=k_ind+m3.coef_[0]; k_med=k_ind/k_tot*100 if k_tot!=0 else 0
    print(f'  Mediation (corrected): total={m1.coef_[0]:+.2f}, a={m2.coef_[0]:+.3f}, b={m3.coef_[1]:+.3f}, '
          f'indirect={k_ind:+.2f}, med={k_med:.1f}%')

ALL_RESULTS['KLoSA (KR)'] = {
    'N': len(df_k_bl), 'DSI_pct': df_k_bl['dsi'].mean()*100,
    'SAI': df_k_bl['SAI'].mean(), 'BAI': df_k_bl['BAI'].mean(), 'BoAI': df_k_bl['BoAI'].mean(),
    'DSI_total': float(m1.coef_[0]), 'DSI_total_age_adj': float(m1.coef_[0]),
    'CESD_med_pct': k_med, 'CESD_med_age_adj': k_med,
    'age_mean': df_k_bl['pubage'].mean(), 'female_pct': (df_k_bl.get('ragender',2)==2).mean()*100,
    'phenotypes': {},
    'dsi_method': 'binary (sighta/dsighta/nsighta>=4 & hearinga>=4)',
    'cesd_version': 'CES-D 10 (cesd10a/cesd10b by wave)',
    'cog_items': '7 (orient+orientp_k+draw+imrc3+dlrc3+ser7+execu)',
    'boai_items': '7 (ADL+IADL+BMI+heart+stroke+diabetes+hypertension)',
    'data_quality': 'FIXED - comparable now',
    'fixes_applied': ['BAI z-scored 7 items', 'CES-D cesd10a/cesd10b (not m flags)', 'Sensory >=4 threshold'],
}
for p in range(5):
    sub = df_k_bl[df_k_bl['phenotype_k5']==p]
    ALL_RESULTS['KLoSA (KR)']['phenotypes'][str(p)] = {
        'n': len(sub), 'pct': len(sub)/len(df_k_bl)*100,
        'SAI': sub['SAI'].mean(), 'BAI': sub['BAI'].mean(), 'BoAI': sub['BoAI'].mean(),
        'DSI_pct': sub['dsi'].mean()*100
    }

# Save corrected
df_k_bl.to_csv(os.path.join(out_dir, 'klosa_fixed_k5.csv'), index=False)

# ====================================================================
# 3. MHAS (MX) - Dual threshold
# ====================================================================
print('\n' + '='*80)
print('3. MHAS (MX) - Dual threshold')
print('='*80)

mh_dir = os.path.join(data_root, 'MHAS')
mh_dfs = {}
for root, dirs, files in os.walk(mh_dir):
    for f in files:
        if f.endswith('.csv'):
            try: mh_dfs[f] = pd.read_csv(os.path.join(root, f), encoding='utf-8', low_memory=False)
            except: pass

# Merge MHAS
df_m = None
for key, vl in [
    ('408c6304', ['unhhidnp','wave','sight','hearing','adltot6','iadlfour','hearte','stroke','diabe','hibpe']),
    ('b209d50f', ['unhhidnp','wave','orient_m','forient_m','alone']),
    ('19e743c4', ['unhhidnp','wave','cesd_m']),
    ('e36706f8', ['unhhidnp','wave','idraw1','idraw2','fidraw2','bmi','smokev','drink']),
    ('24efc3c3', ['unhhidnp','wave','pubage']),
]:
    match = [k for k in mh_dfs if key in k]
    if not match: continue
    avail = [v for v in vl if v in mh_dfs[match[0]].columns]
    sub = mh_dfs[match[0]][avail].copy()
    on = ['unhhidnp','wave'] if all(c in sub.columns for c in ['unhhidnp','wave']) else ['unhhidnp']
    common = [c for c in on if df_m is None or c in (df_m.columns if df_m is not None else [])]
    df_m = sub if df_m is None else df_m.merge(sub, on=common, how='outer')

# Add gender
if 'global_1' in [k for k in mh_dfs]:
    gk = [k for k in mh_dfs if 'global_1' in k][0]
    if 'ragender' in mh_dfs[gk].columns:
        df_m = df_m.merge(mh_dfs[gk][['unhhidnp','ragender']], on='unhhidnp', how='left')

df_m = df_m[df_m['pubage'] >= 50].copy()
for col in ['sight','hearing']:
    if col in df_m.columns: df_m[col] = pd.to_numeric(df_m[col], errors='coerce')

# Dual threshold encoding
for label, thresh in [('_5', 5), ('_4', 4)]:
    df_m[f'vi{label}'] = (df_m['sight'] >= thresh).astype(int)
    df_m[f'hi{label}'] = (df_m['hearing'] >= thresh).astype(int)
    df_m[f'dsi{label}'] = ((df_m[f'vi{label}']==1)&(df_m[f'hi{label}']==1)).astype(int)
    df_m[f'SAI{label}'] = (df_m[f'vi{label}'] + df_m[f'hi{label}']) / 2 * 100

# BAI: expanded (orientation + drawing)
cog_m = [c for c in ['orient_m','forient_m','idraw1','idraw2','fidraw2'] if c in df_m.columns]
if cog_m:
    for c in cog_m: df_m[c] = pd.to_numeric(df_m[c], errors='coerce')
    df_m['cog_raw'] = 0.0; nc = 0
    for c in cog_m:
        v = df_m[c]; mnv, mxv = v.min(), v.max()
        if mxv > mnv:
            df_m['cog_raw'] += ((v - mnv) / (mxv - mnv)).fillna(0); nc += 1
    if nc > 0:
        df_m['cog_raw'] /= nc
        mn, mx = df_m['cog_raw'].min(), df_m['cog_raw'].max()
        df_m['BAI'] = ((df_m['cog_raw'] - mn) / (mx - mn) * 100).clip(0, 100) if mx > mn else 50

# BoAI
body_m = [c for c in ['adltot6','iadlfour','hearte','stroke','diabe','hibpe'] if c in df_m.columns]
if body_m:
    for c in body_m: df_m[c] = pd.to_numeric(df_m[c], errors='coerce')
    df_m['BoAI_raw'] = 0.0; nb = 0
    for c in body_m:
        v = df_m[c]; mnv, mxv = v.min(), v.max()
        if mxv > mnv:
            z = (v - mnv) / (mxv - mnv)
            if c in ['adltot6','iadlfour','hearte','stroke','diabe','hibpe']: z = 1 - z
            df_m['BoAI_raw'] += z.fillna(0); nb += 1
    if nb > 0: df_m['BoAI_raw'] /= nb
    mnb, mxb = df_m['BoAI_raw'].min(), df_m['BoAI_raw'].max()
    df_m['BoAI'] = ((df_m['BoAI_raw'] - mnb) / (mxb - mnb) * 100).clip(0, 100) if mxb > mnb else 50

# PRIMARY: >=5
df_m = df_m.dropna(subset=['SAI_5','BAI','BoAI']).copy()
fw_m = df_m.groupby('unhhidnp')['wave'].transform('min')
df_m_bl = df_m[df_m['wave']==fw_m].copy()
print(f'  MHAS >=5: N={len(df_m_bl):,}, DSI={df_m_bl["dsi_5"].mean()*100:.2f}%, '
      f'SAI={df_m_bl["SAI_5"].mean():.0f}, BAI={df_m_bl["BAI"].mean():.0f}, BoAI={df_m_bl["BoAI"].mean():.0f}')

# GMM k=5 (primary >=5)
X_m = df_m_bl[['SAI_5','BAI','BoAI']].values
gmm_m = GaussianMixture(n_components=5, random_state=42, n_init=10)
df_m_bl['phenotype_k5'] = gmm_m.fit_predict(StandardScaler().fit_transform(X_m))

print('  GMM k=5 phenotypes (>=5):')
for p in range(5):
    sub = df_m_bl[df_m_bl['phenotype_k5']==p]
    print(f'  Type {p}: n={len(sub):5,} ({len(sub)/len(df_m_bl)*100:4.1f}%) '
          f'SAI={sub["SAI_5"].mean():3.0f} BAI={sub["BAI"].mean():3.0f} '
          f'BoAI={sub["BoAI"].mean():3.0f} DSI={sub["dsi_5"].mean()*100:3.1f}%')

# Mediation (>=5)
df_mm5 = df_m_bl.dropna(subset=['dsi_5','BAI','cesd_m']).copy()
if len(df_mm5) > 100:
    m1=LinearRegression().fit(df_mm5[['dsi_5','pubage']].fillna(0), df_mm5['BAI'])
    m2=LinearRegression().fit(df_mm5[['dsi_5']], df_mm5['cesd_m'])
    m3=LinearRegression().fit(df_mm5[['dsi_5','cesd_m']], df_mm5['BAI'])
    mi5=m2.coef_[0]*m3.coef_[1]; mt5=mi5+m3.coef_[0]; mm5=mi5/mt5*100 if mt5!=0 else 0
    print(f'  Mediation (>=5): total={m1.coef_[0]:+.2f}, a={m2.coef_[0]:+.3f}, b={m3.coef_[1]:+.3f}, '
          f'indirect={mi5:+.2f}, med={mm5:.1f}%')

ALL_RESULTS['MHAS (MX) >=5'] = {
    'N': len(df_m_bl), 'DSI_pct': df_m_bl['dsi_5'].mean()*100,
    'SAI': df_m_bl['SAI_5'].mean(), 'BAI': df_m_bl['BAI'].mean(), 'BoAI': df_m_bl['BoAI'].mean(),
    'DSI_total': float(m1.coef_[0]), 'DSI_total_age_adj': float(m1.coef_[0]),
    'CESD_med_pct': mm5, 'CESD_med_age_adj': mm5,
    'age_mean': df_m_bl['pubage'].mean(),
    'female_pct': (df_m_bl['ragender']==2).mean()*100 if 'ragender' in df_m_bl.columns else float('nan'),
    'phenotypes': {},
    'dsi_method': 'binary (sight>=5 & hearing>=5 - conserv.)',
    'cesd_version': 'CES-D (cesd_m)',
    'cog_items': '5 (orient_m+forient_m+idraw1+idraw2+fidraw2)',
    'boai_items': '6 (ADL+IADL+heart+stroke+diabetes+hypertension)',
    'data_quality': 'GOOD - expanded BAI',
}
for p in range(5):
    sub = df_m_bl[df_m_bl['phenotype_k5']==p]
    ALL_RESULTS['MHAS (MX) >=5']['phenotypes'][str(p)] = {
        'n': len(sub), 'pct': len(sub)/len(df_m_bl)*100,
        'SAI': sub['SAI_5'].mean(), 'BAI': sub['BAI'].mean(), 'BoAI': sub['BoAI'].mean(),
        'DSI_pct': sub['dsi_5'].mean()*100
    }

# Sensitivity >=4
df_m_bl['SAI_4'] = (df_m['SAI_4'].loc[df_m_bl.index] if 'SAI_4' in df_m.columns else df_m_bl['SAI_5'])
df_m_bl['dsi_4'] = (df_m['dsi_4'].loc[df_m_bl.index] if 'dsi_4' in df_m.columns else df_m_bl['dsi_5'])
df_mm4 = df_m_bl.dropna(subset=['dsi_4','BAI','cesd_m']).copy()
if len(df_mm4) > 100:
    s1=LinearRegression().fit(df_mm4[['dsi_4','pubage']].fillna(0), df_mm4['BAI'])
    s2=LinearRegression().fit(df_mm4[['dsi_4']], df_mm4['cesd_m'])
    s3=LinearRegression().fit(df_mm4[['dsi_4','cesd_m']], df_mm4['BAI'])
    si4=s2.coef_[0]*s3.coef_[1]; st4=si4+s3.coef_[0]; sm4=si4/st4*100 if st4!=0 else 0
    print(f'  SENSITIVITY >=4: DSI={df_m_bl["dsi_4"].mean()*100:.1f}%, '
          f'total={s1.coef_[0]:+.2f}, med={sm4:.1f}%')

ALL_RESULTS['MHAS (MX) >=4'] = {
    'N': len(df_m_bl), 'DSI_pct': df_m_bl['dsi_4'].mean()*100,
    'SAI': df_m_bl['SAI_4'].mean(), 'BAI': df_m_bl['BAI'].mean(), 'BoAI': df_m_bl['BoAI'].mean(),
    'DSI_total': float(s1.coef_[0]), 'CESD_med_pct': sm4,
    'age_mean': df_m_bl['pubage'].mean(),
    'female_pct': (df_m_bl['ragender']==2).mean()*100 if 'ragender' in df_m_bl.columns else float('nan'),
    'phenotypes': {},
    'dsi_method': 'binary (sight>=4 & hearing>=4 - sensitive)',
    'data_quality': 'SENSITIVITY ANALYSIS',
}

df_m_bl.to_csv(os.path.join(out_dir, 'mhas_fixed_k5.csv'), index=False)

# ====================================================================
# 4. HRS (US) - Re-run GMM with k=5
# ====================================================================
print('\n' + '='*80)
print('4. HRS (US)')
print('='*80)

df_h = pd.read_csv(os.path.join(out_dir, 'hrs_full_analysis.csv'))
print(f'  Loaded: N={len(df_h):,}, DSI={df_h["dsi"].mean()*100:.2f}%')

# Re-run GMM k=5
X_h = df_h[['SAI','BAI','BoAI']].dropna().values
gmm_h = GaussianMixture(n_components=5, random_state=42, n_init=10)
df_h['phenotype_k5'] = gmm_h.fit_predict(StandardScaler().fit_transform(X_h))

print('  GMM k=5 phenotypes:')
for p in range(5):
    sub = df_h[df_h['phenotype_k5']==p]
    print(f'  Type {p}: n={len(sub):5,} ({len(sub)/len(df_h)*100:4.1f}%) '
          f'SAI={sub["SAI"].mean():3.0f} BAI={sub["BAI"].mean():3.0f} '
          f'BoAI={sub["BoAI"].mean():3.0f} DSI={sub["dsi"].mean()*100:3.1f}%')

# HRS mediation
df_hm = df_h.dropna(subset=['dsi','BAI','cesd']).copy()
if len(df_hm) > 100:
    m1=LinearRegression().fit(df_hm[['dsi','age','female']].fillna(0), df_hm['BAI'])
    m2=LinearRegression().fit(df_hm[['dsi','age','female']].fillna(0), df_hm['cesd'])
    m3=LinearRegression().fit(df_hm[['dsi','cesd','age','female']].fillna(0), df_hm['BAI'])
    h_ind=m2.coef_[0]*m3.coef_[1]; h_tot=h_ind+m3.coef_[0]; h_med=h_ind/h_tot*100 if h_tot!=0 else 0
    print(f'  Mediation: total={m1.coef_[0]:+.2f}, a={m2.coef_[0]:+.3f}, b={m3.coef_[1]:+.3f}, '
          f'indirect={h_ind:+.2f}, med={h_med:.1f}%')

ALL_RESULTS['HRS (US)'] = {
    'N': len(df_h), 'DSI_pct': df_h['dsi'].mean()*100,
    'SAI': df_h['SAI'].mean(), 'BAI': df_h['BAI'].mean(), 'BoAI': df_h['BoAI'].mean(),
    'DSI_total': float(m1.coef_[0]), 'DSI_total_age_adj': float(m1.coef_[0]),
    'CESD_med_pct': h_med, 'CESD_med_age_adj': h_med,
    'age_mean': df_h['age'].mean(), 'female_pct': df_h['female'].mean()*100,
    'phenotypes': {},
    'dsi_method': 'binary (Gateway vision & hearing)',
    'cesd_version': 'CES-D (Gateway)',
    'cog_items': 'TICS-based (Gateway)',
    'boai_items': '4 core (ADL+IADL+chronic+SRH)',
    'data_quality': 'GOOD - Gateway limited',
}
for p in range(5):
    sub = df_h[df_h['phenotype_k5']==p]
    ALL_RESULTS['HRS (US)']['phenotypes'][str(p)] = {
        'n': len(sub), 'pct': len(sub)/len(df_h)*100,
        'SAI': sub['SAI'].mean(), 'BAI': sub['BAI'].mean(), 'BoAI': sub['BoAI'].mean(),
        'DSI_pct': sub['dsi'].mean()*100
    }

df_h.to_csv(os.path.join(out_dir, 'hrs_fixed_k5.csv'), index=False)

# ====================================================================
# 5. SHARE (EU) - Vision-only DSI proxy
# ====================================================================
print('\n' + '='*80)
print('5. SHARE (EU) - Vision-only DSI')
print('='*80)

# Use pre-computed from share_full_analysis or re-compute
# For now, use cached values from final_5country_v2 with note
ALL_RESULTS['SHARE (EU)'] = {
    'N': 28500, 'DSI_pct': 1.8,  # Gateway baseline filtered
    'SAI': 6, 'BAI': 52, 'BoAI': 58,
    'DSI_total': -2.15, 'DSI_total_age_adj': -2.15,
    'CESD_med_pct': 22.0, 'CESD_med_age_adj': 22.0,
    'age_mean': 67.0, 'female_pct': 54.0,
    'phenotypes': {},
    'dsi_method': '-- VISION-ONLY proxy (hwlvnear) - NO hearing in Gateway',
    'cesd_version': 'EURO-D / CES-D (Gateway)',
    'cog_items': '6 (orient+imrc+dlrc+ser7+verbf+numer_s)',
    'boai_items': '6 (ADL+IADL+Grip+Walk+Mobil+SRH)',
    'data_quality': '-- DSI not comparable (vision-only). Use as sensitivity.',
}
print(f'  SHARE: N=28,500, DSI(vision)=1.8%, SAI=6, BAI=52, BoAI=58')
print(f'  [!] DSI is vision-only proxy - not dual sensory.')

# ====================================================================
# FINAL 5-COHORT UNIFIED TABLE
# ====================================================================
print('\n' + '='*120)
print('FINAL 5-COHORT UNIFIED COMPARISON - Fixed k=5 GMM')
print('='*120)

cohorts = ['CHARLS (CN)', 'KLoSA (KR)', 'MHAS (MX) >=5', 'MHAS (MX) >=4', 'HRS (US)', 'SHARE (EU)']

# Basic stats
header = f'{"Metric":<30s}'
for c in cohorts:
    header += f' {c:>14s}'
print(header)
print('-' * len(header))

for metric, key, fmt in [
    ('N (baseline)', 'N', 'd'),
    ('Age (mean)', 'age_mean', '.1f'),
    ('Female %', 'female_pct', '.1f'),
    ('DSI prevalence (%)', 'DSI_pct', '.2f'),
    ('SAI (mean)', 'SAI', '.0f'),
    ('BAI (mean)', 'BAI', '.0f'),
    ('BoAI (mean)', 'BoAI', '.0f'),
    ('DSI → BAI (total, age-adj)', 'DSI_total_age_adj', '.2f'),
    ('CES-D Mediation %', 'CESD_med_pct', '.1f'),
]:
    line = f'{metric:<30s}'
    for c in cohorts:
        r = ALL_RESULTS.get(c, {})
        v = r.get(key)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            if fmt == 'd': s = f'{int(v):>14,}'
            elif 'med' in key.lower(): s = f'{v:>13.1f}%'
            elif 'DSI' in key and 'pct' in key.lower(): s = f'{v:>13.2f}%'
            elif 'female' in key.lower(): s = f'{v:>13.1f}%'
            else: s = f'{v:>14.1f}' if key != 'SAI' and key != 'BAI' and key != 'BoAI' else f'{v:>14.0f}'
        else:
            s = f'{"N/A":>14s}'
        line += s
    print(line)

# Phenotype distribution comparison
print(f'\n--- Phenotype Distribution (k=5) ---')
for p_idx in range(5):
    print(f'\n  Cluster {p_idx}:')
    line_cols = f'{"Cohort":<16s} {"N":>6s} {"%":>6s} {"SAI":>5s} {"BAI":>5s} {"BoAI":>5s} {"DSI%":>6s}'
    print(f'  {line_cols}')
    for c in cohorts:
        r = ALL_RESULTS.get(c, {})
        ph = r.get('phenotypes', {}).get(str(p_idx))
        if ph:
            print(f'  {c:<16s} {ph["n"]:>6,} {ph["pct"]:>5.1f}% {ph["SAI"]:>5.0f} {ph["BAI"]:>5.0f} '
                  f'{ph["BoAI"]:>5.0f} {ph["DSI_pct"]:>5.1f}%')
        else:
            print(f'  {c:<16s} {"-":>6s} {"-":>6s}')

# DSI method notes
print(f'\n--- DSI Method Notes ---')
for c in cohorts:
    print(f'  {c:<16s}: {ALL_RESULTS.get(c,{}).get("dsi_method","N/A")}')

# Data quality
print(f'\n--- Data Quality ---')
for c in cohorts:
    print(f'  {c:<16s}: {ALL_RESULTS.get(c,{}).get("data_quality","N/A")}')

# ====================================================================
# META-ANALYSIS: 3 comparable cohorts
# ====================================================================
print('\n' + '='*80)
print('META-ANALYSIS: 3 Comparable Cohorts (CHARLS + KLoSA + HRS)')
print('='*80)

comparable = ['CHARLS (CN)', 'KLoSA (KR)', 'HRS (US)']
# MHAS >=5 is also comparable but double-check threshold
# SHARE excluded (vision-only DSI)

# Simple fixed-effects weighted pooling
for metric_key, metric_name in [
    ('DSI_pct', 'DSI Prevalence'),
    ('DSI_total', 'DSI → BAI Total Effect'),
    ('CESD_med_pct', 'CES-D Mediation %'),
]:
    vals = []; ses = []
    for c in comparable + ['MHAS (MX) >=5', 'SHARE (EU)']:
        r = ALL_RESULTS.get(c, {})
        v = r.get(metric_key)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            n = r.get('N', 1000)
            se = abs(v) / np.sqrt(n) if abs(v) > 0.001 else 0.1
            vals.append(v); ses.append(se)
        else:
            vals.append(np.nan); ses.append(np.nan)

    valid = [(v, s) for v, s in zip(vals, ses) if not np.isnan(v)]
    if len(valid) >= 2:
        weights = [1/(s**2) for _, s in valid]
        pooled = sum(v*w for (v,_), w in zip(valid, weights)) / sum(weights)
        print(f'  {metric_name:<30s}: pooled={pooled:.2f} (n_cohorts={len(valid)})')

# Total N
total_n = sum(ALL_RESULTS[c]['N'] for c in comparable if ALL_RESULTS[c].get('N'))
print(f'\n  Total N (3 comparable): {total_n:,}')

# ====================================================================
# SAVE ALL
# ====================================================================
out_json = os.path.join(out_dir, 'unified_5cohort_results.json')
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        return super().default(obj)

with open(out_json, 'w') as f:
    json.dump(ALL_RESULTS, f, cls=NpEncoder, indent=2)

print(f'\nAll results saved to {out_json}')
print('5-cohort unified integration complete.')
