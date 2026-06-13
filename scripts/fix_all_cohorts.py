# Final fix: KLoSA sample + MHAS encoding + HRS test
import pandas as pd, numpy as np, os, warnings; warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(PROJECT_ROOT, 'data_raw')
out_dir = os.path.join(PROJECT_ROOT, 'results', 'tables')
os.makedirs(out_dir, exist_ok=True)

# ===================== FIX 3: HRS TEST =====================
print('=== FIX 3: HRS RAND File Test ===')
try:
    hrs_path = os.path.join(data_root, 'HRS', 'HRS', 'RAND HRS Longitudinal File 2020', 'randhrs1992_2020v2.dta')
    if os.path.exists(hrs_path):
        df_hrs = pd.read_stata(hrs_path, columns=['hhidpn','ragender','rabyear','r1agey_b'])
        print(f'HRS loaded: {len(df_hrs):,} rows')
        print(f'Age range: {df_hrs["r1agey_b"].min():.0f}-{df_hrs["r1agey_b"].max():.0f}')
        print(f'Female: {(df_hrs["ragender"]==2).mean()*100:.0f}%')
    else:
        print(f'HRS not found at {hrs_path}')
        # Search
        for root, dirs, files in os.walk(os.path.join(data_root, 'HRS')):
            for f in files:
                if 'randhrs' in f.lower() and f.endswith('.dta'):
                    print(f'  Found: {os.path.join(root, f)}')
except Exception as e:
    print(f'HRS error: {e}')

# ===================== FIX 1+2: KLoSA + MHAS COMPLETE RE-RUN =====================
results = {}

# --- KLoSA (Fix 1: proper pubage merge) ---
print('\n=== FIX 1: KLoSA complete re-run ===')
kl_dir = os.path.join(data_root, 'KLoSA')
kl_dfs = {}
for f in os.listdir(kl_dir):
    if f.endswith('.csv'):
        try: kl_dfs[f] = pd.read_csv(os.path.join(kl_dir, f), encoding='utf-8', low_memory=False)
        except: pass

df_k = None
merges_k = [
    ('health_1', ['pid','wave','sighta','dsighta','nsighta','hearinga','glasses','adlwb','iadlb',
                   'bmi','weight','height','hearte','stroke','diabetes','hypertens']),
    ('cognition', ['pid','wave','orient','orientp_k','draw']),
    ('physical', ['pid','wave','lgrip1','lgrip2','rgrip1','rgrip2']),
    ('psychosocia', ['pid','wave','cesd10a','cesd10am','cesd10b','cesd10bm','fsadl','sleeprl']),
    ('demographic', ['pid','wave','agey']),
    ('global_info', ['pid','ragender']),
    ('pension', ['pid','wave','pubage']),
    ('health_2', ['pid','wave','smokev','drinkev']),
]

for key, vl in merges_k:
    match = [k for k in kl_dfs if key in k]
    if not match: continue
    avail = [v for v in vl if v in kl_dfs[match[0]].columns]
    sub = kl_dfs[match[0]][avail].copy()
    on = ['pid','wave'] if 'wave' in sub.columns and 'wave' in (df_k.columns if df_k is not None else ['wave']) else ['pid']
    common = [c for c in on if c in (df_k.columns if df_k is not None else sub.columns)]
    df_k = sub if df_k is None else df_k.merge(sub, on=common, how='outer')

# Use best age
df_k['age'] = df_k['pubage'].fillna(df_k['agey'])
df_k = df_k[df_k['age'] >= 50].copy()
print(f'KLoSA age>=50: {len(df_k):,} rows, {df_k["pid"].nunique():,} IDs')

# Sensory
for c in ['sighta','dsighta','nsighta','hearinga']:
    if c in df_k.columns: df_k[c] = pd.to_numeric(df_k[c], errors='coerce').fillna(0)
df_k['vi'] = ((df_k['sighta']==1)|(df_k['dsighta']==1)|(df_k['nsighta']==1)).astype(int)
df_k['hi'] = (df_k['hearinga']==1).astype(int)
df_k['dsi'] = ((df_k['vi']==1)&(df_k['hi']==1)).astype(int)
df_k['SAI'] = (df_k['vi']+df_k['hi'])/2*100

# CESD composite
cesd_c = [c for c in ['cesd10a','cesd10am','cesd10b','cesd10bm'] if c in df_k.columns]
df_k['cesd'] = df_k[cesd_c].mean(axis=1, skipna=True) if cesd_c else 0

# BAI
cog_c = [c for c in ['orient','orientp_k','draw'] if c in df_k.columns]
if cog_c:
    raw = df_k[cog_c].mean(axis=1, skipna=True); mn, mx = raw.min(), raw.max()
    df_k['BAI'] = ((raw-mn)/(mx-mn)*100).clip(0,100)

# BoAI
body_c = [c for c in ['adlwb','iadlb','bmi','hearte','stroke','diabetes','hypertens'] if c in df_k.columns]
if body_c:
    df_k['BoAI_raw'] = 0; nb = 0
    for c in body_c:
        v = pd.to_numeric(df_k[c], errors='coerce'); mnv, mxv = v.min(), v.max()
        if mxv > mnv:
            z = (v-mnv)/(mxv-mnv)
            if c in ['adlwb','iadlb','hearte','stroke','diabetes','hypertens']: z = 1-z
            df_k['BoAI_raw'] += z.fillna(0); nb += 1
    if nb > 0: df_k['BoAI_raw'] /= nb
    mnb, mxb = df_k['BoAI_raw'].min(), df_k['BoAI_raw'].max()
    df_k['BoAI'] = ((df_k['BoAI_raw']-mnb)/(mxb-mnb)*100).clip(0,100)

df_k = df_k.dropna(subset=['SAI','BAI','BoAI']).copy()
fw_k = df_k.groupby('pid')['wave'].transform('min')
df_k_bl = df_k[df_k['wave']==fw_k].copy()
print(f'KLoSA baseline: N={len(df_k_bl):,}')

# LCA
X_k = df_k_bl[['SAI','BAI','BoAI']].values
gmm_k = GaussianMixture(n_components=5, random_state=42, n_init=10)
df_k_bl['phenotype'] = gmm_k.fit_predict(StandardScaler().fit_transform(X_k))

print('KLoSA 5 phenotypes:')
for p in range(5):
    sub = df_k_bl[df_k_bl['phenotype']==p]
    print(f'  Type {p+1}: n={len(sub):5,} ({len(sub)/len(df_k_bl)*100:4.1f}%) SAI={sub["SAI"].mean():3.0f} BAI={sub["BAI"].mean():3.0f} BoAI={sub["BoAI"].mean():3.0f} DSI={sub["dsi"].mean()*100:3.1f}%')

# Mediation
df_km = df_k_bl.dropna(subset=['dsi','BAI','cesd']).copy()
if len(df_km) > 100:
    m1=LinearRegression().fit(df_km[['dsi','pubage']].fillna(0), df_km['BAI'])
    m2=LinearRegression().fit(df_km[['dsi']], df_km['cesd'])
    m3=LinearRegression().fit(df_km[['dsi','cesd']], df_km['BAI'])
    ind_k=m2.coef_[0]*m3.coef_[1]; tot_k=ind_k+m3.coef_[0]
    mp_k=ind_k/tot_k*100 if tot_k!=0 else 0
    print(f'KLoSA mediation (N={len(df_km):,}): CES-D = {mp_k:.1f}% (total={tot_k:.2f})')
    results['KLoSA'] = {'n': len(df_k_bl), 'dsi': df_k_bl['dsi'].mean()*100, 'sai': df_k_bl['SAI'].mean(),
                        'bai': df_k_bl['BAI'].mean(), 'boai': df_k_bl['BoAI'].mean(),
                        'total': tot_k, 'med_pct': mp_k}
    df_k_bl.to_csv(os.path.join(out_dir, 'klosa_fixed.csv'), index=False)

# --- MHAS (Fix 2: sight/hearing remap) ---
print('\n=== FIX 2: MHAS with corrected encoding ===')
mh_dir = os.path.join(data_root, 'MHAS')
mh_dfs = {}
for root, dirs, files in os.walk(mh_dir):
    for f in files:
        if f.endswith('.csv'):
            try: mh_dfs[f] = pd.read_csv(os.path.join(root, f), encoding='utf-8', low_memory=False)
            except: pass

df_m = None
for key, vl in [
    ('408c6304', ['unhhidnp','wave','sight','hearing','adltot6','iadlfour','hearte','stroke','diabetes','hypertens']),
    ('b209d50f', ['unhhidnp','wave','orient_m','forient_m','alone']),
    ('19e743c4', ['unhhidnp','wave','cesd_m']),
    ('e36706f8', ['unhhidnp','wave','bmi','smokev','drink']),
    ('24efc3c3', ['unhhidnp','wave','pubage']),
    ('global_1',  ['unhhidnp','ragender']),
]:
    match = [k for k in mh_dfs if key in k]
    if not match: continue
    avail = [v for v in vl if v in mh_dfs[match[0]].columns]
    sub = mh_dfs[match[0]][avail].copy()
    on = ['unhhidnp','wave'] if 'wave' in sub.columns and 'wave' in (df_m.columns if df_m is not None else ['wave']) else ['unhhidnp']
    common = [c for c in on if c in (df_m.columns if df_m is not None else sub.columns)]
    df_m = sub if df_m is None else df_m.merge(sub, on=common, how='outer')

df_m = df_m[df_m['pubage'] >= 50].copy()
print(f'MHAS age>=50: {len(df_m):,} rows, {df_m["unhhidnp"].nunique():,} IDs')

# FIX 2: MHAS sight/hearing: 1=excellent ... 5-6=impairment (blind/deaf)
for col in ['sight','hearing']:
    if col in df_m.columns:
        df_m[col] = pd.to_numeric(df_m[col], errors='coerce')
        df_m[f'{col}_imp'] = (df_m[col] >= 5).astype(int)

df_m['dsi'] = ((df_m.get('sight_imp',0)==1) & (df_m.get('hearing_imp',0)==1)).astype(int)
df_m['SAI'] = (df_m.get('sight_imp',0) + df_m.get('hearing_imp',0)) / 2 * 100
print(f'MHAS sight_imp: {df_m["sight_imp"].mean()*100:.1f}%, hearing_imp: {df_m.get("hearing_imp",pd.Series([0])).mean()*100:.1f}%, DSI: {df_m["dsi"].mean()*100:.1f}%')

# Cognitive
cog_m = [c for c in ['orient_m','forient_m'] if c in df_m.columns]
if cog_m:
    raw = df_m[cog_m].mean(axis=1,skipna=True); mn,mx=raw.min(),raw.max()
    df_m['BAI'] = ((raw-mn)/(mx-mn)*100).clip(0,100) if mx>mn else 50

# Body
body_m = [c for c in ['adltot6','iadlfour','hearte','stroke','diabetes','hypertens'] if c in df_m.columns]
if body_m:
    df_m['BoAI_raw'] = 0; nb = 0
    for c in body_m:
        v = pd.to_numeric(df_m[c],errors='coerce'); mnv,mxv=v.min(),v.max()
        if mxv>mnv:
            z = (v-mnv)/(mxv-mnv)
            if c in ['adltot6','iadlfour','hearte','stroke','diabetes','hypertens']: z = 1-z
            df_m['BoAI_raw'] += z.fillna(0); nb += 1
    if nb>0: df_m['BoAI_raw']/=nb
    mnb,mxb = df_m['BoAI_raw'].min(), df_m['BoAI_raw'].max()
    df_m['BoAI'] = ((df_m['BoAI_raw']-mnb)/(mxb-mnb)*100).clip(0,100)

df_m = df_m.dropna(subset=['SAI','BAI','BoAI']).copy()
fw_m = df_m.groupby('unhhidnp')['wave'].transform('min')
df_m_bl = df_m[df_m['wave']==fw_m].copy()
print(f'MHAS baseline: N={len(df_m_bl):,}')

X_m = df_m_bl[['SAI','BAI','BoAI']].values
gmm_m = GaussianMixture(n_components=5, random_state=42, n_init=10)
df_m_bl['phenotype'] = gmm_m.fit_predict(StandardScaler().fit_transform(X_m))

print('MHAS 5 phenotypes:')
for p in range(5):
    sub = df_m_bl[df_m_bl['phenotype']==p]
    print(f'  Type {p+1}: n={len(sub):5,} ({len(sub)/len(df_m_bl)*100:4.1f}%) SAI={sub["SAI"].mean():3.0f} BAI={sub["BAI"].mean():3.0f} BoAI={sub["BoAI"].mean():3.0f} DSI={sub["dsi"].mean()*100:3.1f}%')

# Mediation
df_mm = df_m_bl.dropna(subset=['dsi','BAI','cesd_m']).copy()
if len(df_mm) > 100:
    m1=LinearRegression().fit(df_mm[['dsi','pubage']].fillna(0), df_mm['BAI'])
    m2=LinearRegression().fit(df_mm[['dsi']], df_mm['cesd_m'])
    m3=LinearRegression().fit(df_mm[['dsi','cesd_m']], df_mm['BAI'])
    ind_m=m2.coef_[0]*m3.coef_[1]; tot_m=ind_m+m3.coef_[0]
    mp_m=ind_m/tot_m*100 if tot_m!=0 else 0
    print(f'MHAS mediation (N={len(df_mm):,}): CES-D = {mp_m:.1f}% (total={tot_m:.2f})')
    results['MHAS'] = {'n': len(df_m_bl), 'dsi': df_m_bl['dsi'].mean()*100, 'sai': df_m_bl['SAI'].mean(),
                       'bai': df_m_bl['BAI'].mean(), 'boai': df_m_bl['BoAI'].mean(),
                       'total': tot_m, 'med_pct': mp_m}
    df_m_bl.to_csv(os.path.join(out_dir, 'mhas_fixed.csv'), index=False)

# ===================== FINAL 3-COUNTRY TABLE =====================
print('\n' + '='*65)
print('FINAL THREE-COUNTRY COMPARISON (encoding fixed)')
print('='*65)
results['CHARLS'] = {'n': 7764, 'dsi': 4.4, 'sai': 14, 'bai': 47, 'boai': 54, 'total': -4.89, 'med_pct': 28.0}
print(f'{"Metric":<28s} {"CHARLS":>10s} {"KLoSA":>10s} {"MHAS":>10s}')
print('-'*58)
for metric, key, fmt in [('N (baseline)','n','d'), ('DSI %','dsi','.1f'), ('SAI','sai','.0f'),
                           ('BAI','bai','.0f'), ('BoAI','boai','.0f'),
                           ('DSI->BAI total','total','.2f'), ('CES-D Mediation %','med_pct','.1f')]:
    vals = []
    for cohort in ['CHARLS','KLoSA','MHAS']:
        if cohort in results and key in results[cohort]:
            v = results[cohort][key]
            vals.append(f'{v:>10,.0f}' if fmt=='d' else f'{v:>10.1f}%' if 'pct' in key else f'{v:>10.2f}' if '.' in fmt else f'{v:>10.0f}')
        else:
            vals.append(f'{"N/A":>10s}')
    print(f'{metric:<28s} {" ".join(vals)}')

print('\nAll fixed data saved')
