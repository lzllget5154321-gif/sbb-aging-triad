# Final fix: KLoSA BAI + HRS integration + 4-country comparison
import pandas as pd, numpy as np, os, warnings; warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(PROJECT_ROOT, 'data_raw')
out_dir = os.path.join(PROJECT_ROOT, 'results', 'tables')
os.makedirs(out_dir, exist_ok=True)
results = {}

# ===================== KLoSA (FIX: richer BAI) =====================
print('=== KLoSA: Richer BAI ===')
kl_dir = os.path.join(data_root, 'KLoSA')
kl_dfs = {}
for f in os.listdir(kl_dir):
    if f.endswith('.csv'):
        try: kl_dfs[f] = pd.read_csv(os.path.join(kl_dir, f), encoding='utf-8', low_memory=False)
        except: pass

df_k = None
merges_k = [
    ('health_1', ['pid','wave','sighta','dsighta','nsighta','hearinga','adlwb','iadlb',
                   'bmi','weight','hearte','stroke','diabetes','hypertens']),
    ('cognition', ['pid','wave','imrc3','dlrc3','ser7','orient','orientp_k','draw','execu']),
    ('physical', ['pid','wave','lgrip1','lgrip2','rgrip1','rgrip2']),
    ('psychosocia', ['pid','wave','cesd10a','cesd10am','cesd10b','cesd10bm','fsadl']),
    ('global_info', ['pid','ragender']),
    ('pension', ['pid','wave','pubage']),
    ('health_2', ['pid','wave','smokev','drinkev']),
]
for key, vl in merges_k:
    match = [k for k in kl_dfs if key in k]
    if not match: continue
    avail = [v for v in vl if v in kl_dfs[match[0]].columns]
    sub = kl_dfs[match[0]][avail].copy()
    on = ['pid','wave'] if 'wave' in sub.columns and (df_k is None or 'wave' in df_k.columns) else ['pid']
    common = [c for c in on if df_k is None or c in df_k.columns]
    df_k = sub if df_k is None else df_k.merge(sub, on=common, how='outer')

df_k = df_k[df_k['pubage'] >= 50].copy()
print(f'KLoSA N={len(df_k):,} rows, {df_k["pid"].nunique():,} IDs')

# Sensory
for c in ['sighta','dsighta','nsighta','hearinga']:
    if c in df_k.columns: df_k[c] = pd.to_numeric(df_k[c], errors='coerce').fillna(0)
df_k['vi'] = ((df_k['sighta']==1)|(df_k['dsighta']==1)|(df_k['nsighta']==1)).astype(int)
df_k['hi'] = (df_k['hearinga']==1).astype(int)
df_k['dsi'] = ((df_k['vi']==1)&(df_k['hi']==1)).astype(int)
df_k['SAI'] = (df_k['vi']+df_k['hi'])/2*100

# RICHER BAI: imrc3+dlrc3 (memory) + ser7 (exec/attention) + orient+draw+execu
cog_rich = [c for c in ['imrc3','dlrc3','ser7','orient','orientp_k','draw','execu'] if c in df_k.columns]
print(f'KLoSA BAI components: {cog_rich}')
if cog_rich:
    cog_data = df_k[cog_rich].apply(pd.to_numeric, errors='coerce')
    df_k['cog_raw'] = cog_data.mean(axis=1, skipna=True)
    mn, mx = df_k['cog_raw'].min(), df_k['cog_raw'].max()
    df_k['BAI'] = ((df_k['cog_raw'] - mn) / (mx - mn) * 100).clip(0, 100)
    print(f'  BAI: min={df_k["BAI"].min():.0f}, max={df_k["BAI"].max():.0f}, mean={df_k["BAI"].mean():.0f}, std={df_k["BAI"].std():.0f}')

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

# CESD
cesd_c = [c for c in ['cesd10a','cesd10am','cesd10b','cesd10bm'] if c in df_k.columns]
df_k['cesd'] = df_k[cesd_c].mean(axis=1,skipna=True) if cesd_c else 0

df_k = df_k.dropna(subset=['SAI','BAI','BoAI']).copy()
fw_k = df_k.groupby('pid')['wave'].transform('min')
df_k_bl = df_k[df_k['wave']==fw_k].copy()
print(f'KLoSA baseline: N={len(df_k_bl):,}')

# LCA
X_k = df_k_bl[['SAI','BAI','BoAI']].values
gmm_k = GaussianMixture(n_components=5, random_state=42, n_init=10)
df_k_bl['phenotype'] = gmm_k.fit_predict(StandardScaler().fit_transform(X_k))

for p in range(5):
    sub = df_k_bl[df_k_bl['phenotype']==p]
    print(f'  Type {p+1}: n={len(sub):5,} ({len(sub)/len(df_k_bl)*100:4.1f}%) SAI={sub["SAI"].mean():3.0f} BAI={sub["BAI"].mean():3.0f} BoAI={sub["BoAI"].mean():3.0f} DSI={sub["dsi"].mean()*100:3.1f}%')

# Mediation
df_km = df_k_bl.dropna(subset=['dsi','BAI','cesd']).copy()
m1k=LinearRegression().fit(df_km[['dsi','pubage']].fillna(0), df_km['BAI'])
m2k=LinearRegression().fit(df_km[['dsi']], df_km['cesd'])
m3k=LinearRegression().fit(df_km[['dsi','cesd']], df_km['BAI'])
ind_k=m2k.coef_[0]*m3k.coef_[1]; tot_k=ind_k+m3k.coef_[0]
mp_k=ind_k/tot_k*100 if tot_k!=0 else 0
print(f'KLoSA mediation (N={len(df_km):,}): {mp_k:.1f}% (total={tot_k:.2f}, BAI mean={df_km["BAI"].mean():.0f})')
results['KLoSA'] = {'n':len(df_k_bl),'dsi':df_k_bl['dsi'].mean()*100,'total':tot_k,'med_pct':mp_k,
                     'sai':df_k_bl['SAI'].mean(),'bai':df_k_bl['BAI'].mean(),'boai':df_k_bl['BoAI'].mean()}

# ===================== HRS =====================
print('\n=== HRS (United States) ===')
hrs_path = os.path.join(data_root,'HRS','HRS','RAND HRS Longitudinal File 2020','randhrs1992_2020v2.dta')
if os.path.exists(hrs_path):
    # Load key variables
    cols_to_check = ['hhidpn','ragender','rabyear','r1agey_b','r1height','r1weight','r1smokev',
                     'r1drink','r1adlwa','r1iadlza','r1diab','r1hibpe','r1hearte','r1stroke']
    df_h = pd.read_stata(hrs_path, columns=cols_to_check)
    print(f'HRS loaded: {len(df_h):,} rows')
    # Filter age>=50
    if 'r1agey_b' in df_h.columns:
        df_h = df_h[df_h['r1agey_b'] >= 50].copy()
    print(f'HRS age>=50: {len(df_h):,} rows')
    print(f'Female: {(df_h["ragender"]==2).mean()*100:.0f}%')
    # Placeholder: need full variable mapping for SAI/BAI/BoAI
    print('HRS: Ready for full variable mapping (sensory/cognitive vars need separate HRS core files)')
    results['HRS'] = {'n':len(df_h),'status':'loaded, needs sensory/cognitive vars'}

# ===================== MHAS (reuse fixed from before) =====================
print('\n=== MHAS (already fixed) ===')
mhas_fixed_path = os.path.join(out_dir, 'mhas_fixed.csv')
if os.path.exists(mhas_fixed_path):
    df_m_bl = pd.read_csv(mhas_fixed_path)
    results['MHAS'] = {'n':len(df_m_bl),'dsi':df_m_bl['dsi'].mean()*100,
                       'sai':df_m_bl['SAI'].mean(),'bai':df_m_bl['BAI'].mean(),'boai':df_m_bl['BoAI'].mean()}
    # Re-run mediation
    df_mm = df_m_bl.dropna(subset=['dsi','BAI','cesd_m']).copy()
    if len(df_mm)>100:
        m1=LinearRegression().fit(df_mm[['dsi']].fillna(0),df_mm['BAI'])
        m2=LinearRegression().fit(df_mm[['dsi']],df_mm['cesd_m'])
        m3=LinearRegression().fit(df_mm[['dsi','cesd_m']],df_mm['BAI'])
        im=m2.coef_[0]*m3.coef_[1]; tm=im+m3.coef_[0]
        results['MHAS']['total']=tm; results['MHAS']['med_pct']=im/tm*100 if tm!=0 else 0

# ===================== CHARLS =====================
results['CHARLS'] = {'n':7764,'dsi':4.4,'total':-4.89,'med_pct':28.0,'sai':14,'bai':47,'boai':54}

# ===================== FINAL 4-COUNTRY =====================
print('\n' + '='*72)
print('FINAL FOUR-COUNTRY COMPARISON')
print('='*72)
headers = ['Metric','CHARLS','KLoSA','MHAS','HRS']
print(f'{headers[0]:<28s} {headers[1]:>10s} {headers[2]:>10s} {headers[3]:>10s} {headers[4]:>10s}')
print('-'*68)
for metric, key, fmt in [
    ('N (baseline)','n','d'),('DSI %','dsi','.1f'),('DSI->BAI total','total','.2f'),('CES-D Mediation %','med_pct','.1f')]:
    row = [metric]
    for cohort in ['CHARLS','KLoSA','MHAS','HRS']:
        if cohort in results and key in results[cohort] and results[cohort][key] is not None:
            v = results[cohort][key]
            if isinstance(v, str): row.append(f'{v:>10s}')
            elif fmt=='d': row.append(f'{int(v):>10,}')
            elif '.1f' in fmt: row.append(f'{v:>9.1f}%')
            else: row.append(f'{v:>10.2f}')
        else:
            row.append(f'{"--":>10s}')
    print(' '.join(row))

# Save
df_k_bl.to_csv(os.path.join(out_dir,'klosa_fixed_v2.csv'),index=False)
print(f'\nSaved: KLoSA fixed v2 ({len(df_k_bl):,} rows)')
print('Done')
