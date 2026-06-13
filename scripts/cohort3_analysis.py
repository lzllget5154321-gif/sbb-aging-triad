# MHAS + KLoSA Mediation + 3-Country Summary
import pandas as pd, numpy as np, os, warnings; warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(PROJECT_ROOT, 'data_raw')
out_dir = os.path.join(PROJECT_ROOT, 'results', 'tables')
os.makedirs(out_dir, exist_ok=True)

# ===================== MHAS =====================
print('=== MHAS (Mexico) ===')
mhas_dir = os.path.join(data_root, 'MHAS')
mhas_dfs = {}
for root, dirs, files in os.walk(mhas_dir):
    for f in files:
        if f.endswith('.csv'):
            try: mhas_dfs[f] = pd.read_csv(os.path.join(root, f), encoding='utf-8', low_memory=False)
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
    match = [k for k in mhas_dfs if key in k]
    if not match: continue
    avail = [v for v in vl if v in mhas_dfs[match[0]].columns]
    sub = mhas_dfs[match[0]][avail].copy()
    on = ['unhhidnp','wave'] if 'wave' in sub.columns else ['unhhidnp']
    common = [c for c in on if c in (df_m.columns if df_m is not None else sub.columns)]
    df_m = sub if df_m is None else df_m.merge(sub, on=common, how='outer')

df_m = df_m[df_m['pubage'] >= 50].copy()
for c in ['sight','hearing']: df_m[c] = pd.to_numeric(df_m[c], errors='coerce').fillna(0)
df_m['dsi'] = ((df_m['sight']==1)&(df_m['hearing']==1)).astype(int)
df_m['SAI'] = (df_m['sight']+df_m['hearing'])/2*100
cog_m = [c for c in ['orient_m','forient_m'] if c in df_m.columns]
if cog_m:
    raw = df_m[cog_m].mean(axis=1,skipna=True)
    mn,mx=raw.min(),raw.max()
    df_m['BAI']=((raw-mn)/(mx-mn)*100).clip(0,100) if mx>mn else 50
body_m=[c for c in ['adltot6','iadlfour','hearte','stroke','diabetes','hypertens'] if c in df_m.columns]
if body_m:
    df_m['BoAI_raw']=0; nb=0
    for c in body_m:
        v=pd.to_numeric(df_m[c],errors='coerce'); mn_v,mx_v=v.min(),v.max()
        if mx_v>mn_v:
            z=(v-mn_v)/(mx_v-mn_v)
            if c in ['adltot6','iadlfour','hearte','stroke','diabetes','hypertens']: z=1-z
            df_m['BoAI_raw']+=z.fillna(0); nb+=1
    if nb>0: df_m['BoAI_raw']/=nb; mnb,mxb=df_m['BoAI_raw'].min(),df_m['BoAI_raw'].max()
    df_m['BoAI']=((df_m['BoAI_raw']-mnb)/(mxb-mnb)*100).clip(0,100)

df_m=df_m.dropna(subset=['SAI','BAI','BoAI']).copy()
fw=df_m.groupby('unhhidnp')['wave'].transform('min')
df_m_bl=df_m[df_m['wave']==fw].copy()
print(f'MHAS baseline: N={len(df_m_bl):,}')

X_m=df_m_bl[['SAI','BAI','BoAI']].values
gmm=GaussianMixture(n_components=5,random_state=42,n_init=10)
df_m_bl['phenotype']=gmm.fit_predict(StandardScaler().fit_transform(X_m))
for p in range(5):
    sub=df_m_bl[df_m_bl['phenotype']==p]
    print(f'  Type {p+1}: n={len(sub):5,} ({len(sub)/len(df_m_bl)*100:4.1f}%) SAI={sub["SAI"].mean():3.0f} BAI={sub["BAI"].mean():3.0f} BoAI={sub["BoAI"].mean():3.0f} DSI={sub["dsi"].mean()*100:3.1f}%')

# MHAS Mediation
df_mm=df_m_bl.dropna(subset=['dsi','BAI','cesd_m']).copy()
m1=LinearRegression().fit(df_mm[['dsi']],df_mm['BAI'])
m2=LinearRegression().fit(df_mm[['dsi']],df_mm['cesd_m'])
m3=LinearRegression().fit(df_mm[['dsi','cesd_m']],df_mm['BAI'])
ind_m=m2.coef_[0]*m3.coef_[1]; tot_m=ind_m+m3.coef_[0]
mp_m=ind_m/tot_m*100 if tot_m!=0 else 0
print(f'MHAS mediation: DSI->CESD->BAI = {mp_m:.1f}%')

# ===================== KLoSA MEDIATION =====================
print('\n=== KLoSA Mediation ===')
klosa_dir=os.path.join(data_root,'KLoSA')
kl={}
for f in os.listdir(klosa_dir):
    if f.endswith('.csv'):
        try: kl[f]=pd.read_csv(os.path.join(klosa_dir,f),encoding='utf-8',low_memory=False)
        except: pass

df_k=None
for key,vl in [
    ('health_1',['pid','wave','sighta','dsighta','nsighta','hearinga','adlwb','iadlb','hearte','stroke','diabetes','hypertens']),
    ('cognition',['pid','wave','orient','orientp_k','draw']),
    ('psychosocia',['pid','wave','cesd10a','cesd10am']),
    ('global_info',['pid','ragender']),
    ('pension',['pid','wave','pubage']),
]:
    match=[k for k in kl if key in k]
    if not match: continue
    avail=[v for v in vl if v in kl[match[0]].columns]
    sub=kl[match[0]][avail].copy()
    on=['pid','wave'] if 'wave' in sub.columns else ['pid']
    common=[c for c in on if c in (df_k.columns if df_k is not None else sub.columns)]
    df_k=sub if df_k is None else df_k.merge(sub,on=common,how='outer')

df_k=df_k[df_k['pubage']>=50].copy()
for c in ['sighta','dsighta','nsighta','hearinga']: df_k[c]=pd.to_numeric(df_k[c],errors='coerce').fillna(0)
df_k['vi']=((df_k['sighta']==1)|(df_k.get('dsighta',0)==1)|(df_k.get('nsighta',0)==1)).astype(int)
df_k['hi']=(df_k['hearinga']==1).astype(int)
df_k['dsi']=((df_k['vi']==1)&(df_k['hi']==1)).astype(int)
df_k['SAI']=(df_k['vi']+df_k['hi'])/2*100
cesd_k=[c for c in ['cesd10a','cesd10am'] if c in df_k.columns]
df_k['cesd']=df_k[cesd_k].mean(axis=1,skipna=True) if cesd_k else 0
cog_k=[c for c in ['orient','orientp_k','draw'] if c in df_k.columns]
if cog_k:
    raw_k=df_k[cog_k].mean(axis=1,skipna=True); mnk,mxk=raw_k.min(),raw_k.max()
    df_k['BAI']=((raw_k-mnk)/(mxk-mnk)*100).clip(0,100)

df_k=df_k.dropna(subset=['dsi','BAI','cesd']).copy()
fwk=df_k.groupby('pid')['wave'].transform('min')
df_k_bl=df_k[df_k['wave']==fwk].copy()

mk1=LinearRegression().fit(df_k_bl[['dsi']],df_k_bl['BAI'])
mk2=LinearRegression().fit(df_k_bl[['dsi']],df_k_bl['cesd'])
mk3=LinearRegression().fit(df_k_bl[['dsi','cesd']],df_k_bl['BAI'])
ind_k=mk2.coef_[0]*mk3.coef_[1]; tot_k=ind_k+mk3.coef_[0]
mp_k=ind_k/tot_k*100 if tot_k!=0 else 0
print(f'KLoSA: N={len(df_k_bl):,}, DSI->CESD->BAI mediation = {mp_k:.1f}%')

# ===================== 3-COUNTRY SUMMARY =====================
print('\n=== THREE-COUNTRY COMPARISON ===')
# CHARLS numbers from previous run
c_n, c_dsi, c_tot, c_med = 7764, 4.4, -4.89, 28.0
print(f'{"":30s} {"CHARLS":>10s} {"KLoSA":>10s} {"MHAS":>10s}')
print(f'{"N":30s} {c_n:>10,} {len(df_k_bl):>10,} {len(df_m_bl):>10,}')
print(f'{"DSI %":30s} {c_dsi:>9.1f}% {df_k_bl["dsi"].mean()*100:>9.1f}% {df_m_bl["dsi"].mean()*100:>9.1f}%')
print(f'{"DSI->BAI total":30s} {c_tot:>10.2f} {tot_k:>10.2f} {tot_m:>10.2f}')
print(f'{"CES-D Mediation %":30s} {c_med:>9.1f}% {mp_k:>9.1f}% {mp_m:>9.1f}%')

df_m_bl.to_csv(os.path.join(out_dir,'mhas_phenotypes.csv'),index=False)
df_k_bl.to_csv(os.path.join(out_dir,'klosa_mediation.csv'),index=False)
print('\nAll results saved')
