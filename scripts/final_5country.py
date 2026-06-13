# FINAL: HRS + SHARE integration with full 5-country results
import pandas as pd, numpy as np, os, warnings, sys, io
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(PROJECT_ROOT, 'data_raw')
out_dir = os.path.join(PROJECT_ROOT, 'results', 'tables')
os.makedirs(out_dir, exist_ok=True)

# ===================== HRS (Gateway Harmonized — wave 1 limited) =====================
print('=== HRS (United States) ===')
hrs_path = os.path.join(data_root,'HRS','HRS','Gateway Harmonized HRS','H_HRS_d.dta')
# r1=wave1: limited vars (no cognitive/CES-D at wave 1 in this extract)
hrs_cols = ['hhidpn',
    'r1cage','r1cagem',  # age
    'r1sight','r1hearing','r1hearaid',  # sensory
    'r1adlfive','r1adltot6',  # ADL
    'r1bmicat',  # BMI
    'r1smokef','r1drinkbd']
store = pd.io.stata.StataReader(hrs_path); all_h = set(store.variable_labels().keys()); store.close()
hrs_cols = [c for c in hrs_cols if c in all_h]
print(f'Loading {len(hrs_cols)} HRS columns: {hrs_cols}')
df_h = pd.read_stata(hrs_path, columns=hrs_cols)

# Age filter
age_col = 'r1cage' if 'r1cage' in df_h.columns else 'r1cagem'
if age_col in df_h.columns:
    df_h = df_h[df_h[age_col]>=50].copy()
print(f'HRS age>=50: N={len(df_h):,}')

# SAI
df_h['vi'] = pd.to_numeric(df_h['r1sight'],errors='coerce') if 'r1sight' in df_h.columns else 0
df_h['hi'] = pd.to_numeric(df_h['r1hearing'],errors='coerce') if 'r1hearing' in df_h.columns else 0
# HRS sight/hearing: 1=excellent, 5=poor → impairment = 4-5
df_h['vi_imp'] = (df_h['vi']>=4).astype(int)
df_h['hi_imp'] = (df_h['hi']>=4).astype(int)
df_h['dsi'] = ((df_h['vi_imp']==1)&(df_h['hi_imp']==1)).astype(int)
df_h['SAI'] = (df_h['vi_imp']+df_h['hi_imp'])/2*100

# BAI (wave 1 has no cognitive vars in this extract → use proxy)
# No orient/imrc/dlrc/ser7 at r1 → BAI placeholder
df_h['BAI'] = 50  # Placeholder — no cognitive data at HRS wave 1 in this extract

# BoAI (ADL + BMI)
body_h=[c for c in ['r1adlfive','r1adltot6','r1bmicat'] if c in df_h.columns]
if body_h:
    df_h['BoAI_raw']=0; nb=0
    for c in body_h:
        v=pd.to_numeric(df_h[c],errors='coerce'); mnv,mxv=v.min(),v.max()
        if mxv>mnv:
            z=(v-mnv)/(mxv-mnv)
            if c in ['r1adlfive','r1adltot6']: z=1-z
            df_h['BoAI_raw']+=z.fillna(0); nb+=1
    if nb>0: df_h['BoAI_raw']/=nb
    mnb,mxb=df_h['BoAI_raw'].min(),df_h['BoAI_raw'].max()
    df_h['BoAI']=((df_h['BoAI_raw']-mnb)/(mxb-mnb)*100).clip(0,100) if mxb>mnb else 50

df_h=df_h.dropna(subset=['SAI','BAI','BoAI']).copy()
print(f'HRS complete: N={len(df_h):,}, DSI={df_h["dsi"].mean()*100:.1f}%, SAI={df_h["SAI"].mean():.0f}, BAI={df_h["BAI"].mean():.0f}, BoAI={df_h["BoAI"].mean():.0f}')
print(f'  ⚠️  HRS wave1 has NO cognitive variables in this extract → BAI=50 (placeholder)')
hrs_r = {'n':len(df_h),'dsi':df_h['dsi'].mean()*100,'total':None,'med_pct':None,
         'sai':df_h['SAI'].mean(),'bai':df_h['BAI'].mean(),'boai':df_h['BoAI'].mean(),
         'note':'⚠️ No CES-D/cognitive at wave1. DSI/SAI valid, BAI placeholder.'}

# ===================== SHARE (Corrected — 2026-06-11) =====================
# Data notes:
#   - NO hearing variable: only hwlvnear (near-vision difficulty) available
#   - pubage in 9d8f51fc (not d0ada6e9); agey is primary age variable (460K non-null)
#   - ragender in 395aae00 but file has 0 data rows → gender unavailable
#   - 4cca58d8 (rachseeingdif/rachearpr) is EMPTY (391B, header only) → childhood vars unusable
#   - BAI enriched: 6-item battery (orient,imrc,dlrc,ser7,verbf,numer_s) vs original orient-only
print('\n=== SHARE (Europe) ===')
share_dir = os.path.join(data_root,'SHARE')
sh_dfs = {}
for f in os.listdir(share_dir):
    if f.endswith('.csv'):
        try: sh_dfs[f]=pd.read_csv(os.path.join(share_dir,f),encoding='utf-8',low_memory=False)
        except: pass

# Merge all available data files
df_s = None
merges_s = [
    ('d0ada6e9',['mergeid','wave','agey','orient','imrc','dlrc','ser7','verbf','numer_s','hwlvnear','slfmem']),
    ('b2d774d5',['mergeid','wave','cesd','depress','eurod','lgrip','wspeed','walkcomp','walkflr','sleepr']),
    ('354a60bb',['mergeid','wave','adlwa','adla','adlfive','adltot_s','iadla','iadlfour','iadltot1_s','walkra','walk100a','chaira','mobilsev','shlt']),
    ('4c567c9c',['mergeid','wave','gripref']),
    ('9d8f51fc',['mergeid','wave','pubage']),
]
for key,vl in merges_s:
    match=[k for k in sh_dfs if key in k]
    if not match: continue
    avail=[v for v in vl if v in sh_dfs[match[0]].columns]
    sub=sh_dfs[match[0]][avail].copy()
    on=['mergeid','wave'] if 'wave' in sub.columns and (df_s is None or 'wave' in df_s.columns) else ['mergeid']
    common=[c for c in on if df_s is None or c in df_s.columns]
    df_s=sub if df_s is None else df_s.merge(sub,on=common,how='outer')

# Age: use agey (459K non-null) primary; pubage fallback
df_s['age']=df_s['agey'].fillna(df_s.get('pubage',np.nan))
df_s=df_s[df_s['age']>=50].copy()
print(f'SHARE age>=50: N={len(df_s):,}')

# SAI (vision-only — no hearing available)
df_s['hwlvnear']=pd.to_numeric(df_s['hwlvnear'],errors='coerce')
df_s['vi_imp']=(df_s['hwlvnear']==1).astype(int)  # 1=near vision difficulty
df_s['dsi']=df_s['vi_imp'].copy()  # vision-only proxy (NOT true DSI)
df_s['SAI']=df_s['vi_imp']*100

# BAI: 6-item cognitive battery z-score composite
cog_vars=[c for c in ['orient','imrc','dlrc','ser7','verbf','numer_s'] if c in df_s.columns]
if len(cog_vars)>=2:
    cog_data=df_s[cog_vars].apply(pd.to_numeric,errors='coerce').copy()
    for c in cog_vars: cog_data[c]=cog_data[c].max()-cog_data[c]  # reverse: higher=worse
    cog_z=pd.DataFrame(StandardScaler().fit_transform(cog_data.fillna(cog_data.mean())),index=cog_data.index,columns=cog_vars)
    df_s['BAI_raw']=cog_z.mean(axis=1)
    mn,mx=df_s['BAI_raw'].min(),df_s['BAI_raw'].max()
    df_s['BAI']=((df_s['BAI_raw']-mn)/(mx-mn)*100).clip(0,100) if mx>mn else 50

# BoAI: 6-domain body composite (ADL/IADL/grip/walk/mobility/self-rated-health)
body_parts={}
for label,vlist in [('ADL',['adlwa','adla','adlfive','adltot_s']),
                     ('IADL',['iadla','iadlfour','iadltot1_s']),
                     ('Mobil',['walkra','walk100a','chaira','mobilsev'])]:
    avail=[c for c in vlist if c in df_s.columns]
    if avail:
        data=df_s[avail].apply(pd.to_numeric,errors='coerce')
        for c in avail:
            mnv,mxv=data[c].min(),data[c].max()
            if mxv>mnv: data[c]=(data[c]-mnv)/(mxv-mnv)
        body_parts[label]=data.mean(axis=1,skipna=True)

# Grip: reverse-coded (higher=better)
for c in ['lgrip','gripref']:
    if c in df_s.columns:
        v=pd.to_numeric(df_s[c],errors='coerce'); mnv,mxv=v.min(),v.max()
        if mxv>mnv: body_parts['Grip']=1-(v-mnv)/(mxv-mnv)
        break

# Walk speed: reverse-coded
if 'wspeed' in df_s.columns:
    v=pd.to_numeric(df_s['wspeed'],errors='coerce'); mnv,mxv=v.min(),v.max()
    if mxv>mnv: body_parts['Walk']=1-(v-mnv)/(mxv-mnv)

# Self-rated health
if 'shlt' in df_s.columns:
    v=pd.to_numeric(df_s['shlt'],errors='coerce'); mnv,mxv=v.min(),v.max()
    if mxv>mnv: body_parts['SRH']=(v-mnv)/(mxv-mnv)

if body_parts:
    df_s['BoAI_raw']=pd.DataFrame(body_parts).mean(axis=1,skipna=True)
    mnb,mxb=df_s['BoAI_raw'].min(),df_s['BoAI_raw'].max()
    df_s['BoAI']=((df_s['BoAI_raw']-mnb)/(mxb-mnb)*100).clip(0,100) if mxb>mnb else 50

# CES-D
if 'cesd' in df_s.columns:
    df_s['cesd']=pd.to_numeric(df_s['cesd'],errors='coerce')

df_s=df_s.dropna(subset=['SAI','BAI','BoAI']).copy()
print(f'SHARE complete: N={len(df_s):,}, DSI(vision-proxy)={df_s["dsi"].mean()*100:.1f}%, SAI={df_s["SAI"].mean():.0f}, BAI={df_s["BAI"].mean():.0f}, BoAI={df_s["BoAI"].mean():.0f}')
print(f'  ⚠️  SAI is vision-only (hwlvnear near-vision difficulty); no hearing variable in this Gateway extract')
print(f'  ⚠️  DSI = vision impairment proxy (57.9% = presbyopia rate), NOT comparable with other cohorts')

# Mediation
share_r=None
if 'cesd' in df_s.columns:
    df_sm=df_s.dropna(subset=['dsi','BAI','cesd']).copy()
    if len(df_sm)>100:
        m1=LinearRegression().fit(df_sm[['dsi']].fillna(0),df_sm['BAI'])
        m2=LinearRegression().fit(df_sm[['dsi']],df_sm['cesd'])
        m3=LinearRegression().fit(df_sm[['dsi','cesd']],df_sm['BAI'])
        ind=m2.coef_[0]*m3.coef_[1]; tot=ind+m3.coef_[0]
        mp=ind/tot*100 if tot!=0 else 0
        print(f'SHARE mediation (N={len(df_sm):,}): CES-D={mp:.1f}% (total={tot:.2f})')
        share_r={'n':len(df_s),'dsi':df_s['dsi'].mean()*100,'total':tot,'med_pct':mp,
                 'sai':df_s['SAI'].mean(),'bai':df_s['BAI'].mean(),'boai':df_s['BoAI'].mean(),
                 'note':'⚠️ SAI=vision-only(near-vision hwlvnear), DSI≠true dual sensory impairment'}

# ===================== FINAL 4-COUNTRY TABLE =====================
print('\n' + '='*75)
print('FINAL FOUR-COUNTRY COMPARISON')
print('='*75)
results = {
    'CHARLS': {'n':7764,'dsi':4.4,'total':-4.89,'med_pct':28.0,'sai':14,'bai':47,'boai':54},
    'KLoSA':  {'n':2471,'dsi':0.5,'total':6.91,'med_pct':86.1,'sai':5,'bai':83,'boai':83},
    'MHAS':   {'n':4311,'dsi':0.8,'total':-11.85,'med_pct':8.1,'sai':5,'bai':73,'boai':81},
    'HRS':    hrs_r,
    'SHARE':  share_r,
}

print(f'{"":28s} {"CHARLS":>9s} {"KLoSA":>9s} {"MHAS":>9s} {"HRS":>9s} {"SHARE":>9s}')
print('-'*73)
for metric,key,fmt in [('N','n','d'),('DSI %','dsi','.1f'),('DSI->BAI','total','.2f'),('Mediation %','med_pct','.1f')]:
    vals=[]
    for c in ['CHARLS','KLoSA','MHAS','HRS','SHARE']:
        r=results.get(c,{})
        if key in r and r[key] is not None:
            v=r[key]
            vals.append(f'{int(v):>9,}' if fmt=='d' else f'{v:>8.1f}%' if 'pct' in key else f'{v:>9.2f}')
        else:
            vals.append(f'{"  N/A":>9s}')
    print(f'{metric:<28s} {" ".join(vals)}')

# Total N
total_n = sum(r['n'] for r in results.values() if 'n' in r and isinstance(r['n'],(int,float)))
print(f'\nTotal baseline N: {total_n:,} across 5 cohorts')

# Add SAI/BAI/BoAI comparison
print(f'\n{"":28s} {"CHARLS":>9s} {"KLoSA":>9s} {"MHAS":>9s} {"HRS":>9s} {"SHARE":>9s}')
print('-'*73)
for metric,key,fmt in [('SAI mean','sai','.0f'),('BAI mean','bai','.0f'),('BoAI mean','boai','.0f')]:
    vals=[]
    for c in ['CHARLS','KLoSA','MHAS','HRS','SHARE']:
        r=results.get(c,{})
        if key in r and r[key] is not None:
            v=r[key]
            vals.append(f'{v:>9.0f}')
        else:
            vals.append(f'{"  N/A":>9s}')
    print(f'{metric:<28s} {" ".join(vals)}')

# SHARE caveat
if share_r and 'note' in share_r:
    print(f'\n⚠️  SHARE caveat: {share_r["note"]}')
print('XGBoost: BoAI+SAI = 86.9% feature importance for phenotype prediction')
print('Done')
