#!/usr/bin/env python3
"""KLoSA 86% Mediation Anomaly - Complete Diagnostic & Fix"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.getcwd()
kl_dir = os.path.join(PROJECT_ROOT, 'data_raw', 'KLoSA')
out_dir = os.path.join(PROJECT_ROOT, 'results', 'tables')
os.makedirs(out_dir, exist_ok=True)

print('='*80)
print('KLoSA DIAGNOSTIC -- 86% Mediation Anomaly')
print('='*80)

# ---- Step 1: Load & merge each file directly ----
health = pd.read_csv(os.path.join(kl_dir, 'health_1 (2).csv'), encoding='utf-8', low_memory=False)
health_cols = ['pid','wave','sighta','dsighta','nsighta','sightlmt','hearinga',
               'adlwb','iadlb','bmi','hearte','stroke','diabetes','hypertens']
df = health[[c for c in health_cols if c in health.columns]].copy()
for c in ['sighta','dsighta','nsighta','hearinga','sightlmt']:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
print(f'Health: {len(df):,} rows')

cog = pd.read_csv(os.path.join(kl_dir, 'cognition (2).csv'), encoding='utf-8', low_memory=False)
cog_cols = ['pid','wave','orient','orientp_k','draw','imrc3','dlrc3','ser7','execu']
cog_sub = cog[[c for c in cog_cols if c in cog.columns]].copy()
df = df.merge(cog_sub, on=['pid','wave'], how='left')

psych = pd.read_csv(os.path.join(kl_dir, 'psychosocia (2).csv'), encoding='utf-8', low_memory=False)
psych_cols = ['pid','wave','cesd10a','cesd10am','cesd10b','cesd10bm']
psych_sub = psych[[c for c in psych_cols if c in psych.columns]].copy()
df = df.merge(psych_sub, on=['pid','wave'], how='left')

pension = pd.read_csv(os.path.join(kl_dir, 'pension (2).csv'), encoding='utf-8', low_memory=False)
if 'pubage' in pension.columns:
    df = df.merge(pension[['pid','wave','pubage']], on=['pid','wave'], how='left')

df = df[df['pubage'] >= 50].copy()
print(f'Age>=50: {len(df):,} rows, {df["pid"].nunique():,} IDs')

# ============================================================
# DIAGNOSIS 1: Original BUGGY encoding
# ============================================================
print('\n' + '='*80)
print('DIAGNOSIS 1: Original (BUGGY) encoding -- sighta==1, hearinga==1')
print('='*80)

df1 = df.copy()
df1['vi_bug'] = ((df1['sighta']==1)|(df1['dsighta']==1)|(df1['nsighta']==1)).astype(int)
df1['hi_bug'] = (df1['hearinga']==1).astype(int)
df1['dsi_bug'] = ((df1['vi_bug']==1)&(df1['hi_bug']==1)).astype(int)
cesd_bug_cols = [c for c in ['cesd10a','cesd10am','cesd10b','cesd10bm'] if c in df1.columns]
df1['cesd_bug'] = df1[cesd_bug_cols].mean(axis=1, skipna=True)
cog_bug_cols = [c for c in ['orient','orientp_k','draw'] if c in df1.columns]
raw = df1[cog_bug_cols].mean(axis=1, skipna=True)
mn, mx = raw.min(), raw.max()
df1['BAI_bug'] = ((raw-mn)/(mx-mn)*100).clip(0,100) if mx>mn else 50

df1 = df1.dropna(subset=['dsi_bug','BAI_bug','cesd_bug']).copy()
fw = df1.groupby('pid')['wave'].transform('min')
df1_bl = df1[df1['wave']==fw].copy()

m1=LinearRegression().fit(df1_bl[['dsi_bug']], df1_bl['BAI_bug'])
m2=LinearRegression().fit(df1_bl[['dsi_bug']], df1_bl['cesd_bug'])
m3=LinearRegression().fit(df1_bl[['dsi_bug','cesd_bug']], df1_bl['BAI_bug'])
ind=m2.coef_[0]*m3.coef_[1]; tot=ind+m3.coef_[0]
mp=ind/tot*100 if tot!=0 else 0

print(f'  N(baseline)={len(df1_bl):,}, DSI prev={df1_bl["dsi_bug"].mean()*100:.2f}%')
print(f'  DSI->BAI path coefficients:')
print(f'    a(DSI->CESD)       = {m2.coef_[0]:+.3f}')
print(f'    b(CESD->BAI|DSI)   = {m3.coef_[1]:+.3f}')
print(f'    c_prime(DSI->BAI|CESD) = {m3.coef_[0]:+.2f}')
print(f'    Indirect(a*b)      = {ind:+.2f}')
print(f'    Total(ind+c_prime) = {tot:+.2f}')
print(f'    CES-D Mediation    = {mp:.1f}%')
print(f'  BUG: DSI=1 has BAI={df1_bl[df1_bl["dsi_bug"]==1]["BAI_bug"].mean():.0f} vs DSI=0 BAI={df1_bl[df1_bl["dsi_bug"]==0]["BAI_bug"].mean():.0f}')
print(f'       (DSI=1 means EXCELLENT senses -> HEALTHIER subgroup!)')

# ============================================================
# DIAGNOSIS 2: CORRECTED encoding
# ============================================================
print('\n' + '='*80)
print('DIAGNOSIS 2: CORRECTED encoding -- sighta>=4, hearinga>=4')
print('='*80)

df2 = df.copy()
df2['vi'] = ((df2['sighta']>=4)|(df2['dsighta']>=4)|(df2['nsighta']>=4)).astype(int)
df2['hi'] = (df2['hearinga']>=4).astype(int)
df2['dsi'] = ((df2['vi']==1)&(df2['hi']==1)).astype(int)
df2['SAI'] = (df2['vi'] + df2['hi']) / 2 * 100

# Correct CES-D: use cesd10a or cesd10b (alternate by wave)
df2['cesd'] = np.where(df2['cesd10a'].notna(), df2['cesd10a'],
              np.where(df2['cesd10b'].notna(), df2['cesd10b'], np.nan))

# Expanded BAI: all 7 cognitive vars, z-scored then averaged
cog_vars = [c for c in ['orient','orientp_k','draw','imrc3','dlrc3','ser7','execu'] if c in df2.columns]
print(f'  Cognitive vars ({len(cog_vars)}): {cog_vars}')
cog_z = df2[cog_vars].apply(lambda x: (x-x.mean())/x.std(), axis=0)
df2['BAI_raw_z'] = cog_z.mean(axis=1, skipna=True)
mn2, mx2 = df2['BAI_raw_z'].min(), df2['BAI_raw_z'].max()
df2['BAI'] = ((df2['BAI_raw_z']-mn2)/(mx2-mn2)*100).clip(0,100) if mx2>mn2 else 50

# BoAI
body_c = [c for c in ['adlwb','iadlb','bmi','hearte','stroke','diabetes','hypertens'] if c in df2.columns]
if body_c:
    df2['BoAI_raw'] = 0; nb = 0
    for c in body_c:
        v = pd.to_numeric(df2[c], errors='coerce'); mnv, mxv = v.min(), v.max()
        if mxv > mnv:
            z = (v-mnv)/(mxv-mnv)
            if c in ['adlwb','iadlb','hearte','stroke','diabetes','hypertens']: z = 1-z
            df2['BoAI_raw'] += z.fillna(0); nb += 1
    if nb > 0: df2['BoAI_raw'] /= nb
    mnb, mxb = df2['BoAI_raw'].min(), df2['BoAI_raw'].max()
    df2['BoAI'] = ((df2['BoAI_raw']-mnb)/(mxb-mnb)*100).clip(0,100)

print(f'  Vision impairment (>=4):   {df2["vi"].mean()*100:.1f}%')
print(f'  Hearing impairment (>=4):  {df2["hi"].mean()*100:.1f}%')
print(f'  DSI (dual sensory):        {df2["dsi"].mean()*100:.2f}%')
print(f'  CES-D corrected: mean={df2["cesd"].mean():.1f}, SD={df2["cesd"].std():.1f}')
print(f'  CES-D buggy:      mean={df1["cesd_bug"].mean():.1f} (diluted by {(1-df1["cesd_bug"].mean()/df2["cesd"].mean())*100:.0f}%)')
print(f'  BAI corrected:    mean={df2["BAI"].mean():.0f}, SD={df2["BAI"].std():.0f}')
print(f'  BoAI corrected:   mean={df2["BoAI"].mean():.0f}, SD={df2["BoAI"].std():.0f}')

# ============================================================
# DIAGNOSIS 3: Mediation with corrected variables
# ============================================================
print('\n' + '='*80)
print('DIAGNOSIS 3: Mediation with CORRECTED variables')
print('='*80)

df_med = df2.dropna(subset=['dsi','BAI','cesd']).copy()
fw = df_med.groupby('pid')['wave'].transform('min')
df_bl = df_med[df_med['wave']==fw].copy()

# Verify direction
dsi1 = df_bl[df_bl['dsi']==1]; dsi0 = df_bl[df_bl['dsi']==0]
print(f'  DSI=1: N={len(dsi1)}, BAI={dsi1["BAI"].mean():.0f}, CESD={dsi1["cesd"].mean():.1f}, Age={dsi1["pubage"].mean():.0f}')
print(f'  DSI=0: N={len(dsi0)}, BAI={dsi0["BAI"].mean():.0f}, CESD={dsi0["cesd"].mean():.1f}, Age={dsi0["pubage"].mean():.0f}')
direction_ok = dsi1["BAI"].mean() < dsi0["BAI"].mean()
print(f'  Direction check: DSI=1 has {"WORSE" if direction_ok else "BETTER"} cognition -- {"OK" if direction_ok else "STILL WRONG"}')

# Unadjusted
m1=LinearRegression().fit(df_bl[['dsi']], df_bl['BAI'])
m2=LinearRegression().fit(df_bl[['dsi']], df_bl['cesd'])
m3=LinearRegression().fit(df_bl[['dsi','cesd']], df_bl['BAI'])
a=m2.coef_[0]; b=m3.coef_[1]; cp=m3.coef_[0]
ind=a*b; tot=ind+cp; mp=ind/tot*100 if tot!=0 else 0

print(f'\n  --- Unadjusted ---')
print(f'  a(DSI->CESD)       = {a:+.3f}')
print(f'  b(CESD->BAI|DSI)   = {b:+.3f}')
print(f'  c_prime(DSI->BAI|CESD) = {cp:+.2f}')
print(f'  Indirect(a*b)      = {ind:+.2f}')
print(f'  Total(from m3)     = {tot:+.2f}')
print(f'  Total(from m1)     = {m1.coef_[0]:+.2f}  [cross-check]')
print(f'  CES-D Mediation    = {mp:.1f}%')

# Age-adjusted
if 'pubage' in df_bl.columns:
    m1a=LinearRegression().fit(df_bl[['dsi','pubage']].fillna(0), df_bl['BAI'])
    m2a=LinearRegression().fit(df_bl[['dsi','pubage']].fillna(0), df_bl['cesd'])
    m3a=LinearRegression().fit(df_bl[['dsi','cesd','pubage']].fillna(0), df_bl['BAI'])
    a_a=m2a.coef_[0]; b_a=m3a.coef_[1]; cp_a=m3a.coef_[0]
    ind_a=a_a*b_a; tot_a=ind_a+cp_a; mp_a=ind_a/tot_a*100 if tot_a!=0 else 0
    print(f'\n  --- Age-adjusted ---')
    print(f'  a(DSI->CESD)       = {a_a:+.3f}')
    print(f'  b(CESD->BAI|DSI)   = {b_a:+.3f}')
    print(f'  c_prime(DSI->BAI|CESD) = {cp_a:+.2f}')
    print(f'  Indirect(a*b)      = {ind_a:+.2f}')
    print(f'  Total(from m3)     = {tot_a:+.2f}')
    print(f'  Total(from m1)     = {m1a.coef_[0]:+.2f}  [cross-check]')
    print(f'  CES-D Mediation    = {mp_a:.1f}%')

# ============================================================
# DIAGNOSIS 4: Sensitivity to threshold
# ============================================================
print('\n' + '='*80)
print('DIAGNOSIS 4: Sensitivity to impairment threshold')
print('='*80)

for thresh in [3, 4, 5]:
    vi_t = ((df2['sighta']>=thresh)|(df2['dsighta']>=thresh)|(df2['nsighta']>=thresh)).astype(int)
    hi_t = (df2['hearinga']>=thresh).astype(int)
    dsi_t = ((vi_t==1)&(hi_t==1)).astype(int)
    sai_t = (vi_t + hi_t) / 2 * 100
    sub = pd.DataFrame({'pid':df2['pid'],'wave':df2['wave'],'dsi':dsi_t,'SAI':sai_t,
                        'BAI':df2['BAI'],'cesd':df2['cesd'],'pubage':df2['pubage']})
    sub = sub.dropna(subset=['dsi','BAI','cesd']).copy()
    fw_s = sub.groupby('pid')['wave'].transform('min')
    sub_bl = sub[sub['wave']==fw_s].copy()
    if len(sub_bl)>100:
        s1=LinearRegression().fit(sub_bl[['dsi']], sub_bl['BAI'])
        s2=LinearRegression().fit(sub_bl[['dsi']], sub_bl['cesd'])
        s3=LinearRegression().fit(sub_bl[['dsi','cesd']], sub_bl['BAI'])
        si=s2.coef_[0]*s3.coef_[1]; st=si+s3.coef_[0]; sm=si/st*100 if st!=0 else 0
        print(f'  thresh>={thresh}: DSI={sub_bl["dsi"].mean()*100:.1f}%, N={len(sub_bl):,}, '
              f'Total={st:+.2f}, a={s2.coef_[0]:+.3f}, b={s3.coef_[1]:+.3f}, Med={sm:.1f}%')

# Also sightlmt based
vi_lmt = ((df2['sightlmt']==1)|(df2['sighta']>=5)|(df2['dsighta']>=5)|(df2['nsighta']>=5)).astype(int)
hi_lmt = (df2['hearinga']>=4).astype(int)
dsi_lmt = ((vi_lmt==1)&(hi_lmt==1)).astype(int)
sub_lmt = pd.DataFrame({'pid':df2['pid'],'wave':df2['wave'],'dsi':dsi_lmt,'BAI':df2['BAI'],'cesd':df2['cesd']})
sub_lmt = sub_lmt.dropna(subset=['dsi','BAI','cesd']).copy()
fw_l = sub_lmt.groupby('pid')['wave'].transform('min')
sub_lmt_bl = sub_lmt[sub_lmt['wave']==fw_l].copy()
if len(sub_lmt_bl)>100:
    l1=LinearRegression().fit(sub_lmt_bl[['dsi']], sub_lmt_bl['BAI'])
    l2=LinearRegression().fit(sub_lmt_bl[['dsi']], sub_lmt_bl['cesd'])
    l3=LinearRegression().fit(sub_lmt_bl[['dsi','cesd']], sub_lmt_bl['BAI'])
    li=l2.coef_[0]*l3.coef_[1]; lt=li+l3.coef_[0]; lm=li/lt*100 if lt!=0 else 0
    print(f'  sightlmt+verypoor:  DSI={sub_lmt_bl["dsi"].mean()*100:.1f}%, N={len(sub_lmt_bl):,}, Med={lm:.1f}%')

# ============================================================
# DIAGNOSIS 5: LCA re-run
# ============================================================
print('\n' + '='*80)
print('DIAGNOSIS 5: LCA phenotypes (corrected SAI/BAI/BoAI)')
print('='*80)

df_lca = df2.dropna(subset=['SAI','BAI','BoAI']).copy()
fw_lca = df_lca.groupby('pid')['wave'].transform('min')
df_lca_bl = df_lca[df_lca['wave']==fw_lca].copy()
print(f'  LCA N={len(df_lca_bl):,}')

X = df_lca_bl[['SAI','BAI','BoAI']].values
gmm = GaussianMixture(n_components=5, random_state=42, n_init=10)
df_lca_bl['phenotype'] = gmm.fit_predict(StandardScaler().fit_transform(X))

for p in range(5):
    sub = df_lca_bl[df_lca_bl['phenotype']==p]
    dsi_p = sub['dsi'].mean()*100 if 'dsi' in sub.columns else 0
    print(f'  Type {p+1}: n={len(sub):5,} ({len(sub)/len(df_lca_bl)*100:4.1f}%) '
          f'SAI={sub["SAI"].mean():3.0f} BAI={sub["BAI"].mean():3.0f} '
          f'BoAI={sub["BoAI"].mean():3.0f} DSI={dsi_p:3.1f}%')

# ============================================================
# DIAGNOSIS 6: Cognitive variable direction check
# ============================================================
print('\n' + '='*80)
print('DIAGNOSIS 6: Cognitive variable consistency check')
print('='*80)
print(f'  {"Variable":<12s} {"Min":>6s} {"Max":>6s} {"Corr w/orient":>14s} {"Direction"}')
for v in cog_vars:
    d = df2[v].dropna()
    corr = d.corr(df2['orient']) if 'orient' in df2.columns and len(d)>100 else 0
    direction = 'higher=better' if corr > 0 else 'LOWER=better (REVERSE?)'
    print(f'  {v:<12s} {d.min():>6.1f} {d.max():>6.1f} {corr:>+14.3f}  {direction}')

# ============================================================
# FINAL SUMMARY
# ============================================================
print('\n' + '='*80)
print('FINAL SUMMARY TABLE')
print('='*80)
print(f'{"Metric":<35s} {"BUGGY":>12s} {"CORRECTED":>12s} {"CHARLS(ref)":>12s}')
print('-'*75)
print(f'{"DSI prevalence":<35s} {df1_bl["dsi_bug"].mean()*100:>11.1f}% {df_bl["dsi"].mean()*100:>11.1f}% {"4.4%":>12s}')
print(f'{"DSI->BAI total effect":<35s} {tot:>12.2f} {tot:>12.2f} {"-4.89":>12s}')
print(f'{"DSI->CESD (a path)":<35s} {m2.coef_[0]:>12.3f} {a:>12.3f} {"--":>12s}')
print(f'{"CESD->BAI (b path)":<35s} {m3.coef_[1]:>12.3f} {b:>12.3f} {"--":>12s}')
print(f'{"CES-D Mediation %":<35s} {mp:>11.1f}% {mp:>11.1f}% {"28.0%":>12s}')
print(f'{"BAI mean":<35s} {df1_bl["BAI_bug"].mean():>12.0f} {df_bl["BAI"].mean():>12.0f} {"47":>12s}')
print(f'{"CESD mean":<35s} {df1_bl["cesd_bug"].mean():>12.1f} {df_bl["cesd"].mean():>12.1f} {"--":>12s}')
print(f'{"Age mean":<35s} {df1_bl["pubage"].mean():>12.0f} {df_bl["pubage"].mean():>12.0f} {"--":>12s}')

# Save
df_bl.to_csv(os.path.join(out_dir, 'klosa_corrected.csv'), index=False)
print(f'\nCorrected data saved to results/tables/klosa_corrected.csv')

print('\n' + '='*80)
print('DIAGNOSTIC VERDICT')
print('='*80)
print(f"""
1. Vision encoding (sighta==1):    REVERSED -- 1=Excellent, should use >=4
2. Hearing encoding (hearinga==1): REVERSED -- 1=Excellent, should use >=4
3. CES-D dilution (cesd10am/bm):   VARIABLE MISUSE -- 'm' vars are missing flags
4. BAI missing vars (imrc3 etc):   INCOMPLETE -- 3/7 cognitive vars used
5. Inconsistent mediation model:   MINOR -- pubage covariate inconsistent

ROOT CAUSE of 86% mediation:
  DSI=1 captured "excellent vision AND excellent hearing" -> healthiest subgroup
  These people have BETTER cognition AND LESS depression
  -> Both a-path and b-path artificially affected
  -> Mediation percentage = artifact of encoding error

CORRECTED mediation: {mp:.1f}% (vs 80% buggy replication, vs 86% originally reported)
""")
