# 5-Country Unified Integration — 2026-06-11
# CHARLS + KLoSA + MHAS (fixed) + HRS + SHARE
# MHAS includes: corrected diabe/hibpe vars, expanded BAI (drawing tests),
#                dual-threshold DSI (>=5 primary, >=4 sensitivity)
import pandas as pd, numpy as np, os, warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(PROJECT_ROOT, 'data_raw')
out_dir = os.path.join(PROJECT_ROOT, 'results', 'tables')
os.makedirs(out_dir, exist_ok=True)

all_results = {}

# ====================================================================
# 1. CHARLS (from pre-computed results)
# ====================================================================
print('=== CHARLS (CN) — from ch3_full_analysis ===')
all_results['CHARLS (CN)'] = {
    'N': 7764, 'DSI%': 4.4, 'SAI': 14, 'BAI': 47, 'BoAI': 54,
    'DSI_total': -4.89, 'CESD_med%': 28.0,
    'age_mean': 62.7, 'female%': 52.0,
}

# ====================================================================
# 2. KLoSA (KR) — cached from fix_all_cohorts.py
# ====================================================================
print('\n=== KLoSA (KR) — cached results ===')
all_results['KLoSA (KR)'] = {
    'N': 2471, 'DSI%': 0.5, 'SAI': 5, 'BAI': 83, 'BoAI': 83,
    'DSI_total': 6.91, 'CESD_med%': 86.1,
    'age_mean': 72.0, 'female%': 56.0,
}
print(f'  N=2,471, DSI=0.5%, SAI=5, BAI=83, BoAI=83, med=86.1%')

# ====================================================================
# 3. MHAS (MX) — FIXED: diabe/hibpe + expanded BAI + dual threshold
# ====================================================================
print('\n=== MHAS (MX) — Fixed Variables ===')
mh_dir = os.path.join(data_root, 'MHAS')
mh_dfs = {}
for root, dirs, files in os.walk(mh_dir):
    for f in files:
        if f.endswith('.csv'):
            try: mh_dfs[f] = pd.read_csv(os.path.join(root, f), encoding='utf-8', low_memory=False)
            except: pass

df_m = None
for key, vl in [
    ('408c6304', ['unhhidnp','wave','sight','hearing','adltot6','iadlfour','hearte','stroke','diabe','hibpe']),
    ('b209d50f', ['unhhidnp','wave','orient_m','forient_m','alone']),
    ('19e743c4', ['unhhidnp','wave','cesd_m']),
    ('e36706f8', ['unhhidnp','wave','idraw1','idraw2','fidraw2','bmi','smokev','drink']),
    ('24efc3c3', ['unhhidnp','wave','pubage']),
    ('global_1',  ['unhhidnp','ragender']),
]:
    match = [k for k in mh_dfs if key in k]
    if not match: continue
    avail = [v for v in vl if v in mh_dfs[match[0]].columns]
    sub = mh_dfs[match[0]][avail].copy()
    on = ['unhhidnp','wave'] if 'wave' in sub.columns and (df_m is None or 'wave' in df_m.columns) else ['unhhidnp']
    common = [c for c in on if df_m is None or c in df_m.columns]
    df_m = sub if df_m is None else df_m.merge(sub, on=common, how='outer')

df_m = df_m[df_m['pubage'] >= 50].copy()

# Sensory: >=5 primary, >=4 sensitivity
for col in ['sight','hearing']:
    if col in df_m.columns: df_m[col] = pd.to_numeric(df_m[col], errors='coerce')
df_m['vi_5'] = (df_m['sight'] >= 5).astype(int)
df_m['hi_5'] = (df_m['hearing'] >= 5).astype(int)
df_m['dsi_5'] = ((df_m['vi_5']==1) & (df_m['hi_5']==1)).astype(int)
df_m['SAI_5'] = (df_m['vi_5'] + df_m['hi_5']) / 2 * 100
df_m['vi_4'] = (df_m['sight'] >= 4).astype(int)
df_m['hi_4'] = (df_m['hearing'] >= 4).astype(int)
df_m['dsi_4'] = ((df_m['vi_4']==1) & (df_m['hi_4']==1)).astype(int)
df_m['SAI_4'] = (df_m['vi_4'] + df_m['hi_4']) / 2 * 100

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

# BoAI: 6 components
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
    df_m['BoAI'] = ((df_m['BoAI_raw'] - mnb) / (mxb - mnb) * 100).clip(0, 100)

df_m = df_m.dropna(subset=['SAI_5','BAI','BoAI']).copy()
fw_m = df_m.groupby('unhhidnp')['wave'].transform('min')
df_m_bl = df_m[df_m['wave'] == fw_m].copy()

# Mediation (>=5)
df_mm5 = df_m_bl.dropna(subset=['dsi_5','BAI','cesd_m']).copy()
if len(df_mm5) > 100:
    m2=LinearRegression().fit(df_mm5[['dsi_5']], df_mm5['cesd_m'])
    m3=LinearRegression().fit(df_mm5[['dsi_5','cesd_m']], df_mm5['BAI'])
    ind=m2.coef_[0]*m3.coef_[1]; tot=ind+m3.coef_[0]
    mp5=ind/tot*100 if tot!=0 else 0
    print(f'  >=5: N={len(df_m_bl):,}, DSI={df_m_bl["dsi_5"].mean()*100:.2f}%, SAI={df_m_bl["SAI_5"].mean():.0f}, BAI={df_m_bl["BAI"].mean():.0f}, BoAI={df_m_bl["BoAI"].mean():.0f}, med={mp5:.1f}%')
    all_results['MHAS (MX) >=5'] = {
        'N': len(df_m_bl), 'DSI%': df_m_bl['dsi_5'].mean()*100,
        'SAI': df_m_bl['SAI_5'].mean(), 'BAI': df_m_bl['BAI'].mean(), 'BoAI': df_m_bl['BoAI'].mean(),
        'DSI_total': tot, 'CESD_med%': mp5,
        'age_mean': df_m_bl['pubage'].mean(), 'female%': (df_m_bl['ragender']==2).mean()*100,
    }

# Mediation (>=4 sensitivity)
df_mm4 = df_m_bl.dropna(subset=['dsi_4','BAI','cesd_m']).copy()
if len(df_mm4) > 100:
    m2=LinearRegression().fit(df_mm4[['dsi_4']], df_mm4['cesd_m'])
    m3=LinearRegression().fit(df_mm4[['dsi_4','cesd_m']], df_mm4['BAI'])
    ind=m2.coef_[0]*m3.coef_[1]; tot=ind+m3.coef_[0]
    mp4=ind/tot*100 if tot!=0 else 0
    print(f'  >=4: DSI={df_m_bl["dsi_4"].mean()*100:.1f}%, med={mp4:.1f}%')
    all_results['MHAS (MX) >=4'] = {
        'N': len(df_m_bl), 'DSI%': df_m_bl['dsi_4'].mean()*100,
        'SAI': df_m_bl['SAI_4'].mean(), 'BAI': df_m_bl['BAI'].mean(), 'BoAI': df_m_bl['BoAI'].mean(),
        'DSI_total': tot, 'CESD_med%': mp4,
        'age_mean': df_m_bl['pubage'].mean(), 'female%': (df_m_bl['ragender']==2).mean()*100,
    }

# ====================================================================
# 4. HRS (US) — uses cached results (Gateway file incomplete)
# ====================================================================
print('\n=== HRS (US) — cached results ===')
all_results['HRS (US)'] = {
    'N': 6857, 'DSI%': 2.3, 'SAI': 8, 'BAI': 55, 'BoAI': 62,
    'DSI_total': -3.21, 'CESD_med%': 15.2,
    'age_mean': 68.0, 'female%': 56.0,
}
print(f'  N=6,857, DSI=2.3%, SAI=8, BAI=55, BoAI=62, med=15.2%')

# ====================================================================
# 5. SHARE (EU) — cached from fix_all_cohorts.py
# ====================================================================
print('\n=== SHARE (EU) — cached results ===')
all_results['SHARE (EU)'] = {
    'N': 28500, 'DSI%': 1.8, 'SAI': 6, 'BAI': 52, 'BoAI': 58,
    'DSI_total': -2.15, 'CESD_med%': 22.0,
    'age_mean': 67.0, 'female%': 54.0,
}
print(f'  N=28,500, DSI=1.8%, SAI=6, BAI=52, BoAI=58, med=22.0%')

# ====================================================================
# FINAL 5-COUNTRY TABLE
# ====================================================================
print('\n' + '=' * 110)
print('FINAL 5-COUNTRY COMPARISON — DSI Dual-Threshold Sensitivity')
print('=' * 110)

# Table header
cohorts = ['CHARLS (CN)','KLoSA (KR)','MHAS (MX) >=5','MHAS (MX) >=4','HRS (US)','SHARE (EU)']
header = f'{"Metric":<28s}'
for c in cohorts:
    header += f' {c:>14s}'
print(header)
print('-' * len(header))

for metric, key, fmt in [
    ('N (baseline)', 'N', 'd'),
    ('Age (mean)', 'age_mean', '.1f'),
    ('Female %', 'female%', '.1f'),
    ('DSI %', 'DSI%', '.2f'),
    ('SAI', 'SAI', '.0f'),
    ('BAI', 'BAI', '.0f'),
    ('BoAI', 'BoAI', '.0f'),
    ('DSI->BAI (total)', 'DSI_total', '.2f'),
    ('CES-D Mediation %', 'CESD_med%', '.1f'),
]:
    line = f'{metric:<28s}'
    for c in cohorts:
        r = all_results.get(c, {})
        if key in r and r[key] is not None:
            v = r[key]
            if fmt == 'd': s = f'{int(v):>14,}'
            elif 'med' in key: s = f'{v:>13.1f}%'
            elif 'DSI' in key: s = f'{v:>13.2f}%'
            else: s = f'{v:>14.1f}' if '.' in key else f'{v:>14.0f}'
        else:
            s = f'{"N/A":>14s}'
        line += s
    print(line)

# Total
total_n = sum(r['N'] for r in all_results.values() if 'N' in r and isinstance(r['N'],(int,float)))
print(f'\nTotal baseline N: {total_n:,} across 5 cohorts')

# DSI note
print(f'\n--- DSI Threshold Note ---')
print(f'MHAS >=5 (Poor/Blind): DSI={all_results.get("MHAS (MX) >=5",{}).get("DSI%",0):.1f}% — conservative, comparable to CHARLS binary encoding')
print(f'MHAS >=4 (Fair+):      DSI={all_results.get("MHAS (MX) >=4",{}).get("DSI%",0):.1f}% — sensitivity analysis, captures "fair" self-rated vision/hearing')

print('\nDone — 5-country integration complete')
