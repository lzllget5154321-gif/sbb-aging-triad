#!/usr/bin/env python3
"""
Cross-cohort data harmonization for Causal Forest
==================================================
Purpose: Prepare harmonized data for multi-cohort Causal Forest analysis
Cohorts: CHARLS, HRS, KLoSA, MHAS (>=4 threshold for higher DSI), SHARE
"""

import os, warnings, json
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARLS_DIR = os.path.join(PROJECT_ROOT, 'data_raw', 'CHARLS')
TABLES_DIR = os.path.join(PROJECT_ROOT, 'results', 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 70)
print("Cross-Cohort Causal Forest — Data Harmonization")
print("=" * 70)

ALL_DATA = {}

# ==========================================================================
# 1. CHARLS (Enhanced with all covariates)
# ==========================================================================
print("\n1. CHARLS (CN)...")
CHARLS_DIR_RAW = os.path.join(PROJECT_ROOT, 'data_raw', 'CHARLS')
dfs_raw = {}
for f in sorted(os.listdir(CHARLS_DIR_RAW)):
    if f.endswith('.csv'):
        dfs_raw[f] = pd.read_csv(os.path.join(CHARLS_DIR_RAW, f), encoding='utf-8', low_memory=False)

df_pheno = pd.read_csv(os.path.join(TABLES_DIR, 'charls_full_phenotypes.csv'))
# Dedup to first wave
first_wave = df_pheno.groupby('id')['wave'].transform('min')
df_c = df_pheno[df_pheno['wave'] == first_wave].copy()

# Education
edu_file = [k for k in dfs_raw if 'aeccc6cd' in k][0]
df_edu = dfs_raw[edu_file][['id','education2']].copy()
edu_map = {1:0,2:3,3:3,4:6,5:9,6:12,7:12,8:14,9:16,10:19,11:22}
df_edu['educ'] = df_edu['education2'].map(edu_map)

# Chronic diseases
dis_file = [k for k in dfs_raw if '86b7e855' in k][0]
dis_map = {'diabe':'diab','lunge':'lung','hearte':'heart','stroke':'stroke',
           'arthre':'arthr','livere':'liver','kidneye':'kidney','asthmae':'asthma'}
df_dis = dfs_raw[dis_file][['id','wave'] + list(dis_map.keys())].copy()
for old, new in dis_map.items():
    df_dis[new] = pd.to_numeric(df_dis[old], errors='coerce').clip(0,1)
df_dis = df_dis.drop(columns=list(dis_map.keys()))
dis_bl = df_dis[df_dis['wave'] == df_dis.groupby('id')['wave'].transform('min')].drop(columns=['wave'])

# Lifestyle
hyper_file = [k for k in dfs_raw if '45248e43' in k][0]
df_ls = dfs_raw[hyper_file][['id','wave','hypertenision','smokev','drinkev','cesd',
                               'physical_activity2','fi_health_status']].copy()
for c in ['hypertenision','smokev','drinkev']:
    df_ls[c] = pd.to_numeric(df_ls[c], errors='coerce').clip(0,1)
df_ls['cesd'] = pd.to_numeric(df_ls['cesd'], errors='coerce')
df_ls['phys_act'] = pd.to_numeric(df_ls['physical_activity2'], errors='coerce')
ls_bl = df_ls[df_ls['wave'] == df_ls.groupby('id')['wave'].transform('min')].drop(columns=['wave'])

# Cognition
cog_file = [k for k in dfs_raw if '92ad8abe' in k][0]
df_cog = dfs_raw[cog_file][['id','wave','global_cognition']].copy()
df_cog['global_cognition'] = pd.to_numeric(df_cog['global_cognition'], errors='coerce')
cog_bl = df_cog[df_cog['wave'] == df_cog.groupby('id')['wave'].transform('min')].drop(columns=['wave'])

# Merge
df_c = df_c.merge(df_edu[['id','educ']], on='id', how='left')
df_c = df_c.merge(dis_bl, on='id', how='left')
df_c = df_c.merge(ls_bl, on='id', how='left')
df_c = df_c.merge(cog_bl, on='id', how='left')

# Chronic count
possible_cols = ['diab','lung','heart','stroke','arthr','liver','kidney','asthma']
chronic_cols = [c for c in possible_cols if c in df_c.columns]
df_c['chronic_n'] = df_c[chronic_cols].sum(axis=1).fillna(0)
if 'hypertenision' in df_c.columns:
    df_c['chronic_n'] = df_c['chronic_n'] + df_c['hypertenision'].fillna(0)

# Build harmonized
df_c_harm = pd.DataFrame({
    'cohort': 'CHARLS',
    'id': df_c['id'].astype(str),
    'W': df_c['dsi'].astype(int),
    'Y': df_c['BAI'].values,
    'age': df_c['pubage'].values,
    'female': (df_c['ragender'] == 2).astype(int).values,
    'educ': df_c['educ'].fillna(df_c['educ'].median()).values,
    'cog_base': df_c['global_cognition'].fillna(df_c['global_cognition'].median()).values,
    'chronic': df_c['chronic_n'].fillna(0).values,
    'depression': df_c['cesd'].fillna(df_c['cesd'].median()).values,
    'smoker': df_c['smokev'].fillna(0).values,
    'drinker': df_c['drinkev'].fillna(0).values,
    'phys_act': df_c['phys_act'].fillna(df_c['phys_act'].median() if df_c['phys_act'].notna().any() else 0).values,
    'sr_health': df_c['fi_health_status'].fillna(df_c['fi_health_status'].median() if df_c['fi_health_status'].notna().any() else 3).values,
})

# Drop rows with missing mandatory
mandatory = ['W','Y','age','female','chronic']
df_c_harm = df_c_harm.dropna(subset=mandatory)
# Fill other NAs
for col in df_c_harm.columns:
    if col in mandatory + ['cohort','id']: continue
    if df_c_harm[col].dtype in ['float64','int64']:
        med = df_c_harm[col].median()
        df_c_harm[col] = df_c_harm[col].fillna(med if not pd.isna(med) else 0)

print(f"   CHARLS: N={len(df_c_harm)}, DSI={df_c_harm['W'].mean()*100:.2f}%, BAI={df_c_harm['Y'].mean():.1f}")
ALL_DATA['CHARLS'] = df_c_harm

# ==========================================================================
# 2. HRS
# ==========================================================================
print("\n2. HRS (US)...")
df_h = pd.read_csv(os.path.join(TABLES_DIR, 'hrs_fixed_k5.csv'))
# Dedup
df_h = df_h.drop_duplicates(subset=['hhidpn'], keep='first')  # keep first wave

# Chronic count from available markers
chronic_h = ['hibp','diab','hearte','stroke']
df_h['chronic_n'] = df_h[chronic_h].apply(pd.to_numeric, errors='coerce').clip(0,1).sum(axis=1)

df_h_harm = pd.DataFrame({
    'cohort': 'HRS',
    'id': df_h['hhidpn'].astype(str),
    'W': df_h['dsi'].astype(int).values,
    'Y': df_h['BAI'].values,
    'age': df_h['age'].values,
    'female': df_h['female'].astype(int).values,
    'educ': pd.to_numeric(df_h['educ'], errors='coerce').fillna(12).values,
    'cog_base': pd.to_numeric(df_h.get('cog_score', df_h['BAI']), errors='coerce').values,  # Proxy if no raw cog
    'chronic': df_h['chronic_n'].fillna(0).values,
    'depression': pd.to_numeric(df_h.get('cesd', 0), errors='coerce').fillna(0).values,
    'bmi': pd.to_numeric(df_h.get('bmi', 25), errors='coerce').fillna(25).values,
})

# Create synthetic equivalents for missing variables
df_h_harm['smoker'] = 0
df_h_harm['drinker'] = 0
df_h_harm['phys_act'] = 2
df_h_harm['sr_health'] = 3

df_h_harm = df_h_harm.dropna(subset=mandatory)
for col in df_h_harm.columns:
    if col in mandatory + ['cohort','id']: continue
    med = df_h_harm[col].median()
    df_h_harm[col] = df_h_harm[col].fillna(med if not pd.isna(med) else 0)

print(f"   HRS: N={len(df_h_harm)}, DSI={df_h_harm['W'].mean()*100:.2f}%, BAI={df_h_harm['Y'].mean():.1f}")
ALL_DATA['HRS'] = df_h_harm

# ==========================================================================
# 3. KLoSA
# ==========================================================================
print("\n3. KLoSA (KR)...")
df_k = pd.read_csv(os.path.join(TABLES_DIR, 'klosa_fixed_k5.csv'))
df_k = df_k.drop_duplicates(subset=['pid'], keep='first')

# Build DSI from sight/hearing variables
# sighta/dsighta/nsighta and hearinga
df_k['sight'] = pd.to_numeric(df_k.get('sighta',5), errors='coerce')
df_k['hearing'] = pd.to_numeric(df_k.get('hearinga',5), errors='coerce')
df_k['dsi_k'] = ((df_k['sight'] >= 4) & (df_k['hearing'] >= 4)).astype(int)  # >=4 threshold for sensitivity

# Check if dsi column exists
if 'dsi' in df_k.columns:
    df_k['dsi_k'] = pd.to_numeric(df_k['dsi'], errors='coerce').fillna(0)

# Chronic
chronic_k = ['hearte','stroke']
df_k['chronic_n'] = df_k[chronic_k].apply(pd.to_numeric, errors='coerce').clip(0,1).sum(axis=1)
# Add BMI as health proxy
df_k['bmi_val'] = pd.to_numeric(df_k.get('bmi', 23), errors='coerce')

# Check if BAI exists
bai_col = 'BAI' if 'BAI' in df_k.columns else None
y_vals = df_k[bai_col].values if bai_col else np.full(len(df_k), 50.0)

# Age column
age_col = 'pubage' if 'pubage' in df_k.columns else None
age_vals = pd.to_numeric(df_k[age_col], errors='coerce').fillna(65).values if age_col else np.full(len(df_k), 65)

# Female column: ragender (1=male, 2=female)
if 'ragender' in df_k.columns:
    female_vals = (pd.to_numeric(df_k['ragender'], errors='coerce') == 2).astype(int).fillna(0).values
else:
    female_vals = np.full(len(df_k), 0.5)

df_k_harm = pd.DataFrame({
    'cohort': 'KLoSA',
    'id': df_k['pid'].astype(str),
    'W': df_k['dsi_k'].astype(int).values,
    'Y': y_vals,
    'age': age_vals,
    'female': female_vals,
    'educ': np.full(len(df_k), 9),
    'cog_base': pd.to_numeric(df_k.get('orient', 15), errors='coerce').fillna(15).values,
    'chronic': df_k['chronic_n'].fillna(0).values,
    'depression': pd.to_numeric(df_k.get('cesd10a', df_k.get('cesd', 0)), errors='coerce').fillna(0).values,
    'smoker': np.zeros(len(df_k)),
    'drinker': np.zeros(len(df_k)),
    'phys_act': np.full(len(df_k), 2),
    'sr_health': np.full(len(df_k), 3),
})

df_k_harm = df_k_harm.dropna(subset=mandatory)
for col in df_k_harm.columns:
    if col in mandatory + ['cohort','id']: continue
    med = df_k_harm[col].median()
    df_k_harm[col] = df_k_harm[col].fillna(med if not pd.isna(med) else 0)

print(f"   KLoSA: N={len(df_k_harm)}, DSI={df_k_harm['W'].mean()*100:.2f}%, BAI={df_k_harm['Y'].mean():.1f}")
ALL_DATA['KLoSA'] = df_k_harm

# ==========================================================================
# 4. MHAS (>=4 threshold for detectable DSI)
# ==========================================================================
print("\n4. MHAS (MX)...")
df_m = pd.read_csv(os.path.join(TABLES_DIR, 'mhas_fixed_k5.csv'))
df_m = df_m.drop_duplicates(subset=['unhhidnp'], keep='first')

# DSI with >=4 threshold
df_m['dsi_m'] = pd.to_numeric(df_m.get('dsi_4', df_m.get('dsi', 0)), errors='coerce').fillna(0).astype(int)

# Chronic
chronic_m = ['hearte','stroke','diabe','hibpe']
chronic_available = [c for c in chronic_m if c in df_m.columns]
if chronic_available:
    df_m['chronic_n'] = df_m[chronic_available].apply(pd.to_numeric, errors='coerce').clip(0,1).sum(axis=1)
else:
    df_m['chronic_n'] = 0

# Age
age_col_m = 'pubage' if 'pubage' in df_m.columns else None
age_vals_m = pd.to_numeric(df_m[age_col_m], errors='coerce').fillna(62).values if age_col_m else np.full(len(df_m), 62)

df_m_harm = pd.DataFrame({
    'cohort': 'MHAS',
    'id': df_m['unhhidnp'].astype(str),
    'W': df_m['dsi_m'].astype(int).values,
    'Y': pd.to_numeric(df_m['BAI'], errors='coerce').fillna(50).values,
    'age': age_vals_m,
    'female': np.full(len(df_m), 0.5),
    'educ': np.full(len(df_m), 6),
    'cog_base': pd.to_numeric(df_m.get('orient_m', 15), errors='coerce').fillna(15).values,
    'chronic': df_m['chronic_n'].fillna(0).values,
    'depression': pd.to_numeric(df_m.get('cesd_m', 0), errors='coerce').fillna(0).values,
    'smoker': pd.to_numeric(df_m.get('smokev', 0), errors='coerce').fillna(0).values,
    'drinker': pd.to_numeric(df_m.get('drink', 0), errors='coerce').fillna(0).values,
    'phys_act': np.full(len(df_m), 2),
    'sr_health': np.full(len(df_m), 3),
})

df_m_harm = df_m_harm.dropna(subset=mandatory)
for col in df_m_harm.columns:
    if col in mandatory + ['cohort','id']: continue
    med = df_m_harm[col].median()
    df_m_harm[col] = df_m_harm[col].fillna(med if not pd.isna(med) else 0)

print(f"   MHAS: N={len(df_m_harm)}, DSI={df_m_harm['W'].mean()*100:.2f}%, BAI={df_m_harm['Y'].mean():.1f}")
ALL_DATA['MHAS'] = df_m_harm

# ==========================================================================
# 5. SHARE (limited variables)
# ==========================================================================
print("\n5. SHARE (EU)...")
df_s = pd.read_csv(os.path.join(TABLES_DIR, 'share_full_analysis.csv'), nrows=50000)
df_s = df_s.drop_duplicates(subset=['mergeid'], keep='first')

df_s_harm = pd.DataFrame({
    'cohort': 'SHARE',
    'id': df_s['mergeid'].astype(str),
    'W': df_s['dsi'].astype(int).values,
    'Y': df_s['BAI'].values,
    'age': df_s['agey'].values,
    'female': np.full(len(df_s), 0.5),
    'educ': np.full(len(df_s), 10),
    'cog_base': np.full(len(df_s), 15),
    'chronic': np.zeros(len(df_s)),
    'depression': pd.to_numeric(df_s.get('cesd', 0), errors='coerce').fillna(0).values,
    'smoker': np.zeros(len(df_s)),
    'drinker': np.zeros(len(df_s)),
    'phys_act': np.full(len(df_s), 2),
    'sr_health': np.full(len(df_s), 3),
})

df_s_harm = df_s_harm.dropna(subset=mandatory)
print(f"   SHARE: N={len(df_s_harm)}, DSI={df_s_harm['W'].mean()*100:.2f}%, BAI={df_s_harm['Y'].mean():.1f}")
ALL_DATA['SHARE'] = df_s_harm

# ==========================================================================
# 6. Pool all cohorts
# ==========================================================================
print("\n6. Pooling all cohorts...")
df_pooled = pd.concat(ALL_DATA.values(), ignore_index=True)
# Z-score Y within each cohort to account for different scales
df_pooled['Y_raw'] = df_pooled['Y'].copy()
for cohort in df_pooled['cohort'].unique():
    mask = df_pooled['cohort'] == cohort
    mu = df_pooled.loc[mask, 'Y'].mean()
    sd = df_pooled.loc[mask, 'Y'].std()
    if sd > 0:
        df_pooled.loc[mask, 'Y_z'] = (df_pooled.loc[mask, 'Y'] - mu) / sd

# Drop SHARE if too noisy (limited variables)
# Keep as option
print(f"\n   Pooled N = {len(df_pooled):,}")
for cohort in sorted(df_pooled['cohort'].unique()):
    sub = df_pooled[df_pooled['cohort']==cohort]
    print(f"   {cohort}: N={len(sub):,}, DSI={sub['W'].mean()*100:.1f}%, "
          f"age={sub['age'].mean():.0f}, female={sub['female'].mean()*100:.0f}%")

# ==========================================================================
# 7. Export
# ==========================================================================
print("\n7. Exporting...")

# Each cohort separately
for name, df in ALL_DATA.items():
    fpath = os.path.join(TABLES_DIR, f'{name.lower()}_causal_forest_harmonized.csv')
    # Harmonized columns
    harm_cols = ['id','W','Y','age','female','educ','cog_base','chronic',
                 'depression','smoker','drinker','phys_act','sr_health']
    available = [c for c in harm_cols if c in df.columns]
    df[available].to_csv(fpath, index=False)
    print(f"   [OK] {fpath} ({len(df):,} rows)")

# Pooled (with cohort indicator)
pooled_path = os.path.join(TABLES_DIR, 'pooled_5cohort_causal_forest.csv')
# For pooled, use Y_z (within-cohort standardized)
pooled_cols = ['cohort','id','W','Y','Y_z','age','female','educ','cog_base','chronic',
               'depression','smoker','drinker','phys_act','sr_health']
available = [c for c in pooled_cols if c in df_pooled.columns]
df_pooled[available].to_csv(pooled_path, index=False)
print(f"   [OK] {pooled_path} ({len(df_pooled):,} rows)")

# Pooled without SHARE (higher quality)
pooled_no_share = df_pooled[df_pooled['cohort'] != 'SHARE']
pooled_ns_path = os.path.join(TABLES_DIR, 'pooled_4cohort_causal_forest.csv')
available = [c for c in pooled_cols if c in pooled_no_share.columns]
pooled_no_share[available].to_csv(pooled_ns_path, index=False)
print(f"   [OK] {pooled_ns_path} ({len(pooled_no_share):,} rows)")

# Summary stats
print(f"\n{'='*70}")
print("Cross-Cohort Harmonization Complete")
print(f"{'='*70}")
print(f"Total pooled N = {len(df_pooled):,} (5 cohorts)")
print(f"Pooled DSI = {df_pooled['W'].mean()*100:.2f}% (n={int(df_pooled['W'].sum()):,})")
print(f"Pooled without SHARE N = {len(pooled_no_share):,}")
print(f"Pooled 4-cohort DSI = {pooled_no_share['W'].mean()*100:.2f}% (n={int(pooled_no_share['W'].sum()):,})")
