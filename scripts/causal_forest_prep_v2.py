#!/usr/bin/env python3
"""
Causal Forest 异质性分析 — 数据预处理 (v2)
========================================
Aim 3: 识别 DSI → BAI 效应异质性来源
Treatment: DSI (双感觉障碍)
Outcome: BAI (大脑老化指数)
Heterogeneity variables: 年龄/性别/教育/基线认知/社会参与/慢性病数量
"""

import os, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARLS_DIR = os.path.join(PROJECT_ROOT, 'data_raw', 'CHARLS')
TABLES_DIR = os.path.join(PROJECT_ROOT, 'results', 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 70)
print("CHARLS Causal Forest — Data Preparation v2")
print("=" * 70)

# ============================================================
# 1. Load phenotypes (baseline)
# ============================================================
print("\n1. Loading CHARLS baseline phenotypes...")
df_pheno = pd.read_csv(os.path.join(TABLES_DIR, 'charls_full_phenotypes.csv'))
first_wave = df_pheno.groupby('id')['wave'].transform('min')
df_bl = df_pheno[df_pheno['wave'] == first_wave].copy()
print(f"   Baseline: N={len(df_bl):,}")

# ============================================================
# 2. Load raw CHARLS data
# ============================================================
print("\n2. Loading raw CHARLS data...")
dfs_raw = {}
for f in sorted(os.listdir(CHARLS_DIR)):
    if f.endswith('.csv'):
        dfs_raw[f] = pd.read_csv(os.path.join(CHARLS_DIR, f), encoding='utf-8', low_memory=False)

# 2a. Education
edu_file = [k for k in dfs_raw if 'aeccc6cd' in k][0]
df_edu = dfs_raw[edu_file][['id','education2']].copy()
# Education mapping: 1=illiterate(0y), 2=partial primary(3y), 3=Sishu(3y),
# 4=elementary(6y), 5=middle(9y), 6=high(12y), 7=vocational(12y),
# 8=college(14y), 9=bachelor(16y), 10=master(19y), 11=PhD(22y)
edu_map = {1:0, 2:3, 3:3, 4:6, 5:9, 6:12, 7:12, 8:14, 9:16, 10:19, 11:22}
df_edu['education_yrs'] = df_edu['education2'].map(edu_map)
df_edu = df_edu.drop(columns=['education2']).dropna(subset=['education_yrs'])

# 2b. Chronic diseases (86b7e855)
disease_file = [k for k in dfs_raw if '86b7e855' in k][0]
disease_map = {
    'diabe':'chronic_diabetes','lunge':'chronic_lung','hearte':'chronic_heart',
    'stroke':'chronic_stroke','psyche':'chronic_psychiatric','arthre':'chronic_arthritis',
    'livere':'chronic_liver','kidneye':'chronic_kidney','asthmae':'chronic_asthma'
}
df_disease = dfs_raw[disease_file][['id','wave'] + list(disease_map.keys())].copy()
for old, new in disease_map.items():
    df_disease[new] = pd.to_numeric(df_disease[old], errors='coerce').clip(0, 1)
df_disease = df_disease.drop(columns=list(disease_map.keys()))

# 2c. Lifestyle & health (45248e43) — largest file
hyper_file = [k for k in dfs_raw if '45248e43' in k][0]
lifestyle_cols = ['id','wave','hypertenision','physical_activity2','smokev','drinkev',
                  'fi_health_status','fi_depression','cesd','sleep_night','short_sleep']
df_lifestyle = dfs_raw[hyper_file][[c for c in lifestyle_cols if c in dfs_raw[hyper_file].columns]].copy()
for c in ['hypertenision','smokev','drinkev','fi_depression','short_sleep']:
    if c in df_lifestyle.columns:
        df_lifestyle[c] = pd.to_numeric(df_lifestyle[c], errors='coerce').clip(0, 1)
for c in ['physical_activity2','fi_health_status','cesd','sleep_night']:
    if c in df_lifestyle.columns:
        df_lifestyle[c] = pd.to_numeric(df_lifestyle[c], errors='coerce')

# 2d. Baseline cognition (92ad8abe)
cog_file = [k for k in dfs_raw if '92ad8abe' in k][0]
df_cog = dfs_raw[cog_file][['id','wave','global_cognition','orient','draw']].copy()
for c in ['global_cognition','orient','draw']:
    df_cog[c] = pd.to_numeric(df_cog[c], errors='coerce')

# ============================================================
# 3. Merge at baseline
# ============================================================
print("\n3. Merging at baseline...")

def get_baseline(df, id_col='id', wave_col='wave'):
    """Keep first wave per ID"""
    first = df.groupby(id_col)[wave_col].transform('min')
    return df[df[wave_col] == first].drop(columns=[wave_col])

disease_bl = get_baseline(df_disease)
lifestyle_bl = get_baseline(df_lifestyle)
cog_bl = get_baseline(df_cog)

df = df_bl.merge(df_edu, on='id', how='left')
df = df.merge(disease_bl, on='id', how='left')
df = df.merge(lifestyle_bl, on='id', how='left')
df = df.merge(cog_bl, on='id', how='left')

# Count chronic diseases
chronic_cols = [c for c in df.columns if c.startswith('chronic_')]
df['chronic_n'] = df[chronic_cols].sum(axis=1).fillna(0)
if 'hypertenision' in df.columns:
    df['chronic_n'] = df['chronic_n'] + df['hypertenision'].fillna(0)

print(f"   After merge: N={len(df):,}")

# ============================================================
# 4. Prepare & impute analysis variables
# ============================================================
print("\n4. Preparing & imputing variables...")

# Core variables
df['W'] = df['dsi'].astype(int)       # Treatment: DSI
df['Y'] = df['BAI']                    # Outcome: BAI
df['age'] = df['pubage']               # Age
df['female'] = (df['ragender'] == 2).astype(int)  # Sex
df['edu_yrs'] = df['education_yrs']    # Education (years)
df['cog_base'] = df['global_cognition']  # Baseline cognition
df['chronic'] = df['chronic_n']        # Chronic disease count

# Social participation/lifestyle variables
if 'physical_activity2' in df.columns:
    df['phys_act'] = df['physical_activity2'].fillna(df['physical_activity2'].median())
else:
    df['phys_act'] = 2

if 'smokev' in df.columns:
    df['smoker'] = df['smokev'].fillna(0)
else:
    df['smoker'] = 0

if 'drinkev' in df.columns:
    df['drinker'] = df['drinkev'].fillna(0)
else:
    df['drinker'] = 0

if 'cesd' in df.columns:
    df['depression'] = df['cesd'].fillna(df['cesd'].median())
elif 'cesd10' in df.columns:
    df['depression'] = df['cesd10'].fillna(df['cesd10'].median())
else:
    df['depression'] = 0

if 'fi_health_status' in df.columns:
    df['sr_health'] = df['fi_health_status'].fillna(df['fi_health_status'].median())
else:
    df['sr_health'] = 3

# Composite social engagement score (physical activity + drinking + smoking inverse)
df['social_score'] = (
    df['phys_act'].fillna(2) * 0.5 +
    df['drinker'].fillna(0) * 1.0 +
    (1 - df['smoker'].fillna(0)) * 0.5
)

# ============================================================
# 5. Clean and export
# ============================================================
print("\n5. Cleaning and exporting...")

# Analysis-ready columns
out_cols = ['id', 'W', 'Y', 'age', 'female', 'edu_yrs', 'cog_base',
            'chronic', 'phys_act', 'smoker', 'drinker', 'depression',
            'sr_health', 'social_score', 'SAI', 'BoAI']

df_out = df[out_cols].copy()

# Drop rows with missing in mandatory variables
mandatory = ['W', 'Y', 'age', 'female', 'edu_yrs', 'cog_base', 'chronic']
n_before = len(df_out)
df_out = df_out.dropna(subset=mandatory)
n_after = len(df_out)
print(f"   After mandatory dropna: {n_before:,} -> {n_after:,} (dropped {n_before-n_after})")

# Fill remaining NAs with median/mode
for col in df_out.columns:
    if col in mandatory or col == 'id':
        continue
    if df_out[col].isna().sum() > 0:
        if df_out[col].dtype in ['float64', 'int64']:
            med = df_out[col].median()
            df_out[col] = df_out[col].fillna(med if not pd.isna(med) else 0)
            print(f"   Imputed {col}: {df_out[col].isna().sum()} NAs -> median={med:.1f}")
        else:
            mode = df_out[col].mode()
            fill_val = mode.iloc[0] if len(mode) > 0 else 0
            df_out[col] = df_out[col].fillna(fill_val)

# Summary
print(f"\n{'='*60}")
print("Final Analysis Dataset")
print(f"{'='*60}")
print(f"  N = {n_after:,}")
print(f"  DSI (W=1) = {df_out['W'].mean()*100:.2f}% (n={int(df_out['W'].sum())})")
print(f"  BAI (Y)   = {df_out['Y'].mean():.1f} ± {df_out['Y'].std():.1f}")
print(f"  Age       = {df_out['age'].mean():.1f} ± {df_out['age'].std():.1f}")
print(f"  Female    = {df_out['female'].mean()*100:.1f}%")
print(f"  Education = {df_out['edu_yrs'].mean():.1f} ± {df_out['edu_yrs'].std():.1f} yrs")
print(f"  Cognition = {df_out['cog_base'].mean():.1f} ± {df_out['cog_base'].std():.1f}")
print(f"  Chronic   = {df_out['chronic'].mean():.1f} ± {df_out['chronic'].std():.1f}")
print(f"  Depression = {df_out['depression'].mean():.1f} ± {df_out['depression'].std():.1f}")
print(f"  Social score = {df_out['social_score'].mean():.1f} ± {df_out['social_score'].std():.1f}")

# Quick check: DSI vs non-DSI BAI
from scipy import stats as sp_stats
dsi1 = df_out[df_out['W']==1]['Y']
dsi0 = df_out[df_out['W']==0]['Y']
t_stat, p_val = sp_stats.ttest_ind(dsi1, dsi0)
print(f"\n  BAI: DSI=0 {dsi0.mean():.1f} vs DSI=1 {dsi1.mean():.1f}")
print(f"  Diff = {dsi1.mean()-dsi0.mean():.1f}, t={t_stat:.2f}, p={p_val:.4f}")

# Export
out_path = os.path.join(TABLES_DIR, 'charls_causal_forest_data.csv')
df_out.to_csv(out_path, index=False)
print(f"\n  [OK] Exported: {out_path}")
print(f"  [OK] Columns ({len(out_cols)}): {out_cols}")

print("\n=== Data Preparation Complete ===")
