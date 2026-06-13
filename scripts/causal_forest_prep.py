#!/usr/bin/env python3
"""
Causal Forest 异质性分析 — 数据预处理
========================================
Aim 3: DSI → BAI 效应异质性来源识别
Treatment: DSI (双感觉障碍)
Outcome: BAI (大脑老化指数)
Heterogeneity variables: 年龄/性别/教育/基线认知/社会参与/慢性病数量

产出: charls_causal_forest_data.csv → R grf 分析
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
print("CHARLS Causal Forest — Data Preparation")
print("=" * 70)

# ============================================================
# 1. Load phenotypes (baseline from full analysis)
# ============================================================
print("\n1. Loading CHARLS baseline phenotypes...")
df_pheno = pd.read_csv(os.path.join(TABLES_DIR, 'charls_full_phenotypes.csv'))
print(f"   Baseline phenotypes: N={len(df_pheno):,}")

# Keep only baseline wave per ID (first wave)
first_wave = df_pheno.groupby('id')['wave'].transform('min')
df_bl = df_pheno[df_pheno['wave'] == first_wave].copy()
print(f"   After dedup (first wave): N={len(df_bl):,}")

# ============================================================
# 2. Load raw data for additional covariates
# ============================================================
print("\n2. Loading raw CHARLS data...")
dfs_raw = {}
for f in sorted(os.listdir(CHARLS_DIR)):
    if f.endswith('.csv'):
        dfs_raw[f] = pd.read_csv(os.path.join(CHARLS_DIR, f), encoding='utf-8', low_memory=False)

# 2a. Education & demographics
print("   2a. Education & demographics...")
edu_file = [k for k in dfs_raw if 'aeccc6cd' in k][0]
edu_file = [k for k in dfs_raw if 'aeccc6cd' in k][0]
df_edu = dfs_raw[edu_file][['id','education2']].copy()
# education2: 1=No education, 2=Did not finish primary, 3=Sishu, 4=Elementary,
#              5=Middle school, 6=High school, 7=Vocational, 8=2/3yr college,
#              9=Bachelor, 10=Master, 11=PhD
df_edu['education_years'] = df_edu['education2'].map({
    1: 0, 2: 3, 3: 3, 4: 6, 5: 9, 6: 12, 7: 12, 8: 14, 9: 16, 10: 19, 11: 22
})
df_edu = df_edu.dropna(subset=['education_years'])

# 2b. Chronic diseases
print("   2b. Chronic disease indicators...")
disease_file = [k for k in dfs_raw if '86b7e855' in k][0]
disease_cols_map = {
    'diabe': 'chronic_diabetes',
    'lunge': 'chronic_lung',
    'hearte': 'chronic_heart',
    'stroke': 'chronic_stroke',
    'psyche': 'chronic_psychiatric',
    'arthre': 'chronic_arthritis',
    'livere': 'chronic_liver',
    'kidneye': 'chronic_kidney',
    'asthmae': 'chronic_asthma',
}
df_disease = dfs_raw[disease_file][['id','wave'] + list(disease_cols_map.keys())].copy()
for old, new in disease_cols_map.items():
    df_disease[new] = pd.to_numeric(df_disease[old], errors='coerce').clip(0, 1)
df_disease = df_disease.drop(columns=list(disease_cols_map.keys()))

# 2c. Hypertension & physical activity
print("   2c. Hypertension & physical activity...")
hyper_file = [k for k in dfs_raw if '45248e43' in k][0]
df_hyper = dfs_raw[hyper_file][['id','wave','hypertenision','physical_activity2',
                                  'vgact_c','mdact_c','ltact_c']].copy()
df_hyper['hypertenision'] = pd.to_numeric(df_hyper['hypertenision'], errors='coerce').clip(0, 1)
# physical_activity2: 1=vigorous, 2=moderate, 3=light, coding varies
df_hyper['pa_level'] = pd.to_numeric(df_hyper['physical_activity2'], errors='coerce')
# Higher = more active (reverse if needed)
df_hyper['social_activity'] = df_hyper[['vgact_c','mdact_c','ltact_c']].notna().sum(axis=1)
df_hyper = df_hyper.drop(columns=['vgact_c','mdact_c','ltact_c'])

# 2d. Baseline cognition scores
print("   2d. Cognition scores...")
cog_file = [k for k in dfs_raw if '92ad8abe' in k][0]
df_cog = dfs_raw[cog_file][['id','wave','global_cognition','orient','draw']].copy()
for c in ['global_cognition','orient','draw']:
    df_cog[c] = pd.to_numeric(df_cog[c], errors='coerce')

# ============================================================
# 3. Merge all covariates
# ============================================================
print("\n3. Merging covariates...")
# Merge disease data at baseline
df_disease_bl = df_disease[df_disease['wave'] == df_disease.groupby('id')['wave'].transform('min')]
df_disease_bl = df_disease_bl.drop(columns=['wave'])

# Merge hyper/PA at baseline
df_hyper_bl = df_hyper[df_hyper['wave'] == df_hyper.groupby('id')['wave'].transform('min')]
df_hyper_bl = df_hyper_bl.drop(columns=['wave'])

# Merge cognition at baseline
df_cog_bl = df_cog[df_cog['wave'] == df_cog.groupby('id')['wave'].transform('min')]
df_cog_bl = df_cog_bl.drop(columns=['wave'])

# Sequential merge
df = df_bl.merge(df_edu, on='id', how='left')
df = df.merge(df_disease_bl, on='id', how='left')
df = df.merge(df_hyper_bl, on='id', how='left')
df = df.merge(df_cog_bl, on='id', how='left')

# Count chronic diseases
chronic_cols = [c for c in df.columns if c.startswith('chronic_')]
df['chronic_count'] = df[chronic_cols].sum(axis=1)

# Add hypertension to count
df['chronic_count'] = df['chronic_count'] + df['hypertenision'].fillna(0)

print(f"   After merge: N={len(df):,}")

# ============================================================
# 4. Prepare analysis variables
# ============================================================
print("\n4. Preparing analysis variables...")

# Treatment: DSI (binary)
df['W'] = df['dsi'].astype(int)

# Outcome: BAI
df['Y'] = df['BAI']

# Covariates for heterogeneity
df['age'] = df['pubage']
df['female'] = (df['ragender'] == 2).astype(int)
df['education_yrs'] = df['education_years']
df['baseline_cog'] = df['global_cognition']
df['social_pa'] = df['pa_level']  # Physical activity as social engagement proxy
df['chronic_n'] = df['chronic_count']

# CES-D as depression covariate
df['cesd'] = df['cesd10']

# ============================================================
# 5. Clean and export
# ============================================================
print("\n5. Cleaning and exporting...")

# Select analysis variables
analysis_vars = ['id', 'W', 'Y', 'age', 'female', 'education_yrs',
                 'baseline_cog', 'social_pa', 'chronic_n', 'cesd',
                 'SAI', 'BoAI', 'pubage', 'ragender']

df_out = df[analysis_vars].copy()

# Drop rows with missing in key variables
key_vars = ['W', 'Y', 'age', 'female', 'education_yrs', 'baseline_cog', 'chronic_n']
n_before = len(df_out)
df_out = df_out.dropna(subset=key_vars)
n_after = len(df_out)
print(f"   Before dropna: {n_before:,} → After: {n_after:,} (dropped {n_before-n_after})")

# Summary statistics
print(f"\n{'='*60}")
print("Analysis Dataset Summary")
print(f"{'='*60}")
print(f"  N = {n_after:,}")
print(f"  DSI prevalence = {df_out['W'].mean()*100:.2f}%")
print(f"  BAI mean (SD) = {df_out['Y'].mean():.1f} ({df_out['Y'].std():.1f})")
print(f"  Age mean (SD)  = {df_out['age'].mean():.1f} ({df_out['age'].std():.1f})")
print(f"  Female %        = {df_out['female'].mean()*100:.1f}%")
print(f"  Education mean  = {df_out['education_yrs'].mean():.1f} yrs")
print(f"  Chronic diseases mean = {df_out['chronic_n'].mean():.1f}")

# Export
out_path = os.path.join(TABLES_DIR, 'charls_causal_forest_data.csv')
df_out.to_csv(out_path, index=False)
print(f"\n  [OK] Exported: {out_path}")
print(f"  [OK] Columns: {list(df_out.columns)}")

# ============================================================
# 6. Quick sanity checks
# ============================================================
print(f"\n{'='*60}")
print("Sanity Checks")
print(f"{'='*60}")

# Check: DSI group should have higher BAI (worse cognition)
dsi0 = df_out[df_out['W']==0]['Y']
dsi1 = df_out[df_out['W']==1]['Y']
from scipy import stats
t_stat, p_val = stats.ttest_ind(dsi1, dsi0)
print(f"  BAI: DSI=0 mean={dsi0.mean():.1f}, DSI=1 mean={dsi1.mean():.1f}")
print(f"  Difference: {dsi1.mean()-dsi0.mean():.1f}, t={t_stat:.2f}, p={p_val:.4f}")

# Check: DSI prevalence by age tertile
df_out['age_tertile'] = pd.qcut(df_out['age'], 3, labels=['Young','Mid','Old'])
print(f"\n  DSI prevalence by age:")
for tier in ['Young','Mid','Old']:
    sub = df_out[df_out['age_tertile']==tier]
    print(f"    {tier:<6s}: DSI={sub['W'].mean()*100:.2f}%, N={len(sub):,}")

print("\n=== Data Preparation Complete ===")
