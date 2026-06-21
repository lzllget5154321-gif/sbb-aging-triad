#!/usr/bin/env python
# =============================================================================
# generate_demo_data.py — Generate simulated CHARLS-format demo dataset
# =============================================================================
# Purpose: Create a synthetic dataset (N=200) with distributions matching
#          the real CHARLS baseline data for third-party code verification.
#          Contains NO real participant information.
# Usage:   python generate_demo_data.py
# Output:  demo_data/charls_demo_full.csv
# =============================================================================

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 200

# ============================================================
# 1. Demographics
# ============================================================
# Age: mean=63.7, std=8.5, range [50, 99]
age = np.clip(np.random.normal(63.7, 8.5, N), 50, 95).astype(float)
# Sex: 1=male, 2=female (~50% female)
sex = np.random.choice([1, 2], N, p=[0.48, 0.52])
female = (sex == 2).astype(int)

# Education years: mean ~6 with right skew
edu_raw = np.random.lognormal(1.6, 0.6, N).astype(float)
edu_raw = np.clip(edu_raw, 0, 22)
education_years = np.round(edu_raw).astype(float)

# ============================================================
# 2. Sensory variables
# ============================================================
# vision_impairment: ~16% prevalence
vision_impairment = np.random.binomial(1, 0.16, N).astype(float)
# hearing_impairment: ~25% prevalence
hearing_impairment = np.random.binomial(1, 0.25, N).astype(float)
# vision_imp: alised
vision_imp = vision_impairment.copy()
hearing_imp = hearing_impairment.copy()

# DSI: ~3% prevalence (vision_imp AND hearing_imp)
# Make it somewhat correlated but not fully
dsi_probs = np.where(
    (vision_impairment == 1) & (hearing_impairment == 1),
    0.65,  # 65% of dual-impaired have DSI
    0.005  # 0.5% baseline
)
DSI = np.random.binomial(1, np.clip(dsi_probs, 0, 1)).astype(int)

# ============================================================
# 3. Body variables (for BoAI construction)
# ============================================================
# BMI: mean=23.7, std=3.5 (trimmed — real data has extreme outliers)
bmi = np.clip(np.random.normal(23.7, 3.5, N), 15, 40)

# ADL sum: mean ~0.07, mostly 0
adl_sum = np.random.binomial(1, 0.07, N).astype(float)
adl_sum = np.where(np.random.random(N) < 0.02, 2, adl_sum)  # few with 2

# IADL sum (if present)
iadl_sum = np.random.binomial(5, 0.04, N).astype(float)

# Chronic count: mean ~1.5
chronic_count = np.clip(np.random.poisson(1.5, N), 0, 8).astype(float)

# Self-rated health: ~31% poor/fair
sr_health = np.random.binomial(1, 0.31, N).astype(float)

# ============================================================
# 4. Cognitive & mental health
# ============================================================
# global_cognition: mean=14.3, std=4.7, range [0, 28]
global_cognition = np.clip(
    np.random.normal(14.3, 4.7, N), 0, 28
).astype(float)

# CES-D: mean=8.3, std=6.2, range [0, 30]
cesd = np.clip(
    np.random.normal(8.3, 6.2, N), 0, 30
).astype(float)
# depression_flag: CES-D >= 10 (38% prevalence)
depression_flag = (cesd >= 10).astype(float)

# ============================================================
# 5. Construct derived indices (SAI/BAI/BoAI)
# ============================================================
# SAI: based on vision_imp + hearing_imp (0-100)
vision_imp_val = np.where(vision_impairment > 0.5,
    np.random.uniform(1, 5, N),
    np.random.uniform(0, 1, N))
hearing_imp_val = np.where(hearing_impairment > 0.5,
    np.random.uniform(1, 5, N),
    np.random.uniform(0, 1, N))
sai_raw = (vision_imp_val + hearing_imp_val) / 2
SAI_raw = sai_raw
sai_z = (SAI_raw - np.mean(SAI_raw)) / np.std(SAI_raw)
SAI = (sai_z - sai_z.min()) / (sai_z.max() - sai_z.min()) * 100

# BAI: based on global_cognition (scale 0-100, higher=better)
cog_z = (global_cognition - np.mean(global_cognition)) / np.std(global_cognition)
BAI_raw = cog_z
BAI = (cog_z - cog_z.min()) / (cog_z.max() - cog_z.min()) * 100

# BoAI: based on adl_sum, iadl_sum, chronic_count, sr_health
adl_z = (adl_sum - np.mean(adl_sum)) / np.std(adl_sum)
iadl_z = (iadl_sum - np.mean(iadl_sum)) / np.std(iadl_sum)
chronic_z = (chronic_count - np.mean(chronic_count)) / np.std(chronic_count)
sr_z = (sr_health - np.mean(sr_health)) / np.std(sr_health)
boai_components = np.column_stack([adl_z, iadl_z, chronic_z, sr_z])
BoAI_raw = np.mean(boai_components, axis=1)
BoAI = (BoAI_raw - BoAI_raw.min()) / (BoAI_raw.max() - BoAI_raw.min()) * 100

# ============================================================
# 6. Other health variables
# ============================================================
# Heart disease: ~19%
heart_disease = np.random.binomial(1, 0.19, N).astype(float)
# Stroke: ~6%
stroke = np.random.binomial(1, 0.06, N).astype(float)
# Kidney disease: ~9%
kidneye = np.random.binomial(1, 0.09, N).astype(float)
# Liver disease: ~6%
livere = np.random.binomial(1, 0.06, N).astype(float)
# Smoking (ever): ~43%
smoke_ever = np.random.binomial(1, 0.43, N).astype(float)
# Smoking (current): ~24%
smoke_now = np.random.binomial(1, 0.24, N).astype(float)
# Drinking (ever): ~44%
drink_ever = np.random.binomial(1, 0.44, N).astype(float)

# Physical activity: 1-3 scale
physical_activity = np.random.choice([1, 2, 3], N, p=[0.30, 0.35, 0.35]).astype(float)

# Grip strength: mean=26, std=10
grip_strength = np.clip(np.random.normal(26, 10, N), 5, 60)
grip1 = grip_strength * np.random.uniform(0.9, 1.1, N)
grip2 = grip_strength * np.random.uniform(0.9, 1.1, N)

# Walk speed: mean ~4.0, std ~2.0
walk_speed1 = np.clip(np.random.normal(4.0, 2.0, N), 0.5, 12)
walk_speed2 = np.clip(np.random.normal(3.9, 1.8, N), 0.5, 12)

# Height: mean=1.56, std=0.10
height = np.clip(np.random.normal(1.56, 0.10, N), 1.35, 1.90)
# Weight: mean=56.6, std=11.5
weight = np.clip(np.random.normal(56.6, 11.5, N), 35, 100)

# Sleep hours: 1-4 coded (actual hours ~6-8)
sleep_hours = np.random.choice([1, 2, 3, 4], N, p=[0.05, 0.15, 0.35, 0.45]).astype(float)
sleep_night = (sleep_hours * 2 + 2).astype(float)  # approximate actual hours

# Balance impairment: ~15%
balance_imp = np.random.binomial(1, 0.15, N).astype(float)

# ============================================================
# 7. Longitudinal structure (3 waves per participant)
# ============================================================
# Make it longitudinal: 3 waves × 200 participants = 600 rows
# But for demo simplicity, just 1 baseline wave per participant
wave = np.full(N, 2011.0)
id_vals = np.arange(1, N + 1)

# ============================================================
# 8. Assemble DataFrame
# ============================================================
data = {
    'id': id_vals,
    'wave': wave,
    'age': age,
    'sex': sex,
    'female': female,
    'education_years': education_years,
    'education': education_years,  # alias
    'vision_impairment': vision_impairment,
    'hearing_impairment': hearing_impairment,
    'vision_imp': vision_imp,
    'hearing_imp': hearing_imp,
    'DSI': DSI,
    'SAI': SAI,
    'BAI': BAI,
    'BoAI': BoAI,
    'global_cognition': global_cognition,
    'cesd': cesd,
    'depression_flag': depression_flag,
    'bmi': bmi,
    'adl_sum': adl_sum,
    'iadl_sum': iadl_sum,
    'chronic_count': chronic_count,
    'sr_health': sr_health,
    'heart_disease': heart_disease,
    'stroke': stroke,
    'kidneye': kidneye,
    'livere': livere,
    'smoke_ever': smoke_ever,
    'smoke_now': smoke_now,
    'drink_ever': drink_ever,
    'physical_activity': physical_activity,
    'grip_strength': grip_strength,
    'grip1': grip1,
    'grip2': grip2,
    'walk_speed1': walk_speed1,
    'walk_speed2': walk_speed2,
    'height': height,
    'weight': weight,
    'sleep_hours': sleep_hours,
    'sleep_night': sleep_night,
    'balance_imp': balance_imp,
    'walk_report': np.random.binomial(1, 0.12, N).astype(float),
    'walk_complete': np.random.binomial(1, 0.70, N).astype(float),
    # Placeholder columns (not used in demo analysis but present in real data)
    'sarcopenia': np.full(N, np.nan),
    'low_strength': np.full(N, np.nan),
    'low_mass': np.full(N, np.nan),
    'short_sleep': np.full(N, np.nan),
    'sleep_lunch': np.full(N, np.nan),
    'drinkl': np.full(N, np.nan),
    'drinkn_c': np.full(N, np.nan),
    'smokef': np.full(N, np.nan),
    'tyg_bmi': np.full(N, np.nan),
    'heartf': np.full(N, np.nan),
    'liverf': np.full(N, np.nan),
    'education3': np.full(N, np.nan),
}

df = pd.DataFrame(data)

# ============================================================
# 9. Save
# ============================================================
outdir = os.path.dirname(os.path.abspath(__file__))
outpath = os.path.join(outdir, 'charls_demo_full.csv')
df.to_csv(outpath, index=False)

print(f'[OK] Demo data generated: {outpath}')
print(f'   Rows: {len(df)}, Columns: {len(df.columns)}')
print(f'   DSI prevalence: {df["DSI"].mean()*100:.1f}%')
print(f'   Age: {df["age"].mean():.1f} +/- {df["age"].std():.1f}')
print(f'   Female: {df["female"].mean()*100:.1f}%')
print(f'   CES-D: {df["cesd"].mean():.1f} +/- {df["cesd"].std():.1f}')
print(f'   SAI: {df["SAI"].mean():.1f} +/- {df["SAI"].std():.1f}')
print(f'   BAI: {df["BAI"].mean():.1f} +/- {df["BAI"].std():.1f}')
print(f'   BoAI: {df["BoAI"].mean():.1f} +/- {df["BoAI"].std():.1f}')
print(f'   File size: {os.path.getsize(outpath):,} bytes')
print()
print('WARNING: THIS IS SIMULATED DATA -- NO REAL PARTICIPANT INFORMATION.')
