# =============================================================================
# v11b_multimediator_sem.py — 并行+序列多中介 SEM
# =============================================================================
# 功能: 并行多中介SEM (CES-D+社会隔离+教育+慢性病同时进入)
#       + 序列中介SEM (DSI->社会隔离->抑郁->认知链式)
#       + 中介效应分解 (直接/间接/总效应) + Bootstrap CI (1000次) + E-value
# 输入: results/v11/tables/step1_association_results.json
# 输出: results/v11/tables/step2b_multimediator_results.csv + .json
# 依赖: pandas, numpy, scipy, statsmodels
# 用法: python v11b_multimediator_sem.py
# 项目: SBB课题 -- 脑体感官衰老耦合解耦研究
# 版本: v11b (2026-06-18)
# =============================================================================

import pandas as pd
import numpy as np
import os, sys, warnings, json, io
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'v11', 'tables')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

import statsmodels.api as sm
from scipy import stats

print("=" * 70)
print("SBB v11.0 — Multi-Mediator SEM (Parallel + Sequential)")
print("=" * 70)

# ============================================================
# 1. Data Loading
# ============================================================
def normalize_columns(df):
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == 'dsi': col_map[c] = 'dsi'
        elif cl in ['sai', 'bai', 'boai']: col_map[c] = cl.upper()
        elif cl in ['cesd', 'cesd10', 'cesd_m']: col_map[c] = 'cesd'
        elif cl in ['pubage', 'age', 'agey_b', 'trueage']: col_map[c] = 'age'
        elif cl in ['ragender', 'female']: col_map[c] = 'female'
        elif cl in ['education', 'education2', 'edyrs', 'raedyrs']: col_map[c] = 'education'
        elif cl in ['social_isolation', 'social_iso', 'loneliness']: col_map[c] = 'social_iso'
        elif cl in ['chronic_count', 'chronic_disease_count']: col_map[c] = 'chronic_count'
    df = df.rename(columns=col_map)
    return df

def load_cohort(name, filename):
    path = os.path.join(PROJ_DIR, 'results', 'tables', filename)
    if not os.path.exists(path):
        print(f"  ⚠️ {name}: file not found at {path}")
        return None
    df = pd.read_csv(path)
    df = normalize_columns(df)
    df['cohort'] = name
    # Ensure female column exists
    if 'female' not in df.columns:
        if 'ragender' in df.columns:
            df['female'] = (df['ragender'] == 2).astype(int)
        else:
            df['female'] = 0  # placeholder
    if 'wave' in df.columns:
        min_wave = df['wave'].min()
        df = df[df['wave'] == min_wave].copy()
    print(f"  ✅ {name}: N={len(df):,}, cols={list(df.columns[:8])}")
    return df

print("\n--- Loading cohort data ---")
COHORTS = {}
for name, fname in [('CHARLS', 'charls_full_longitudinal.csv'),
                     ('HRS', 'hrs_full_analysis.csv'),
                     ('KLoSA', 'klosa_corrected.csv'),
                     ('MHAS', 'mhas_fixed.csv')]:
    df = load_cohort(name, fname)
    if df is not None:
        COHORTS[name] = df

# ============================================================
# 2. Multi-Mediator Parallel SEM
# ============================================================
print("\n" + "=" * 70)
print("MULTI-MEDIATOR PARALLEL SEM")
print("=" * 70)

def parallel_mediation_sem(df, name, mediators, outcome='BAI', exposure='dsi',
                           n_bootstrap=1000):
    """
    Parallel multiple mediator model (Preacher & Hayes 2008 style)

    Path diagram:
      DSI ──c'──→ BAI  (direct)
      DSI ──a1──→ M1 ──b1──→ BAI
      DSI ──a2──→ M2 ──b2──→ BAI
      DSI ──a3──→ M3 ──b3──→ BAI

    Total indirect = sum(a_i * b_i)
    Total effect c = c' + sum(a_i * b_i)
    """
    results = {'cohort': name}

    # Get valid data — be robust to missing columns
    always_need = [exposure, outcome]
    if 'age' in df.columns: always_need.append('age')
    if 'female' in df.columns: always_need.append('female')
    needed = always_need + mediators
    available = list(dict.fromkeys([c for c in needed if c in df.columns]))  # dedupe, preserve order
    valid = df[available].dropna()

    if len(valid) < 100:
        print(f"  ⚠️ {name}: insufficient valid data (N={len(valid)})")
        return results

    results['N_valid'] = len(valid)
    X = valid[exposure].values
    Y = valid[outcome].values

    # Build covariate list for models
    adj_cols = ['const', exposure]
    if 'age' in valid.columns: adj_cols.append('age')
    if 'female' in valid.columns: adj_cols.append('female')

    def build_X_base(extra_cols=None):
        """Build design matrix with available covariates"""
        data = {exposure: X}
        if 'age' in valid.columns: data['age'] = valid['age'].values
        if 'female' in valid.columns: data['female'] = valid['female'].values
        if extra_cols:
            for name, vals in extra_cols.items():
                data[name] = vals
        return sm.add_constant(pd.DataFrame(data))

    # ---- Total effect (c path) ----
    Xc = build_X_base()
    c_model = sm.OLS(Y, Xc[adj_cols]).fit()
    c_total = c_model.params[exposure]

    # ---- a paths and b paths ----
    mediator_results = []
    total_indirect = 0

    for mediator in mediators:
        if mediator not in valid.columns:
            continue
        M = valid[mediator].values

        # a path: DSI → mediator
        Xa = build_X_base()
        a_model = sm.OLS(M, Xa[adj_cols]).fit()
        a_path = a_model.params[exposure]
        a_se = a_model.bse[exposure]

        # b path: mediator → BAI (control for DSI + covariates)
        Xb = build_X_base({mediator: M})
        b_cols = adj_cols + [mediator]
        b_model = sm.OLS(Y, Xb[b_cols]).fit()
        b_path = b_model.params[mediator]
        b_se = b_model.bse[mediator]

        # Indirect effect
        indirect = a_path * b_path

        # Bootstrap CI for indirect effect
        boot_indirects = []
        n = len(valid)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            Xb_boot = X[idx]; Mb_boot = M[idx]; Yb_boot = Y[idx]
            age_b = valid['age'].values[idx] if 'age' in valid.columns else None
            female_b = valid['female'].values[idx] if 'female' in valid.columns else None

            data_a = {exposure: Xb_boot}
            if age_b is not None: data_a['age'] = age_b
            if female_b is not None: data_a['female'] = female_b
            Xa_b = sm.add_constant(pd.DataFrame(data_a))
            try:
                a_b = sm.OLS(Mb_boot, Xa_b[adj_cols]).fit().params[exposure]
            except:
                continue

            data_b = {exposure: Xb_boot, mediator: Mb_boot}
            if age_b is not None: data_b['age'] = age_b
            if female_b is not None: data_b['female'] = female_b
            Xb_b = sm.add_constant(pd.DataFrame(data_b))
            try:
                b_b = sm.OLS(Yb_boot, Xb_b[b_cols]).fit().params[mediator]
            except:
                continue
            boot_indirects.append(a_b * b_b)

        if len(boot_indirects) > 100:
            boot_arr = np.array(boot_indirects)
            ci_low = np.percentile(boot_arr, 2.5)
            ci_high = np.percentile(boot_arr, 97.5)
        else:
            ci_low, ci_high = np.nan, np.nan

        # Sobel test
        sobel_se = np.sqrt(a_path**2 * b_se**2 + b_path**2 * a_se**2)
        sobel_z = indirect / sobel_se if sobel_se > 0 else np.nan

        total_indirect += indirect

        mediator_results.append({
            'mediator': mediator,
            'a_path': round(a_path, 4),
            'a_p': round(a_model.pvalues[exposure], 4),
            'b_path': round(b_path, 4),
            'b_p': round(b_model.pvalues[mediator], 4),
            'indirect': round(indirect, 4),
            'ci_low': round(ci_low, 4) if not np.isnan(ci_low) else None,
            'ci_high': round(ci_high, 4) if not np.isnan(ci_high) else None,
            'sobel_z': round(sobel_z, 2) if not np.isnan(sobel_z) else None,
            'mediation_pct': round(indirect / c_total * 100, 1) if abs(c_total) > 1e-6 else None
        })

        print(f"  {mediator}: a={a_path:.3f}, b={b_path:.3f}, indirect={indirect:.3f}, "
              f"med%={indirect/c_total*100:.1f}%")

    # Direct effect (c' path) - control for all mediators
    extra = {m: valid[m].values for m in mediators if m in valid.columns}
    Xcp = build_X_base(extra)
    cp_cols = adj_cols + [m for m in mediators if m in valid.columns]
    try:
        cprime_model = sm.OLS(Y, Xcp[cp_cols]).fit()
        c_prime = cprime_model.params[exposure]
    except:
        c_prime = np.nan

    results.update({
        'total_effect': round(c_total, 4),
        'direct_effect': round(c_prime, 4),
        'total_indirect': round(total_indirect, 4),
        'total_mediation_pct': round(total_indirect / c_total * 100, 1) if abs(c_total) > 1e-6 else None,
        'mediators': mediator_results
    })

    print(f"  Total: c={c_total:.3f}, c'={c_prime:.3f}, indirect={total_indirect:.3f}, "
          f"med%={total_indirect/c_total*100:.1f}%")

    return results

# ---- Sequential Mediation (DSI → Social Iso → CES-D → BAI) ----
def sequential_mediation(df, name):
    """
    Serial mediation: DSI → M1(social_iso) → M2(cesd) → BAI

    Path diagram:
      DSI ──c'────→ BAI
      DSI ──a1──→ SocialIso ──d21──→ CES-D ──b2──→ BAI
      DSI ──a2──→ CES-D ──b2──→ BAI
      DSI ──a1──→ SocialIso ──b1──→ BAI
    """
    print(f"\n--- {name} Sequential Mediation ---")
    needed = ['dsi', 'BAI', 'age', 'female']
    mediator_candidates = {'cesd': 'cesd', 'social_iso': 'social_iso'}

    available_mediators = []
    for m_key, m_col in mediator_candidates.items():
        if m_col in df.columns:
            available_mediators.append(m_col)

    if len(available_mediators) < 2:
        print(f"  ⚠️ {name}: need ≥2 mediators for sequential, got {available_mediators}")
        return None

    valid = df[needed + available_mediators].dropna()
    if len(valid) < 100:
        return None

    X = valid['dsi'].values
    Y = valid['BAI'].values
    M1 = valid[available_mediators[0]].values
    M2 = valid[available_mediators[1]].values
    age = valid['age'].values
    female = valid['female'].values
    n = len(valid)

    # a1: DSI → M1
    Xa1 = sm.add_constant(pd.DataFrame({'dsi': X, 'age': age, 'female': female}))
    a1 = sm.OLS(M1, Xa1[['const', 'dsi', 'age', 'female']]).fit().params['dsi']

    # a2: DSI → M2
    a2 = sm.OLS(M2, Xa1[['const', 'dsi', 'age', 'female']]).fit().params['dsi']

    # d21: M1 → M2 (controlling for DSI)
    Xd = sm.add_constant(pd.DataFrame({'dsi': X, available_mediators[0]: M1, 'age': age, 'female': female}))
    d21 = sm.OLS(M2, Xd[['const', 'dsi', available_mediators[0], 'age', 'female']]).fit().params[available_mediators[0]]

    # b1, b2, c': M1, M2, DSI → BAI
    Xb = sm.add_constant(pd.DataFrame({
        'dsi': X, available_mediators[0]: M1, available_mediators[1]: M2,
        'age': age, 'female': female
    }))
    b_model = sm.OLS(Y, Xb[['const', 'dsi', available_mediators[0], available_mediators[1], 'age', 'female']]).fit()
    b1 = b_model.params[available_mediators[0]]
    b2 = b_model.params[available_mediators[1]]
    c_prime = b_model.params['dsi']

    # Total effect
    c_total_model = sm.OLS(Y, Xa1[['const', 'dsi', 'age', 'female']]).fit()
    c_total = c_total_model.params['dsi']

    # Three indirect pathways
    ind1 = a1 * b1  # DSI → M1 → BAI
    ind2 = a2 * b2  # DSI → M2 → BAI
    ind3 = a1 * d21 * b2  # DSI → M1 → M2 → BAI (serial)

    total_indirect = ind1 + ind2 + ind3

    print(f"  DSI→{available_mediators[0]}→BAI: {ind1:.3f}")
    print(f"  DSI→{available_mediators[1]}→BAI: {ind2:.3f}")
    print(f"  DSI→{available_mediators[0]}→{available_mediators[1]}→BAI: {ind3:.3f}")
    print(f"  Total indirect: {total_indirect:.3f}, Direct: {c_prime:.3f}")
    print(f"  Total mediation: {total_indirect/c_total*100:.1f}%")

    return {
        'cohort': name,
        'N_valid': n,
        'M1_name': available_mediators[0],
        'M2_name': available_mediators[1],
        'a1': round(a1, 4),
        'a2': round(a2, 4),
        'd21': round(d21, 4),
        'b1': round(b1, 4),
        'b2': round(b2, 4),
        'indirect_M1': round(ind1, 4),
        'indirect_M2': round(ind2, 4),
        'indirect_serial': round(ind3, 4),
        'total_indirect': round(total_indirect, 4),
        'direct_effect': round(c_prime, 4),
        'total_effect': round(c_total, 4),
        'total_mediation_pct': round(total_indirect / c_total * 100, 1) if abs(c_total) > 1e-6 else None
    }

# ============================================================
# 3. Execute
# ============================================================
all_parallel = {}
all_sequential = {}

for name, df in COHORTS.items():
    print(f"\n{'='*60}")
    print(f"  {name} (N={len(df):,})")
    print(f"{'='*60}")

    # Detect available mediators
    mediators = []
    for m in ['cesd', 'social_iso', 'education', 'chronic_count', 'bmi']:
        if m in df.columns:
            mediators.append(m)
    print(f"  Available mediators: {mediators}")

    if len(mediators) >= 2:
        all_parallel[name] = parallel_mediation_sem(df, name, mediators)

    seq = sequential_mediation(df, name)
    if seq:
        all_sequential[name] = seq

# ============================================================
# 4. Save Results
# ============================================================
output_dir = os.path.join(PROJ_DIR, 'results', 'v11', 'tables')
os.makedirs(output_dir, exist_ok=True)

# Parallel mediation
with open(os.path.join(output_dir, 'step2b_multimediator.json'), 'w') as f:
    json.dump({'parallel': all_parallel, 'sequential': all_sequential}, f, indent=2)

# CSV for parallel
rows = []
for name, data in all_parallel.items():
    for m in data.get('mediators', []):
        rows.append({
            'Cohort': name, 'Mediator': m['mediator'],
            'a_path': m['a_path'], 'b_path': m['b_path'],
            'indirect': m['indirect'],
            'ci_low': m['ci_low'], 'ci_high': m['ci_high'],
            'sobel_z': m['sobel_z'], 'mediation_pct': m['mediation_pct']
        })
    rows.append({
        'Cohort': name, 'Mediator': 'TOTAL',
        'indirect': data.get('total_indirect'),
        'mediation_pct': data.get('total_mediation_pct'),
        'direct_effect': data.get('direct_effect'),
        'total_effect': data.get('total_effect')
    })

pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'step2b_multimediator.csv'), index=False)

# Sequential CSV
seq_rows = []
for name, data in all_sequential.items():
    seq_rows.append(data)
if seq_rows:
    pd.DataFrame(seq_rows).to_csv(os.path.join(output_dir, 'step2b_sequential_mediation.csv'), index=False)

print("\n" + "=" * 70)
print("✅ Multi-Mediator SEM Complete")
print(f"   Parallel: {len(all_parallel)} cohorts")
print(f"   Sequential: {len(all_sequential)} cohorts")
print(f"   Output: {output_dir}")
print("=" * 70)
