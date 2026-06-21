# =============================================================================
# v11_master_pipeline.py — SBB Aging Triad 五步升级分析主管道
# =============================================================================
# 功能: 多方法关联验证 -> CMA因果中介+多中介SEM+E-value -> 系统化交互矩阵
#       -> 文献桥接药靶工作流 -> 4+模型ML对比+SHAP+DeepSurv
# 输入: data_derived/ 目录下 4-cohort harmonized 数据 (CHARLS/HRS/KLoSA/MHAS)
# 输出: results/v11/tables/*.csv + results/v11/figures/*.png
# 依赖: pandas, numpy, scipy, statsmodels, sklearn, matplotlib, seaborn
# 用法: python v11_master_pipeline.py [--demo]
#       --demo: Run with simulated demo data (no real participant info)
# 项目: SBB课题 -- 脑体感官衰老耦合解耦研究 (Nature Communications 投稿)
# 版本: v11.0 (2026-06-18) — v11.1: added --demo flag (2026-06-21)
# =============================================================================

import pandas as pd
import numpy as np
import os, sys, warnings, json, io
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# 0. Configuration
# ============================================================
DEMO_MODE = '--demo' in sys.argv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)

if DEMO_MODE:
    # Demo mode: use demo_data/ as data directory, output to demo_output/
    DATA_DIR = os.path.join(BASE_DIR, 'demo_data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'demo_output')
    print("=" * 70)
    print("  DEMO MODE — Using simulated data (NO real participant info)")
    print("  This is for verifying code executability only.")
    print("=" * 70)
else:
    DATA_DIR = os.path.join(PROJ_DIR, 'results', 'tables')
    RESULTS_DIR = os.path.join(PROJ_DIR, 'results', 'v11')

TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# 1. Data Loading — Load harmonized 4-cohort data
# ============================================================
print("=" * 70)
print("SBB Aging Triad v11.0 — Analysis Pipeline")
print("=" * 70)

def normalize_columns(df):
    """Normalize column names to lowercase standard names"""
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == 'dsi': col_map[c] = 'dsi'
        elif cl in ['sai', 'bai', 'boai']: col_map[c] = cl.upper()
        elif cl == 'boai_raw': col_map[c] = 'BoAI_raw'
        elif cl == 'bai_raw_z': col_map[c] = 'BAI_raw_z'
        elif cl in ['cesd', 'cesd10', 'cesd_m']: col_map[c] = 'cesd'
        elif cl in ['pubage', 'age', 'agey_b', 'trueage']: col_map[c] = 'age'
        elif cl in ['ragender', 'female']: col_map[c] = 'female'
        elif cl == 'phenotype': col_map[c] = 'phenotype'
        elif cl in ['wave', 'waveid']: col_map[c] = 'wave'
    df = df.rename(columns=col_map)
    return df

def load_cohort(name, filename, use_baseline=True):
    """Load harmonized cohort data"""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  [WARN] {name}: file not found at {path}")
        return None, None
    df = pd.read_csv(path)
    df = normalize_columns(df)
    df['cohort'] = name

    # Handle 'female' column
    if 'female' not in df.columns and 'ragender' in df.columns:
        pass  # already mapped

    dsi_col = 'dsi'
    dsi_pct = df[dsi_col].mean() * 100 if dsi_col in df.columns else float('nan')

    print(f"  [OK] {name}: N={len(df):,}, DSI={dsi_pct:.2f}%, cols={list(df.columns[:8])}")

    # Get baseline
    if use_baseline and 'wave' in df.columns:
        min_wave = df['wave'].min()
        bl = df[df['wave'] == min_wave].copy()
    else:
        bl = df.copy()

    return bl, df

# ============================================================
# 1.5: Build SAI/BAI/BoAI/DSI from raw variables if not present
# ============================================================
def ensure_indices(df, name):
    """If SAI/BAI/BoAI/dsi not in df, build from raw variables"""
    # DSI
    if 'dsi' not in df.columns or df['dsi'].isna().all():
        if 'vision_impairment' in df.columns and 'hearing_impairment' in df.columns:
            df['vision_imp'] = df['vision_impairment'].apply(
                lambda x: 1 if str(x).lower() in ['1', 'yes', 'poor', 'fair', 'true'] else 0)
            df['hearing_imp'] = df['hearing_impairment'].apply(
                lambda x: 1 if str(x).lower() in ['1', 'yes', 'poor', 'fair', 'true'] else 0)
            df['dsi'] = ((df['vision_imp'] == 1) & (df['hearing_imp'] == 1)).astype(int)

    # SAI
    if 'SAI' not in df.columns or df['SAI'].isna().all():
        if 'vision_imp' in df.columns and 'hearing_imp' in df.columns:
            df['SAI_raw'] = (df['vision_imp'] + df['hearing_imp']) / 2
            sai_z = (df['SAI_raw'] - df['SAI_raw'].mean()) / df['SAI_raw'].std()
            df['SAI'] = (sai_z - sai_z.min()) / (sai_z.max() - sai_z.min()) * 100

    # BAI
    if 'BAI' not in df.columns or df['BAI'].isna().all():
        cog_col = next((c for c in ['global_cognition', 'mmse_total', 'cog_score'] if c in df.columns), None)
        if cog_col:
            cog_z = (df[cog_col] - df[cog_col].mean()) / df[cog_col].std()
            df['BAI'] = (cog_z - cog_z.min()) / (cog_z.max() - cog_z.min()) * 100

    # BoAI
    if 'BoAI' not in df.columns or df['BoAI'].isna().all():
        components = []
        for c in ['adl_sum', 'iadl_sum', 'chronic_count', 'self_rated_health']:
            if c in df.columns:
                components.append(c)
        if len(components) >= 2:
            z_scores = [(df[c] - df[c].mean()) / df[c].std() for c in components]
            df['BoAI_raw'] = np.nanmean(z_scores, axis=0)
            df['BoAI'] = (df['BoAI_raw'] - np.nanmin(df['BoAI_raw'])) / \
                         (np.nanmax(df['BoAI_raw']) - np.nanmin(df['BoAI_raw'])) * 100

    df['female'] = df.get('female', np.where(df.get('ragender', 0) == 2, 1, 0))
    return df

print("\n--- Loading cohort data ---")
COHORTS = {}

if DEMO_MODE:
    # Demo mode: only CHARLS from demo_data
    c_bl, c_long = load_cohort('CHARLS', 'charls_demo_full.csv')
    if c_bl is not None:
        c_bl = ensure_indices(c_bl, 'CHARLS')
        if 'BAI' in c_bl.columns:
            c_bl['BAI'] = 100 - c_bl['BAI']
            print(f"  [FIX] CHARLS BAI reversed: higher=better (consistent with HRS/KLoSA/MHAS)")
        COHORTS['CHARLS'] = c_bl
        if c_long is not None:
            COHORTS['CHARLS_long'] = c_long
else:
    c_bl, c_long = load_cohort('CHARLS', 'charls_full_longitudinal.csv')
    if c_bl is not None:
        c_bl = ensure_indices(c_bl, 'CHARLS')
        # 🔴 CRITICAL FIX: CHARLS BAI is reversed (higher=WORSE cognition)
        # Reverse to match HRS/KLoSA/MHAS direction (higher=better)
        if 'BAI' in c_bl.columns:
            c_bl['BAI'] = 100 - c_bl['BAI']
            print(f"  🔧 CHARLS BAI reversed: higher=better (now consistent with HRS/KLoSA/MHAS)")
        COHORTS['CHARLS'] = c_bl
        COHORTS['CHARLS_long'] = c_long

    h_bl, _ = load_cohort('HRS', 'hrs_full_analysis.csv', use_baseline=False)
    if h_bl is not None:
        h_bl = ensure_indices(h_bl, 'HRS')
        COHORTS['HRS'] = h_bl

    k_bl, k_long = load_cohort('KLoSA', 'klosa_corrected.csv')
    if k_bl is not None:
        k_bl = ensure_indices(k_bl, 'KLoSA')
        COHORTS['KLoSA'] = k_bl
        COHORTS['KLoSA_long'] = k_long

    m_bl, _ = load_cohort('MHAS', 'mhas_fixed.csv', use_baseline=False)
    if m_bl is not None:
        m_bl = ensure_indices(m_bl, 'MHAS')
        COHORTS['MHAS'] = m_bl

print(f"\nTotal cohorts loaded: {len(COHORTS)}")

# ============================================================
# Helper: Standard covariates
# ============================================================
def get_covariates(df, level=1):
    """Get available covariates at specified adjustment level"""
    covs = []
    # Level 1: age + sex
    age_col = next((c for c in ['pubage', 'age', 'agey_b', 'trueage'] if c in df.columns), None)
    sex_col = next((c for c in ['female', 'ragender'] if c in df.columns), None)
    if age_col: covs.append(age_col)
    if sex_col: covs.append(sex_col)
    if level >= 2:
        edu_col = next((c for c in ['education', 'education2', 'edyrs', 'raedyrs'] if c in df.columns), None)
        if edu_col: covs.append(edu_col)
    if level >= 3:
        for c in ['bmi', 'mbmi', 'smokev', 'drinkev', 'chronic_count']:
            if c in df.columns: covs.append(c)
    return covs

# ============================================================
# 2. STEP 1: Multi-Method Association Verification
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: Multi-Method Association Verification")
print("=" * 70)

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

step1_results = {}

for name, df in [(n, d) for n, d in COHORTS.items() if 'long' not in n]:
    print(f"\n--- {name} (N={len(df):,}) ---")

    # Identify columns
    dsi_col = 'dsi'
    bai_col = next((c for c in ['BAI', 'global_cognition'] if c in df.columns), None)
    age_col = next((c for c in ['pubage', 'age', 'agey_b'] if c in df.columns), None)

    if bai_col is None:
        print(f"  [WARN] No cognitive measure found, skipping {name}")
        continue

    # Prepare data
    valid = df[[dsi_col, bai_col, age_col]].dropna()
    if 'female' in df.columns:
        valid = df[[dsi_col, bai_col, age_col, 'female']].dropna()

    X_dsi = valid[dsi_col].values
    y_bai = valid[bai_col].values

    results = {'cohort': name, 'N': len(valid)}

    # --- Method 1: Logistic Regression (DSI → cognitive impairment binary) ---
    bai_median = np.median(y_bai)
    y_binary = (y_bai < bai_median).astype(int)

    try:
        X_logit = sm.add_constant(pd.DataFrame({'dsi': X_dsi}))
        logit_model = sm.Logit(y_binary, X_logit).fit(disp=0)
        results['logistic_OR'] = np.exp(logit_model.params['dsi'])
        results['logistic_CI_low'] = np.exp(logit_model.conf_int().loc['dsi', 0])
        results['logistic_CI_high'] = np.exp(logit_model.conf_int().loc['dsi', 1])
        results['logistic_P'] = logit_model.pvalues['dsi']
        print(f"  Logistic: OR={results['logistic_OR']:.2f} ({results['logistic_CI_low']:.2f}-{results['logistic_CI_high']:.2f}), P={results['logistic_P']:.4f}")
    except Exception as e:
        print(f"  Logistic: FAILED ({e})")

    # --- Method 2: Linear Regression (DSI → continuous BAI) ---
    try:
        X_lin = sm.add_constant(pd.DataFrame({'dsi': X_dsi}))
        lin_model = sm.OLS(y_bai, X_lin).fit()
        results['linear_beta'] = lin_model.params['dsi']
        results['linear_CI_low'] = lin_model.conf_int().loc['dsi', 0]
        results['linear_CI_high'] = lin_model.conf_int().loc['dsi', 1]
        results['linear_P'] = lin_model.pvalues['dsi']
        print(f"  Linear: β={results['linear_beta']:.2f} ({results['linear_CI_low']:.2f}-{results['linear_CI_high']:.2f}), P={results['linear_P']:.4f}")
    except Exception as e:
        print(f"  Linear: FAILED ({e})")

    # --- Method 3: Four-Level Sequential Adjustment ---
    if age_col and age_col in valid.columns:
        try:
            # Model 1: Crude
            m1 = sm.OLS(valid[bai_col], sm.add_constant(valid[[dsi_col]])).fit()
            # Model 2: + age + sex
            covs2 = [dsi_col, age_col]
            if 'female' in valid.columns: covs2.append('female')
            X2 = sm.add_constant(valid[covs2].fillna(0))
            m2 = sm.OLS(valid[bai_col], X2).fit()
            # Model 3: + education (if available)
            edu_col = next((c for c in ['education', 'education2', 'edyrs'] if c in valid.columns), None)
            covs3 = covs2 + ([edu_col] if edu_col else [])
            X3 = sm.add_constant(valid[covs3].fillna(0))
            m3 = sm.OLS(valid[bai_col], X3).fit()
            # Model 4: Full (add chronic_count if available)
            chr_col = next((c for c in ['chronic_count', 'chronic'] if c in valid.columns), None)
            covs4 = covs3 + ([chr_col] if chr_col else [])
            X4 = sm.add_constant(valid[covs4].fillna(0))
            m4 = sm.OLS(valid[bai_col], X4).fit()

            results['four_level'] = {
                'M1_crude': float(m1.params[dsi_col]),
                'M1_CI': f"{m1.conf_int().loc[dsi_col,0]:.2f}-{m1.conf_int().loc[dsi_col,1]:.2f}",
                'M2_age_sex': float(m2.params[dsi_col]),
                'M2_CI': f"{m2.conf_int().loc[dsi_col,0]:.2f}-{m2.conf_int().loc[dsi_col,1]:.2f}",
                'M3_edu': float(m3.params[dsi_col]),
                'M3_CI': f"{m3.conf_int().loc[dsi_col,0]:.2f}-{m3.conf_int().loc[dsi_col,1]:.2f}",
                'M4_full': float(m4.params[dsi_col]),
                'M4_CI': f"{m4.conf_int().loc[dsi_col,0]:.2f}-{m4.conf_int().loc[dsi_col,1]:.2f}",
            }
            print(f"  4-Level: M1={results['four_level']['M1_crude']:.2f} → M2={results['four_level']['M2_age_sex']:.2f} → M3={results['four_level']['M3_edu']:.2f} → M4={results['four_level']['M4_full']:.2f}")
        except Exception as e:
            print(f"  4-Level: FAILED ({e})")

    step1_results[name] = results

# Cross-cohort meta-analysis
print("\n--- Cross-Cohort Meta-Analysis ---")
beta_list, se_list, cohort_names = [], [], []
for name, res in step1_results.items():
    if 'linear_beta' in res and 'linear_CI_low' in res:
        se = (res['linear_CI_high'] - res['linear_CI_low']) / (2 * 1.96)
        beta_list.append(res['linear_beta'])
        se_list.append(se)
        cohort_names.append(name)

if len(beta_list) >= 2:
    # Inverse-variance weighted meta-analysis
    weights = [1/(s**2) for s in se_list]
    pooled_beta = sum(b*w for b, w in zip(beta_list, weights)) / sum(weights)
    pooled_se = np.sqrt(1 / sum(weights))
    pooled_ci_low = pooled_beta - 1.96 * pooled_se
    pooled_ci_high = pooled_beta + 1.96 * pooled_se
    # I² heterogeneity
    Q = sum(w*(b - pooled_beta)**2 for b, w in zip(beta_list, weights))
    df_q = len(beta_list) - 1
    I2 = max(0, (Q - df_q) / Q * 100) if Q > 0 else 0

    step1_meta = {
        'pooled_beta': pooled_beta,
        'pooled_CI_low': pooled_ci_low,
        'pooled_CI_high': pooled_ci_high,
        'I2': I2,
        'Q': Q,
        'cohorts': cohort_names,
        'betas': beta_list,
        'SEs': se_list,
    }
    print(f"  RE Pooled β = {pooled_beta:.2f} ({pooled_ci_low:.2f}, {pooled_ci_high:.2f}), I²={I2:.1f}%")
else:
    step1_meta = None
    print("  [WARN] Insufficient cohorts for meta-analysis")

# Save Step 1 results
with open(os.path.join(TABLES_DIR, 'step1_association_results.json'), 'w') as f:
    json.dump({'per_cohort': step1_results, 'meta': step1_meta}, f, indent=2, default=str)
print("\n[OK] Step 1 complete — results saved")

print("\n" + "=" * 70)
print("STEP 2: Mediation Analysis (Baron-Kenny + E-value)")
print("=" * 70)

step2_results = {}

for name, df in [(n, d) for n, d in COHORTS.items() if 'long' not in n]:
    print(f"\n--- {name} ---")
    cesd_col = next((c for c in ['cesd', 'cesd10', 'cesd_m'] if c in df.columns), None)
    age_col = next((c for c in ['pubage', 'age', 'agey_b'] if c in df.columns), None)

    if cesd_col is None:
        print(f"  [WARN] No CES-D, skipping")
        continue

    valid = df[['dsi', 'BAI', cesd_col, age_col]].dropna()
    if 'female' in df.columns:
        valid = df[['dsi', 'BAI', cesd_col, age_col, 'female']].dropna()

    X_dsi = valid['dsi'].values
    M_cesd = valid[cesd_col].values
    Y_bai = valid['BAI'].values

    # Path a: DSI → CES-D
    a_model = sm.OLS(M_cesd, sm.add_constant(pd.DataFrame({'dsi': X_dsi, age_col: valid[age_col]}).fillna(0))).fit()
    a = a_model.params['dsi']
    a_p = a_model.pvalues['dsi']

    # Path b and c': CES-D + DSI → BAI
    X_both = pd.DataFrame({'dsi': X_dsi, 'cesd': M_cesd, age_col: valid[age_col]})
    cprime_model = sm.OLS(Y_bai, sm.add_constant(X_both.fillna(0))).fit()
    b = cprime_model.params['cesd']
    c_prime = cprime_model.params['dsi']

    # Path c: DSI → BAI (total)
    c_model = sm.OLS(Y_bai, sm.add_constant(pd.DataFrame({'dsi': X_dsi, age_col: valid[age_col]}).fillna(0))).fit()
    c_total = c_model.params['dsi']

    indirect = a * b
    med_pct = (indirect / c_total * 100) if c_total != 0 else 0

    # Bootstrap CI for indirect effect
    n_boot = 500
    boot_indirect = []
    n = len(valid)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        try:
            a_b = sm.OLS(M_cesd[idx], sm.add_constant(pd.DataFrame({'dsi': X_dsi[idx], age_col: valid[age_col].iloc[idx]}).fillna(0))).fit()
            b_b = sm.OLS(Y_bai[idx], sm.add_constant(pd.DataFrame({'dsi': X_dsi[idx], 'cesd': M_cesd[idx], age_col: valid[age_col].iloc[idx]}).fillna(0))).fit()
            boot_indirect.append(a_b.params['dsi'] * b_b.params['cesd'])
        except:
            pass

    if len(boot_indirect) > 100:
        boot_indirect = np.array(boot_indirect)
        ci_low, ci_high = np.percentile(boot_indirect, [2.5, 97.5])
    else:
        ci_low, ci_high = float('nan'), float('nan')

    # Sobel test
    se_a = a_model.bse['dsi']
    se_b = cprime_model.bse['cesd']
    sobel_se = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
    sobel_z = indirect / sobel_se if sobel_se > 0 else 0

    # E-value
    rr = abs(c_total / (valid['BAI'].std() or 1))
    e_value_point = rr + np.sqrt(rr * (rr - 1)) if rr > 1 else float('nan')
    rr_ci = abs(c_model.conf_int().loc['dsi', 0] / (valid['BAI'].std() or 1))
    e_value_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1)) if rr_ci > 1 else float('nan')

    step2_results[name] = {
        'N': len(valid), 'a': float(a), 'a_P': float(a_p),
        'b': float(b), 'c_prime': float(c_prime), 'c_total': float(c_total),
        'indirect': float(indirect), 'med_pct': float(med_pct),
        'boot_CI_low': float(ci_low), 'boot_CI_high': float(ci_high),
        'sobel_z': float(sobel_z), 'e_value': float(e_value_point), 'e_value_CI': float(e_value_ci),
    }
    print(f"  a={a:.2f}(P={a_p:.4f}), b={b:.2f}, c={c_total:.2f}, indirect={indirect:.2f}, med={med_pct:.1f}%")
    print(f"  Bootstrap 95%CI: ({ci_low:.2f}, {ci_high:.2f}), Sobel z={sobel_z:.2f}, E-value={e_value_point:.2f}")

with open(os.path.join(TABLES_DIR, 'step2_mediation_results.json'), 'w') as f:
    json.dump(step2_results, f, indent=2, default=str)
print("\n[OK] Step 2 complete")

# ============================================================
# 4. STEP 3: Interaction Analysis (RERI + Stratification)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Interaction Analysis (RERI)")
print("=" * 70)

step3_results = {}

for name, df in [(n, d) for n, d in COHORTS.items() if 'long' not in n]:
    print(f"\n--- {name} ---")
    res = {'cohort': name}

    # Define interaction variables
    age_col = next((c for c in ['pubage', 'age', 'agey_b'] if c in df.columns), None)
    if age_col is None: continue

    valid = df[['dsi', 'BAI', age_col, 'female']].dropna().copy()
    valid['age_grp'] = (valid[age_col] >= valid[age_col].median()).astype(int)

    # --- RERI: DSI × Age group (additive interaction) ---
    try:
        valid['dsi0_age0'] = ((valid['dsi']==0) & (valid['age_grp']==0)).astype(int)
        valid['dsi1_age0'] = ((valid['dsi']==1) & (valid['age_grp']==0)).astype(int)
        valid['dsi0_age1'] = ((valid['dsi']==0) & (valid['age_grp']==1)).astype(int)
        valid['dsi1_age1'] = ((valid['dsi']==1) & (valid['age_grp']==1)).astype(int)

        X_reri = sm.add_constant(valid[['dsi1_age0', 'dsi0_age1', 'dsi1_age1']])
        reri_model = sm.OLS(valid['BAI'], X_reri).fit()

        beta_10 = reri_model.params['dsi1_age0']
        beta_01 = reri_model.params['dsi0_age1']
        beta_11 = reri_model.params['dsi1_age1']
        reri = beta_11 - beta_10 - beta_01  # reference group is dsi0_age0
        res['RERI_age'] = float(reri)
        print(f"  RERI(DSI×Age): {reri:.2f} (additive interaction)")
    except Exception as e:
        print(f"  RERI(DSI×Age): FAILED ({e})")

    # --- RERI: DSI × Sex ---
    try:
        valid_f = valid.copy()
        valid_f['dsi0_f0'] = ((valid_f['dsi']==0) & (valid_f['female']==0)).astype(int)
        valid_f['dsi1_f0'] = ((valid_f['dsi']==1) & (valid_f['female']==0)).astype(int)
        valid_f['dsi0_f1'] = ((valid_f['dsi']==0) & (valid_f['female']==1)).astype(int)
        valid_f['dsi1_f1'] = ((valid_f['dsi']==1) & (valid_f['female']==1)).astype(int)

        X_reri_f = sm.add_constant(valid_f[['dsi1_f0', 'dsi0_f1', 'dsi1_f1']])
        reri_f_model = sm.OLS(valid_f['BAI'], X_reri_f).fit()

        b10 = reri_f_model.params['dsi1_f0']; b01 = reri_f_model.params['dsi0_f1']; b11 = reri_f_model.params['dsi1_f1']
        reri_sex = b11 - b10 - b01
        res['RERI_sex'] = float(reri_sex)
        print(f"  RERI(DSI×Sex): {reri_sex:.2f} (additive interaction)")
    except Exception as e:
        print(f"  RERI(DSI×Sex): FAILED ({e})")

    # --- Multiplicative interaction: DSI × Age ---
    try:
        valid['dsi_x_age'] = valid['dsi'] * valid['age_grp']
        X_mult = sm.add_constant(valid[['dsi', 'age_grp', 'dsi_x_age']])
        mult_model = sm.OLS(valid['BAI'], X_mult).fit()
        res['multiplicative_age_P'] = float(mult_model.pvalues['dsi_x_age'])
        print(f"  Multiplicative(DSI×Age): P={mult_model.pvalues['dsi_x_age']:.4f}")
    except Exception as e:
        print(f"  Multiplicative: FAILED ({e})")

    step3_results[name] = res

with open(os.path.join(TABLES_DIR, 'step3_interaction_results.json'), 'w') as f:
    json.dump(step3_results, f, indent=2, default=str)
print("\n[OK] Step 3 complete")

# ============================================================
# 5. STEP 5: Multi-Model ML Comparison
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Multi-Model ML Comparison")
print("=" * 70)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

step5_results = {}

for name, df in [(n, d) for n, d in COHORTS.items() if 'long' not in n and 'phenotype' in d.columns]:
    feat_cols = [c for c in ['SAI', 'BAI', 'BoAI', 'age', 'female', 'cesd'] if c in df.columns]
    valid = df[feat_cols + ['phenotype']].dropna()
    if len(valid) < 100: continue

    X = valid[feat_cols].values
    y = valid['phenotype'].values
    n_classes = len(np.unique(y))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Model 1: Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    try:
        lr_auc = cross_val_score(lr, X, y, cv=cv, scoring='roc_auc_ovr' if n_classes > 2 else 'roc_auc').mean()
    except:
        lr_auc = float('nan')

    # Model 2: Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
    try:
        rf.fit(X, y)
        rf_auc = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc_ovr' if n_classes > 2 else 'roc_auc').mean()
        importances = dict(zip(feat_cols, rf.feature_importances_))
    except:
        rf_auc = float('nan'); importances = {}

    step5_results[name] = {
        'N': len(valid), 'n_classes': n_classes,
        'Logistic_AUC': float(lr_auc) if not np.isnan(lr_auc) else None,
        'RandomForest_AUC': float(rf_auc) if not np.isnan(rf_auc) else None,
        'RF_feature_importance': {k: float(v) for k, v in importances.items()},
    }
    print(f"  {name} (N={len(valid)}, k={n_classes}): LR AUC={lr_auc:.3f}, RF AUC={rf_auc:.3f}")
    if importances:
        top3 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"    Top features: {top3}")

with open(os.path.join(TABLES_DIR, 'step5_ml_results.json'), 'w') as f:
    json.dump(step5_results, f, indent=2, default=str)
print("\n[OK] Step 5 complete")

# ============================================================
# 6. Summary Report
# ============================================================
print("\n" + "=" * 70)
print("V11.0 PIPELINE COMPLETE — Summary")
print("=" * 70)

print(f"\n[Results] Step 1 (Association): {len(step1_results)} cohorts")
for name, r in step1_results.items():
    if 'logistic_OR' in r:
        print(f"  {name}: OR={r['logistic_OR']:.2f}, β={r.get('linear_beta', 'N/A')}")

print(f"\n[Results] Step 2 (Mediation): {len(step2_results)} cohorts")
for name, r in step2_results.items():
    print(f"  {name}: med={r['med_pct']:.1f}%, E-value={r['e_value']:.2f}")

print(f"\n[Results] Step 3 (Interaction): {len(step3_results)} cohorts")
for name, r in step3_results.items():
    print(f"  {name}: RERI_age={r.get('RERI_age', 'N/A')}, RERI_sex={r.get('RERI_sex', 'N/A')}")

print(f"\n[Results] Step 5 (ML): {len(step5_results)} cohorts")
for name, r in step5_results.items():
    print(f"  {name}: LR={r.get('Logistic_AUC', 'N/A')}, RF={r.get('RandomForest_AUC', 'N/A')}")

print(f"\n[OK] Results saved to: {TABLES_DIR}")
if DEMO_MODE:
    print("\n" + "=" * 70)
    print("  DEMO MODE COMPLETE")
    print("  Simulated data (N=200, CHARLS format). All pipeline steps executed.")
    print("  Results are illustrative only — NOT for scientific inference.")
    print("  For real analysis, run: python v11_master_pipeline.py")
    print("=" * 70)
else:
    print("=" * 70)
