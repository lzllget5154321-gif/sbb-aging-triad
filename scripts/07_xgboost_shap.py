#!/usr/bin/env python3
# ============================================================
# 07_xgboost_shap.py — XGBoost + SHAP: Predict DSI (Final)
# SBB Aging Triad Project | v3.2 | 2026-06-12
#
# Strategy:
#   Train on CHARLS (11 features) with scale_pos_weight
#   Optuna Bayesian tuning (50 trials)
#   Validate on HRS (7 common features)
#   Also test KLoSA/MHAS where possible
#   NO SMOTE (caused severe overfitting: CV 0.99 -> holdout 0.51)
#   NO SAI (tautological: DSI constructed from same vision/hearing)
# ============================================================

"""XGBoost + SHAP + Optuna: DSI prediction with cross-cohort validation."""

import numpy as np
import pandas as pd
import os, sys, warnings, json, gc
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

from sklearn.model_selection import (train_test_split, cross_val_score,
                                      StratifiedKFold, RepeatedStratifiedKFold)
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import xgboost as xgb
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 0. Configuration
# ============================================================
RANDOM_SEED = 20260612
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

OUTCOME = 'dsi'

print("=" * 70)
print("XGBoost + SHAP + Optuna: DSI Prediction (scale_pos_weight)")
print("=" * 70)

# ============================================================
# 1. Load CHARLS (Training Set) — ALL available features
# ============================================================
print("\n=== 1. Loading CHARLS (Training) ===")

df_pheno = pd.read_csv(os.path.join(TABLES_DIR, 'charls_full_phenotypes.csv'))
df_cov = pd.read_csv(os.path.join(TABLES_DIR, 'charls_causal_forest_harmonized.csv'))

df = df_pheno[['id', 'dsi', 'BAI', 'BoAI', 'pubage', 'ragender', 'cesd10']].copy()
df = df.rename(columns={'pubage': 'age', 'ragender': 'female_raw', 'cesd10': 'depression'})
df['female'] = (df['female_raw'] == 2).astype(int)
df = df.drop(columns=['female_raw'])

cov_cols = ['id', 'educ', 'chronic', 'smoker', 'drinker', 'phys_act', 'sr_health', 'cog_base']
cov_avail = [c for c in cov_cols if c in df_cov.columns]
df = df.merge(df_cov[cov_avail], on='id', how='left')

for col in cov_avail:
    if col == 'id': continue
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median() if df[col].dtype in ['float64','int64'] else 0)

charls_features = [c for c in df.columns if c not in ['id', OUTCOME]]
df = df.dropna(subset=charls_features + [OUTCOME])

print(f"  CHARLS: {len(df):,} rows, DSI={df[OUTCOME].mean()*100:.1f}%")
print(f"  Features ({len(charls_features)}): {charls_features}")

# ============================================================
# 2. Load HRS (Primary External Validation)
# ============================================================
print("\n=== 2. Loading External Cohorts ===")

def load_hrs():
    d = pd.read_csv(os.path.join(TABLES_DIR, 'hrs_full_analysis.csv'))
    d = d.rename(columns={'cesd': 'depression', 'hhidpn': 'id'})
    d['chronic'] = d[[c for c in ['hibp','diab','hearte','stroke'] if c in d.columns]].sum(axis=1)
    if 'wave' in d.columns:
        d = d.loc[d.groupby('id')['wave'].idxmin()]
    return d

def load_klosa():
    dp = pd.read_csv(os.path.join(TABLES_DIR, 'klosa_phenotypes.csv'))
    dc = pd.read_csv(os.path.join(TABLES_DIR, 'klosa_causal_forest_harmonized.csv'))
    dp = dp.loc[dp.groupby('pid')['wave'].idxmin()]
    dp = dp.rename(columns={'pid': 'id'})
    dc_avail = [c for c in ['id','age','female','educ','chronic','depression'] if c in dc.columns]
    dp = dp.merge(dc[dc_avail], on='id', how='left')
    for c in ['age','female','educ','chronic','depression']:
        if c in dp.columns and dp[c].isna().sum() > 0:
            dp[c] = dp[c].fillna(dp[c].median() if dp[c].dtype in ['float64','int64'] else 0)
    return dp

def load_mhas():
    dp = pd.read_csv(os.path.join(TABLES_DIR, 'mhas_phenotypes.csv'))
    dp = dp.rename(columns={'unhhidnp': 'id', 'pubage': 'age', 'ragender': 'female_raw',
                             'cesd_m': 'depression'})
    dp['female'] = (dp['female_raw'] == 2).astype(int)
    dp = dp.drop(columns=['female_raw'], errors='ignore')
    dis_cols = [c for c in ['hearte','stroke','hibpe','diabe'] if c in dp.columns]
    if dis_cols:
        dp['chronic'] = dp[dis_cols].sum(axis=1)
    if 'wave' in dp.columns:
        dp = dp.loc[dp.groupby('id')['wave'].idxmin()]
    return dp

cohorts = {}
for name, loader in [('HRS', load_hrs), ('KLoSA', load_klosa), ('MHAS', load_mhas)]:
    try:
        d = loader()
        avail = [f for f in charls_features if f in d.columns]
        d = d.dropna(subset=avail + [OUTCOME])
        if d[OUTCOME].mean() > 0.5:
            print(f"  {name:8s}: SKIPPED (DSI={d[OUTCOME].mean()*100:.0f}% — likely coding issue)")
            continue
        if d[OUTCOME].sum() < 10:
            print(f"  {name:8s}: SKIPPED (only {int(d[OUTCOME].sum())} DSI cases)")
            continue
        cohorts[name] = {'df': d, 'features': avail}
        print(f"  {name:8s}: {len(d):,} rows, DSI={d[OUTCOME].mean()*100:.1f}%, "
              f"features={len(avail)}, common={len([f for f in avail if f in charls_features])}")
    except Exception as e:
        print(f"  {name:8s}: SKIPPED ({e})")

# ============================================================
# 3. Optuna + XGBoost (scale_pos_weight, NO SMOTE)
# ============================================================
print("\n=== 3. Optuna Hyperparameter Optimization ===")

X_raw = df[charls_features].values
y_raw = df[OUTCOME].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=RANDOM_SEED, stratify=y_raw
)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
print(f"  Scale pos weight: {scale_pos_weight:.1f}")
print(f"  Train: {len(X_train):,} (DSI={y_train.mean()*100:.1f}%)")
print(f"  Test:  {len(X_test):,} (DSI={y_test.mean()*100:.1f}%)")

# Optuna
print(f"\n  Running 50 Optuna trials...")

def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 0.5, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 0.5, log=True),
        'scale_pos_weight': scale_pos_weight,
        'random_state': RANDOM_SEED,
        'verbosity': 0,
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=RANDOM_SEED)
    scores = []
    for tr_idx, val_idx in cv.split(X_train_s, y_train):
        m = xgb.XGBClassifier(**params)
        m.fit(X_train_s[tr_idx], y_train[tr_idx],
              eval_set=[(X_train_s[val_idx], y_train[val_idx])],
              verbose=False)
        yp = m.predict_proba(X_train_s[val_idx])[:, 1]
        scores.append(roc_auc_score(y_train[val_idx], yp))
    return np.mean(scores)

study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=50, show_progress_bar=False)

best_params = study.best_params
print(f"  Best trial #{study.best_trial.number}: CV AUC={study.best_value:.4f}")
print(f"  Best params: {best_params}")

# Train final model
final_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': scale_pos_weight,
    'random_state': RANDOM_SEED,
    'verbosity': 0,
    **best_params
}

model = xgb.XGBClassifier(**final_params)
model.fit(X_train_s, y_train,
          eval_set=[(X_test_s, y_test)],
          verbose=False)

# Internal validation
y_pred_proba = model.predict_proba(X_test_s)[:, 1]
y_pred_class = model.predict(X_test_s)
auc_internal = roc_auc_score(y_test, y_pred_proba)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
sens = tp / (tp + fn) if (tp + fn) > 0 else 0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n  CHARLS Internal Validation:")
print(f"    AUC:          {auc_internal:.4f}")
print(f"    Sensitivity:   {sens:.4f}")
print(f"    Specificity:   {spec:.4f}")
print(f"    TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# Cross-validation on full training data
cv_params = {k: v for k, v in final_params.items() if k not in ['n_estimators']}
cv_params['n_estimators'] = best_params.get('n_estimators', 200)
cv_m = xgb.XGBClassifier(**cv_params)
cv_aucs = cross_val_score(cv_m, X_train_s, y_train,
                          cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED),
                          scoring='roc_auc')
print(f"    5-fold CV AUC: {cv_aucs.mean():.4f} +/- {cv_aucs.std():.4f}")

# ============================================================
# 4. SHAP on Full CHARLS
# ============================================================
print("\n=== 4. SHAP Analysis ===")

X_full_s = scaler.fit_transform(X_raw)
full_params = {k: v for k, v in final_params.items()}
model_full = xgb.XGBClassifier(**full_params)
model_full.fit(X_full_s, y_raw, verbose=False)

explainer = shap.TreeExplainer(model_full)
shap_subset = np.random.choice(len(X_full_s), min(2000, len(X_full_s)), replace=False)
shap_values = explainer.shap_values(X_full_s[shap_subset])

importance_df = pd.DataFrame({
    'feature': charls_features,
    'shap_importance': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_importance', ascending=False)

print("  Top features:")
for i, row in importance_df.iterrows():
    print(f"    {row['feature']:20s} {row['shap_importance']:.4f}")

# SHAP Summary
fig, ax = plt.subplots(figsize=(10, 8))
X_display = scaler.inverse_transform(X_full_s[shap_subset])
X_display_df = pd.DataFrame(X_display, columns=charls_features)
shap.summary_plot(shap_values, X_display_df, max_display=20, show=False)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'figure5a_shap_summary_final.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> results/figures/figure5a_shap_summary_final.png")

# SHAP Dependence
top3 = importance_df.head(3)['feature'].values
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, feat in enumerate(top3):
    bai_idx = list(charls_features).index('BAI') if 'BAI' in charls_features else 0
    shap.dependence_plot(feat, shap_values, X_display_df,
                         feature_names=charls_features,
                         interaction_index=bai_idx,
                         ax=axes[idx], show=False)
    axes[idx].set_title(f'SHAP Dependence: {feat} x BAI', fontsize=12, fontweight='bold')
plt.tight_layout()
fig_path2 = os.path.join(FIGURES_DIR, 'figure5b_shap_dependence_final.png')
plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> results/figures/figure5b_shap_dependence_final.png")

# Importance bar
fig, ax = plt.subplots(figsize=(10, 6))
top_imp = importance_df.iloc[::-1]
colors = ['#2166ac' if v > importance_df['shap_importance'].median() else '#92c5de'
          for v in top_imp['shap_importance']]
ax.barh(range(len(top_imp)), top_imp['shap_importance'], color=colors)
ax.set_yticks(range(len(top_imp)))
ax.set_yticklabels(top_imp['feature'])
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('SHAP Feature Importance — CHARLS (Optuna Tuned)', fontweight='bold')
plt.tight_layout()
fig_path3 = os.path.join(FIGURES_DIR, 'figure5c_shap_importance_final.png')
plt.savefig(fig_path3, dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> results/figures/figure5c_shap_importance_final.png")

# ============================================================
# 5. External Validation
# ============================================================
print("\n=== 5. External Validation ===")

all_results = []
for name, info in cohorts.items():
    d = info['df']
    avail = [f for f in charls_features if f in d.columns]

    # Build aligned feature matrix: pad missing features with 0
    X_ext_aligned = np.zeros((len(d), len(charls_features)))
    for j, feat in enumerate(charls_features):
        if feat in d.columns:
            X_ext_aligned[:, j] = d[feat].values
        # else: stays 0 (missing feature)

    X_ext_s = scaler.transform(X_ext_aligned)
    y_ext = d[OUTCOME].values

    y_pred = model_full.predict_proba(X_ext_s)[:, 1]
    auc_ext = roc_auc_score(y_ext, y_pred)

    y_pred_c = model_full.predict(X_ext_s)
    tn_e, fp_e, fn_e, tp_e = confusion_matrix(y_ext, y_pred_c).ravel()
    sens_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0
    spec_e = tn_e / (tn_e + fp_e) if (tn_e + fp_e) > 0 else 0

    all_results.append({
        'Cohort': name, 'N': len(d), 'DSI_pct': y_ext.mean()*100,
        'Common_Features': len(avail),
        'AUC': auc_ext, 'AUC_Drop': auc_internal - auc_ext,
        'Sensitivity': sens_e, 'Specificity': spec_e
    })
    arrow = "v" if auc_ext < auc_internal else "^"
    print(f"  {name:8s}: AUC={auc_ext:.4f} ({arrow} {auc_internal:.4f}, "
          f"drop={auc_internal-auc_ext:+.4f}), DSI={y_ext.mean()*100:.1f}%, "
          f"N={len(d):,}, features={len(avail)}/{len(charls_features)}")

# Multi-cohort ROC
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#2166ac', '#b2182b', '#4daf4a', '#ff7f00']
fpr_c, tpr_c, _ = roc_curve(y_test, y_pred_proba)
ax.plot(fpr_c, tpr_c, '-', color=colors[0], linewidth=2.5,
        label=f'CHARLS (internal) AUC={auc_internal:.3f}')

for i, (name, info) in enumerate(cohorts.items()):
    d = info['df']
    avail = [f for f in charls_features if f in d.columns]
    X_ext_aligned = np.zeros((len(d), len(charls_features)))
    for j, feat in enumerate(charls_features):
        if feat in d.columns:
            X_ext_aligned[:, j] = d[feat].values
    X_ext_s = scaler.transform(X_ext_aligned)
    y_ext = d[OUTCOME].values
    y_pred = model_full.predict_proba(X_ext_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_ext, y_pred)
    ax.plot(fpr, tpr, '--', color=colors[i+1], linewidth=2,
            label=f'{name} AUC={roc_auc_score(y_ext, y_pred):.3f}')

ax.plot([0, 1], [0, 1], 'k:', alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves: CHARLS -> External Cohorts\n(Optuna, scale_pos_weight)', fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig_path4 = os.path.join(FIGURES_DIR, 'figure5d_roc_multicohort_final.png')
plt.savefig(fig_path4, dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> results/figures/figure5d_roc_multicohort_final.png")

# ============================================================
# 6. Cross-Cohort SHAP Consistency
# ============================================================
print("\n=== 6. SHAP Consistency ===")

shap_consistency = []
for name, info in cohorts.items():
    d = info['df']
    avail = [f for f in charls_features if f in d.columns]
    X_ext_aligned = np.zeros((len(d), len(charls_features)))
    for j, feat in enumerate(charls_features):
        if feat in d.columns:
            X_ext_aligned[:, j] = d[feat].values
    X_ext_s = scaler.transform(X_ext_aligned)
    shap_ext = explainer.shap_values(X_ext_s[:min(2000, len(X_ext_s))])
    imp_ext = pd.DataFrame({
        'feature': charls_features,
        f'shap_{name}': np.abs(shap_ext).mean(axis=0)
    }).sort_values('feature')

    imp_charls = importance_df.sort_values('feature')
    merged = imp_charls.merge(imp_ext, on='feature')
    rho, pval = spearmanr(merged['shap_importance'], merged[f'shap_{name}'])

    shap_consistency.append({
        'Cohort': name, 'Spearman_rho': rho, 'Spearman_p': pval,
        'N_features': len(merged)
    })
    print(f"  CHARLS vs {name:8s}: rho={rho:.4f} (p={pval:.4f}), "
          f"n_features={len(merged)} {'[PASS]' if rho>0.6 else '[FAIL]'}")

# ============================================================
# 7. Validation Gate
# ============================================================
print("\n=== 7. Validation Gate ===")

checks = []
checks.append(('CHARLS CV AUC >= 0.70', cv_aucs.mean() >= 0.70, f'{cv_aucs.mean():.3f}'))
checks.append(('CHARLS Holdout AUC >= 0.65', auc_internal >= 0.65, f'{auc_internal:.3f}'))
for r in all_results:
    checks.append((f'{r["Cohort"]} AUC drop < 0.15',
                    abs(r['AUC_Drop']) < 0.15, f'{r["AUC_Drop"]:+.3f}'))
for c in shap_consistency:
    checks.append((f'{c["Cohort"]} SHAP rho > 0.6',
                    c['Spearman_rho'] > 0.6, f'{c["Spearman_rho"]:.3f}'))

all_pass = True
for name, passed, value in checks:
    status = '[PASS]' if passed else '[FAIL]'
    print(f"  {status}: {name} = {value}")
    if not passed:
        all_pass = False

# Summary table
print(f"\n  Summary:")
print(f"  {'Cohort':10s} {'N':>6s} {'DSI%':>6s} {'AUC':>7s} {'Drop':>7s} {'Sens':>7s} {'Spec':>7s} {'rho':>7s}")
print(f"  {'CHARLS':10s} {len(df):>6,} {df[OUTCOME].mean()*100:>5.1f}% {auc_internal:>7.4f} {'--':>7s} {sens:>7.4f} {spec:>7.4f} {'--':>7s}")
for r, c in zip(all_results, shap_consistency):
    print(f"  {r['Cohort']:10s} {r['N']:>6,} {r['DSI_pct']:>5.1f}% {r['AUC']:>7.4f} {r['AUC_Drop']:>+7.4f} {r['Sensitivity']:>7.4f} {r['Specificity']:>7.4f} {c['Spearman_rho']:>7.4f}")

# Save report
report = {
    'timestamp': '2026-06-12',
    'version': 'v3.2-final',
    'approach': 'scale_pos_weight + Optuna, no SMOTE, no SAI',
    'outcome': 'DSI',
    'charls_features': charls_features,
    'charls_n': len(df),
    'charls_dsi_pct': float(df[OUTCOME].mean()*100),
    'charls_auc': float(auc_internal),
    'charls_cv_auc_mean': float(cv_aucs.mean()),
    'charls_cv_auc_std': float(cv_aucs.std()),
    'optuna_best_params': best_params,
    'optuna_best_cv_auc': study.best_value,
    'external_results': all_results,
    'shap_consistency': shap_consistency,
    'all_checks_passed': all_pass,
}

with open(os.path.join(RESULTS_DIR, 'xgboost_shap_final_report.json'), 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

print(f"\n{'[ALL CHECKS PASSED]' if all_pass else '[SOME CHECKS FAILED]'}")
print(f"  Report -> results/xgboost_shap_final_report.json")
print("\n=== Pipeline Complete (v3.2) ===")
