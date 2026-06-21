# =============================================================================
# v11b_enhanced_pipeline.py — 增强 ML 分析管道
# =============================================================================
# 功能: RCS剂量-反应 + LMEM纵向 + Cox PH生存
#       + XGBoost/LightGBM + SHAP + Calibration + DCA
# 输入: data_derived/ 4-cohort harmonized 数据
# 输出: results/v11b/tables/*.csv + results/v11b/figures/*.png
# 依赖: pandas, numpy, xgboost, lightgbm, sklearn, shap, lifelines, matplotlib, seaborn
# 用法: python v11b_enhanced_pipeline.py
# 项目: SBB课题 -- 脑体感官衰老耦合解耦研究
# 版本: v11b (2026-06-18)
# =============================================================================

import pandas as pd
import numpy as np
import os, sys, warnings, json, io

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ============================================================
# 0. Config
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJ_DIR, "results", "v11b")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# ============================================================
# 1. Data Loading (same as v11.0)
# ============================================================
def normalize_columns(df):
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "dsi": col_map[c] = "dsi"
        elif cl in ["sai", "bai", "boai"]: col_map[c] = cl.upper()
        elif cl == "boai_raw": col_map[c] = "BoAI_raw"
        elif cl == "bai_raw_z": col_map[c] = "BAI_raw_z"
        elif cl in ["cesd", "cesd10", "cesd_m"]: col_map[c] = "cesd"
        elif cl in ["pubage", "age", "agey_b", "trueage"]: col_map[c] = "age"
        elif cl in ["ragender", "female"]: col_map[c] = "female"
        elif cl == "phenotype": col_map[c] = "phenotype"
        elif cl in ["wave", "waveid"]: col_map[c] = "wave"
    return df.rename(columns=col_map)


def load_cohort(name, filename, use_baseline=True):
    path = os.path.join(PROJ_DIR, "results", "tables", filename)
    if not os.path.exists(path):
        print(f"  ⚠️ {name}: file not found at {path}")
        return None, None
    df = pd.read_csv(path)
    df = normalize_columns(df)
    df["cohort"] = name
    if use_baseline and "wave" in df.columns:
        min_wave = df["wave"].min()
        bl = df[df["wave"] == min_wave].copy()
    else:
        bl = df.copy()
    print(f"  ✅ {name}: N={len(df):,}, DSI={df['dsi'].mean()*100:.2f}%")
    return bl, df


def ensure_indices(df, name):
    if "dsi" not in df.columns or df["dsi"].isna().all():
        if "vision_impairment" in df.columns and "hearing_impairment" in df.columns:
            df["vision_imp"] = df["vision_impairment"].apply(
                lambda x: 1 if str(x).lower() in ["1", "yes", "poor", "fair", "true"] else 0
            )
            df["hearing_imp"] = df["hearing_impairment"].apply(
                lambda x: 1 if str(x).lower() in ["1", "yes", "poor", "fair", "true"] else 0
            )
            df["dsi"] = ((df["vision_imp"] == 1) & (df["hearing_imp"] == 1)).astype(int)
    if "SAI" not in df.columns or df["SAI"].isna().all():
        if "vision_imp" in df.columns and "hearing_imp" in df.columns:
            df["SAI_raw"] = (df["vision_imp"] + df["hearing_imp"]) / 2
            sai_z = (df["SAI_raw"] - df["SAI_raw"].mean()) / df["SAI_raw"].std()
            df["SAI"] = (sai_z - sai_z.min()) / (sai_z.max() - sai_z.min()) * 100
    if "BAI" not in df.columns or df["BAI"].isna().all():
        cog_col = next((c for c in ["global_cognition", "mmse_total", "cog_score"] if c in df.columns), None)
        if cog_col:
            cog_z = (df[cog_col] - df[cog_col].mean()) / df[cog_col].std()
            df["BAI"] = (cog_z - cog_z.min()) / (cog_z.max() - cog_z.min()) * 100
    if "BoAI" not in df.columns or df["BoAI"].isna().all():
        components = []
        for c in ["adl_sum", "iadl_sum", "chronic_count", "self_rated_health"]:
            if c in df.columns: components.append(c)
        if len(components) >= 2:
            z_scores = [(df[c] - df[c].mean()) / df[c].std() for c in components]
            df["BoAI_raw"] = np.nanmean(z_scores, axis=0)
            df["BoAI"] = (
                (df["BoAI_raw"] - np.nanmin(df["BoAI_raw"]))
                / (np.nanmax(df["BoAI_raw"]) - np.nanmin(df["BoAI_raw"]))
                * 100
            )
    df["female"] = df.get("female", np.where(df.get("ragender", 0) == 2, 1, 0))
    return df


print("=" * 70)
print("SBB Aging Triad v11b — Enhanced Analysis Pipeline")
print("=" * 70)

COHORTS = {}
LONG_DATA = {}

c_bl, c_long = load_cohort("CHARLS", "charls_full_longitudinal.csv")
if c_bl is not None:
    c_bl = ensure_indices(c_bl, "CHARLS")
    c_long = ensure_indices(c_long, "CHARLS") if c_long is not None else None
    if "BAI" in c_bl.columns:
        c_bl["BAI"] = 100 - c_bl["BAI"]
        if c_long is not None and "BAI" in c_long.columns:
            c_long["BAI"] = 100 - c_long["BAI"]
        print("  🔧 CHARLS BAI reversed")
    COHORTS["CHARLS"] = c_bl
    if c_long is not None:
        LONG_DATA["CHARLS"] = c_long

h_bl, _ = load_cohort("HRS", "hrs_full_analysis.csv", use_baseline=False)
if h_bl is not None:
    h_bl = ensure_indices(h_bl, "HRS")
    COHORTS["HRS"] = h_bl

k_bl, k_long = load_cohort("KLoSA", "klosa_corrected.csv")
if k_bl is not None:
    k_bl = ensure_indices(k_bl, "KLoSA")
    COHORTS["KLoSA"] = k_bl
    if k_long is not None:
        k_long = ensure_indices(k_long, "KLoSA") if k_long is not None else None
        LONG_DATA["KLoSA"] = k_long

m_bl, _ = load_cohort("MHAS", "mhas_fixed.csv", use_baseline=False)
if m_bl is not None:
    m_bl = ensure_indices(m_bl, "MHAS")
    COHORTS["MHAS"] = m_bl

print(f"\nLoaded {len(COHORTS)} baseline cohorts + {len(LONG_DATA)} longitudinal")

# ============================================================
# 1A. RCS Dose-Response (Step 1 enhanced)
# ============================================================
print("\n" + "=" * 70)
print("STEP 1+: RCS Dose-Response Curves")
print("=" * 70)

import statsmodels.api as sm
from scipy import stats as scipy_stats

rcs_results = {}

fig, axes = plt.subplots(1, len(COHORTS), figsize=(6 * len(COHORTS), 5))
if len(COHORTS) == 1:
    axes = [axes]

for idx, (name, df) in enumerate([(n, d) for n, d in COHORTS.items()]):
    print(f"\n--- {name} ---")
    age_col = next((c for c in ["pubage", "age", "agey_b"] if c in df.columns), None)
    if age_col is None:
        continue

    valid = df[["dsi", "BAI", age_col]].dropna()
    if "female" in df.columns:
        valid = df[["dsi", "BAI", age_col, "female"]].dropna()

    # Use SAI as continuous exposure (preferred over binary DSI)
    sai_col = "SAI" if "SAI" in df.columns else "dsi"

    # RCS with 4 knots
    try:
        from patsy import dmatrix
        import statsmodels.formula.api as smf

        # Build design matrix with natural spline
        knot_positions = np.percentile(valid[sai_col], [5, 35, 65, 95])
        spline_basis = dmatrix(
            f"cr({sai_col}, df=3, knots=list({list(knot_positions[1:3])}))",
            valid,
            return_type="dataframe",
        )

        # Build full model
        cov_list = ["female"] if "female" in valid.columns else []
        X_cols = list(spline_basis.columns[1:])
        for c in cov_list:
            X_cols.append(c)

        X_rcs = pd.concat([spline_basis.iloc[:, 1:], valid[cov_list].reset_index(drop=True)], axis=1)
        X_rcs = sm.add_constant(X_rcs)
        rcs_model = sm.OLS(valid["BAI"], X_rcs.astype(float)).fit()

        # Predict across SAI range
        sai_range = np.linspace(valid[sai_col].min(), valid[sai_col].max(), 100)
        pred_df = pd.DataFrame({sai_col: sai_range})
        pred_spline = dmatrix(
            f"cr({sai_col}, df=3, knots=list({list(knot_positions[1:3])}))",
            pred_df,
            return_type="dataframe",
        )
        for c in cov_list:
            pred_df[c] = valid[c].median()

        X_pred = pd.concat([pred_spline.iloc[:, 1:], pred_df[cov_list].reset_index(drop=True)], axis=1)
        X_pred = sm.add_constant(X_pred)
        preds = rcs_model.predict(X_pred.astype(float))
        pred_se = rcs_model.get_prediction(X_pred.astype(float)).se_mean
        ci_low = preds - 1.96 * pred_se
        ci_high = preds + 1.96 * pred_se

        rcs_results[name] = {
            "sai_range": sai_range.tolist(),
            "preds": preds.tolist(),
            "ci_low": ci_low.tolist(),
            "ci_high": ci_high.tolist(),
            "model_R2": float(rcs_model.rsquared),
            "nonlinear_P": float(rcs_model.f_test(np.eye(len(X_rcs.columns))[1:3]).pvalue),
        }

        ax = axes[idx]
        ax.plot(sai_range, preds, "b-", linewidth=2)
        ax.fill_between(sai_range, ci_low, ci_high, alpha=0.2, color="blue")
        ax.scatter(valid[sai_col], valid["BAI"], alpha=0.05, s=5, color="gray")
        ax.set_xlabel("SAI (continuous)" if sai_col == "SAI" else "DSI (binary)")
        ax.set_ylabel("BAI (Cognitive Score)")
        ax.set_title(f"{name}\n(N={len(valid):,}, R²={rcs_model.rsquared:.3f})", fontweight="bold")
        ax.grid(alpha=0.3)
        print(f"  RCS: R²={rcs_model.rsquared:.3f}, nonlinear P={rcs_model.f_test(np.eye(len(X_rcs.columns))[1:3]).pvalue:.4f}")
    except Exception as e:
        print(f"  RCS: FAILED ({e})")

fig.suptitle("Step 1+: RCS Dose-Response — DSI/SAI → Cognitive Score", fontsize=16, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/figS1_rcs_dose_response.png", dpi=300, facecolor="white")
plt.close()
print("\n✅ RCS figure saved")

# ============================================================
# 1B. LMEM Longitudinal Trajectory (Step 1 enhanced)
# ============================================================
print("\n" + "=" * 70)
print("STEP 1+: LMEM Longitudinal Trajectory Modeling")
print("=" * 70)

lmem_results = {}

try:
    from statsmodels.regression.mixed_linear_model import MixedLM

    for name, df_long in LONG_DATA.items():
        print(f"\n--- {name} ---")
        age_col = next((c for c in ["pubage", "age", "agey_b"] if c in df_long.columns), None)
        wave_col = next((c for c in ["wave", "waveid"] if c in df_long.columns), None)

        if age_col is None or wave_col is None:
            print(f"  ⚠️ Missing age/wave, skipping")
            continue

        valid = df_long[["dsi", "BAI", age_col, wave_col]].dropna().copy()

        # Create time variable (wave index starting from 0)
        waves = sorted(valid[wave_col].unique())
        wave_map = {w: i for i, w in enumerate(waves)}
        valid["time"] = valid[wave_col].map(wave_map)

        # Fit LMEM: BAI ~ dsi * time + age + (1 | wave)
        try:
            model = MixedLM(
                valid["BAI"],
                valid[["dsi", "time", age_col]],
                groups=valid[wave_col],
                re_formula="1",
            )
            result = model.fit(reml=True, method=["powell", "lbfgs"])

            lmem_results[name] = {
                "N_obs": len(valid),
                "N_waves": len(waves),
                "dsi_beta": float(result.fe_params["dsi"]),
                "dsi_p": float(result.pvalues["dsi"]),
                "time_beta": float(result.fe_params["time"]),
                "time_p": float(result.pvalues["time"]),
                "logLik": float(result.llf),
            }
            print(f"  LMEM: DSI β={result.fe_params['dsi']:.2f} (P={result.pvalues['dsi']:.4f}), "
                  f"Time β={result.fe_params['time']:.2f} (P={result.pvalues['time']:.4f})")
        except Exception as e:
            print(f"  LMEM: FAILED ({e})")
            lmem_results[name] = {"error": str(e)}
except ImportError:
    print("  ⚠️ statsmodels MixedLM not available")

# Plot LMEM trajectories
if LONG_DATA and lmem_results:
    fig, axes = plt.subplots(1, len(LONG_DATA), figsize=(6 * len(LONG_DATA), 5))
    if len(LONG_DATA) == 1:
        axes = [axes]

    for idx, (name, df_long) in enumerate(LONG_DATA.items()):
        ax = axes[idx]
        age_col = next((c for c in ["pubage", "age", "agey_b"] if c in df_long.columns), None)
        wave_col = next((c for c in ["wave", "waveid"] if c in df_long.columns), None)
        if age_col is None or wave_col is None:
            continue

        valid = df_long[[age_col, "BAI", "dsi", wave_col]].dropna()
        waves = sorted(valid[wave_col].unique())

        for dsi_val, label, color in [(0, "DSI=0", "#3498DB"), (1, "DSI=1", "#E74C3C")]:
            subset = valid[valid["dsi"] == dsi_val]
            means = subset.groupby(wave_col)["BAI"].agg(["mean", "sem"])
            x = range(len(means))
            ax.errorbar(x, means["mean"], yerr=means["sem"] * 1.96, fmt="o-",
                       color=color, label=label, capsize=5, linewidth=2)

        ax.set_xlabel("Wave")
        ax.set_ylabel("BAI (Cognitive Score)")
        ax.set_title(f"{name} (N={len(valid):,})", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Step 1+: LMEM Longitudinal Cognitive Trajectories by DSI Status", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/figS2_lmem_trajectories.png", dpi=300, facecolor="white")
    plt.close()
    print("\n✅ LMEM figure saved")

# ============================================================
# 1C. Cox PH Survival Analysis (Step 1 enhanced)
# ============================================================
print("\n" + "=" * 70)
print("STEP 1+: Cox Proportional Hazards — Cognitive Impairment Onset")
print("=" * 70)

cox_results = {}

try:
    from lifelines import CoxPHFitter

    for name, df_long in LONG_DATA.items():
        print(f"\n--- {name} ---")
        age_col = next((c for c in ["pubage", "age", "agey_b"] if c in df_long.columns), None)
        wave_col = next((c for c in ["wave", "waveid"] if c in df_long.columns), None)

        if age_col is None or wave_col is None:
            print(f"  ⚠️ Missing age/wave, skipping")
            continue

        valid = df_long[["dsi", "BAI", age_col, wave_col, "female"]].dropna().copy()
        valid = valid[valid["female"].isin([0, 1])]

        # Define "event" = BAI drops below median at any follow-up
        bai_baseline = valid.groupby(wave_col)["BAI"].transform("first")
        bai_median = valid["BAI"].median()
        valid["event"] = (valid["BAI"] < bai_median).astype(int)

        # Create survival data: time to first cognitive impairment
        time_var = valid.groupby(wave_col).cumcount()
        valid["duration"] = time_var + 1

        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(
                valid[["duration", "event", "dsi", age_col, "female"]],
                duration_col="duration",
                event_col="event",
            )

            cox_results[name] = {
                "N": len(valid),
                "dsi_HR": float(np.exp(cph.params_["dsi"])),
                "dsi_HR_CI_low": float(np.exp(cph.confidence_intervals_.loc["dsi", "95% lower-bound"])),
                "dsi_HR_CI_high": float(np.exp(cph.confidence_intervals_.loc["dsi", "95% upper-bound"])),
                "dsi_P": float(cph.summary.loc["dsi", "p"]),
            }
            hr = cox_results[name]["dsi_HR"]
            print(f"  Cox: DSI HR={hr:.2f} ({cox_results[name]['dsi_HR_CI_low']:.2f}-{cox_results[name]['dsi_HR_CI_high']:.2f}), "
                  f"P={cox_results[name]['dsi_P']:.4f}")
        except Exception as e:
            print(f"  Cox: FAILED ({e})")
            cox_results[name] = {"error": str(e)}
except ImportError:
    print("  ⚠️ lifelines not available")

# KM-style plot
if cox_results:
    fig, axes = plt.subplots(1, len(LONG_DATA), figsize=(6 * len(LONG_DATA), 5))
    if len(LONG_DATA) == 1:
        axes = [axes]

    for idx, (name, df_long) in enumerate(LONG_DATA.items()):
        ax = axes[idx]
        wave_col = next((c for c in ["wave", "waveid"] if c in df_long.columns), None)
        if wave_col is None: continue

        valid = df_long[["dsi", "BAI", wave_col]].dropna()
        bai_median = valid["BAI"].median()

        for dsi_val, label, color in [(0, "DSI=0", "#3498DB"), (1, "DSI=1", "#E74C3C")]:
            subset = valid[valid["dsi"] == dsi_val]
            waves = sorted(subset[wave_col].unique())
            prop_impaired = [np.mean(subset[subset[wave_col] == w]["BAI"] < bai_median) for w in waves]
            ax.plot(waves, prop_impaired, "o-", color=color, label=label, linewidth=2)

        ax.set_xlabel("Wave")
        ax.set_ylabel("Proportion Cognitively Impaired")
        ax.set_title(f"{name}", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Step 1+: Cumulative Cognitive Impairment Incidence by DSI Status", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/figS3_cox_survival.png", dpi=300, facecolor="white")
    plt.close()
    print("\n✅ Cox survival figure saved")

# ============================================================
# 5A. Enhanced ML: XGBoost + LightGBM + SHAP + Calibration + DCA
# ============================================================
print("\n" + "=" * 70)
print("STEP 5+: Enhanced ML (XGBoost + LightGBM + SHAP + Calibration + DCA)")
print("=" * 70)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

ml_results = {}

# Try importing optional packages
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ⚠️ xgboost not installed")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  ⚠️ lightgbm not installed")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("  ⚠️ shap not installed")

for name, df in [(n, d) for n, d in COHORTS.items()]:
    feat_cols = [c for c in ["SAI", "BAI", "BoAI", "age", "female", "cesd"] if c in df.columns]

    # Use binary outcome (DSI as target if no phenotype)
    if "phenotype" in df.columns:
        y_col = "phenotype"
        valid = df[feat_cols + [y_col]].dropna()
    else:
        y_col = "dsi"
        valid = df[feat_cols + [y_col]].dropna()

    if len(valid) < 100:
        continue

    X = valid[feat_cols].values
    y = valid[y_col].values
    n_classes = len(np.unique(y))

    # Binary classification: use positive class
    if n_classes > 2:
        # For multiclass, convert to binary (highest class vs rest)
        y_bin = (y == y.max()).astype(int)
    else:
        y_bin = y.astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.3, random_state=RANDOM_STATE, stratify=y_bin
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scorer = "roc_auc"

    res = {"N": len(valid), "n_classes": n_classes, "features": feat_cols}
    models = {}
    probs = {}

    # LR
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    try:
        lr_auc_cv = cross_val_score(lr, X, y_bin, cv=cv, scoring=scorer).mean()
        lr.fit(X_train, y_train)
        lr_auc_test = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
        lr_brier = brier_score_loss(y_test, lr.predict_proba(X_test)[:, 1])
        res["LR_AUC_CV"] = float(lr_auc_cv)
        res["LR_AUC_test"] = float(lr_auc_test)
        res["LR_Brier"] = float(lr_brier)
        models["Logistic"] = lr
        probs["Logistic"] = lr.predict_proba(X_test)[:, 1]
        print(f"  {name}: LR AUC(cv)={lr_auc_cv:.3f}, AUC(test)={lr_auc_test:.3f}, Brier={lr_brier:.3f}")
    except Exception as e:
        print(f"  {name}: LR FAILED ({e})")

    # RF
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
    try:
        rf_auc_cv = cross_val_score(rf, X, y_bin, cv=cv, scoring=scorer).mean()
        rf.fit(X_train, y_train)
        rf_auc_test = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
        rf_brier = brier_score_loss(y_test, rf.predict_proba(X_test)[:, 1])
        res["RF_AUC_CV"] = float(rf_auc_cv)
        res["RF_AUC_test"] = float(rf_auc_test)
        res["RF_Brier"] = float(rf_brier)
        res["RF_importance"] = {f: float(v) for f, v in zip(feat_cols, rf.feature_importances_)}
        models["RandomForest"] = rf
        probs["RandomForest"] = rf.predict_proba(X_test)[:, 1]
        print(f"  {name}: RF AUC(cv)={rf_auc_cv:.3f}, AUC(test)={rf_auc_test:.3f}, Brier={rf_brier:.3f}")
    except Exception as e:
        print(f"  {name}: RF FAILED ({e})")

    # XGBoost
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False
        )
        try:
            xgb_auc_cv = cross_val_score(xgb, X, y_bin, cv=cv, scoring=scorer).mean()
            xgb.fit(X_train, y_train)
            xgb_auc_test = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
            xgb_brier = brier_score_loss(y_test, xgb.predict_proba(X_test)[:, 1])
            res["XGB_AUC_CV"] = float(xgb_auc_cv)
            res["XGB_AUC_test"] = float(xgb_auc_test)
            res["XGB_Brier"] = float(xgb_brier)
            models["XGBoost"] = xgb
            probs["XGBoost"] = xgb.predict_proba(X_test)[:, 1]
            print(f"  {name}: XGB AUC(cv)={xgb_auc_cv:.3f}, AUC(test)={xgb_auc_test:.3f}, Brier={xgb_brier:.3f}")
        except Exception as e:
            print(f"  {name}: XGB FAILED ({e})")

    # LightGBM
    if HAS_LGB:
        lgb = LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=-1
        )
        try:
            lgb_auc_cv = cross_val_score(lgb, X, y_bin, cv=cv, scoring=scorer).mean()
            lgb.fit(X_train, y_train)
            lgb_auc_test = roc_auc_score(y_test, lgb.predict_proba(X_test)[:, 1])
            lgb_brier = brier_score_loss(y_test, lgb.predict_proba(X_test)[:, 1])
            res["LGB_AUC_CV"] = float(lgb_auc_cv)
            res["LGB_AUC_test"] = float(lgb_auc_test)
            res["LGB_Brier"] = float(lgb_brier)
            models["LightGBM"] = lgb
            probs["LightGBM"] = lgb.predict_proba(X_test)[:, 1]
            print(f"  {name}: LGB AUC(cv)={lgb_auc_cv:.3f}, AUC(test)={lgb_auc_test:.3f}, Brier={lgb_brier:.3f}")
        except Exception as e:
            print(f"  {name}: LGB FAILED ({e})")

    # --- SHAP (on RF model) ---
    if HAS_SHAP and "RandomForest" in models:
        try:
            # Use a subset for SHAP (computationally expensive)
            n_shap = min(200, len(X_train))
            X_shap = X_train[:n_shap]
            explainer = shap.TreeExplainer(models["RandomForest"])
            shap_values = explainer.shap_values(X_shap)

            # If multiclass, take first class
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            # SHAP bar
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.summary_plot(shap_values, X_shap, feature_names=feat_cols, show=False, plot_type="bar")
            ax.set_title(f"{name} — SHAP Feature Importance (RF)", fontweight="bold")
            plt.tight_layout()
            fig.savefig(f"{FIGURES_DIR}/figS4a_shap_{name}.png", dpi=300, facecolor="white", bbox_inches="tight")
            plt.close()

            # SHAP beeswarm
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.summary_plot(shap_values, X_shap, feature_names=feat_cols, show=False)
            ax.set_title(f"{name} — SHAP Beeswarm (RF)", fontweight="bold")
            plt.tight_layout()
            fig.savefig(f"{FIGURES_DIR}/figS4b_shap_beeswarm_{name}.png", dpi=300, facecolor="white", bbox_inches="tight")
            plt.close()
            print(f"  {name}: SHAP figures saved")
        except Exception as e:
            print(f"  {name}: SHAP FAILED ({e})")

    ml_results[name] = res

# --- Combined Calibration Plot ---
if ml_results:
    fig, axes = plt.subplots(1, len(ml_results), figsize=(6 * len(ml_results), 5))
    if len(ml_results) == 1:
        axes = [axes]

    for idx, (name, res) in enumerate(ml_results.items()):
        ax = axes[idx]

        # Get test data for this cohort
        df = COHORTS.get(name)
        if df is None: continue

        feat_cols = res.get("features", [])
        y_col = next((c for c in ["phenotype", "dsi"] if c in df.columns), None)
        if y_col is None: continue

        valid = df[feat_cols + [y_col]].dropna()
        X_cohort = valid[feat_cols].values
        y_cohort = valid[y_col].values
        n_classes = len(np.unique(y_cohort))
        y_bin = (y_cohort == y_cohort.max()).astype(int) if n_classes > 2 else y_cohort.astype(int)

        # Fit RF on this cohort for calibration
        rf_cal = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
        X_tr, X_te, y_tr, y_te = train_test_split(X_cohort, y_bin, test_size=0.3, random_state=RANDOM_STATE, stratify=y_bin)
        rf_cal.fit(X_tr, y_tr)
        prob_true, prob_pred = calibration_curve(y_te, rf_cal.predict_proba(X_te)[:, 1], n_bins=10)

        ax.plot(prob_pred, prob_true, "s-", color="#E74C3C", linewidth=2, label="RF")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Proportion")
        ax.set_title(f"{name}", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Step 5+: Calibration Curves — Random Forest", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/figS5_calibration.png", dpi=300, facecolor="white")
    plt.close()
    print("\n✅ Calibration figure saved")

# --- DCA Decision Curve ---
if ml_results:
    fig, axes = plt.subplots(1, len(ml_results), figsize=(6 * len(ml_results), 5))
    if len(ml_results) == 1:
        axes = [axes]

    for idx, (name, res) in enumerate(ml_results.items()):
        ax = axes[idx]
        df = COHORTS.get(name)
        if df is None: continue

        feat_cols = res.get("features", [])
        y_col = next((c for c in ["phenotype", "dsi"] if c in df.columns), None)
        if y_col is None: continue

        valid = df[feat_cols + [y_col]].dropna()
        X_cohort = valid[feat_cols].values
        y_cohort = valid[y_col].values
        n_classes = len(np.unique(y_cohort))
        y_bin = (y_cohort == y_cohort.max()).astype(int) if n_classes > 2 else y_cohort.astype(int)

        # DCA: Net benefit = (TP - w * FP) / N
        X_tr, X_te, y_tr, y_te = train_test_split(X_cohort, y_bin, test_size=0.3, random_state=RANDOM_STATE, stratify=y_bin)
        rf_cal = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
        rf_cal.fit(X_tr, y_tr)
        preds = rf_cal.predict_proba(X_te)[:, 1]

        thresholds = np.linspace(0.01, 0.99, 99)
        nb_model, nb_all, nb_none = [], [], []
        prev = y_te.mean()

        for t in thresholds:
            tp = np.sum((preds >= t) & (y_te == 1))
            fp = np.sum((preds >= t) & (y_te == 0))
            n = len(y_te)
            w = t / (1 - t)
            nb_model.append((tp - w * fp) / n)
            nb_all.append(prev - (1 - prev) * w)
            nb_none.append(0)

        ax.plot(thresholds, nb_model, "b-", linewidth=2, label="RF Model")
        ax.plot(thresholds, nb_all, "k--", linewidth=1.5, label="Treat All")
        ax.plot(thresholds, nb_none, "k:", linewidth=1.5, label="Treat None")
        ax.set_xlabel("Threshold Probability")
        ax.set_ylabel("Net Benefit")
        ax.set_title(f"{name}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Step 5+: Decision Curve Analysis (DCA)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/figS6_dca.png", dpi=300, facecolor="white")
    plt.close()
    print("\n✅ DCA figure saved")

# ============================================================
# 6. Save All Results
# ============================================================
all_results = {
    "rcs": rcs_results,
    "lmem": lmem_results,
    "cox": cox_results,
    "ml": {k: {kk: vv for kk, vv in v.items() if not callable(vv)} for k, v in ml_results.items()},
}

with open(os.path.join(TABLES_DIR, "v11b_all_results.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print("\n" + "=" * 70)
print("V11B ENHANCED PIPELINE COMPLETE")
print("=" * 70)
print(f"Results: {TABLES_DIR}")
print(f"Figures: {FIGURES_DIR}")
for f in sorted(os.listdir(FIGURES_DIR)):
    print(f"  ✅ {f}")
