# Reproducible Pipeline — SBB Aging Triad Study

> **项目**: 脑体感官衰老耦合解耦研究 (Sensory-Body-Brain Aging Triad)
> **投稿目标**: Nature Communications
> **版本**: v1.0 (2026-06-21)
> **脚本数**: 11 (6 Python + 5 R)
> **许可证**: MIT

---

## 项目目的

本研究使用来自四个国家的五项老龄化队列 (CHARLS/HRS/KLoSA/MHAS/SHARE) 的 harmonized 个人数据，
系统检验双感觉损伤 (DSI, 视听同时受损) 与认知衰退之间的关联、中介机制和药物靶点。
分析管道涵盖五步升级框架: 关联分析 → 因果中介 → 机器学习预测 → 轨迹建模 → 药靶桥接。

---

## 目录结构

```
reproducible_pipeline/
├── README.md                          ← 本文件
├── environment.yml                    ← Conda 环境配置 (Python 3.10)
├── requirements.txt                   ← Pip 依赖清单
├── R_requirements.txt                 ← R 包清单 (R ≥ 4.2.0)
├── v11_master_pipeline.py             ← 主管道 (Step 1-5: 关联→中介→ML→轨迹→药靶)
├── v11b_multimediator_sem.py          ← 多中介 SEM (并行+序列, 四库)
├── v11b_drug_target_pipeline.py       ← 药靶桥接管道 (pQTL MR + 六步筛选)
├── v11b_enhanced_pipeline.py          ← 增强 ML (RCS + LMEM + Cox + XGBoost/LightGBM + SHAP + DCA)
├── 03_gbtm_trajectories_v3.py         ← GBTM 轨迹建模 (GMM-based, 方案D诚实版)
├── 03b_validate_gbtm.py               ← GBTM 三验证套件 (稳定性 + Bootstrap + 逆概率)
├── causal_forest_extended.R           ← Causal Forest 跨库 (HTE + 倾向修剪 + 校准)
├── compute_evalues.R                  ← E-value 敏感性分析 (四库未测量混杂评估)
├── 08_missing_imputation.R            ← MICE 多重插补 (SHARE缺失处理, m=20)
├── utils/
│   ├── harmonize_functions.R          ← 五库变量映射函数库
│   └── visualization_theme.R          ← Nature Comms 风格 ggplot2 主题
├── demo_data/                         ← 模拟数据集 (待 Task 39 创建)
└── _archive/                          ← 原始项目脚本归档
```

---



---
## Demo Data (for third-party verification)

A simulated CHARLS-format dataset (N=200) is provided in `demo_data/` for code executability verification:

| File | Description |
|------|-------------|
| `demo_data/charls_demo_full.csv` | N=200 simulated participants, 55 harmonized variables (76 KB) |
| `demo_data/README.md` | Full data dictionary with column definitions |
| `demo_data/generate_demo_data.py` | Reproducible generator script (seed=42) |

**Quick test** (no real data required):

```bash
conda activate sbb-aging-triad
python v11_master_pipeline.py --demo
```

The pipeline automatically detects `demo_data/charls_demo_full.csv` and runs all steps with simulated data. Expected runtime: ~2 min. All five analytical steps execute successfully, producing illustrative output in `demo_output/`. See `demo_data/README.md` for expected console output.

> ⚠️ **WARNING**: Demo data is entirely synthetic (seed=42). Results are illustrative only — NOT for scientific inference. No real participant information is included.
## 数据来源

原始数据因各数据库使用协议限制**未包含在此代码包中**。
第三方需自行向各数据库申请访问权限:

| 数据库 | 国家/地区 | 基线年份 | N (基线) | 数据获取 |
|--------|---------|:--:|:--:|------|
| **CHARLS** | 中国 | 2011 | ~17,708 | [charls.pku.edu.cn](https://charls.pku.edu.cn) — 注册后免费下载 |
| **HRS** | 美国 | 1992 | ~20,000 | [hrs.isr.umich.edu](https://hrs.isr.umich.edu) — 注册申请 |
| **KLoSA** | 韩国 | 2006 | ~10,254 | [survey.keis.or.kr](https://survey.keis.or.kr) — 注册申请 |
| **MHAS** | 墨西哥 | 2001 | ~16,970 | [mhasweb.org](https://mhasweb.org) — 注册后免费下载 |
| **SHARE** | 欧洲多国 | 2004 | ~140,000 | [share-eric.eu](https://share-eric.eu) — 研究申请 |

数据准备后，按 `utils/harmonize_functions.R` 中的变量映射表构建统一的 harmonized 数据集，
放入 `data_derived/` 目录。详细变量定义见稿件 **Supplementary Table S1**。

---

## 环境配置

### Python 环境 (方式A: Conda — 推荐)

```bash
# 创建环境 (~5 min, ~2.5 GB)
conda env create -f environment.yml

# 激活环境
conda activate sbb-aging-triad

# 验证安装
python -c "
import pandas, numpy, scipy, statsmodels, sklearn
import xgboost, lightgbm, shap, lifelines, matplotlib, seaborn
print('All 12 packages OK')
"
```

### Python 环境 (方式B: Pip-only)

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### R 环境

```r
# R ≥ 4.2.0 推荐, RStudio ≥ 2024.04
install.packages(c(
  "grf", "EValue",
  "mice", "naniar", "finalfit", "mitools",
  "data.table", "dplyr", "ggplot2", "jsonlite", "survival"
), repos = "https://cloud.r-project.org")
```

### 依赖总览

| 类别 | Python (conda) | R |
|------|:--:|:--:|
| 数据操作 | pandas, numpy, scipy, pyarrow | data.table, dplyr |
| 统计建模 | statsmodels, lifelines | survival |
| 机器学习 | scikit-learn, xgboost, lightgbm | grf |
| 可视化 | matplotlib, seaborn | ggplot2, naniar |
| 可解释性 | shap | — |
| 敏感性 | — | EValue |
| 缺失插补 | — | mice, finalfit, mitools |
| I/O 工具 | openpyxl, pyyaml | jsonlite |
| 可复现研究 | jupyterlab, ipykernel | — |
| 网络分析 | networkx | — |

---

## 完整分析流程

### Step 1: 数据准备

```bash
# 1a. 加载变量映射函数库
Rscript -e 'source("utils/harmonize_functions.R")'

# 1b. SHARE数据库缺失插补 (MICE, m=20)
#     输入: data_derived/share_raw.csv
#     输出: data_derived/share_imputed.csv + results/tables/missing_pattern.pdf
Rscript 08_missing_imputation.R
```

### Step 2: 主分析管道

```bash
# 2a. 五步升级主管道 (核心分析, ~2-4h)
#     步骤: (1) 关联分析 → (2) 因果中介 → (3) ML预测 → (4) 轨迹建模 → (5) 药靶筛选
#     输入: data_derived/charls_hrs_klosa_mhas_harmonized.csv
#     输出: results/tables/step1_association/ step2_mediation/ step3_ml/ ...
python v11_master_pipeline.py

# 2b. 多中介 SEM (并行+序列, 四库, ~30 min)
#     输入: data_derived/ (同主管道)
#     输出: results/v11b/tables/mediation_bootstrap.csv
python v11b_multimediator_sem.py

# 2c. 增强 ML 分析 (RCS + LMEM + Cox + XGBoost/LightGBM + SHAP + DCA, ~30 min)
#     输入: data_derived/ (同主管道)
#     输出: results/v11b/tables/ + results/v11b/figures/
python v11b_enhanced_pipeline.py
```

### Step 3: 补充分析

```bash
# 3a. GBTM 轨迹建模 (GMM-based, 方案D诚实版, ~10 min)
#     输入: data_derived/charls_longitudinal.csv
#     输出: results/tables/gbtm_trajectories_k5.csv + results/figures/gbtm_*
python 03_gbtm_trajectories_v3.py

# 3b. GBTM 验证套件 (稳定性 + Bootstrap + 逆概率, ~20 min)
#     输入: data_derived/charls_longitudinal.csv
#     输出: results/tables/gbtm_validation_bootstrap.csv
python 03b_validate_gbtm.py

# 3c. Causal Forest 跨库 (HTE + 倾向修剪 + 校准, ~30 min)
#     输入: data_derived/charls_hrs_klosa_mhas_harmonized.csv
#     输出: results/tables/causal_forest_hte.csv + results/figures/cf_*
Rscript causal_forest_extended.R

# 3d. E-value 敏感性分析 (未测量混杂评估, ~2 min)
#     输入: results/tables/step2_mediation_pooled.csv
#     输出: results/tables/evalue_summary.csv
Rscript compute_evalues.R

# 3e. 药靶桥接管道 (独立模块, ~20 min)
#     MR中介 + 六步药靶筛选: 表型中介物 → pQTL MR → colocalization
#     → druggability (DrugBank/ChEMBL) → PheWAS safety → docking
#     输入: data_derived/gwas_dsi_sumstats.csv + data_derived/pqtl_sun2023.csv
#     输出: results/v11/tables/step3_drug_target_pipeline.json
python v11b_drug_target_pipeline.py
```

### Step 4: 可视化与汇总

```bash
# 加载可视化主题, 各脚本内置图表已在上方步骤中生成
Rscript -e 'source("utils/visualization_theme.R")'
```

---

## 预期输出

| 分析步骤 | 输出格式 | 输出位置 | 关键产出 |
|---------|---------|------|------|
| Step 1 关联分析 | CSV, JSON | `results/tables/step1_*` | Pooled β, I², 四库OR/β/CI表 |
| Step 2 因果中介 | CSV | `results/tables/step2_*` | CES-D中介比例, Bootstrap CI |
| Step 3 ML预测 | CSV, PNG | `results/tables/step3_*`, `results/figures/` | AUC, SHAP beeswarm, DCA curves |
| Step 4 轨迹建模 | CSV, PNG | `results/tables/gbtm_*`, `results/figures/gbtm_*` | 5-trajectory profiles, 验证指标 |
| Step 5 药靶 | JSON, CSV | `results/v11/tables/step3_*` | 6-target prioritization scores |
| 多中介 SEM | CSV | `results/v11b/tables/` | Multi-mediator direct/indirect/total |
| 增强 ML | CSV, PNG | `results/v11b/tables/`, `results/v11b/figures/` | RCS curves, LMEM, Cox HR, XGBoost feature importance |
| Causal Forest | CSV, PNG | `results/tables/cf_*`, `results/figures/cf_*` | HTE estimates, calibration plots |
| E-value | CSV | `results/tables/evalue_*` | Point + CI E-values (four cohorts) |
| 缺失插补 | CSV, PDF | `results/tables/missing_*` | Imputed dataset, convergence diagnostics |

---

## 预计资源消耗

| 脚本 | 运行时间 | 内存峰值 | 并行化 |
|------|:--:|:--:|:--:|
| `v11_master_pipeline.py` | 2-4 h | ~8 GB | 否 (顺序执行) |
| `v11b_multimediator_sem.py` | 20-40 min | ~4 GB | 否 |
| `v11b_enhanced_pipeline.py` | 20-40 min | ~6 GB | 否 (XGBoost/LightGBM 内置多线程) |
| `03_gbtm_trajectories_v3.py` | 5-15 min | ~2 GB | 否 |
| `03b_validate_gbtm.py` | 10-25 min | ~3 GB | Bootstrap 可并行 (手动) |
| `v11b_drug_target_pipeline.py` | 10-30 min | ~4 GB | 否 |
| `causal_forest_extended.R` | 15-40 min | ~6 GB | grf 内置多线程 |
| `compute_evalues.R` | 1-3 min | ~0.5 GB | 否 |
| `08_missing_imputation.R` | 5-15 min | ~2 GB | 否 (MICE串行) |
| **全部合计** | **~5-9 h** | **~8 GB** | — |

> **推荐硬件**: 16 GB RAM, 4+ CPU cores, 50 GB 磁盘空间。
> 部分脚本 (GBTM验证Bootstrap, Causal Forest) 可通过修改脚本内的并行参数大幅加速。

---

## 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|------|
| `ModuleNotFoundError: No module named 'xgboost'` | 环境未激活 | `conda activate sbb-aging-triad` |
| `Rscript: command not found` | R 未安装或未加入 PATH | 安装 R (≥4.2.0), 确保 `Rscript` 在 PATH 中 |
| `Error in library(grf): there is no package called 'grf'` | R包未安装 | `install.packages("grf")` |
| `FileNotFoundError: data_derived/...` | 数据未准备 | 按「数据来源」章节获取数据, 按 harmonize_functions.R 构建 |
| `MemoryError` / `std::bad_alloc` | 内存不足 | 减小 Bootstrap 重采样次数 (B=500→100), 关闭其他应用 |
| 运行时间远超预估 | 单核串行执行 | 检查脚本内并行参数 (n_jobs, num.threads) |
| XGBoost/LightGBM 不可用但脚本继续 | 设计如此——脚本有降级逻辑 | 输出中会标注 `[skip]`, 不影响其他模块 |

---

## 引用

若使用此代码包，请引用:

> [Author List]. Sensory-body-brain aging triad: dual sensory impairment and cognitive decline
> across five international cohorts. *Nature Communications* (under review). 2026.
>
> Code & reproducible pipeline: [https://github.com/lzllget5154321-gif/sbb-aging-triad](https://github.com/lzllget5154321-gif/sbb-aging-triad)
> Archived version: [Zenodo DOI — 待 Task 41 填入]

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE) 文件 (待 Task 40 创建)。

Copyright (c) 2026 SBB Aging Triad Study Authors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to use, copy, modify,
merge, publish, and distribute the Software, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

---

*由 SBB课题 Task 36-38 生成 — 2026-06-21*
