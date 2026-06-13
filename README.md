# Sensory-Brain-Body Aging Triad (感官-脑-体三元衰老研究)

> **Sensory-Brain-Body Aging Triad: Decoupling Patterns, Social Mediation, and Cross-National Determinants of Cognitive Resilience**
>
> CHARLS · HRS · SHARE · KLoSA · MHAS · CLHLS | ~82,000 participants | ≥2 waves follow-up

---

## Quick Start

```bash
# 1. Read the full research plan
cat docs/研究方案_感官脑体三元衰老_v2.0.md

# 2. Check data readiness
ls data_raw/

# 3. Run CHARLS pilot analysis
Rscript scripts/00_setup_environment.R
Rscript scripts/01_build_indices.R
Rscript scripts/02_lca_phenotypes.R
Rscript scripts/03_gbtm_trajectories.R
Rscript scripts/04_causal_mediation.R
Rscript scripts/05_causal_forest.R
python scripts/07_xgboost_shap.py
```

---

## Project Structure

```
第三个课题--脑体感官衰老耦合解耦研究/
├── README.md                          # ← 你在这
├── START_HERE.md                      # 新对话启动指南
├── .gitignore
│
├── docs/                              # 📋 研究文档
│   ├── 研究方案_感官脑体三元衰老_v2.0.md    # 完整研究方案（v2.0深度重构版）
│   ├── 变量映射矩阵_6库harmonization.md    # 六库变量精确映射
│   ├── 方法学详细规格.md                   # GBTM/LCA/CMA/CF/XGBoost详细参数
│   └── 发表策略与目标期刊分析.md            # 分层发表计划
│
├── data_raw/                          # 📊 原始数据（从E:\各类指导文件\new数据库\new数据库\解压后）
│   ├── CHARLS/
│   ├── HRS/
│   ├── SHARE/
│   ├── KLoSA/
│   ├── MHAS/
│   └── CLHLS/
│
├── scripts/                           # 💻 分析脚本
│   ├── 00_setup_environment.R         # 环境配置+包安装
│   ├── 01_build_indices.R             # SAI/BAI/BoAI三大指数构建
│   ├── 02_lca_phenotypes.R            # LCA表型聚类（Aim 1）
│   ├── 03_gbtm_trajectories.R         # GBTM纵向轨迹（Aim 1）
│   ├── 04_causal_mediation.R          # 因果中介分析（Aim 2）
│   ├── 05_markov_transitions.R        # 多状态Markov模型（Aim 3）
│   ├── 06_causal_forest.R             # 因果森林+效应异质性（Aim 3）
│   ├── 07_xgboost_shap.py             # XGBoost+SHAP+跨国验证（Aim 4）
│   ├── 08_meta_analysis.R             # 跨国Meta-analysis（Aim 2-4）
│   ├── 09_cross_national_validation.R # 跨国外部验证
│   └── utils/
│       ├── harmonize_functions.R      # 变量harmonization函数库
│       ├── visualization_theme.R      # 统一可视化主题
│       └── table_one.R               # Table 1自动生成
│
├── results/                           # 📈 分析结果
│   ├── tables/
│   │   ├── table1_baseline.csv
│   │   ├── lca_phenotypes.csv
│   │   ├── mediation_results.csv
│   │   └── meta_results.csv
│   └── figures/
│       ├── figure1_radar_phenotypes.png
│       ├── figure2_trajectories.png
│       ├── figure3_mediation_path.png
│       ├── figure4_causal_forest.png
│       ├── figure5_shap_summary.png
│       └── figure6_cross_national.png
│
├── manuscript/                        # ✍️ 手稿
│   ├── main_text.md
│   ├── figures/
│   ├── supplementary/
│   └── cover_letter.md
│
├── literature/                        # 📚 参考文献
│   ├── key_papers.md                  # 200+篇核心文献数据库
│   └── reading_notes/                 # 精读笔记
│
└── validation/                        # ✅ 验证与测试
    ├── test_scripts.md                # 脚本测试记录
    ├── code_review_checklist.md       # 代码审查清单
    └── simulation_report.md           # 模拟验证报告
```

---

## Core Scientific Question

> **Are sensory, brain, and body aging synchronized or decoupled?**
>
> Is sensory impairment the "canary in the coal mine" for brain aging?
> Does social isolation mediate the sensory→cognitive pathway (~40% effect)?
> What modifiable lifestyle factors promote phenotype transitions from "global aging" to "cognitive resilience"?

---

## Five Aging Phenotypes (LCA)

```
         Body Young          Body Old
       ┌──────────────┬──────────────┐
Brain  │ 🟢 Type A     │ 🟡 Type C    │ ← Brain-Resilient
Young  │ Successful    │ Brain-       │   (Body aged but
       │ Aging         │ Resilient    │    cognition young)
       ├──────────────┼──────────────┤
Brain  │ 🟡 Type D     │ 🔴 Type E    │
Old    │ Body-         │ Global       │
       │ Resilient     │ Accelerated  │
       └──────────────┴──────────────┘
       + Sensory dimension overlay:
       🟡 Type B: Sensory-First Decline (sensory ↓, brain/body OK)
```

---

## Key Innovations

| # | Innovation | Status Quo | Our Breakthrough |
|:--:|-----------|-----------|-----------------|
| 1 | **Triadic Aging Theory** | Single-domain aging | First integration of sensory-brain-body as unified phenotype system |
| 2 | **"Sensory Sentinel" Concept** | Sensory→Cognition unidirectional | Sensory as earliest behavioral biomarker of brain aging |
| 3 | **Resilience Framework** | "Who declines faster?" | "Who stays cognitively intact despite sensory loss?" |
| 4 | **Causal Forest Personalization** | Average treatment effects | "Who benefits most from what intervention?" |
| 5 | **Cross-National AI Transportability** | Single-country ML models | 5-country independent training + external validation |

---

## Data Summary

| Database | Country | Waves | Age | Est. Sample | Unique Strength |
|----------|---------|-------|-----|:-----------:|----------------|
| CHARLS | 🇨🇳 China | 5 | 45+ | ~10,000 | Richest physical function (grip/gait/balance/sarcopenia) |
| HRS | 🇺🇸 US | 15 | 50+ | ~13,000 | Longest follow-up; genomics sub-study |
| SHARE | 🇪🇺 28 EU | 9 | 50+ | ~45,000 | Largest sample; cross-national policy variation |
| KLoSA | 🇰🇷 Korea | 8 | 45+ | ~6,000 | East Asian comparison; employment history |
| MHAS | 🇲🇽 Mexico | 6 | 50+ | ~8,000 | Latin American representation |
| CLHLS | 🇨🇳 China | 8 | 65+ | ~100,000 | 🔬 40+ blood biomarkers for validation |

---

## Target Journals

| Journal | IF | Rationale |
|---------|-----|-----------|
| **Nature Communications** | 14.7 | Cross-national design + AI methods + large sample |
| **Lancet Healthy Longevity** | 13.1 | Aging/longevity theme perfect match |
| **Neurology** | 9.9 | Brain aging + sensory neuroscience |
| **JAMA Neurology** | 29.0 | Highest IF neuro journal; strong methods needed |

---

## Timeline

| Phase | Weeks | Milestone |
|-------|-------|-----------|
| **0: Data Prep** | 1-2 | Unzip 6 databases + variable harmonization |
| **1: CHARLS Pilot** | 3-6 | LCA + GBTM + mediation |
| **2: Multi-Cohort** | 7-12 | HRS/SHARE/KLoSA/MHAS replication |
| **3: AI Modeling** | 13-16 | XGBoost+SHAP+Causal Forest+Meta |
| **4: Manuscript** | 17-20 | Full manuscript + submission |
| **5: CLHLS Sub** | 21-24 | Blood biomarker validation |

---

## Quick Links

- 📖 [Full Research Plan v2.0](docs/研究方案_感官脑体三元衰老_v2.0.md)
- 📊 [Variable Mapping Matrix](docs/变量映射矩阵_6库harmonization.md)
- 📚 [Literature Database](literature/key_papers.md) (200+ papers)
- 🧪 [Method Specifications](docs/方法学详细规格.md)
- 📝 [Wiki Research Page](../../../wiki/洞察/感官脑体三元衰老课题研究方案.md)

---

*Project initialized: 2026-06-09 | v2.0 Deep Restructure*
