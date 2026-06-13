# STATUS.md — UKB 课题三：脑体感官衰老耦合解耦研究

> **基线建立**：2026-06-12 | Phase 0 知识库全功能激活计划
> **追踪目标**：从当前的 "投稿不可" 状态 → 投稿就绪状态
> **关联方案**：[[洞察/知识库工作流潜力深度开发方案]]

---

## 当前里程碑

| 里程碑 | 状态 | 完成日期 | 备注 |
|------|:--:|------|------|
| 研究方案设计 | ✅ | 2026-06-09 | v1.0 完整方案（730行，12节，4 Aim） |
| 六库变量映射 | ✅ | 2026-06-09 | CHARLS/HRS/SHARE/KLoSA/MHAS/CLHLS |
| 分析脚本开发 | ✅ | 2026-06-12 | 28个脚本（R+Python） |
| 多库分析执行 | ✅ | 2026-06-11 | 五库横向对比报告 |
| 稿件初稿 | ✅ | 2026-06-12 | 22页，21引用，评分 6.0/10 |
| 内部审稿 | ✅ | 2026-06-12 | 5 P0 / 7 P1 / 5 P2 |
| 🔴 P0修复 | ⏳ | — | **5项阻塞** |
| 🟡 P1改进 | ⏳ | — | 7项建议 |
| 🟢 P2打磨 | ⏳ | — | 5项润色 |
| **投稿就绪** | ❌ | 目标：1-2周 | 需先清空P0 |

---

## P0 阻塞问题（投稿前必改）

| # | 问题 | 严重性 | 修复状态 | 预计工作量 |
|:--:|------|:--:|:--:|:--:|
| P0-1 | GBTM轨迹结果与内部验证不一致（Section 3.3报告"加速认知衰老10.7%"但v3验证证实为噪声伪影） | 🔴 致命 | ⏳ | 3-5小时 |
| P0-2 | Causal Forest单库校准不显著(p=0.888)但亚组CATE作为主要发现呈现 | 🔴 严重 | ⏳ | 2-3小时 |
| P0-3 | XGBoost外部验证AUC 0.44-0.66——模型无运输能力 | 🔴 严重 | ⏳ | 2-3小时 |
| P0-4 | 代码和数据可用性缺失（声明"接受后提供"不可接受） | 🔴 前提 | 🟡 修复路径已明确 | 1-2小时 (用户需上传GitHub/Zenodo——路径已写入data_availability.tex) |
| P0-5 | 参考文献仅21篇（目标期刊预期40-70篇） | 🔴 严重 | ✅ 已达标 | 46条 (33 DOI, 72%) — Phase 1 完成 |

---

## 项目资产清单

### 分析脚本（28个）

| 文件 | 语言 | 功能 | 状态 |
|------|:--:|------|:--:|
| 00_setup_environment.R | R | 环境配置+包安装 | ✅ |
| 01_build_indices.R | R | SAI/BAI/BoAI三大指数构建 | ✅ |
| 03_gbtm_trajectories.py | Python | GBTM纵向轨迹分析 | ⚠️ v2→v3修正 |
| 03_gbtm_trajectories_v2.py | Python | GBTM v2（噪声版本） | ❌ 已证伪 |
| 03_gbtm_trajectories_v3.py | Python | GBTM v3（诚实版本） | ✅ |
| 03b_validate_gbtm.py | Python | GBTM验证（E1/E2/E3） | ✅ |
| 07_xgboost_shap.py | Python | XGBoost+SHAP | ⚠️ AUC问题 |
| 08_generate_figures.py | Python | 图表生成 | ✅ |
| causal_forest_analysis.R | R | Causal Forest 单库 | ⚠️ p=0.888 |
| causal_forest_extended.R | R | Causal Forest 四库合并 | ✅ p<0.001 |
| causal_forest_prep.py/v2.py | Python | CF数据准备 | ✅ |
| ipw_robustness_check.R | R | IPW稳健性 | ✅ |
| charls_full_analysis.py | Python | CHARLS完整分析 | ✅ |
| hrs_full_analysis.py | Python | HRS完整分析 | ✅ |
| share_full_analysis.py | Python | SHARE完整分析 | ✅ |
| mhas_reanalysis.py | Python | MHAS重分析 | ✅ |
| klosa_diagnostic.py | Python | KLoSA诊断 | ✅ |
| final_5country.py/v2.py | Python | 五国汇总 | ✅ |
| unified_5cohort_final.py | Python | 五库统一 | ✅ |
| multi_cohort_replication.py | Python | 多库复现 | ✅ |
| cross_cohort_prep.py | Python | 跨库准备 | ✅ |
| fix_all_cohorts.py | Python | 全库修正 | ✅ |
| charls_pilot.py/v2.py | Python | CHARLS试点 | ✅ |
| utils/harmonize_functions.R | R | 变量映射工具 | ✅ |
| utils/visualization_theme.R | R | 可视化主题 | ✅ |

### 结果产出（139个文件）
- 70+ 图表（PNG/PDF）
- 30+ 数据表（CSV/JSON）
- 报告：CHARLS完整分析、五库横向对比汇总

### 稿件组件（42个文件）
- 主稿件：main_text.md + LaTeX源码（6节）
- 补充材料：supplementary PDF
- 投稿材料：cover letter, ethics, data/code availability
- 6张主图 + 参考文献bib

### 文献资产
- key_papers.md：200+篇文献数据库（18主题）
- 30篇核心论文阅读计划（0/30完成）

---

## 知识库工作流使用记录

> 🔴 以下记录本次全功能激活计划中使用的方法——用于追踪哪些进化方法被首次使用。

| 进化方法 | 使用状态 | 首次使用日期 | 备注 |
|------|:--:|------|------|
| anti-hallucination V2 (HalluRef引用验证) | ⏳ 待使用 | — | Phase 1 |
| stop-slop-zh AI味审校 | ⏳ 待使用 | — | Phase 2 |
| academic-writing 全管道 | ⏳ 待使用 | — | Phase 2 |
| journal-select 选刊匹配 | ⏳ 待使用 | — | Phase 3 |
| grant-writing 基金标书 | ⏳ 待使用 | — | Phase 3 |
| experimental-design 实验设计 | ⏳ 待使用 | — | Phase 3 |
| Agent Writer+Critic | ⏳ 待使用 | — | Phase 2+5 |
| skill-orchestration 模板6（深度研究） | ⏳ 待使用 | — | Phase 1 |
| skill-orchestration 模板2（论文写作） | ⏳ 待使用 | — | Phase 2 |
| skill-orchestration 模板10（PPT生成） | ⏳ 待使用 | — | Phase 4 |

---

## 变更日志

| 日期 | 变更 | 来源 |
|------|------|------|
| 2026-06-12 | Phase 0 基线建立——STATUS.md 创建 | 知识库全功能激活计划 |
