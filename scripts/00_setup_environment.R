#!/usr/bin/env Rscript
# ============================================================
# 00_setup_environment.R — 环境配置与包安装
# SBB Aging Triad Project | v1.0 | 2026-06-09
# ============================================================

cat("\n=== SBB Aging Triad: Environment Setup ===\n\n")

# ---- 1. Package list ----
packages <- c(
  # Core data manipulation
  "tidyverse", "data.table", "haven", "lubridate",
  # LCA & clustering
  "poLCA", "tidySEM", "mclust",
  # Trajectory modeling
  "lcmm", "traj",
  # Mediation
  "lavaan", "mediation", "medsens",
  # Causal inference & ML
  "grf", "econml",  # causal forest
  # Multistate Markov
  "msm", "mstate",
  # Meta-analysis
  "metafor",
  # Joint models
  "JMbayes2",
  # Visualization
  "ggplot2", "ggpubr", "patchwork", "ggalluvial", "ggradar",
  "corrplot", "ComplexHeatmap", "RColorBrewer",
  # Table output
  "gtsummary", "flextable", "kableExtra",
  # Reporting
  "rmarkdown", "knitr"
)

# ---- 2. Install missing packages ----
installed <- rownames(installed.packages())
to_install <- packages[!packages %in% installed]

if(length(to_install) > 0) {
  cat("Installing", length(to_install), "missing packages...\n")
  install.packages(to_install, repos = "https://cloud.r-project.org")
}

# ---- 3. Load all packages ----
invisible(lapply(packages, library, character.only = TRUE))

# ---- 4. Global settings ----
options(
  scipen = 999,
  digits = 3,
  stringsAsFactors = FALSE,
  mc.cores = parallel::detectCores() - 1
)

set.seed(20260609)  # Project seed

# ---- 5. Paths ----
PATHS <- list(
  data_raw = "data_raw",
  results = "results",
  figures = "results/figures",
  tables  = "results/tables"
)

for(p in PATHS) dir.create(p, showWarnings = FALSE, recursive = TRUE)

# ---- 6. Custom functions ----
source("scripts/utils/harmonize_functions.R")

cat("\n✅ Environment ready. Project paths:\n")
str(PATHS)
cat("\nLoaded packages:", length(packages), "\n")
cat("R version:", R.version.string, "\n")
