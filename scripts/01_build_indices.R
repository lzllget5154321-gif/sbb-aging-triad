#!/usr/bin/env Rscript
# ============================================================
# 01_build_indices.R — SAI/BAI/BoAI 三大系统老化指数构建
# SBB Aging Triad Project | v1.0 | 2026-06-09
# ============================================================

source("scripts/00_setup_environment.R")

cat("\n=== Building Sensory-Brain-Body Aging Indices ===\n")

# ---- 1. Helper functions ----

#' Build Sensory Aging Index (SAI)
#' @param df data.frame with vision_impairment and hearing_impairment columns
#' @return data.frame with added SAI and DSI columns
build_SAI <- function(df, cohort_name = "") {
  cat("  Building SAI for", cohort_name, "...\n")

  df <- df %>%
    mutate(
      vision_imp = case_when(
        vision_impairment %in% c(1, "yes", "poor", "fair") ~ 1,
        vision_impairment %in% c(0, "no", "good", "excellent") ~ 0,
        TRUE ~ NA_real_
      ),
      hearing_imp = case_when(
        hearing_impairment %in% c(1, "yes", "poor", "fair") ~ 1,
        hearing_impairment %in% c(0, "no", "good", "excellent") ~ 0,
        TRUE ~ NA_real_
      ),
      dsi = as.integer(vision_imp == 1 & hearing_imp == 1),
      SAI_raw = (vision_imp + hearing_imp) / 2
    )

  # Standardize to 0-100 (higher = more sensory aging)
  sai_z <- scale(df$SAI_raw)
  df$SAI <- (sai_z - min(sai_z, na.rm = TRUE)) /
            (max(sai_z, na.rm = TRUE) - min(sai_z, na.rm = TRUE)) * 100

  cat("    DSI prevalence:", round(mean(df$dsi, na.rm = TRUE) * 100, 1), "%\n")
  df
}

#' Build Brain Aging Index (BAI)
build_BAI <- function(df, cohort_name = "") {
  cat("  Building BAI for", cohort_name, "...\n")

  # Cognitive composite — varies by cohort
  # Fallback: use available cognitive score directly
  if("global_cognition" %in% names(df)) {
    bai_z <- scale(df$global_cognition)
  } else if("mmse_total" %in% names(df)) {
    bai_z <- scale(df$mmse_total)
  } else {
    stop("No cognitive measure found for ", cohort_name)
  }

  df$BAI <- (bai_z - min(bai_z, na.rm = TRUE)) /
            (max(bai_z, na.rm = TRUE) - min(bai_z, na.rm = TRUE)) * 100
  df
}

#' Build Body Aging Index (BoAI) — Core version
build_BoAI <- function(df, cohort_name = "") {
  cat("  Building BoAI for", cohort_name, "...\n")

  components <- c()
  if("adl_sum" %in% names(df)) components <- c(components, "adl_sum")
  if("iadl_sum" %in% names(df)) components <- c(components, "iadl_sum")
  if("chronic_count" %in% names(df)) components <- c(components, "chronic_count")
  if("self_rated_health" %in% names(df)) components <- c(components, "self_rated_health")

  if(length(components) < 3) {
    warning("  ⚠️ Only ", length(components), " BoAI components available for ", cohort_name)
  }

  # Z-score each and average
  z_scores <- sapply(df[components], scale)
  df$BoAI_raw <- rowMeans(z_scores, na.rm = TRUE)
  df$BoAI <- (df$BoAI_raw - min(df$BoAI_raw, na.rm = TRUE)) /
             (max(df$BoAI_raw, na.rm = TRUE) - min(df$BoAI_raw, na.rm = TRUE)) * 100
  df
}

# ---- 2. Process each cohort ----

cohorts <- c("CHARLS", "HRS", "SHARE", "KLoSA", "MHAS")
results <- list()

for(cohort in cohorts) {
  cat("\n--- Processing", cohort, "---\n")

  # Load data (adapt path as needed)
  data_path <- file.path(PATHS$data_raw, cohort)
  # df <- read_dta(file.path(data_path, "...")))  # adapt to actual file format

  # For now: placeholder
  cat("  ⚠️ Data not yet available. Using placeholder structure.\n")
  cat("  Expected variables: vision_impairment, hearing_impairment,\n")
  cat("    global_cognition, adl_sum, iadl_sum, chronic_count, self_rated_health\n")
}

cat("\n=== Indices building complete ===\n")
cat("Next step: Run 02_lca_phenotypes.R\n")
