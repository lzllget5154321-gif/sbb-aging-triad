#!/usr/bin/env Rscript
# ============================================================================
# Causal Forest 扩展分析 — 跨库验证 + 倾向修剪 + 交互项 + 合并库
# ============================================================================
library(grf)
library(data.table)
library(dplyr)

PROJECT_ROOT <- "D:/科研相关项目/程全老师课题组--UKB组/第三个课题--脑体感官衰老耦合解耦研究"
TABLES_DIR <- file.path(PROJECT_ROOT, "results", "tables")
FIGURES_DIR <- file.path(PROJECT_ROOT, "results", "figures")
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# Helper function: run causal forest on one dataset
# ============================================================================
run_causal_forest <- function(df, cohort_name, X_vars) {
  cat(sprintf("\n%s\n", paste(rep("=", 60), collapse="")))
  cat(sprintf("  Causal Forest: %s (N=%d, DSI=%.1f%%)\n",
              cohort_name, nrow(df), mean(df$W)*100))
  cat(sprintf("%s\n", paste(rep("=", 60), collapse="")))

  W <- df$W
  Y <- df$Y
  X <- as.matrix(df[, X_vars, with=FALSE])
  X <- X[, apply(X, 2, sd) > 1e-8, drop=FALSE]  # Remove constant columns

  if (ncol(X) < 3 || sum(W==1) < 20) {
    cat("  SKIP: insufficient data (N_DSI < 20 or < 3 covariates)\n")
    return(NULL)
  }

  set.seed(42)
  cf <- causal_forest(
    X = X, Y = Y, W = W,
    num.trees = 2000, min.node.size = 5,
    sample.fraction = 0.5, honesty = TRUE,
    tune.parameters = "all"
  )

  tau_hat <- predict(cf)$predictions
  ate <- average_treatment_effect(cf, target.sample = "treated")

  # Variable importance
  var_imp <- variable_importance(cf)
  vi_names <- colnames(X)
  vi_df <- data.frame(Variable=vi_names, Importance=var_imp)
  vi_df <- vi_df[order(-vi_df$Importance), ]

  cat(sprintf("  ATE: %.3f (SE=%.3f, p=%.4f)\n",
              ate[1], ate[2], 2*pnorm(-abs(ate[1]/ate[2]))))
  cat(sprintf("  CATE range: [%.3f, %.3f], mean=%.3f\n",
              min(tau_hat), max(tau_hat), mean(tau_hat)))
  cat("  Top 5 variables:\n")
  for (i in 1:min(5, nrow(vi_df))) {
    cat(sprintf("    %d. %-15s %.4f\n", i, vi_df$Variable[i], vi_df$Importance[i]))
  }

  # Calibration test
  suppressWarnings({
    ct <- test_calibration(cf)
  })
  cat(sprintf("  Calibration: mean.pred=%.3f(p=%.3f), diff.pred=%.3f(p=%.3f)\n",
              ct[1,1], ct[1,4], ct[2,1], ct[2,4]))

  list(cf=cf, tau=tau_hat, ate=ate, vi=vi_df, calib=ct, X=X, Y=Y, W=W)
}

# ============================================================================
# Helper: propensity trimming
# ============================================================================
run_with_trimming <- function(df, cohort_name, X_vars, trim_threshold=0.05) {
  cat(sprintf("\n--- %s with propensity trimming (threshold=%.2f) ---\n",
              cohort_name, trim_threshold))

  W <- df$W; Y <- df$Y
  X <- as.matrix(df[, X_vars, with=FALSE])
  X <- X[, apply(X, 2, sd) > 1e-8, drop=FALSE]

  # Estimate propensity
  pf <- regression_forest(X, W, num.trees=500)
  W_hat <- predict(pf)$predictions
  W_hat <- pmax(pmin(W_hat, 0.99), 0.01)

  # Trim extreme propensities
  keep <- (W_hat >= trim_threshold) & (W_hat <= (1 - trim_threshold))
  cat(sprintf("  Before trim: N=%d, After trim: N=%d (%.1f%% retained)\n",
              nrow(df), sum(keep), mean(keep)*100))

  if (sum(keep) < 100 || sum(W[keep]==1) < 10) {
    cat("  SKIP: insufficient data after trimming\n")
    return(NULL)
  }

  X_t <- X[keep, , drop=FALSE]
  Y_t <- Y[keep]; W_t <- W[keep]

  set.seed(42)
  cf_t <- causal_forest(
    X = X_t, Y = Y_t, W = W_t,
    num.trees = 2000, min.node.size = 5,
    sample.fraction = 0.5, honesty = TRUE
  )

  tau_t <- predict(cf_t)$predictions
  ate_t <- average_treatment_effect(cf_t, target.sample = "treated")
  vi_t <- variable_importance(cf_t)

  cat(sprintf("  ATE (trimmed): %.3f (SE=%.3f)\n", ate_t[1], ate_t[2]))
  cat(sprintf("  CATE range: [%.3f, %.3f]\n", min(tau_t), max(tau_t)))

  vi_names <- colnames(X)
  vi_t_df <- data.frame(Variable=vi_names, Importance=vi_t)
  vi_t_df <- vi_t_df[order(-vi_t_df$Importance), ]
  cat("  Top 3 (trimmed):\n")
  for (i in 1:min(3, nrow(vi_t_df))) {
    cat(sprintf("    %d. %-15s %.4f\n", i, vi_t_df$Variable[i], vi_t_df$Importance[i]))
  }

  list(cf=cf_t, tau=tau_t, ate=ate_t, vi=vi_t_df, keep=keep)
}

# ============================================================================
# Helper: traditional interaction validation
# ============================================================================
interaction_validation <- function(df, cohort_name, top_vars, n_vars=3) {
  cat(sprintf("\n--- %s: Interaction term validation ---\n", cohort_name))

  vars_to_test <- top_vars[1:min(n_vars, length(top_vars))]
  results <- list()

  for (v in vars_to_test) {
    x <- df[[v]]
    x_scaled <- scale(x)

    # Model 1: Y ~ W + X
    m1 <- lm(Y ~ W + x_scaled, data=df)
    # Model 2: Y ~ W + X + W:X
    m2 <- lm(Y ~ W * x_scaled, data=df)

    # F-test for interaction
    ft <- anova(m1, m2)
    p_int <- ft[2, "Pr(>F)"]
    coef_int <- coef(m2)["W:x_scaled"]

    cat(sprintf("  %-15s: interaction=%.3f, F-test p=%.4f %s\n",
                v, coef_int, p_int,
                ifelse(p_int < 0.05, "*", ifelse(p_int < 0.10, ".", ""))))

    results[[v]] <- list(coef=coef_int, p=p_int)
  }
  results
}

# ============================================================================
# MAIN ANALYSIS
# ============================================================================
cat("========================================\n")
cat("EXTENDED CAUSAL FOREST ANALYSIS\n")
cat("========================================\n")

# Covariates for analysis
X_vars_common <- c("age", "female", "educ", "cog_base", "chronic",
                   "depression", "smoker", "drinker", "phys_act", "sr_health")

ALL_RESULTS <- list()

# ============================================================================
# 1. CHARLS — Enhanced with trimming + interaction
# ============================================================================
cat("\n\n########## 1. CHARLS (Enhanced) ##########\n")
df_c <- fread(file.path(TABLES_DIR, "charls_causal_forest_harmonized.csv"))
available_c <- X_vars_common[X_vars_common %in% names(df_c)]

# 1a. Standard
res_c <- run_causal_forest(df_c, "CHARLS", available_c)
ALL_RESULTS[["CHARLS"]] <- list(std=res_c)

# 1b. Propensity trimming
res_ct <- run_with_trimming(df_c, "CHARLS", available_c)
ALL_RESULTS[["CHARLS"]][["trimmed"]] <- res_ct

# 1c. Interaction validation
if (!is.null(res_c)) {
  top3_c <- res_c$vi$Variable[1:3]
  int_c <- interaction_validation(df_c, "CHARLS", top3_c)
  ALL_RESULTS[["CHARLS"]][["interactions"]] <- int_c
}

# ============================================================================
# 2. HRS
# ============================================================================
cat("\n\n########## 2. HRS ##########\n")
df_h <- fread(file.path(TABLES_DIR, "hrs_causal_forest_harmonized.csv"))
available_h <- X_vars_common[X_vars_common %in% names(df_h)]
res_h <- run_causal_forest(df_h, "HRS", available_h)
ALL_RESULTS[["HRS"]] <- list(std=res_h)

if (!is.null(res_h)) {
  tryCatch({
    run_with_trimming(df_h, "HRS", available_h)
  }, error=function(e) cat("  Trimming failed:", e$message, "\n"))

  top3_h <- res_h$vi$Variable[1:3]
  tryCatch({
    interaction_validation(df_h, "HRS", top3_h)
  }, error=function(e) cat("  Interaction failed:", e$message, "\n"))
}

# ============================================================================
# 3. KLoSA
# ============================================================================
cat("\n\n########## 3. KLoSA ##########\n")
df_k <- fread(file.path(TABLES_DIR, "klosa_causal_forest_harmonized.csv"))
available_k <- X_vars_common[X_vars_common %in% names(df_k)]
res_k <- run_causal_forest(df_k, "KLoSA", available_k)
ALL_RESULTS[["KLoSA"]] <- list(std=res_k)

if (!is.null(res_k)) {
  top3_k <- res_k$vi$Variable[1:3]
  tryCatch({
    interaction_validation(df_k, "KLoSA", top3_k)
  }, error=function(e) cat("  Interaction failed:", e$message, "\n"))
}

# ============================================================================
# 4. MHAS (>=4 threshold)
# ============================================================================
cat("\n\n########## 4. MHAS ##########\n")
df_m <- fread(file.path(TABLES_DIR, "mhas_causal_forest_harmonized.csv"))
available_m <- X_vars_common[X_vars_common %in% names(df_m)]
res_m <- run_causal_forest(df_m, "MHAS", available_m)
ALL_RESULTS[["MHAS"]] <- list(std=res_m)

if (!is.null(res_m)) {
  top3_m <- res_m$vi$Variable[1:3]
  tryCatch({
    interaction_validation(df_m, "MHAS", top3_m)
  }, error=function(e) cat("  Interaction failed:", e$message, "\n"))
}

# ============================================================================
# 5. POOLED — 4 cohorts (CHARLS + HRS + KLoSA + MHAS)
# ============================================================================
cat("\n\n########## 5. POOLED (4 cohorts) ##########\n")
df_pool <- fread(file.path(TABLES_DIR, "pooled_4cohort_causal_forest.csv"))

# Use Y_z (within-cohort z-score) for pooled analysis
if ("Y_z" %in% names(df_pool)) {
  df_pool$Y <- df_pool$Y_z
}

available_p <- X_vars_common[X_vars_common %in% names(df_pool)]
res_pool <- run_causal_forest(df_pool, "POOLED (CHARLS+HRS+KLoSA+MHAS)", available_p)
ALL_RESULTS[["POOLED"]] <- list(std=res_pool)

if (!is.null(res_pool)) {
  # Propensity trimming on pooled
  run_with_trimming(df_pool, "POOLED", available_p)

  # Interaction validation
  top3_p <- res_pool$vi$Variable[1:3]
  interaction_validation(df_pool, "POOLED", top3_p)
}

# ============================================================================
# 6. Cross-cohort comparison figures
# ============================================================================
cat("\n\n########## 6. Cross-Cohort Comparison ##########\n")

# Collect ATE estimates
ate_comparison <- data.frame(
  Cohort = character(),
  ATE = numeric(), SE = numeric(),
  N = integer(), DSI_pct = numeric(),
  stringsAsFactors = FALSE
)

# Collect ATE estimates
ate_list <- list()
for (nm in names(ALL_RESULTS)) {
  res <- ALL_RESULTS[[nm]]
  if (is.null(res) || is.null(res[["std"]])) next
  r <- res[["std"]]
  ate_list[[length(ate_list) + 1]] <- data.frame(
    Cohort = nm, ATE = r$ate[1], SE = r$ate[2],
    N = length(r$W), DSI_pct = mean(r$W) * 100,
    stringsAsFactors = FALSE
  )
}
ate_df <- do.call(rbind, ate_list)

# ATE forest plot
png(file.path(FIGURES_DIR, "CausalForest_CrossCohort_ATE.png"),
    width = 900, height = 600, res = 100)
par(mar = c(5, 10, 3, 2))
plot(ate_df$ATE, 1:nrow(ate_df),
     xlim = range(c(ate_df$ATE - 2*ate_df$SE,
                    ate_df$ATE + 2*ate_df$SE), na.rm=TRUE),
     pch = 19, col = "#E74C3C", cex = 1.5,
     xlab = "ATE (DSI -> BAI)", ylab = "",
     yaxt = "n", main = "Cross-Cohort: DSI Effect on Brain Aging")
abline(v = 0, lty = 2, col = "gray50")
for (i in 1:nrow(ate_df)) {
  segments(ate_df$ATE[i] - 1.96*ate_df$SE[i], i,
           ate_df$ATE[i] + 1.96*ate_df$SE[i], i,
           lwd = 2, col = "#E74C3C")
}
axis(2, at = 1:nrow(ate_df),
     labels = sprintf("%s\n(N=%s, DSI=%.1f%%)",
                      ate_df$Cohort,
                      format(ate_df$N, big.mark=","),
                      ate_df$DSI_pct),
     las = 1, cex.axis = 0.8)
dev.off()
cat("  [OK] Saved: CausalForest_CrossCohort_ATE.png\n")

# Variable importance comparison
# Collect top-3 per cohort
vi_tables <- list()
for (cohort_name in names(ALL_RESULTS)) {
  res <- ALL_RESULTS[[cohort_name]]$std
  if (is.null(res)) next
  vi_tables[[cohort_name]] <- res$vi
}

# Write comparison table
vi_compare <- data.frame(Rank = 1:3)
for (nm in names(vi_tables)) {
  vi_compare[[nm]] <- paste0(vi_tables[[nm]]$Variable[1:3],
                              " (", round(vi_tables[[nm]]$Importance[1:3], 3), ")")
}
write.csv(vi_compare, file.path(TABLES_DIR, "causal_forest_cross_cohort_vi_comparison.csv"),
          row.names = FALSE)
cat("  [OK] Saved: cross_cohort_vi_comparison.csv\n")

# ============================================================================
# 7. Summary
# ============================================================================
cat("\n\n========================================\n")
cat("EXTENDED ANALYSIS COMPLETE\n")
cat("========================================\n")
cat("\nCross-Cohort ATE Summary:\n")
print(ate_df, row.names=FALSE)

cat("\nVariable Importance Comparison (Top 3 per cohort):\n")
print(vi_compare, row.names=FALSE)

cat(sprintf("\nOutput directory: %s\n", FIGURES_DIR))
