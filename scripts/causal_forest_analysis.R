#!/usr/bin/env Rscript
# ============================================================================
# Causal Forest 异质性分析 — CHARLS DSI → BAI
# Aim 3: 感觉障碍对脑衰老效应的异质性来源识别
#
# Treatment: DSI (双感觉障碍, binary)
# Outcome:   BAI (大脑老化指数, 0-100, higher=better)
# Method:    grf::causal_forest()
# ============================================================================

library(grf)
library(data.table)
library(dplyr)

# ============================================================================
# 0. Setup
# ============================================================================
PROJECT_ROOT <- normalizePath("D:/科研相关项目/程全老师课题组--UKB组/第三个课题--脑体感官衰老耦合解耦研究", mustWork=FALSE)
TABLES_DIR <- file.path(PROJECT_ROOT, "results", "tables")
FIGURES_DIR <- file.path(PROJECT_ROOT, "results", "figures")
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

cat("========================================\n")
cat("CHARLS Causal Forest Analysis\n")
cat("========================================\n\n")

# ============================================================================
# 1. Load data
# ============================================================================
cat("1. Loading data...\n")
data_path <- file.path(TABLES_DIR, "charls_causal_forest_data.csv")
df <- fread(data_path)
cat(sprintf("   N = %d, DSI = %.2f%%\n", nrow(df), mean(df$W) * 100))

# ============================================================================
# 2. Define treatment, outcome, covariates
# ============================================================================
cat("\n2. Preparing variables...\n")

# Treatment
W <- df$W

# Outcome
Y <- df$Y

# Covariate matrix (heterogeneity variables)
X_vars <- c("age", "female", "edu_yrs", "cog_base", "chronic",
            "phys_act", "smoker", "drinker", "depression",
            "sr_health", "social_score")
X <- as.matrix(df[, ..X_vars])

cat(sprintf("   Covariates (%d): %s\n", length(X_vars), paste(X_vars, collapse=", ")))
cat(sprintf("   Dimension: %d x %d\n", nrow(X), ncol(X)))

# ============================================================================
# 3. Fit causal forest
# ============================================================================
cat("\n3. Fitting causal_forest...\n")
set.seed(42)

cf <- causal_forest(
  X = X,
  Y = Y,
  W = W,
  num.trees = 2000,
  min.node.size = 5,
  sample.fraction = 0.5,
  honesty = TRUE,
  honesty.fraction = 0.5,
  ci.group.size = 1,
  tune.parameters = "all"
)

cat(sprintf("   Trees: %d\n", cf$`_num.trees`))
cat(sprintf("   Tuned params: alpha=%.4f, imbalance.penalty=%.4f\n",
            cf$tunable.params["alpha"], cf$tunable.params["imbalance.penalty"]))

# ============================================================================
# 4. CATE estimates
# ============================================================================
cat("\n4. Computing CATE...\n")

tau_hat <- predict(cf)$predictions
cat(sprintf("   CATE range: [%.3f, %.3f]\n", min(tau_hat), max(tau_hat)))
cat(sprintf("   CATE mean (SD): %.3f (%.3f)\n", mean(tau_hat), sd(tau_hat)))
cat(sprintf("   Fraction negative CATE: %.1f%%\n", mean(tau_hat < 0) * 100))

# ATE (average treatment effect) — use target.sample due to low DSI prevalence
ate <- average_treatment_effect(cf, target.sample = "treated")
cat(sprintf("\n   ATE (treated) = %.3f (SE=%.3f, 95%% CI: [%.3f, %.3f])\n",
            ate[1], ate[2], ate[1] - 1.96*ate[2], ate[1] + 1.96*ate[2]))

# ============================================================================
# 5. Variable Importance
# ============================================================================
cat("\n5. Variable importance...\n")

var_imp <- variable_importance(cf)
var_imp_df <- data.frame(
  Variable = X_vars,
  Importance = var_imp
)
var_imp_df <- var_imp_df[order(-var_imp_df$Importance), ]

cat("   Variable Importance Ranking:\n")
for (i in 1:nrow(var_imp_df)) {
  cat(sprintf("   %2d. %-15s %.4f\n", i, var_imp_df$Variable[i], var_imp_df$Importance[i]))
}

# Variable importance plot
png(file.path(FIGURES_DIR, "CausalForest_VariableImportance.png"),
    width = 800, height = 600, res = 100)
par(mar = c(4, 10, 3, 2))
bar_cols <- ifelse(var_imp_df$Importance > mean(var_imp_df$Importance),
                   "#E74C3C", "#3498DB")
bp <- barplot(var_imp_df$Importance, horiz = TRUE,
              names.arg = var_imp_df$Variable,
              col = bar_cols, border = NA,
              main = "Causal Forest: Variable Importance\n(DSI -> BAI Heterogeneity)",
              xlab = "Importance", las = 1, cex.names = 0.9)
abline(v = mean(var_imp_df$Importance), lty = 2, col = "gray50")
text(mean(var_imp_df$Importance) + 0.02, max(bp) * 0.9,
     "mean", col = "gray50", cex = 0.7)
dev.off()
cat("   [OK] Saved: CausalForest_VariableImportance.png\n")

# ============================================================================
# 6. Best Linear Projection (BLP)
# ============================================================================
cat("\n6. Best Linear Projection...\n")

# Test if CATE estimates have meaningful heterogeneity
suppressWarnings({
  blp_test <- test_calibration(cf)
})
cat("   Raw calibration test output:\n")
print(blp_test)
cat(sprintf("\n   Mean forest coefficient: %.3f (SE=%.3f, t=%.3f)\n",
            blp_test[2,1], blp_test[2,2], blp_test[2,3]))

# Simple HTE check: estimate ATE by CATE quantile
cate_quint <- ntile(tau_hat, 5)
cat("\n   ATE by CATE quintile:\n")
for (q in 1:5) {
  idx <- which(cate_quint == q)
  if (sum(W[idx]==1) >= 5) {
    ate_q <- tryCatch(average_treatment_effect(cf, subset = idx, target.sample = "treated"),
                       error = function(e) c(NA, NA))
    if (is.na(ate_q[1])) {
      # Fallback: simple difference
      y1 <- mean(Y[idx][W[idx]==1])
      y0 <- mean(Y[idx][W[idx]==0])
      ate_q <- c(y1 - y0, sqrt(var(Y[idx][W[idx]==1])/sum(W[idx]==1) +
                               var(Y[idx][W[idx]==0])/sum(W[idx]==0)))
    }
    cat(sprintf("     Q%d (CATE>%.1f): ATE=%.2f (SE=%.2f), N=%d, DSI=%d\n",
                q, min(tau_hat[idx]), ate_q[1], ate_q[2], length(idx), sum(W[idx]==1)))
  }
}

# ============================================================================
# 7. Partial dependence plots (Top 3 variables)
# ============================================================================
cat("\n7. Partial dependence plots...\n")

top3 <- var_imp_df$Variable[1:3]
cat(sprintf("   Top 3: %s\n", paste(top3, collapse=", ")))

# Custom partial dependence calculation
for (var_name in top3) {
  var_idx <- which(X_vars == var_name)
  x_vals <- X[, var_idx]
  x_grid <- seq(quantile(x_vals, 0.05), quantile(x_vals, 0.95), length.out = 50)

  # For binary variables, use discrete points
  if (length(unique(x_vals)) <= 2) {
    x_grid <- sort(unique(x_vals))
  }

  pd_est <- sapply(x_grid, function(xv) {
    X_temp <- X
    X_temp[, var_idx] <- xv
    mean(predict(cf, newdata = X_temp)$predictions)
  })

  # For binary, also compute by-group CATE
  if (length(unique(x_vals)) <= 2) {
    cat(sprintf("   %s: CATE(0)=%.3f, CATE(1)=%.3f\n",
                var_name, pd_est[1], pd_est[2]))
  }

  png(file.path(FIGURES_DIR, sprintf("CausalForest_PDP_%s.png", var_name)),
      width = 800, height = 600, res = 100)

  ylab_text <- "Predicted CATE (DSI -> BAI)"
  if (length(unique(x_vals)) <= 2) {
    plot(x_grid, pd_est, type = "b", pch = 19, col = "#E74C3C", lwd = 2,
         main = sprintf("Partial Dependence: %s\n(DSI -> BAI)", var_name),
         xlab = var_name, ylab = ylab_text,
         xaxt = "n", ylim = range(pd_est) * c(0.8, 1.2))
    axis(1, at = x_grid, labels = x_grid)
  } else {
    plot(x_grid, pd_est, type = "l", lwd = 3, col = "#E74C3C",
         main = sprintf("Partial Dependence: %s\n(DSI -> BAI)", var_name),
         xlab = var_name, ylab = ylab_text)
    # Add rug plot
    rug(x_vals, col = rgb(0, 0, 0, 0.1))
  }
  abline(h = mean(tau_hat), lty = 2, col = "gray50")
  legend("topright", legend = c("CATE", "Mean CATE"),
         col = c("#E74C3C", "gray50"), lty = c(1, 2), lwd = c(3, 1), cex = 0.8)
  dev.off()
  cat(sprintf("   [OK] Saved: CausalForest_PDP_%s.png\n", var_name))
}

# Multi-panel PDP for all top variables
png(file.path(FIGURES_DIR, "CausalForest_PDP_All.png"),
    width = 1000, height = 800, res = 100)
par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

for (i in 1:min(6, nrow(var_imp_df))) {
  var_name <- var_imp_df$Variable[i]
  var_idx <- which(X_vars == var_name)
  x_vals <- X[, var_idx]

  if (length(unique(x_vals)) <= 2) {
    x_grid <- sort(unique(x_vals))
  } else {
    x_grid <- seq(quantile(x_vals, 0.05), quantile(x_vals, 0.95), length.out = 30)
  }

  pd_est <- sapply(x_grid, function(xv) {
    X_temp <- X
    X_temp[, var_idx] <- xv
    mean(predict(cf, newdata = X_temp)$predictions)
  })

  imp_score <- round(var_imp_df$Importance[i], 3)
  if (length(unique(x_vals)) <= 2) {
    plot(x_grid, pd_est, type="b", pch=19, col="#E74C3C", lwd=2,
         main=sprintf("%s (Imp=%.3f)", var_name, imp_score),
         xlab=var_name, ylab="CATE", xaxt="n")
    axis(1, at=x_grid, labels=x_grid)
  } else {
    plot(x_grid, pd_est, type="l", lwd=3, col="#E74C3C",
         main=sprintf("%s (Imp=%.3f)", var_name, imp_score),
         xlab=var_name, ylab="CATE")
  }
  abline(h=mean(tau_hat), lty=2, col="gray50")
}
dev.off()
cat("   [OK] Saved: CausalForest_PDP_All.png\n")

# ============================================================================
# 8. Subgroup analysis — Top 3 effect modifiers
# ============================================================================
cat("\n8. Subgroup analysis...\n")

# Attach CATE to dataframe
df$cate <- tau_hat

# Define subgroups
subgroup_results <- list()

for (i in 1:3) {
  var_name <- var_imp_df$Variable[i]
  var_idx <- which(X_vars == var_name)
  x_vals <- X[, var_idx]

  if (length(unique(x_vals)) <= 2) {
    # Binary: by group
    for (g in sort(unique(x_vals))) {
      idx <- which(x_vals == g)
      subgroup_results[[sprintf("%s=%s", var_name, g)]] <- c(
        n = length(idx),
        cate_mean = mean(tau_hat[idx]),
        cate_se = sd(tau_hat[idx]) / sqrt(length(idx))
      )
    }
  } else {
    # Continuous: tertile split
    tert <- cut(x_vals, breaks = quantile(x_vals, c(0, 1/3, 2/3, 1)),
                labels = c("Low", "Mid", "High"), include.lowest = TRUE)
    for (g in c("Low", "Mid", "High")) {
      idx <- which(tert == g)
      subgroup_results[[sprintf("%s=%s", var_name, g)]] <- c(
        n = length(idx),
        cate_mean = mean(tau_hat[idx]),
        cate_se = sd(tau_hat[idx]) / sqrt(length(idx)),
        x_range = if (g == "Low") paste0("[", round(min(x_vals[idx]),1), "-", round(max(x_vals[idx]),1), "]")
                  else if (g == "High") paste0("[", round(min(x_vals[idx]),1), "-", round(max(x_vals[idx]),1), "]")
                  else paste0("[", round(min(x_vals[idx]),1), "-", round(max(x_vals[idx]),1), "]")
      )
    }
  }
}

# Print subgroup table
cat("\n   Subgroup CATE Analysis:\n")
cat(sprintf("   %-25s %6s %12s %12s %s\n", "Subgroup", "N", "CATE", "SE", "Range"))
cat(sprintf("   %s\n", paste(rep("-", 70), collapse="")))
for (nm in names(subgroup_results)) {
  r <- subgroup_results[[nm]]
  xr <- if ("x_range" %in% names(r)) as.character(r["x_range"]) else ""
  cat(sprintf("   %-25s %6.0f %12.3f %12.3f %s\n",
              nm, as.numeric(r["n"]), as.numeric(r["cate_mean"]), as.numeric(r["cate_se"]), xr))
}

# ============================================================================
# 9. Sex-specific analysis
# ============================================================================
cat("\n9. Sex-stratified CATE...\n")

for (sex_val in c(0, 1)) {
  sex_label <- ifelse(sex_val == 0, "Male", "Female")
  idx_sex <- which(df$female == sex_val)

  if (length(idx_sex) < 50) next

  # Re-estimate on subset? No — just compute subgroup CATE from full model
  cate_sex <- tau_hat[idx_sex]
  cat(sprintf("   %s: N=%d, CATE=%.3f (SE=%.3f)\n",
              sex_label, length(idx_sex),
              mean(cate_sex), sd(cate_sex)/sqrt(length(idx_sex))))

  # Distribution
  qs <- quantile(cate_sex, c(0.25, 0.5, 0.75))
  cat(sprintf("      Q25=%.3f, Q50=%.3f, Q75=%.3f\n", qs[1], qs[2], qs[3]))
}

# ============================================================================
# 10. Calibration test
# ============================================================================
cat("\n10. Calibration test...\n")

# Mean forest for calibration
mf <- regression_forest(X, Y, num.trees = 1000)
Y_hat <- predict(mf)$predictions

# Compare CATE prediction vs actual difference
# Split by predicted CATE quartile
cate_q <- ntile(tau_hat, 4)
calib_df <- data.frame(
  Quartile = 1:4,
  N = tapply(df$id, cate_q, length),
  Pred_CATE = tapply(tau_hat, cate_q, mean)
)

# Within each quartile, compute actual Y difference W=1 vs W=0
for (q in 1:4) {
  idx <- which(cate_q == q)
  y1 <- mean(Y[idx][W[idx] == 1])
  y0 <- mean(Y[idx][W[idx] == 0])
  n1 <- sum(W[idx] == 1)
  calib_df$Actual_Diff[q] <- y1 - y0
  calib_df$N_W1[q] <- n1
}

cat("\n   Calibration (by predicted CATE quartile):\n")
cat(sprintf("   %s\n", paste(rep("-", 65), collapse="")))
cat(sprintf("   %-10s %6s %6s %12s %12s\n", "Quartile", "N", "N(W=1)", "Pred CATE", "Actual Diff"))
for (q in 1:4) {
  cat(sprintf("   Q%d        %6d %6d %12.3f %12.3f\n",
              q, calib_df$N[q], calib_df$N_W1[q],
              calib_df$Pred_CATE[q], calib_df$Actual_Diff[q]))
}

# Correlation between predicted and actual
if (sum(!is.na(calib_df$Actual_Diff)) >= 3) {
  cal_cor <- cor(calib_df$Pred_CATE, calib_df$Actual_Diff,
                 use = "complete.obs")
  cat(sprintf("\n   Calibration cor(pred, actual) = %.3f\n", cal_cor))
}

# ============================================================================
# 11. Save results
# ============================================================================
cat("\n11. Saving results...\n")

# Variable importance table
write.csv(var_imp_df, file.path(TABLES_DIR, "causal_forest_variable_importance.csv"),
          row.names = FALSE)

# Full CATE estimates
cate_out <- data.frame(
  id = df$id,
  cate = tau_hat,
  cate_quartile = cate_q,
  W = W,
  Y = Y
)
write.csv(cate_out, file.path(TABLES_DIR, "causal_forest_cate_estimates.csv"),
          row.names = FALSE)

# Subgroup results
sub_df <- do.call(rbind, lapply(names(subgroup_results), function(nm) {
  r <- subgroup_results[[nm]]
  parts <- strsplit(nm, "=")[[1]]
  data.frame(
    Subgroup = nm,
    Variable = parts[1],
    Level = parts[2],
    N = r["n"],
    CATE = r["cate_mean"],
    SE = r["cate_se"],
    stringsAsFactors = FALSE
  )
}))
write.csv(sub_df, file.path(TABLES_DIR, "causal_forest_subgroup_analysis.csv"),
          row.names = FALSE)

cat(sprintf("   [OK] variable_importance.csv\n"))
cat(sprintf("   [OK] cate_estimates.csv\n"))
cat(sprintf("   [OK] subgroup_analysis.csv\n"))

# ============================================================================
# 12. Summary
# ============================================================================
cat("\n========================================\n")
cat("Analysis Complete\n")
cat("========================================\n")
cat(sprintf("Sample: N=%d (DSI=%d, non-DSI=%d)\n",
            nrow(df), sum(W==1), sum(W==0)))
cat(sprintf("ATE: %.2f (SE=%.2f, p=%.4f)\n", ate[1], ate[2], 2*pnorm(-abs(ate[1]/ate[2]))))
cat(sprintf("CATE range: [%.2f, %.2f]\n", min(tau_hat), max(tau_hat)))
cat(sprintf("Top 3 modifiers: %s\n",
            paste(var_imp_df$Variable[1:3], collapse=", ")))
cat(sprintf("Output: %s\n", FIGURES_DIR))
