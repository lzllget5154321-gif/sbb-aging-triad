#!/usr/bin/env Rscript
# ============================================================================
# IPW (Inverse Probability Weighting) robustness check
# Alternative to DoubleML for verifying Causal Forest results
# ============================================================================
library(grf)
library(data.table)

PROJECT_ROOT <- "D:/科研相关项目/程全老师课题组--UKB组/第三个课题--脑体感官衰老耦合解耦研究"
TABLES_DIR <- file.path(PROJECT_ROOT, "results", "tables")

cat("========================================\n")
cat("IPW Robustness Check\n")
cat("========================================\n\n")

# ============================================================================
# 1. CHARLS IPW
# ============================================================================
cat("1. CHARLS IPW Analysis\n")
df <- fread(file.path(TABLES_DIR, "charls_causal_forest_harmonized.csv"))

# Estimate propensity score
X <- as.matrix(df[, c("age","female","educ","cog_base","chronic","depression",
                       "smoker","drinker","phys_act","sr_health"), with=FALSE])
X <- X[, apply(X, 2, sd) > 1e-8, drop=FALSE]
W <- df$W
Y <- df$Y

pf <- regression_forest(X, W, num.trees=1000, honesty=TRUE)
e_hat <- predict(pf)$predictions
e_hat <- pmax(pmin(e_hat, 0.99), 0.01)

# IPW weights
w <- ifelse(W == 1, 1/e_hat, 1/(1-e_hat))
# Normalize weights
w <- w / mean(w)

# Weighted ATE
ate_ipw <- weighted.mean(Y[W==1], w[W==1]) - weighted.mean(Y[W==0], w[W==0])
# Bootstrap SE
boot_ates <- replicate(500, {
  idx <- sample(1:length(Y), replace=TRUE)
  Yi <- Y[idx]; Wi <- W[idx]; wi <- w[idx]
  weighted.mean(Yi[Wi==1], wi[Wi==1]) - weighted.mean(Yi[Wi==0], wi[Wi==0])
})
se_ipw <- sd(boot_ates)

cat(sprintf("  IPW ATE = %.3f (SE=%.3f, p=%.4f)\n",
            ate_ipw, se_ipw, 2*pnorm(-abs(ate_ipw/se_ipw))))

# Stratified IPW by age
age_tert <- cut(df$age, breaks=quantile(df$age, c(0,1/3,2/3,1)),
                labels=c("Young","Mid","Old"), include.lowest=TRUE)
for (tier in levels(age_tert)) {
  idx <- which(age_tert == tier & !is.na(age_tert))
  if (sum(W[idx]==1) < 5) next
  ate_tier <- weighted.mean(Y[idx][W[idx]==1], w[idx][W[idx]==1]) -
              weighted.mean(Y[idx][W[idx]==0], w[idx][W[idx]==0])
  cat(sprintf("  Age %s: IPW ATE = %.3f (N=%d, DSI=%d)\n",
              tier, ate_tier, length(idx), sum(W[idx]==1)))
}

# ============================================================================
# 2. Overlap diagnostics
# ============================================================================
cat("\n2. Overlap diagnostics\n")
cat(sprintf("  Propensity range: [%.4f, %.4f]\n", min(e_hat), max(e_hat)))
cat(sprintf("  Mean propensity (DSI=0): %.4f\n", mean(e_hat[W==0])))
cat(sprintf("  Mean propensity (DSI=1): %.4f\n", mean(e_hat[W==1])))

# Check if any extreme weights
max_w <- max(w)
cat(sprintf("  Max IPW weight: %.1f (threshold=10)\n", max_w))
if (max_w > 10) {
  cat("  WARNING: Some weights exceed 10 -- consider trimming\n")
}

# Effective sample size
ess <- (sum(w))^2 / sum(w^2)
cat(sprintf("  Effective sample size: %.0f / %d (%.1f%%)\n",
            ess, length(w), ess/length(w)*100))

# ============================================================================
# 3. Comparison with Causal Forest
# ============================================================================
cat("\n3. Comparison with Causal Forest\n")
cat("  Method           | ATE (SE)\n")
cat("  -----------------|----------\n")
cat(sprintf("  Causal Forest    | -2.365 (0.789)\n"))
cat(sprintf("  IPW (untrimmed)  | %.3f (%.3f)\n", ate_ipw, se_ipw))
cat(sprintf("  CF (trimmed 5%%)  | -2.281 (1.073)\n"))

# ============================================================================
# 4. POOLED IPW (with cohort fixed effects)
# ============================================================================
cat("\n4. POOLED IPW Analysis\n")
df_p <- fread(file.path(TABLES_DIR, "pooled_4cohort_causal_forest.csv"))

# Use Y_z for comparability
Y_p <- df_p$Y_z
W_p <- df_p$W
X_p <- as.matrix(df_p[, c("age","female","educ","cog_base","chronic",
                            "depression"), with=FALSE])
X_p <- X_p[, apply(X_p, 2, sd) > 1e-8, drop=FALSE]

pf_p <- regression_forest(X_p, W_p, num.trees=1000, honesty=TRUE)
e_hat_p <- predict(pf_p)$predictions
e_hat_p <- pmax(pmin(e_hat_p, 0.99), 0.01)

w_p <- ifelse(W_p == 1, 1/e_hat_p, 1/(1-e_hat_p))
w_p <- w_p / mean(w_p)

ate_ipw_p <- weighted.mean(Y_p[W_p==1], w_p[W_p==1]) -
             weighted.mean(Y_p[W_p==0], w_p[W_p==0])

# Quick bootstrap (200 reps)
boot_p <- replicate(200, {
  idx <- sample(1:length(Y_p), replace=TRUE)
  Yi <- Y_p[idx]; Wi <- W_p[idx]; wi <- w_p[idx]
  weighted.mean(Yi[Wi==1], wi[Wi==1]) - weighted.mean(Yi[Wi==0], wi[Wi==0])
})
se_ipw_p <- sd(boot_p)

cat(sprintf("  Pooled IPW ATE = %.3f SD (SE=%.3f, p=%.4f)\n",
            ate_ipw_p, se_ipw_p, 2*pnorm(-abs(ate_ipw_p/se_ipw_p))))

cat(sprintf("\n  Causal Forest POOLED: -0.074 SD (SE=0.013)\n"))
cat(sprintf("  IPW POOLED:          %.3f SD (SE=%.3f)\n", ate_ipw_p, se_ipw_p))

# ============================================================================
# Summary
# ============================================================================
cat("\n========================================\n")
cat("IPW Robustness Check Complete\n")
cat("========================================\n")
cat("\nConclusion:\n")
if (abs(ate_ipw + 2.365) < 2 * max(se_ipw, 0.789)) {
  cat("  IPW and Causal Forest estimates are consistent\n")
  cat("  DSI -> BAI effect is robust to estimation method\n")
} else {
  cat("  IPW and Causal Forest estimates differ\n")
  cat("  Further sensitivity analysis recommended\n")
}
