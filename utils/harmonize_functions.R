# =============================================================================
# harmonize_functions.R — 五库变量映射工具函数
# =============================================================================
# 功能: harmonize_sensory() -- 跨库标准化感官损伤编码 (0/1)
#       harmonize_cognition() -- 跨库标准化认知评分 (z-score)
#       harmonize_covariates() -- 跨库标准化协变量 (年龄/性别/教育等)
#       build_dsi() -- 从vision+hearing构建DSI二元变量
# 输入: 各库原始数据 (CHARLS/HRS/KLoSA/MHAS/SHARE)
# 输出: 含 harmonized_* 列的标准DataFrame
# 依赖: base R (无额外包依赖)
# 用法: source('utils/harmonize_functions.R')
# 项目: SBB课题 -- 脑体感官衰老耦合解耦研究
# 版本: v1.0 (2026-06-09)
# =============================================================================

harmonize_sensory <- function(df, vision_col, hearing_col, cohort) {
  switch(cohort,
    "CHARLS" = {
      df$vision_impairment <- ifelse(df[[vision_col]] == 1, 1, 0)
      df$hearing_impairment <- ifelse(df[[hearing_col]] == 1, 1, 0)
    },
    "KLoSA" = {
      # sighta/dsighta/nsighta -> any difficulty = impairment
      df$vision_impairment <- ifelse(
        rowSums(df[, c("sighta", "dsighta", "nsighta")] == 1, na.rm = TRUE) > 0, 1, 0
      )
      df$hearing_impairment <- ifelse(df$hearinga == 1, 1, 0)
    },
    "MHAS" = {
      df$vision_impairment <- ifelse(df$sight == 1, 1, 0)
      df$hearing_impairment <- ifelse(df$hearing == 1, 1, 0)
    },
    "SHARE" = {
      # Placeholder — exact variable names TBD
      df$vision_impairment <- NA
      df$hearing_impairment <- NA
    }
  )
  df$dsi <- as.integer(df$vision_impairment == 1 & df$hearing_impairment == 1)
  df
}

#' Standardize cognitive score across databases
#' @param df data.frame
#' @param cohort Cohort name
#' @return data.frame with harmonized global_cognition z-score
harmonize_cognition <- function(df, cohort) {
  switch(cohort,
    "CHARLS" = {
      df$global_cognition_z <- scale(df$global_cognition)[,1]
    },
    "KLoSA" = {
      # Derive from orient + draw
      cog <- rowSums(df[, c("orient", "orientp_k", "draw")], na.rm = TRUE)
      df$global_cognition_z <- scale(cog)[,1]
    },
    "MHAS" = {
      cog <- rowSums(df[, c("orient_m", "forient_m")], na.rm = TRUE)
      df$global_cognition_z <- scale(cog)[,1]
    },
    "SHARE" = {
      df$global_cognition_z <- scale(df$orient)[,1]
    }
  )
  df
}

#' Standardize ADL/IADL across databases
harmonize_adl_iadl <- function(df, cohort) {
  switch(cohort,
    "CHARLS" = {
      # dadliv is derived total — use directly
      df$adl_sum <- df$dadliv
      df$iadl_sum <- NULL  # TBD
    },
    "KLoSA" = {
      df$adl_sum <- rowSums(df[, grep("^adl", names(df))], na.rm = TRUE)
      df$iadl_sum <- rowSums(df[, grep("^iadl", names(df))], na.rm = TRUE)
    },
    "MHAS" = {
      df$adl_sum <- df$adltot6
      df$iadl_sum <- df$iadlfour
    },
    "SHARE" = {
      # adlwa/adlwam/adlwaa — use most recent wave
      df$adl_sum <- df$adlwa
      df$iadl_sum <- df$iadla
    }
  )
  df
}

#' Build 3-system aging indices for a cohort
build_sbb_indices <- function(df, cohort) {
  # SAI
  sai_z <- scale(df$SAI_raw <- (df$vision_impairment + df$hearing_impairment) / 2)
  df$SAI <- 100 * (sai_z - min(sai_z, na.rm=TRUE)) / diff(range(sai_z, na.rm=TRUE))

  # BAI
  df$BAI <- 100 * (df$global_cognition_z - min(df$global_cognition_z, na.rm=TRUE)) /
            diff(range(df$global_cognition_z, na.rm=TRUE))

  # BoAI
  components <- c("adl_sum", "iadl_sum", "chronic_count", "self_rated_health")
  available <- intersect(components, names(df))
  boai_z <- scale(rowMeans(sapply(df[available], scale), na.rm=TRUE))
  df$BoAI <- 100 * (boai_z - min(boai_z, na.rm=TRUE)) / diff(range(boai_z, na.rm=TRUE))

  df
}
