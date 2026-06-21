# =============================================================================
# visualization_theme.R — 统一可视化主题 + 调色板
# =============================================================================
# 功能: Nature Communications 风格 ggplot2 theme
#       + 5表型统一调色板 (PHENOTYPE_COLORS)
#       + 5库统一调色板 (COHORT_COLORS)
#       + save_sbb_figure() -- 统一导出函数 (300 DPI)
# 输入: ggplot2 plot 对象
# 输出: results/figures/*.png (300 DPI, 白色背景)
# 依赖: R packages: ggplot2
# 用法: source('utils/visualization_theme.R'); plot + theme_sbb()
# 项目: SBB课题 -- 脑体感官衰老耦合解耦研究
# 版本: v1.0 (2026-06-09)
# =============================================================================

library(ggplot2)

# Unified color palette for 5 phenotypes
PHENOTYPE_COLORS <- c(
  "Successful Aging"     = "#2ECC71",  # Green
  "Sensory-First"        = "#F39C12",  # Orange
  "Brain-Resilient"      = "#3498DB",  # Blue
  "Body-Resilient"       = "#9B59B6",  # Purple
  "Global Accelerated"   = "#E74C3C"   # Red
)

# Unified color palette for cohorts
COHORT_COLORS <- c(
  "CHARLS" = "#E74C3C",
  "HRS"    = "#3498DB",
  "SHARE"  = "#2ECC71",
  "KLoSA"  = "#F39C12",
  "MHAS"   = "#9B59B6"
)

# ggplot2 theme consistent with Nature Communications style
theme_sbb <- theme_minimal() +
  theme(
    text = element_text(size = 10, family = "sans"),
    plot.title = element_text(size = 12, face = "bold"),
    plot.subtitle = element_text(size = 10, color = "grey40"),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 8),
    legend.position = "bottom",
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 8),
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 9, face = "bold"),
    plot.margin = margin(10, 10, 10, 10)
  )

# Save figure with consistent settings
save_sbb_figure <- function(plot, filename, width = 7, height = 5, dpi = 300) {
  ggsave(
    file.path("results/figures", filename),
    plot = plot,
    width = width, height = height, dpi = dpi,
    bg = "white"
  )
  cat("Saved:", filename, "\n")
}
