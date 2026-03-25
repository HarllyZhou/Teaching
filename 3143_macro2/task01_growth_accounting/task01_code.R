# ============================
# Solow data task (no alpha calibration)
# Countries: US, Japan, India, Nigeria
# Years: 1990-2020
# Y = cgdpo, L = pop, K = cn
# Note: variables are in millions -> multiply by 1e6 before logs
# TFP measure = regression residual:
#   lnTFP_t = residual from lnY ~ lnK + lnL
#   gTFP_t  = residual from gY  ~ gK  + gL
# ============================

rm(list = ls())
graphics.off()

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(car)
})

get_script_dir <- function() {
  cmd <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd, value = TRUE)
  
  if (length(file_arg) > 0) {
    # running via Rscript
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  
  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    # running inside RStudio / Cursor editor
    path <- tryCatch(rstudioapi::getActiveDocumentContext()$path,
                     error = function(e) "")
    if (nzchar(path)) {
      return(dirname(normalizePath(path)))
    }
  }
  
  # fallback
  return(getwd())
}

script_dir <- get_script_dir()

# Put your PWT csv in the same folder as this script and name it "pwt.csv"
data_path <- file.path(script_dir, "pwt.csv")
stopifnot(file.exists(data_path))

pwt <- read_csv(data_path, show_col_types = FALSE)

countries <- c("United States", "Japan", "India", "Nigeria")
years <- 1990:2020
scale_million <- 1e6

run_country_task <- function(df_country, country_name) {
  
  df <- df_country %>%
    filter(year %in% years) %>%
    transmute(
      country = country_name,
      year = year,
      Y = cgdpo * scale_million,
      K = cn    * scale_million,
      L = pop   * scale_million
    ) %>%
    filter(is.finite(Y), is.finite(K), is.finite(L), Y > 0, K > 0, L > 0) %>%
    arrange(year) %>%
    mutate(
      lnY = log(Y),
      lnK = log(K),
      lnL = log(L),
      gY = lnY - dplyr::lag(lnY),
      gK = lnK - dplyr::lag(lnK),
      gL = lnL - dplyr::lag(lnL)
    )
  
  # ---- 1) Level regression ----
  reg_level <- lm(lnY ~ lnK + lnL, data = df)
  df <- df %>%
    mutate(
      lnTFP_hat = resid(reg_level)  # interpreted as ln(A_t) up to a constant
    )
  
  # ---- 2) Growth regression ----
  df_g <- df %>% filter(!is.na(gY) & !is.na(gK) & !is.na(gL))
  reg_growth <- lm(gY ~ gK + gL, data = df_g)
  df_g <- df_g %>%
    mutate(
      gTFP_hat = resid(reg_growth)  # interpreted as g_A,t
    )
  
  list(df_levels = df, df_growth = df_g, reg_level = reg_level, reg_growth = reg_growth)
}

# Collect results for all countries
all_results <- list()
for (cty in countries) {
  df_cty <- pwt %>% filter(country == cty)
  if (nrow(df_cty) == 0) {
    warning(sprintf("No rows found for country='%s' (check spelling in your PWT file).", cty))
    next
  }
  all_results[[cty]] <- run_country_task(df_cty, cty)
}

# ---- Report 1: Coefficients summary ----
report_coef <- file.path(script_dir, "report_coefficients.txt")
sink(report_coef)
cat("========================================\n")
cat("Report 1: Regression Coefficients Summary\n")
cat("========================================\n\n")
for (cty in names(all_results)) {
  res <- all_results[[cty]]
  bL <- coef(res$reg_level)
  bG <- coef(res$reg_growth)
  cat("Country:", cty, "\n")
  cat("  Level regression (lnY ~ lnK + lnL):  lnK =", round(bL["lnK"], 4),
      " lnL =", round(bL["lnL"], 4),
      " sum =", round(bL["lnK"] + bL["lnL"], 4), "\n")
  cat("  Growth regression (gY ~ gK + gL):    gK  =", round(bG["gK"], 4),
      " gL  =", round(bG["gL"], 4),
      " sum =", round(bG["gK"] + bG["gL"], 4), "\n\n")
}
sink()
cat("Written:", report_coef, "\n")

# ---- Report 2: Tests for coefficients sum to 1 ----
report_test <- file.path(script_dir, "report_test_sum_to_one.txt")
sink(report_test)
cat("========================================\n")
cat("Report 2: Tests H0: coefficients sum to 1\n")
cat("========================================\n\n")
for (cty in names(all_results)) {
  res <- all_results[[cty]]
  cat("Country:", cty, "\n")
  cat("\n  Level regression: H0: lnK + lnL = 1\n")
  print(linearHypothesis(res$reg_level, "lnK + lnL = 1"))
  cat("\n  Growth regression: H0: gK + gL = 1\n")
  print(linearHypothesis(res$reg_growth, "gK + gL = 1"))
  cat("\n----------------------------------------\n\n")
}
sink()
cat("Written:", report_test, "\n")

cat("\nDone.\n")
