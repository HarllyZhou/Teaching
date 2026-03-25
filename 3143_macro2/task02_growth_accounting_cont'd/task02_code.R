# ============================================================
# Task: Why simple regressions do not identify alpha, and how
#       to calibrate alpha using factor income shares
# ============================================================

rm(list = ls())
graphics.off()

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(tidyr)
})

# ----------------------------
# Flexible script directory
# ----------------------------
get_script_dir <- function() {
  cmd <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd, value = TRUE)

  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }

  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    path <- tryCatch(rstudioapi::getActiveDocumentContext()$path,
                     error = function(e) "")
    if (nzchar(path)) {
      return(dirname(normalizePath(path)))
    }
  }

  return(getwd())
}

script_dir <- get_script_dir()

# ----------------------------
# File paths
# ----------------------------
data_path <- file.path(script_dir, "pwt.csv")
stopifnot(file.exists(data_path))

# ----------------------------
# Load data
# ----------------------------
pwt <- read_csv(data_path, show_col_types = FALSE)

countries <- c("United States", "Japan", "India", "Nigeria")
start_year <- 1990
end_year <- 2020

# ----------------------------
# Keep relevant sample
# ----------------------------
df <- pwt %>%
  filter(country %in% countries,
         year >= start_year,
         year <= end_year) %>%
  transmute(
    country = country,
    year    = year,
    Y       = cgdpo * 1e6,
    K       = cn * 1e6,
    N       = pop * 1e6,
    labsh   = labsh
  ) %>%
  filter(is.finite(Y), is.finite(K), is.finite(N),
         Y > 0, K > 0, N > 0) %>%
  arrange(country, year) %>%
  group_by(country) %>%
  mutate(
    lnY = log(Y),
    lnK = log(K),
    lnN = log(N),
    gY  = lnY - lag(lnY),
    gK  = lnK - lag(lnK),
    gN  = lnN - lag(lnN),
    t   = row_number(),
    gY_lag = lag(gY),
    alpha_t = 1 - labsh
  ) %>%
  ungroup()

# ============================================================
# Part 1. US log levels in same figure, normalized
# ============================================================

df_us <- df %>% filter(country == "United States")

df_us_levels_plot <- df_us %>%
  transmute(
    year = year,
    lnY_norm = lnY - first(lnY),
    lnK_norm = lnK - first(lnK),
    lnN_norm = lnN - first(lnN)
  ) %>%
  pivot_longer(cols = c(lnY_norm, lnK_norm, lnN_norm),
               names_to = "series",
               values_to = "value")

p1 <- ggplot(df_us_levels_plot, aes(x = year, y = value, color = series)) +
  geom_line(linewidth = 1) +
  labs(
    title = "United States: Normalized Log Levels",
    x = "Year",
    y = "Log level normalized to initial year",
    color = ""
  ) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA)
  )

ggsave(
  filename = file.path(script_dir, "us_log_levels_normalized.png"),
  plot = p1,
  width = 8,
  height = 5,
  dpi = 300,
  bg = "white"
)

# ============================================================
# Part 2. US growth rates in same figure
# ============================================================

df_us_growth_plot <- df_us %>%
  select(year, gY, gK, gN) %>%
  pivot_longer(cols = c(gY, gK, gN),
               names_to = "series",
               values_to = "value")

p2 <- ggplot(df_us_growth_plot, aes(x = year, y = value, color = series)) +
  geom_line(linewidth = 1, na.rm = TRUE) +
  labs(
    title = "United States: Growth Rates",
    x = "Year",
    y = "Growth rate (log difference)",
    color = ""
  ) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA)
  )

ggsave(
  filename = file.path(script_dir, "us_growth_rates.png"),
  plot = p2,
  width = 8,
  height = 5,
  dpi = 300,
  bg = "white"
)

# ============================================================
# Part 3. Level regression with time trend
# ============================================================

level_trend_results <- list()
level_trend_models <- list()

for (cty in countries) {
  df_cty <- df %>% filter(country == cty)

  reg <- lm(lnY ~ lnK + lnN + t, data = df_cty)
  level_trend_models[[cty]] <- reg

  co <- coef(summary(reg))

  out <- data.frame(
    country   = cty,
    term      = rownames(co),
    estimate  = co[, "Estimate"],
    std_error = co[, "Std. Error"],
    t_value   = co[, "t value"],
    p_value   = co[, "Pr(>|t|)"],
    row.names = NULL
  )

  level_trend_results[[cty]] <- out
}

level_trend_table <- bind_rows(level_trend_results)

# ============================================================
# Part 4. Growth regression with lagged dependent variable
# ============================================================

growth_lag_results <- list()
growth_lag_models <- list()

for (cty in countries) {
  df_cty <- df %>%
    filter(country == cty) %>%
    filter(!is.na(gY), !is.na(gK), !is.na(gN), !is.na(gY_lag))

  reg <- lm(gY ~ gK + gN + gY_lag, data = df_cty)
  growth_lag_models[[cty]] <- reg

  co <- coef(summary(reg))

  out <- data.frame(
    country   = cty,
    term      = rownames(co),
    estimate  = co[, "Estimate"],
    std_error = co[, "Std. Error"],
    t_value   = co[, "t value"],
    p_value   = co[, "Pr(>|t|)"],
    row.names = NULL
  )

  growth_lag_results[[cty]] <- out
}

growth_lag_table <- bind_rows(growth_lag_results)

# ============================================================
# Part 5c. alpha_t = 1 - labsh
# ============================================================

alpha_timeseries <- df %>%
  select(country, year, alpha_t) %>%
  filter(!is.na(alpha_t))

alpha_mean <- alpha_timeseries %>%
  group_by(country) %>%
  summarise(
    mean_alpha = mean(alpha_t, na.rm = TRUE),
    min_alpha  = min(alpha_t, na.rm = TRUE),
    max_alpha  = max(alpha_t, na.rm = TRUE),
    n_obs      = n(),
    .groups = "drop"
  )

p3 <- ggplot(alpha_timeseries, aes(x = year, y = alpha_t, color = country)) +
  geom_line(linewidth = 1, na.rm = TRUE) +
  labs(
    title = expression(paste("Capital Share ", alpha[t], " = 1 - labsh")),
    x = "Year",
    y = expression(alpha[t]),
    color = "Country"
  ) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA)
  )

ggsave(
  filename = file.path(script_dir, "alpha_timeseries_4countries.png"),
  plot = p3,
  width = 8,
  height = 5,
  dpi = 300,
  bg = "white"
)

# ============================================================
# TXT output
# ============================================================

txt_path <- file.path(script_dir, "task_shares02_output.txt")

sink(txt_path)

cat("============================================================\n")
cat("Task shares02 output\n")
cat("Countries: United States, Japan, India, Nigeria\n")
cat("Sample: 1990-2020\n")
cat("============================================================\n\n")

cat("PART 3. LEVEL REGRESSION WITH TIME TREND\n")
cat("Model: lnY = beta0 + beta1 lnK + beta2 lnN + beta3 t + u\n\n")

for (cty in countries) {
  cat("------------------------------------------------------------\n")
  cat("Country:", cty, "\n")
  cat("------------------------------------------------------------\n")
  print(summary(level_trend_models[[cty]]))
  cat("\n")
}

cat("\n============================================================\n")
cat("PART 4. GROWTH REGRESSION WITH LAGGED DEPENDENT VARIABLE\n")
cat("Model: gY_t = gamma0 + gamma1 gK_t + gamma2 gN_t + gamma3 gY_{t-1} + v_t\n\n")

for (cty in countries) {
  cat("------------------------------------------------------------\n")
  cat("Country:", cty, "\n")
  cat("------------------------------------------------------------\n")
  print(summary(growth_lag_models[[cty]]))
  cat("\n")
}

cat("\n============================================================\n")
cat("PART 5C. CAPITAL SHARE alpha_t = 1 - labsh\n")
cat("============================================================\n\n")

cat("Alpha time series (first several rows):\n")
print(head(alpha_timeseries, 20))
cat("\n")

cat("Mean alpha by country:\n")
print(alpha_mean)
cat("\n")

sink()

cat("Saved TXT output to:\n", txt_path, "\n", sep = "")
cat("Saved PNG files with white background.\n")
cat("Done.\n")