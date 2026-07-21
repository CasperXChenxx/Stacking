## Collect per-configuration outputs into paper-formatted Tables 1-6.
## Run from the repository root after the experiment scripts finish.

if (!dir.exists("results")) {
  stop("No results directory found. Run the experiment scripts first.")
}

n_reps <- suppressWarnings(as.integer(Sys.getenv("N_REPS", unset = "1000")))
if (is.na(n_reps) || n_reps < 1L) stop("N_REPS must be a positive integer.")

output_dir <- file.path("results", "paper-tables")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

read_required_csv <- function(path) {
  if (!file.exists(path)) {
    stop(sprintf("Missing result file: %s", path))
  }
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}

write_paper_table <- function(table, number) {
  path <- file.path(output_dir, sprintf("table-%d.csv", number))
  write.csv(table, path, row.names = FALSE, quote = FALSE)
  cat(sprintf("Wrote %s\n", path))
}

collect_simulation <- function(num_coef, table_number) {
  path <- file.path(
    "results", "table-1-2-backward",
    sprintf("backward_p50_reps%d_num%d.csv", n_reps, num_coef)
  )
  x <- read_required_csv(path)
  x <- x[x$f_norms %in% 1:5, , drop = FALSE]
  x <- x[order(x$f_norms), , drop = FALSE]
  if (nrow(x) != 5L) stop(sprintf("Expected five integer signal strengths in %s", path))

  values <- rbind(
    AIC = 1000 * x$mean_mse_best,
    Oracle = 1000 * x$min_mse_oracle,
    Stacking = 1000 * x$mean_mse_stack,
    Mallows = 1000 * x$mean_mse_Mallow
  )
  relative <- 100 * (values["AIC", ] - values["Stacking", ]) / values["AIC", ]
  values <- rbind(values, "Relative Improvement (%)" = relative)
  values <- round(values, 3)

  out <- data.frame(Method = rownames(values), values, check.names = FALSE)
  colnames(out)[-1] <- sprintf("||f|| = %d", 1:5)
  write_paper_table(out, table_number)
}

extract_tree_mse <- function(path) {
  x <- read_required_csv(path)
  if (!all(c("Method", "MSE") %in% names(x))) {
    stop(sprintf("Unexpected columns in %s", path))
  }

  find_one <- function(pattern) {
    values <- x$MSE[grepl(pattern, x$Method, fixed = TRUE)]
    if (length(values) != 1L) stop(sprintf("Could not identify '%s' in %s", pattern, path))
    values
  }

  c(
    AIC = find_one("Best single model"),
    Stacking = find_one("Stacking"),
    Mallows = find_one("Mallows Model Averaging")
  )
}

collect_real_data <- function(result_subdir, file_prefix, table_number) {
  noise_grid <- seq(0L, 60L, by = 10L)
  values <- vapply(noise_grid, function(m_noise) {
    path <- file.path(
      "results", result_subdir,
      sprintf("%snoise_m%d_nosplit%s.csv", file_prefix, m_noise,
              if (identical(result_subdir, "table-4-california-housing")) "_log1pY" else "")
    )
    1000 * extract_tree_mse(path)
  }, numeric(3))

  relative <- 100 * (values["AIC", ] - values["Stacking", ]) / values["AIC", ]
  values <- rbind(round(values, 1), "Relative Improvement (%)" = round(relative, 3))

  out <- data.frame(Method = rownames(values), values, check.names = FALSE)
  colnames(out)[-1] <- c("Baseline", sprintf("+%d Features", noise_grid[-1]))
  write_paper_table(out, table_number)
}

collect_simulation(num_coef = 20L, table_number = 1L)
collect_simulation(num_coef = 35L, table_number = 2L)

collect_real_data(
  result_subdir = "table-3-online-news",
  file_prefix = "online_news_tree_",
  table_number = 3L
)
collect_real_data(
  result_subdir = "table-4-california-housing",
  file_prefix = "california_housing_tree_",
  table_number = 4L
)
collect_real_data(
  result_subdir = "table-5-superconductivity",
  file_prefix = "superconductivity_tree_",
  table_number = 5L
)
collect_real_data(
  result_subdir = "table-6-communities-crime",
  file_prefix = "communities_crime_tree_",
  table_number = 6L
)
