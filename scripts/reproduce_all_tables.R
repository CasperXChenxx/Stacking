## Run from the repository root:
##   Rscript scripts/reproduce_all_tables.R
##
## Set N_REPS to a smaller positive integer for a quick simulation smoke test.
## The paper uses N_REPS=1000 (the default).

if (!dir.exists("Simulation codes") || !dir.exists("Real data") || !dir.exists("data")) {
  stop("Run this script from the repository root.")
}

rscript <- file.path(
  R.home("bin"),
  if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript"
)

run_script <- function(script, env = character()) {
  cat(sprintf("\nRunning %s", script))
  if (length(env)) cat(sprintf(" [%s]", paste(env, collapse = ", ")))
  cat("\n")

  status <- system2(rscript, args = shQuote(script), env = env)
  if (!identical(status, 0L)) {
    stop(sprintf("%s failed with exit status %s.", script, status))
  }
}

n_reps <- Sys.getenv("N_REPS", unset = "1000")
if (is.na(suppressWarnings(as.integer(n_reps))) || as.integer(n_reps) < 1L) {
  stop("N_REPS must be a positive integer.")
}

# Tables 1 and 2: backward deletion simulations.
for (num_coef in c(20L, 35L)) {
  run_script(
    file.path("Simulation codes", "Backward.R"),
    env = c(sprintf("NUM_COEF=%d", num_coef), sprintf("N_REPS=%s", n_reps))
  )
}

# Tables 3-6: baseline plus 10, 20, ..., 60 independent noise features.
real_data_scripts <- c(
  file.path("Real data", "TREE_Newsdata_with_noise_splitting.R"),
  file.path("Real data", "TREE_Calihousing_with_noise_splitting.R"),
  file.path("Real data", "TREE_Superconductivity_with_noise_splitting.R"),
  file.path("Real data", "TREE_community_crime_with_noise_splitting.R")
)

for (script in real_data_scripts) {
  for (m_noise in seq(0L, 60L, by = 10L)) {
    run_script(script, env = sprintf("M_NOISE=%d", m_noise))
  }
}

run_script(
  file.path("scripts", "collect_results.R"),
  env = sprintf("N_REPS=%s", n_reps)
)

cat("\nAll runs completed. Paper-formatted CSV files are in results/paper-tables/.\n")
