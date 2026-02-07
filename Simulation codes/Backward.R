library(Iso)
library(matrixStats)
library(rmutil)
library(leaps)     # backward selection (regsubsets)

# Plotting / saving
library(ggplot2)
library(tidyr)
library(dplyr)

setwd("C:/Users/xchen/Desktop/Study/Research/Jason Klusowski/Tree")

# ----------------------------
# Simulation setup
# ----------------------------
n <- 1000  # number of samples
p <- 50    # number of features  (CHANGED to 50)
set.seed(123)

# Define f_norms (9 values)
f_norms <- seq(1, 5, length.out = 9)

# Keep ONLY 4 results:
# 1) data-selected best single model
# 2) oracle best single model
# 3) stacking
# 4) Mallows model averaging
results <- data.frame(
  f_norms = f_norms,
  mean_mse_best   = numeric(length(f_norms)),
  min_mse_oracle  = numeric(length(f_norms)),
  mean_mse_stack  = numeric(length(f_norms)),
  mean_mse_Mallow = numeric(length(f_norms))
)

# Tuning parameters (kept as in your original)
lambda <- 2
lambda2 <- 1
tau <- 1
gamma <- min(1 / tau, 1 / lambda)

eta <- 1/2  # (kept but not used below)

# Simulation parameters
s <- 3     # noise sd
B <- 1000  # iterations

# Nested model sizes 1..p
df <- seq_len(p)

# True model uses first num_coef coordinates (must be <= p now)
num_coef <- 20
stopifnot(num_coef <= p)

# ----------------------------
# Loop over f_norm values
# ----------------------------
for (f_index in seq_along(f_norms)) {
  f_norm <- f_norms[f_index]
  
  # build the true linear function on the first num_coef coordinates
  coefficients <- runif(num_coef, min = -5, max = 5)
  coefficients <- f_norm * coefficients / sqrt(sum(coefficients^2))
  
  f1 <- function(x) sum(coefficients * x)
  
  # storage over B repetitions
  mse_best   <- numeric(B)
  mse_stack  <- numeric(B)
  mse_Mallow <- numeric(B)
  mse_oracle <- matrix(NA, nrow = B, ncol = length(df))
  
  for (i in 1:B) {
    cat(sprintf("f_norm %d/%d (%.3f) | Iteration %d/%d\n",
                f_index, length(f_norms), f_norm, i, B))
    
    # Generate random design X (n x p)
    X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    colnames(X) <- paste0("X", seq_len(p))
    
    # noiseless signal and noisy response
    z <- apply(X[, 1:num_coef, drop = FALSE], 1, f1)
    ytest <- z
    y <- z + rnorm(n, 0, s)
    
    # ------------------------------------------------------------
    # NO honest split:
    # Use FULL DATA for:
    # (i) selecting nested models (backward selection)
    # (ii) fitting each model
    # (iii) stacking
    # ------------------------------------------------------------
    
    # Backward selection on FULL DATA to define nested models (sizes 1..p)
    df_full <- data.frame(y = y, X)
    fit_backward <- regsubsets(
      y ~ .,
      data   = df_full,
      nvmax  = p,
      method = "backward"
    )
    
    sum_back <- summary(fit_backward)
    which_mat <- sum_back$which[, -1, drop = FALSE]  # drop intercept
    var_names <- colnames(which_mat)
    
    Kmodels <- nrow(which_mat)  # should be p
    
    # Predictions U and empirical risks R computed on FULL DATA
    U <- matrix(NA, nrow = n, ncol = Kmodels)
    R <- numeric(Kmodels)
    
    for (k in seq_len(Kmodels)) {
      if (k == 1 || k %% 10 == 0 || k == Kmodels) {
        cat(sprintf("  Building nested model %d/%d\n", k, Kmodels))
      }
      
      vars_k <- var_names[which_mat[k, ]]
      
      Xk <- as.data.frame(X[, vars_k, drop = FALSE])
      names(Xk) <- make.names(vars_k)
      
      # Fit WITHOUT intercept, matching your earlier "-1" style
      model <- lm(y ~ . - 1, data = cbind(y = y, Xk))
      
      U[, k] <- predict(model)
      R[k] <- mean((y - U[, k])^2)
      
      # Oracle MSE for this model size (evaluate vs noiseless signal)
      mse_oracle[i, k] <- mean((ytest - U[, k])^2)
    }
    
    # gamma_hat computed from FULL-DATA risk sequence
    deld <- diff(c(0, df))
    r <- c(mean(y^2), R)
    gamma_hat <- pava((deld) / -diff(r), -diff(r), decreasing = FALSE) * (s^2 / n)
    
    W <- cbind(0, U)
    
    # ----------------------------
    # Best single model (data-selected)
    # ----------------------------
    pred_best <- rowDiffs(W) %*% as.numeric(gamma_hat < 1 / lambda)
    mse_best[i] <- mean((ytest - pred_best)^2)
    
    # ----------------------------
    # Stacking
    # ----------------------------
    weights_stack <- as.numeric(gamma_hat < gamma) * (1 - tau * gamma_hat)
    pred_stack <- rowDiffs(W) %*% weights_stack
    mse_stack[i] <- mean((ytest - pred_stack)^2)
    
    # ----------------------------
    # Mallows model averaging
    # ----------------------------
    tau_tilde <- rep(0, length(df))
    tau_tilde[-1] <- pava((deld / -diff(r))[-1], (-diff(r))[-1], decreasing = FALSE)
    weights_Mallow <- pmax(1 - tau_tilde * (s^2) / n, 0)
    pred_Mallow <- rowDiffs(W) %*% weights_Mallow
    mse_Mallow[i] <- mean((ytest - pred_Mallow)^2)
  }
  
  # Record aggregated results for this f_norm
  results$mean_mse_best[f_index]   <- mean(mse_best)
  results$min_mse_oracle[f_index]  <- min(colMeans(mse_oracle, na.rm = TRUE))
  results$mean_mse_stack[f_index]  <- mean(mse_stack)
  results$mean_mse_Mallow[f_index] <- mean(mse_Mallow)
}

results_full <- results

# ----------------------------
# Plotting (only 4 methods)
# ----------------------------
pdf("plot_backward_full_data_p50_reps1000_num20.pdf", width = 7, height = 7)

results_long <- results %>%
  pivot_longer(cols = -f_norms,
               names_to = "Model",
               values_to = "MSE")

results_long$Model <- factor(
  results_long$Model,
  levels = c("mean_mse_best", "min_mse_oracle", "mean_mse_stack", "mean_mse_Mallow")
)

ggplot(results_long, aes(x = f_norms, y = MSE, color = Model, shape = Model)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_color_manual(
    values = c("red", "blue", "black", "orange"),
    labels = c(
      "Data-selected Best Single Model",
      "Oracle Best Single Model",
      "Stacking",
      "Mallows Model Averaging"
    )
  ) +
  scale_shape_manual(
    values = c(16, 17, 18, 19),
    labels = c(
      "Data-selected Best Single Model",
      "Oracle Best Single Model",
      "Stacking",
      "Mallows Model Averaging"
    )
  ) +
  labs(x = "Norm of f", y = "MSE") +
  theme_minimal(base_size = 15) +
  theme(
    legend.position = "bottom",
    legend.text.align = 0
  ) +
  guides(shape = guide_legend(nrow = 2), color = guide_legend(nrow = 2))

dev.off()

# ----------------------------
# Save CSV to folder
# ----------------------------
folder <- "simulation save"
dir.create(folder, showWarnings = FALSE, recursive = TRUE)

outfile_csv <- file.path(folder, "plot_backward_full_data_p50_reps1000_num20.csv")
write.csv(results_full, outfile_csv, row.names = FALSE)

cat(sprintf("\nSaved CSV to: %s\n", outfile_csv))
cat("Saved plot to: plot_backward_full_data_p50_reps1000.pdf\n")
