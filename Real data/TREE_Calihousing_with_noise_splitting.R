library(Iso)
library(matrixStats)
library(rmutil)
library(stats)
library(rpart)   # nested trees (Breiman pruning path)

setwd("C:/Users/xchen/Desktop/Study/Research/Jason Klusowski/Tree")

# ============================================================
# USER OPTIONS
# ============================================================

## A) Data file / target
data_file  <- "housing.csv"           # <-- your local file
target_col <- "median_house_value"    # housing target

## B) Remove the categorical column ocean_proximity
remove_ocean_proximity <- TRUE        # set TRUE as requested

## C) Add independent irrelevant (noise) features?
add_noise_features <- FALSE
m_noise    <- 60
sd_min     <- 1
sd_max     <- 10
noise_seed <- 111

## D) Internal split: model selection vs stacking (refit + risks + weights)
use_internal_split   <- FALSE
sel_prop             <- 0.20
internal_split_seed  <- 222

## E) Train/Test split
test_seed <- 111
ntest     <- 500

## F) Optional target transform
use_log1p_target <- TRUE  # TRUE: y <- log1p(y)

## G) Tree path thinning: keep internal nodes multiple of K
tree_keep_multiple <- 10

## H) If you previously used regsubsets: keep this FALSE (no need now)
drop_linear_dependencies <- FALSE

# ============================================================
# Helpers
# ============================================================

safe_chol <- function(G) {
  eps <- 1e-10
  for (k in 0:6) {
    add <- if (k == 0) 0 else (10^(k-6))
    out <- try(chol(G + diag(add + eps, ncol(G))), silent = TRUE)
    if (!inherits(out, "try-error")) return(out)
  }
  stop("Cholesky failed; matrix may be singular. Try removing collinear columns.")
}

# ============================================================
# 0) Load housing data + preprocessing (numeric-only, impute)
# ============================================================

raw_data <- read.csv(data_file, stringsAsFactors = FALSE)

if (!target_col %in% names(raw_data)) {
  stop(sprintf("Target column '%s' not found. Columns are:\n%s",
               target_col, paste(names(raw_data), collapse = ", ")))
}

# Remove ocean_proximity as requested (if present)
if (remove_ocean_proximity && "ocean_proximity" %in% names(raw_data)) {
  raw_data$ocean_proximity <- NULL
  cat("Removed column: ocean_proximity\n\n")
}

# Optional: add independent noise features BEFORE splitting
if (add_noise_features) {
  set.seed(noise_seed)
  ndata <- nrow(raw_data)
  noise_features <- sapply(seq_len(m_noise), function(i) {
    sd_i <- runif(1, min = sd_min, max = sd_max)
    rnorm(ndata, mean = 0, sd = sd_i)
  })
  noise_features <- as.data.frame(noise_features)
  colnames(noise_features) <- paste0("noise_var", seq_len(m_noise))
  raw_data <- cbind(noise_features, raw_data)
  cat(sprintf("Added %d independent noise features (sd in [%.2f, %.2f]).\n\n",
              m_noise, sd_min, sd_max))
} else {
  m_noise <- 0
  cat("No noise features added.\n\n")
}

# Coerce everything to numeric (now should be numeric-only)
raw_data <- as.data.frame(lapply(raw_data, function(col) as.numeric(as.character(col))))

# ============================================================
# 1) Train/Test split
# ============================================================

set.seed(test_seed)
n_all <- nrow(raw_data)

test_idx <- sample(seq_len(n_all), size = ntest, replace = FALSE)
train_idx <- setdiff(seq_len(n_all), test_idx)

train_df <- raw_data[train_idx, , drop = FALSE]
test_df  <- raw_data[test_idx,  , drop = FALSE]

y_train <- train_df[[target_col]]
y_test  <- test_df[[target_col]]

if (use_log1p_target) {
  y_train <- log1p(y_train)
  y_test  <- log1p(y_test)
  cat("Applied y <- log1p(y) to target.\n\n")
}

X_train <- as.matrix(train_df[, setdiff(names(train_df), target_col), drop = FALSE])
X_test  <- as.matrix(test_df[,  setdiff(names(test_df),  target_col), drop = FALSE])

# Median impute using TRAIN medians (no leakage)
train_meds <- apply(X_train, 2, function(v) median(v, na.rm = TRUE))
for (j in seq_len(ncol(X_train))) {
  X_train[is.na(X_train[, j]), j] <- train_meds[j]
  X_test[is.na(X_test[, j]), j]   <- train_meds[j]
}

# Ensure column names exist and are syntactic/consistent
if (is.null(colnames(X_train))) colnames(X_train) <- paste0("V", seq_len(ncol(X_train)))
if (is.null(colnames(X_test)))  colnames(X_test)  <- colnames(X_train)

colnames(X_train) <- make.names(colnames(X_train), unique = TRUE)
colnames(X_test)  <- colnames(X_train)

n_train <- nrow(X_train)
p <- ncol(X_train)

cat(sprintf("Train/Test split: train=%d, test=%d | p(numeric)=%d\n\n",
            n_train, ntest, p))

# ============================================================
# 2) Internal split: model selection vs stacking (refit + risks)
# ============================================================

if (use_internal_split) {
  set.seed(internal_split_seed)
  perm <- sample(seq_len(n_train))
  n_sel <- max(30, min(n_train - 30, floor(sel_prop * n_train)))
  
  sel_local_idx   <- perm[seq_len(n_sel)]
  stack_local_idx <- perm[(n_sel + 1):n_train]
  
  X_sel <- X_train[sel_local_idx, , drop = FALSE]
  y_sel <- y_train[sel_local_idx]
  
  X_stack <- X_train[stack_local_idx, , drop = FALSE]
  y_stack <- y_train[stack_local_idx]
  
  cat(sprintf("Internal split ON: selection=%d, stacking=%d (sel_prop=%.2f)\n\n",
              n_sel, n_train - n_sel, sel_prop))
} else {
  X_sel <- X_train
  y_sel <- y_train
  X_stack <- X_train
  y_stack <- y_train
  cat(sprintf("Internal split OFF: using ALL training for selection+stacking: n_train=%d\n\n",
              n_train))
}

n2 <- nrow(X_stack)

# ============================================================
# 3) Whitening on STACK part (then apply to STACK + TEST)
# ============================================================

col_means_stack <- colMeans(X_stack)
Xc_stack <- sweep(X_stack, 2, col_means_stack, FUN = "-")
Xc_test  <- sweep(X_test,  2, col_means_stack, FUN = "-")

G <- crossprod(Xc_stack)
Rchol <- safe_chol(G)
Rinv <- backsolve(Rchol, diag(ncol(G)))  # R^{-1}

X_stack_whitened <- Xc_stack %*% Rinv
X_test_whitened  <- Xc_test  %*% Rinv

colnames(X_stack_whitened) <- colnames(X_train)
colnames(X_test_whitened)  <- colnames(X_train)

# ============================================================
# 4) Estimate sigma^2 on STACK part
# ============================================================

fit_sigma <- lm(y_stack ~ ., data = data.frame(X_stack, y_stack))
sigma2_hat <- summary(fit_sigma)$sigma^2
cat(sprintf("sigma2_hat (from STACK part)=%.6f\n\n", sigma2_hat))

# ============================================================
# 5) Tuning parameters (same as before)
# ============================================================

lambda  <- 2
lambda2 <- 1
tau  <- 1
tau2 <- 1/3
tau3 <- 1.5

gamma        <- min(1/tau,  1/lambda)
gamma2       <- min(1/tau2, 1/lambda)
gamma3       <- min(1/tau3, 1/lambda)
gamma_Mallow <- min(1/tau,  1/lambda2)

eta <- 1/2

########################################################
## PIPELINE: NESTED TREES (Breiman pruning path)       ##
## keep only d_k = 10,20,30,... (internal nodes)       ##
########################################################

# (B1) Grow/pruning path on SELECTION part only
df_sel_tree <- data.frame(y = y_sel, X_sel)

full_tree <- rpart(
  y ~ .,
  data   = df_sel_tree,
  method = "anova",
  control = rpart.control(
    cp       = 0,
    xval     = 0,
    minsplit = 20,
    maxdepth = 30
  )
)

cp_table <- full_tree$cptable
cp_table <- cp_table[cp_table[, "nsplit"] > 0, , drop = FALSE]
cp_table <- cp_table[order(cp_table[, "nsplit"]), , drop = FALSE]

d_internal_all <- cp_table[, "nsplit"]
keep_idx <- which(d_internal_all %% tree_keep_multiple == 0)
if (length(keep_idx) == 0) {
  warning(sprintf("No trees with internal nodes multiple of %d found; using all subtrees.",
                  tree_keep_multiple))
  keep_idx <- seq_len(nrow(cp_table))
}

cp_table  <- cp_table[keep_idx, , drop = FALSE]
df2       <- cp_table[, "nsplit"]  # internal nodes
cp_values <- cp_table[, "CP"]

cat(sprintf("\nNested trees selected (on SELECTION part): total=%d\n\n", length(df2)))

# (B2) Evaluate the pruned subtrees on STACK and TEST; risks computed on STACK
tr_df_tree_stack <- data.frame(X_stack)
te_df_tree_test  <- data.frame(X_test)

U_tree_stack <- matrix(nrow = n2,    ncol = length(df2))
Utest_tree   <- matrix(nrow = ntest, ncol = length(df2))
R_tree <- numeric(length(df2))

for (kk in seq_along(df2)) {
  cat(sprintf("Tree path iteration: %d / %d | internal nodes d_k: %d\n",
              kk, length(df2), df2[kk]))
  
  tree_k <- prune(full_tree, cp = cp_values[kk])
  
  U_tree_stack[, kk] <- predict(tree_k, newdata = tr_df_tree_stack)
  Utest_tree[, kk]   <- predict(tree_k, newdata = te_df_tree_test)
  
  R_tree[kk] <- mean((y_stack - U_tree_stack[, kk])^2)
}

deld2 <- diff(c(0, df2))
r2 <- c(mean(y_stack^2), R_tree)
gamma_hat_tree <- pava((deld2) / -diff(r2), -diff(r2), decreasing = FALSE) * (sigma2_hat / n2)

Wtest_tree <- cbind(0, Utest_tree)

# Best single model (tree)
pred_best_tree <- rowDiffs(Wtest_tree) %*% as.numeric(gamma_hat_tree < 1 / lambda)
mse_best_tree <- mean((y_test - pred_best_tree)^2)

# Stacking (tree)
weights_stack_tree <- as.numeric(gamma_hat_tree < gamma) * (1 - tau * gamma_hat_tree)
pred_stack_tree <- rowDiffs(Wtest_tree) %*% weights_stack_tree
mse_stack_tree <- mean((y_test - pred_stack_tree)^2)

# Mallows Model Average (tree)
tau_tilde_tree <- rep(0, length(df2))
tau_tilde_tree[-1] <- pava((deld2 / -diff(r2))[-1], (-diff(r2))[-1], decreasing = FALSE)
weights_Mallow_tree <- pmax(1 - tau_tilde_tree * sigma2_hat / n2, 0)
pred_Mallow_tree <- rowDiffs(Wtest_tree) %*% weights_Mallow_tree
mse_Mallow_tree <- mean((y_test - pred_Mallow_tree)^2)

mse_results_tree <- data.frame(
  Method = c("Tree(Breiman) - Best single model",
             "Tree(Breiman) - Stacking",
             "Tree(Breiman) - Mallows Model Averaging"),
  MSE = c(mse_best_tree, mse_stack_tree, mse_Mallow_tree)
)

cat("\nFINAL RESULTS (TREES ONLY):\n")
print(mse_results_tree, row.names = FALSE)

# ============================================================
# Save results
# ============================================================

results_dir <- "results_housing_no_ocean_proximity"
if (!dir.exists(results_dir)) dir.create(results_dir)

tag_split <- if (use_internal_split) sprintf("split_sel%.2f", sel_prop) else "nosplit"
tag_noise <- if (add_noise_features) sprintf("noise_m%d", m_noise) else "noise_m0"
tag_log   <- if (use_log1p_target) "log1pY" else "rawY"

file_name <- file.path(results_dir, paste0("housing_treeonly_results_", tag_noise, "_",
                                           tag_split, "_", tag_log, ".txt"))

write.table(mse_results_tree, file = file_name,
            row.names = FALSE, col.names = TRUE, quote = FALSE)

cat(sprintf("\nSaved results to: %s\n", file_name))
