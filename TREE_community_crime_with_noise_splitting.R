############################################################
## Communities & Crime (UCI) - TREES ONLY
##  - add_noise_features option (m_noise, sd range)
##  - Train/Test split (ntest, seeds)
##  - optional internal split (kept as option; default OFF)
##  - Whitening on STACK part
##  - sigma^2 estimated on STACK part
##  - Pipeline: nested trees (Breiman pruning path), keep d_k multiples
##  - Save results
############################################################

library(Iso)
library(matrixStats)
library(rmutil)
library(stats)
library(rpart)   # nested trees

setwd("C:/Users/xchen/Desktop/Study/Research/Jason Klusowski/Tree")

# ============================================================
# USER OPTIONS
# ============================================================

## A) Data file (UCI Communities & Crime)
data_file <- "communities.data"

## B) Add independent irrelevant (noise) features?
add_noise_features <- TRUE     # TRUE/FALSE
m_noise <- 60                  # how many noise features to add (ignored if add_noise_features=FALSE)
sd_min <- 1                    # noise sd range
sd_max <- 10
noise_seed <- 111              # seed used only for noise generation

## C) Internal split for model selection vs stacking?
use_internal_split <- FALSE    # TRUE/FALSE
sel_prop <- 0.20               # fraction of TRAIN used for model selection only
internal_split_seed <- 222     # seed for internal split

## D) Train/Test split
test_seed <- 111
ntest <- 300

## E) Preprocessing options
standardize_numeric <- FALSE   # keep FALSE since your pipeline does whitening later
impute_missing <- TRUE         # median-impute missing values if any

## F) Tree path thinning (keep internal nodes multiples of this)
tree_keep_multiple <- 10

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

median_impute_vec <- function(v) {
  v <- as.numeric(v)
  if (all(is.na(v))) return(v)  # should be removed earlier
  v[is.na(v)] <- median(v, na.rm = TRUE)
  v
}

# ============================================================
# 0) Load + PREPROCESS (match your provided preprocessing)
#    - read communities.data, na.strings="?"
#    - remove first 5 nonpredictive columns
#    - target = last column (ViolentCrimesPerPop)
#    - drop all-NA predictor columns
#    - numeric coercion + median imputation
#    - drop rows with missing y
# ============================================================

raw0 <- read.table(
  file = data_file,
  sep = ",",
  header = FALSE,
  stringsAsFactors = FALSE,
  na.strings = "?",
  fill = TRUE,
  comment.char = ""
)

cat(sprintf("Loaded '%s': n=%d, p=%d (incl nonpredictive + target)\n\n",
            data_file, nrow(raw0), ncol(raw0)))

if (ncol(raw0) < 7) stop("Data seems too small; please check the file format.")

# Remove nonpredictive vars: first 5 columns
raw0 <- raw0[, -(1:5), drop = FALSE]
cat(sprintf("After removing first 5 nonpredictive cols: p=%d\n\n", ncol(raw0)))

# Target is the last column
y_col <- ncol(raw0)
y <- as.numeric(raw0[[y_col]])
X_df <- raw0[, -y_col, drop = FALSE]

# Drop predictor columns that are all NA
all_na <- sapply(X_df, function(col) all(is.na(col)))
if (any(all_na)) {
  cat(sprintf("Dropping %d predictor columns that are all NA.\n\n", sum(all_na)))
  X_df <- X_df[, !all_na, drop = FALSE]
}

# Coerce to numeric
X_df <- as.data.frame(lapply(X_df, as.numeric))

# Drop rows with missing y
if (any(is.na(y))) {
  keep <- !is.na(y)
  cat(sprintf("Dropping %d rows with missing y.\n\n", sum(!keep)))
  X_df <- X_df[keep, , drop = FALSE]
  y <- y[keep]
}

# Median impute predictors (if requested)
if (impute_missing) {
  X_df <- as.data.frame(lapply(X_df, median_impute_vec))
}

# Optional standardize (not needed if whitening later)
if (standardize_numeric) {
  mu <- sapply(X_df, mean)
  sdv <- sapply(X_df, sd)
  sdv[sdv == 0] <- 1
  X_df <- as.data.frame(scale(X_df, center = mu, scale = sdv))
}

# ============================================================
# 0b) Add independent noise features BEFORE splitting
# ============================================================

if (add_noise_features) {
  set.seed(noise_seed)
  ndata <- nrow(X_df)
  
  noise_features <- sapply(seq_len(m_noise), function(i) {
    sd_i <- runif(1, min = sd_min, max = sd_max)
    rnorm(ndata, mean = 0, sd = sd_i)
  })
  noise_features <- as.data.frame(noise_features)
  colnames(noise_features) <- paste0("noise_var", seq_len(m_noise))
  
  X_df <- cbind(noise_features, X_df)
  
  cat(sprintf("Added %d independent noise features (sd in [%.2f, %.2f]).\n\n",
              m_noise, sd_min, sd_max))
} else {
  m_noise <- 0
  cat("No noise features added.\n\n")
}

# Build design matrix (no intercept)
X <- model.matrix(~ . - 1, data = X_df)

n <- nrow(X)
p <- ncol(X)
cat(sprintf("Final design matrix: n=%d, p=%d (includes %d noise vars)\n\n", n, p, m_noise))

# ============================================================
# 1) Train/Test split
# ============================================================

set.seed(test_seed)
ntest <- min(ntest, n - 10)  # safety
test_idx <- sample(seq_len(n), size = ntest, replace = FALSE)
train_idx <- setdiff(seq_len(n), test_idx)

X_train <- X[train_idx, , drop = FALSE]
y_train <- y[train_idx]
X_test  <- X[test_idx,  , drop = FALSE]
y_test  <- y[test_idx]

cat(sprintf("Train/Test split: train=%d, test=%d | p=%d\n\n",
            length(train_idx), length(test_idx), p))

# ============================================================
# 2) Internal split (selection vs stacking)
# ============================================================

n_train <- nrow(X_train)

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
# 3) Whitening computed on STACK part (then applied to STACK + TEST)
# ============================================================

col_means_stack <- colMeans(X_stack)
Xc_stack <- sweep(X_stack, 2, col_means_stack, FUN = "-")
Xc_test  <- sweep(X_test,  2, col_means_stack, FUN = "-")

G <- crossprod(Xc_stack)
Rchol <- safe_chol(G)
Rinv <- backsolve(Rchol, diag(ncol(G))) # R^{-1}

X_stack_whitened <- Xc_stack %*% Rinv
X_test_whitened  <- Xc_test  %*% Rinv

orig_names <- colnames(X_train)
colnames(X_stack_whitened) <- orig_names
colnames(X_test_whitened)  <- orig_names

# ============================================================
# 4) Estimate sigma^2 on STACK part
# ============================================================

fit_sigma <- lm(y_stack ~ ., data = data.frame(X_stack, y_stack))
sigma2_hat <- summary(fit_sigma)$sigma^2
cat(sprintf("sigma2_hat (from STACK part)=%.6f\n\n", sigma2_hat))

# ============================================================
# 5) Tuning parameters (same as your previous code)
# ============================================================

lambda <- 2
lambda2 <- 1
tau <- 1
tau2 <- 1/3
tau3 <- 1.5

gamma <- min(1/tau, 1/lambda)
gamma2 <- min(1/tau2, 1/lambda)
gamma3 <- min(1/tau3, 1/lambda)
gamma_Mallow <- min(1/tau, 1/lambda2)

eta <- 1/2

########################################################
## PIPELINE: NESTED TREES (Breiman pruning path)       ##
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
  warning(sprintf("No trees with internal nodes multiple of %d found; using all available subtrees.",
                  tree_keep_multiple))
  keep_idx <- seq_len(nrow(cp_table))
}

cp_table  <- cp_table[keep_idx, , drop = FALSE]
df2       <- cp_table[, "nsplit"]  # internal nodes
cp_values <- cp_table[, "CP"]

cat(sprintf("\nNested trees selected: total=%d\n\n", length(df2)))

# (B2) Evaluate pruned subtrees on STACK and TEST; risks computed on STACK
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

results_dir <- "results_communities_crime"
if (!dir.exists(results_dir)) dir.create(results_dir)

tag_split <- if (use_internal_split) sprintf("split_sel%.2f", sel_prop) else "nosplit"
tag_noise <- if (add_noise_features) sprintf("noise_m%d", m_noise) else "noise_m0"

file_name <- file.path(results_dir, paste0("communities_crime_treeonly_result_", tag_noise, "_", tag_split, ".txt"))

write.table(mse_results_tree, file = file_name,
            row.names = FALSE, col.names = TRUE, quote = FALSE)

cat(sprintf("\nSaved results to: %s\n", file_name))
