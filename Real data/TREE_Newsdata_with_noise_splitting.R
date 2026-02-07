library(Iso)
library(matrixStats)
library(rmutil)
library(stats)
library(rpart)   # nested trees (Breiman pruning path)

setwd("C:/Users/xchen/Desktop/Study/Research/Jason Klusowski/Tree")

# ============================================================
# USER OPTIONS
# ============================================================

## A) Add independent irrelevant (noise) features?
add_noise_features <- FALSE     # TRUE/FALSE
m_noise <- 180                 # how many noise features to add (ignored if add_noise_features=FALSE)
sd_min <- 1                    # noise sd range
sd_max <- 10
noise_seed <- 111              # seed used only for noise generation

## B) Internal split for model selection vs stacking?
use_internal_split <- FALSE    # TRUE/FALSE
sel_prop <- 0.20               # fraction of TRAIN used for model selection only (ignored if use_internal_split=FALSE)
internal_split_seed <- 222     # seed for internal split

## C) Train/Test split
test_seed <- 111
ntest <- 500

# ============================================================
# 0) Read and preprocess data
# ============================================================

raw_data <- read.csv("News.csv", stringsAsFactors = FALSE)
raw_data <- raw_data[, !colnames(raw_data) %in% c("weekday_is_sunday", "is_weekend", "LDA_04")]
raw_data <- raw_data[, -c(1, 2)]

raw_data <- as.data.frame(lapply(raw_data, function(col) as.numeric(as.character(col))))

# ------------------------------------------------------------
# Optional: add new irrelevant (independent) features BEFORE transform/design matrix
# ------------------------------------------------------------
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
  cat(sprintf("Added %d independent noise features (sd in [%.2f, %.2f]).\n\n", m_noise, sd_min, sd_max))
} else {
  m_noise <- 0
  cat("No noise features added.\n\n")
}

# ------------------------------------------------------------
# Transform the target
# ------------------------------------------------------------
cols_to_transform <- c("shares")
for (col in cols_to_transform) {
  if (!col %in% names(raw_data)) {
    warning(sprintf("Column %s not found in raw_data", col))
    next
  }
  x <- raw_data[[col]]
  z <- rep(0, length(x))
  idx <- x > 0
  z[idx] <- log(x[idx])
  raw_data[[col]] <- z
}

# y is the target
y <- raw_data[[ncol(raw_data)]]

# X as a numeric design matrix (no intercept)
X_df <- raw_data[, -ncol(raw_data), drop = FALSE]
X <- model.matrix(~ . - 1, data = X_df)

# ============================================================
# 1) Train/Test split
# ============================================================

set.seed(test_seed)
n <- nrow(X)
p <- ncol(X)

test_idx <- sample(seq_len(n), size = ntest, replace = FALSE)
train_idx <- setdiff(seq_len(n), test_idx)

X_train <- X[train_idx, , drop = FALSE]
y_train <- y[train_idx]
X_test  <- X[test_idx, , drop = FALSE]
y_test  <- y[test_idx]

cat(sprintf("Train/Test split: train=%d, test=%d\n\n", length(train_idx), length(test_idx)))

# ============================================================
# 2) Internal split (selection vs stacking)
# ============================================================

n_train <- nrow(X_train)

if (use_internal_split) {
  set.seed(internal_split_seed)
  perm <- sample(seq_len(n_train))
  n_sel <- max(30, min(n_train - 30, floor(sel_prop * n_train))) # keep both sides non-trivial
  
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
  cat(sprintf("Internal split OFF: using ALL training for selection+stacking: n_train=%d\n\n", n_train))
}

n2 <- nrow(X_stack)  # stacking sample size

# ============================================================
# 3) Whitening computed on STACK part (then applied to STACK + TEST)
# ============================================================

col_means_stack <- colMeans(X_stack)
Xc_stack <- sweep(X_stack, 2, col_means_stack, FUN = "-")
Xc_test  <- sweep(X_test,  2, col_means_stack, FUN = "-")

safe_chol <- function(G) {
  eps <- 1e-10
  for (k in 0:6) {
    add <- if (k == 0) 0 else (10^(k-6))
    out <- try(chol(G + diag(add + eps, ncol(G))), silent = TRUE)
    if (!inherits(out, "try-error")) return(out)
  }
  stop("Cholesky failed; matrix may be singular. Try removing collinear columns.")
}

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
# 5) Tuning parameters
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

eta <- 1/2  # kept

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

keep_idx <- which(d_internal_all %% 10 == 0)
if (length(keep_idx) == 0) {
  warning("No trees with internal nodes multiple of 10 found; using all available subtrees.")
  keep_idx <- seq_len(nrow(cp_table))
}

cp_table  <- cp_table[keep_idx, , drop = FALSE]
df2       <- cp_table[, "nsplit"]        # internal nodes
cp_values <- cp_table[, "CP"]
k_leaves  <- df2 + 1

cat(sprintf("\nNested trees selected (on SELECTION part): total=%d\n\n", length(df2)))

# (B2) Evaluate the pruned subtrees on STACK and TEST; risks computed on STACK
tr_df_tree_stack <- data.frame(X_stack)
te_df_tree_test  <- data.frame(X_test)

U_tree_stack <- matrix(nrow = n2,    ncol = length(df2))
Utest_tree   <- matrix(nrow = ntest, ncol = length(df2))
R_tree <- numeric(length(df2))

for (kk in seq_along(df2)) {
  cat(sprintf(
    "Tree path iteration: %d / %d | leaves: %d | internal nodes d_k: %d\n",
    kk, length(df2), k_leaves[kk], df2[kk]
  ))
  
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

results_dir <- "results_with_noise_and_split"
if (!dir.exists(results_dir)) dir.create(results_dir)

tag_split <- if (use_internal_split) sprintf("split_sel%.2f", sel_prop) else "nosplit"
tag_noise <- if (add_noise_features) sprintf("noise_m%d", m_noise) else "noise_m0"

file_name <- file.path(results_dir, paste0("Newsdata_treeonly_result_", tag_noise, "_", tag_split, ".txt"))

write.table(mse_results_tree, file = file_name,
            row.names = FALSE, col.names = TRUE, quote = FALSE)

cat(sprintf("\nSaved results to: %s\n", file_name))
