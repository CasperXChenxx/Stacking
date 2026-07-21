## Compare freshly generated display tables with Tables 1-6 in the current
## manuscript (Stacking_Models.pdf, pages 25, 27, and 28).
## This script does not run experiments or supply missing results.

make_table <- function(methods, columns, values) {
  matrix(
    values,
    nrow = length(methods),
    byrow = TRUE,
    dimnames = list(methods, columns)
  )
}

simulation_methods <- c(
  "AIC", "Oracle", "Stacking", "Mallows", "Relative Improvement (%)"
)
simulation_columns <- sprintf("||f|| = %d", 1:5)
real_methods <- c("AIC", "Stacking", "Mallows", "Relative Improvement (%)")
real_columns <- c("Baseline", sprintf("+%d Features", seq(10L, 60L, by = 10L)))

expected <- list(
  `1` = make_table(
    simulation_methods,
    simulation_columns,
    c(
      368.918, 355.273, 347.459, 339.870, 335.534,
      339.818, 302.689, 271.508, 219.512, 196.351,
      294.790, 293.268, 286.487, 270.678, 262.148,
      293.591, 295.611, 290.534, 275.414, 268.597,
      20.093, 17.453, 17.548, 20.358, 21.872
    )
  ),
  `2` = make_table(
    simulation_methods,
    simulation_columns,
    c(
      468.922, 449.454, 425.647, 421.455, 406.208,
      445.302, 442.259, 422.856, 418.182, 386.408,
      403.352, 430.364, 400.175, 400.218, 369.462,
      386.177, 419.486, 394.218, 393.773, 366.109,
      13.983, 4.247, 5.984, 5.039, 9.046
    )
  ),
  `3` = make_table(
    real_methods,
    real_columns,
    c(
      1112.3, 1112.0, 1116.6, 1259.2, 1193.1, 1187.3, 1217.8,
      1018.1, 1012.4, 1027.8, 1148.9, 1098.2, 1085.5, 1108.7,
      1018.5, 1007.1, 1025.2, 1148.6, 1098.5, 1088.9, 1114.7,
      8.471, 8.958, 7.954, 8.756, 7.953, 8.580, 8.963
    )
  ),
  `4` = make_table(
    real_methods,
    real_columns,
    c(
      78.8, 83.3, 89.8, 88.4, 90.9, 89.6, 88.0,
      74.2, 77.6, 81.4, 80.4, 82.0, 80.7, 80.6,
      74.1, 77.9, 82.2, 80.6, 82.0, 80.8, 80.8,
      5.838, 6.846, 9.339, 9.036, 9.846, 9.943, 8.429
    )
  ),
  `5` = make_table(
    real_methods,
    real_columns,
    c(
      398.6, 454.0, 470.9, 484.6, 484.3, 498.8, 501.6,
      384.9, 430.5, 446.5, 457.9, 458.5, 471.5, 474.2,
      381.9, 426.0, 442.2, 456.6, 457.2, 471.4, 472.4,
      3.434, 5.179, 5.171, 5.513, 5.346, 5.473, 5.451
    )
  ),
  `6` = make_table(
    real_methods,
    real_columns,
    c(
      28.9, 29.9, 32.0, 32.2, 30.4, 30.6, 31.2,
      27.7, 28.7, 30.5, 30.7, 29.3, 29.5, 30.0,
      27.7, 28.8, 30.6, 30.6, 29.2, 29.5, 30.1,
      4.048, 4.123, 4.472, 4.644, 3.740, 3.518, 3.720
    )
  )
)

tolerance <- 1e-9
checked_cells <- 0L

for (number in names(expected)) {
  path <- file.path("results", "paper-tables", sprintf("table-%s.csv", number))
  if (!file.exists(path)) {
    stop(sprintf("Missing generated table: %s", path))
  }

  x <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  actual <- as.matrix(x[-1])
  storage.mode(actual) <- "double"
  rownames(actual) <- x$Method
  target <- expected[[number]]

  if (!identical(dimnames(actual), dimnames(target))) {
    stop(sprintf("Table %s row or column labels differ from the manuscript.", number))
  }

  differences <- abs(actual - target)
  bad <- which(differences > tolerance, arr.ind = TRUE)
  if (nrow(bad)) {
    details <- apply(bad, 1L, function(index) {
      row <- index[[1L]]
      column <- index[[2L]]
      sprintf(
        "Table %s, %s, %s: generated %.12g; manuscript %.12g",
        number,
        rownames(actual)[row],
        colnames(actual)[column],
        actual[row, column],
        target[row, column]
      )
    })
    stop(paste(details, collapse = "\n"))
  }

  checked_cells <- checked_cells + length(target)
  cat(sprintf("Table %s: %d numeric cells match.\n", number, length(target)))
}

cat(sprintf(
  "Verified %d numeric cells across Tables 1-6; no differences found.\n",
  checked_cells
))
