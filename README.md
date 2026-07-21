# Reproducibility code for *When Does Stacking Outperform Single Model Selection?*

This repository contains the R code and data snapshots used for the simulation and real-data experiments in the paper. It reproduces Tables 1-6: backward-deletion simulations, Online News Popularity, California Housing, Superconductivity, and Communities and Crime.

All paths are relative to the repository. Run every command below from the repository root.

## Repository layout

```text
.
|-- Simulation codes/
|   `-- Backward.R
|-- Real data/
|   |-- TREE_Newsdata_with_noise_splitting.R
|   |-- TREE_Calihousing_with_noise_splitting.R
|   |-- TREE_Superconductivity_with_noise_splitting.R
|   `-- TREE_community_crime_with_noise_splitting.R
|-- data/
|   |-- README.md
|   |-- News.csv
|   |-- housing.csv
|   |-- train.csv
|   `-- communities.data
|-- scripts/
|   |-- reproduce_all_tables.R
|   `-- collect_results.R
`-- results/                 # created when the scripts run
```

## Software requirements

- R 4.x
- R packages: `Iso`, `matrixStats`, `rmutil`, `leaps`, `ggplot2`, `tidyr`, `dplyr`, and `rpart`
- Base/recommended R package: `stats`

The validation reported for this repository used R 4.1.3 with `Iso` 0.0-18.1, `matrixStats` 0.63.0, `rmutil` 1.1.10, `leaps` 3.1, `ggplot2` 3.4.4, `tidyr` 1.2.0, `dplyr` 1.1.1, and `rpart` 4.1-16.

Install the required packages once with:

```r
install.packages(c(
  "Iso", "matrixStats", "rmutil", "leaps",
  "ggplot2", "tidyr", "dplyr", "rpart"
))
```

For an exact software record, save `sessionInfo()` after a run:

```sh
Rscript -e "writeLines(capture.output(sessionInfo()), 'results/sessionInfo.txt')"
```

## Data

The four input data files used by the scripts are included under [`data/`](data/), so no manual download or machine-specific path is required. [`data/README.md`](data/README.md) records the upstream source, citation, license information, row count, file size, and SHA-256 checksum for each snapshot.

## Reproduce all paper tables

The following command runs every configuration for Tables 1-6 and then creates paper-formatted CSV files under `results/paper-tables/`:

```sh
Rscript scripts/reproduce_all_tables.R
```

The full run is computationally intensive: it includes 1,000 Monte Carlo replications for each simulation setting and 28 large-tree fits for the real-data tables. The scripts print progress to the terminal.

For a quick simulation smoke test, reduce the replication count. For example:

```sh
# macOS/Linux
N_REPS=2 Rscript scripts/reproduce_all_tables.R

# Windows PowerShell
$env:N_REPS=2; Rscript scripts/reproduce_all_tables.R
```

The paper results use the default `N_REPS=1000`.

## Table-by-table instructions

### Tables 1 and 2: backward deletion

Table 1 uses 20 active covariates and Table 2 uses 35. Each run writes a CSV and PDF to `results/table-1-2-backward/`.

```sh
# macOS/Linux
NUM_COEF=20 N_REPS=1000 Rscript "Simulation codes/Backward.R"  # Table 1
NUM_COEF=35 N_REPS=1000 Rscript "Simulation codes/Backward.R"  # Table 2
```

```powershell
# Windows PowerShell
$env:NUM_COEF=20; $env:N_REPS=1000; Rscript "Simulation codes/Backward.R"  # Table 1
$env:NUM_COEF=35; $env:N_REPS=1000; Rscript "Simulation codes/Backward.R"  # Table 2
```

The simulation records signal norms from 1 to 5 in increments of 0.5. The paper tables use the integer-norm rows and report MSE multiplied by 1,000.

### Tables 3-6: real-data tree pruning

Each table uses the baseline data and then adds 10, 20, ..., 60 independent Gaussian noise features. `M_NOISE=0` is the baseline. Run the seven settings for the corresponding script:

| Paper table | Dataset | Script | Per-run output directory |
|---|---|---|---|
| Table 3 | Online News Popularity | `Real data/TREE_Newsdata_with_noise_splitting.R` | `results/table-3-online-news/` |
| Table 4 | California Housing | `Real data/TREE_Calihousing_with_noise_splitting.R` | `results/table-4-california-housing/` |
| Table 5 | Superconductivity | `Real data/TREE_Superconductivity_with_noise_splitting.R` | `results/table-5-superconductivity/` |
| Table 6 | Communities and Crime | `Real data/TREE_community_crime_with_noise_splitting.R` | `results/table-6-communities-crime/` |

Example for Table 3:

```sh
# macOS/Linux
for n in 0 10 20 30 40 50 60; do
  M_NOISE=$n Rscript "Real data/TREE_Newsdata_with_noise_splitting.R"
done
```

```powershell
# Windows PowerShell
0,10,20,30,40,50,60 | ForEach-Object {
  $env:M_NOISE=$_
  Rscript "Real data/TREE_Newsdata_with_noise_splitting.R"
}
```

Use the same loop with the script listed in the table for Tables 4-6. All scripts use the seeds and train/test sizes reported in the paper. The baseline and noise-feature runs therefore remain deterministic for a fixed R and `rpart` version.

After the required runs finish, assemble the six display tables with:

```sh
Rscript scripts/collect_results.R
```

The collector multiplies MSE by 1,000, selects the integer signal norms for Tables 1-2, and computes

```text
Relative Improvement (%) = 100 * (AIC - Stacking) / AIC.
```

The final files are `results/paper-tables/table-1.csv` through `table-6.csv`.

## Reproducibility notes

- The simulation seed is `123`.
- The real-data test-split seed and noise-feature seed are both `111`.
- The optional internal selection/stacking split is disabled, matching the paper experiments.
- Regression trees use `rpart` with `cp = 0`, `xval = 0`, `minsplit = 20`, and `maxdepth = 30`. The retained pruning-path models have numbers of internal nodes that are multiples of 10.
- Raw per-run MSE values are saved before the paper's `x 10^3` display scaling.
