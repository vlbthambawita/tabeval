# Tabular Data Evaluation

A Python pipeline for evaluating synthetic tabular data generation. It loads tabular datasets, creates stratified train/test splits, trains SDV (Synthetic Data Vault) synthesizers on subsamples, generates synthetic data, and evaluates quality via visualizations and ML augmentation metrics.

## Quickstart: useful command batches

- **Run full evaluation on Adult-like mixed data (synthetic + ML + privacy + quality)**:

  ```bash
  ./run_tabular_evaluation_cat_and_num.sh
  ```

- **Run full evaluation on medical-style categorical data (child demo)**:

  ```bash
  ./run_tabular_evaluation_cat_only.sh
  ```

- **Run full evaluation on news/text-style label data (news demo)**:

  ```bash
  ./run_tabular_evaluation_num_only.sh
  ```

- **Use your own CSV with the Adult-style config** (replace dataset path and columns as needed):

  ```bash
  # In run_tabular_evaluation_cat_and_num.sh
  DATA_SOURCE="--data-path /path/to/your/data.csv"
  SDV_DATASET="adult"          # unused when --data-path is set
  STRATIFY_COLUMN="label"      # change to your column
  PREDICTION_COLUMN="--prediction-column label"
  MINORITY_CLASS="--minority-class-label >50K"
  ```

- **Minimal quality-only evaluation from the CLI (no ML, no privacy)**:

  ```bash
  python3 tabular_evaluation.py \
    --sdv-demo \
    --sdv-modality single_table \
    --sdv-dataset adult \
    --stratify-column label \
    --test-size 1000 \
    --subsample-sizes 400,200 \
    --train-synthesizer gaussian_copula \
    --eval-quality \
    --eval-quality-subsample 500 \
    --save-datasets \
    -o output
  ```

---

## Features

- **Data loading**: CSV/Excel files or [SDV demo datasets](https://docs.sdv.dev/sdv/datasets/demos)
- **Stratified splitting**: Train/test split and subsamples for low-data scenarios
- **Synthesizers**: Gaussian Copula (fast), CTGAN, TVAE
- **Visualizations**:
  - Bar plots for each subsample
  - Comparative plots across subsamples (and optionally synthetic data)
  - Real vs synthetic distribution plots (per column, with optional K-run mean±std overlays)
- **ML augmentation metrics**:
  - **Classification**: BinaryClassifierPrecisionEfficacy and BinaryClassifierRecallEfficacy (via SDMetrics)
  - **Regression**: LinearRegression and MLPRegressor ML efficacy scores when the prediction target is numerical
- **Privacy metrics** (via SDMetrics):
  - DCRBaselineProtection, DCROverfittingProtection
  - Optional DisclosureProtection with configurable attacker-known and sensitive columns
- **Quality metrics** (via SDMetrics):
  - KSComplement / TVComplement (marginal distributions)
  - ContingencySimilarity (categorical/mixed pairs)
  - CorrelationSimilarity (numerical pairs; Pearson or Spearman)
  - Configurable subsampling and real-association/correlation thresholds

---

## Requirements

- Python 3.9+
- pandas, scikit-learn
- matplotlib, seaborn (for plots)
- [SDV](https://github.com/sdv-dev/SDV) (for synthesizers and demo data)
- xgboost (for ML augmentation metrics)

```bash
pip install pandas scikit-learn matplotlib seaborn sdv xgboost
```

---

## Core Script: `tabular_evaluation.py`

The main Python script that powers all shell wrappers. It provides:

### Data loading and splitting

| Option | Description |
|--------|-------------|
| `--data-path PATH` | Load from CSV/Excel file. Supported extensions: `.csv`, `.xlsx`, `.xls`. |
| `--sdv-demo` | Use an SDV demo dataset instead of a file. |
| `--sdv-modality` | SDV modality (e.g. `single_table`, `multi_table`). |
| `--sdv-dataset` | Demo dataset name: `child`, `news`, `adult`, etc. |
| `--stratify-column` | Column for stratified sampling (also the default prediction target). Numeric columns are automatically binned into quantiles for stable stratification. |
| `--test-size` | **Number of rows** in the held-out test set (not a fraction). |
| `--subsample-sizes` | Comma-separated subsample sizes (e.g. `400,200`) drawn from the training set, stratified on `--stratify-column`. |
| `--random-state` | Random seed for reproducible splits and synthesizer training. |
| `-o`, `--output-dir` | Base output directory (default: `output`). |
| `--save-datasets` | Save `train_data.csv`, `test_data.csv`, and all `subsample_*.csv` files. |
| `-q`, `--quiet` | Reduce console verbosity. |

### Basic plots and comparative plots

| Option | Description |
|--------|-------------|
| `--no-plots` | Skip per-subsample bar plots. |
| `--comparative-plots` | Generate comparative plots per column showing all subsamples (and, when synthesizers are enabled, also synthetic variants). Categorical columns use percentage bar charts; numerical columns use box plots. |

### Synthesizer training and synthetic data

| Option | Description |
|--------|-------------|
| `--train-synthesizer {gaussian_copula,ctgan,tvae}` | Enable SDV single-table synthesizer training for each subsample. |
| `--save-synthetic` | Save generated synthetic datasets as CSV files under `output/synthetic/<dataset>/<synthesizer>/`. |
| `--synthesizer-epochs` | Training epochs for CTGAN/TVAE (ignored for GaussianCopula). |
| `--eval-k-runs K` | Train the synthesizer K times per subsample, producing K synthetic datasets (`*_synthetic_run0.csv`, `run1.csv`, …). All evaluation metrics aggregate scores across these runs (mean±std). |

### Real vs synthetic distribution plots

| Option | Description |
|--------|-------------|
| `--eval-visualizations` | Generate column-wise real vs synthetic distribution plots for each subsample under `eval_plots/`. For categorical columns, synthetic bars can include mean±std across K runs; for numerical columns, KDE plots compare real vs synthetic densities. |
| `--eval-plot-format {pdf,png}` | File format for evaluation plots (default: `pdf`). |

### ML augmentation evaluation

These metrics answer: **if we augment the real training data with synthetic data, how well do downstream models perform on real held-out test data?**

| Option | Description |
|--------|-------------|
| `--eval-ml-augmentation` | Enable ML augmentation evaluation. For categorical targets, runs **BinaryClassifierPrecisionEfficacy** and **BinaryClassifierRecallEfficacy**; for numerical targets, runs **LinearRegression** and **MLPRegressor** ML efficacy metrics. |
| `--prediction-column COL` | Target column name. Defaults to `--stratify-column`. |
| `--minority-class-label VALUE` | Required for **categorical** targets with `--eval-ml-augmentation`; defines the positive/minority class (e.g. `>50K`). Not required when the prediction column is numerical (regression metrics are used). |
| `--ml-label-encode` | Label-encode categorical features to integers prior to ML evaluation. This avoids XGBoost `enable_categorical` issues when there are many categorical columns. |

ML augmentation outputs are saved to `ml_augmentation_eval.json` (see **Output Structure** below).

### Privacy evaluation

| Option | Description |
|--------|-------------|
| `--eval-privacy` | Enable privacy evaluation across synthetic datasets using SDMetrics. Computes **DCRBaselineProtection** and **DCROverfittingProtection** (when test/validation data is available). |
| `--eval-privacy-subsample N` | Optionally subsample N rows when computing DCR metrics, to speed up evaluation on large datasets. |
| `--eval-privacy-disclosure` | Additionally compute **DisclosureProtection**, which estimates how well an attacker could infer sensitive attributes from known attributes. |
| `--eval-privacy-disclosure-known COL1,COL2,...` | Comma-separated list of attacker-known columns for DisclosureProtection (e.g. demographics). |
| `--eval-privacy-disclosure-sensitive COL1,COL2,...` | Comma-separated list of sensitive target columns for DisclosureProtection (e.g. disease status). |
| `--eval-privacy-disclosure-continuous COL1,COL2,...` | Optional list of continuous columns that should be discretized for DisclosureProtection. |
| `--eval-privacy-disclosure-computation {cap,generalized_cap,zero_cap}` | CAP computation method for DisclosureProtection (default: `cap`). |

Privacy outputs are saved to `privacy_eval.json` (see **Output Structure** below).

### Quality evaluation (statistical similarity)

| Option | Description |
|--------|-------------|
| `--eval-quality` | Enable statistical quality metrics comparing real vs synthetic data. |
| `--eval-quality-subsample N` | Optionally subsample N rows per evaluation to reduce runtime on large datasets. |
| `--eval-quality-threshold FLOAT` | Real association threshold for **ContingencySimilarity**; only column pairs whose real association exceeds this threshold are included (others are set to NaN). Recommended values: `0.3` or higher. |
| `--eval-quality-correlation-coefficient {Pearson,Spearman}` | Correlation coefficient used by **CorrelationSimilarity** for numerical pairs (default: `Pearson`). |
| `--eval-quality-correlation-threshold FLOAT` | Real correlation threshold for **CorrelationSimilarity**; pairs with \|r\| below this threshold are ignored (set to NaN). Recommended values: `0.4` or higher. |

Quality outputs are saved to `quality_eval.json` (see **Output Structure** below).

---

## Shell Scripts: Use Cases

Preconfigured shell scripts run `tabular_evaluation.py` with different datasets and settings. Edit the variables at the top of each script, then run:

```bash
./run_tabular_evaluation_cat_only.sh
./run_tabular_evaluation_cat_only.sh
./run_tabular_evaluation_num_only.sh
./run_tabular_evaluation_cat_and_num.sh
```

### 1. `run_tabular_evaluation_cat_only.sh` — Categorical Medical/Health Data (`child` demo)

| Setting | Value |
|---------|-------|
| **Dataset** | `child` (SDV demo) |
| **Stratify** | `Disease` |
| **Prediction** | `Sick` (binary: yes/no) |
| **Minority class** | `yes` |
| **Evaluations** | ML augmentation (binary classification), privacy (including DisclosureProtection), optional quality metrics |

**Use case**: Datasets with mostly categorical columns and medical-style outcomes (e.g., diseases, diagnoses, yes/no flags). This script is configured to:

- use the `child` demo as a proxy for pediatric/health datasets
- evaluate ML augmentation performance on a binary outcome (`Sick`)
- evaluate privacy, including DisclosureProtection with:
  - attacker-known columns: `Disease`
  - sensitive columns: `Age`

```bash
./run_tabular_evaluation_cat_only.sh
```

---

### 2. `run_tabular_evaluation_num_only.sh` — News / Text-Derived Data (`news` demo)

| Setting | Value |
|---------|-------|
| **Dataset** | `news` (SDV demo) |
| **Stratify** | `label` |
| **Prediction** | `label` |
| **Minority class** | `50000+` (adjust for your label semantics) |
| **Evaluations** | ML augmentation (binary classification), privacy, optional quality metrics |

**Use case**: Datasets from news or similar text-derived sources where the prediction target is a single categorical label column. Configure `MINORITY_CLASS` to match your positive class.

```bash
./run_tabular_evaluation_num_only.sh
```

---

### 3. `run_tabular_evaluation_cat_and_num.sh` — Adult / Mixed Categorical + Numerical (`adult` demo)

| Setting | Value |
|---------|-------|
| **Dataset** | `adult` (SDV demo) |
| **Stratify** | `label` (income) |
| **Prediction** | `label` |
| **Minority class** | `>50K` |
| **ML label encode** | `--ml-label-encode` ✅ |
| **Evaluations** | ML augmentation (binary classification), privacy, optional quality metrics |

**Use case**: Datasets with both categorical and numerical features (e.g., Adult income: age, education, occupation, etc.). Uses `--ml-label-encode` for high-cardinality categoricals to ensure reliable ML augmentation evaluation.

```bash
./run_tabular_evaluation_cat_and_num.sh
```

---

## Output Structure

```
output/
├── <dataset>/                    # e.g. child, news, adult
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── subsample_400.csv
│   ├── subsample_200.csv
├── plots/
│   └── <dataset>/
│       ├── subsample_*.png       # Bar plots per subsample (if not --no-plots)
│       └── comparative/          # Comparative plots per column across subsamples (and optionally synthetic)
│           └── *.png
└── synthetic/<dataset>/<synthesizer>/
    ├── subsample_*_synthetic_run0.csv
    ├── subsample_*_synthetic_run1.csv
    ├── subsample_*_synthetic_run*.csv
    ├── subsample_*_metadata.json
    ├── eval_plots/               # Real vs synthetic plots (if --eval-visualizations)
    │   └── subsample_*/
    │       └── column_*.pdf/.png
    ├── ml_augmentation_eval.json # If --eval-ml-augmentation
    ├── privacy_eval.json         # If --eval-privacy
    └── quality_eval.json         # If --eval-quality
```

### JSON output files

- **`ml_augmentation_eval.json`** (per dataset & synthesizer)
  - Top-level keys: subsample names (e.g. `subsample_400`, `subsample_200`).
  - Each subsample contains metric objects, e.g.:
    - `BinaryClassifierPrecisionEfficacy`, `BinaryClassifierRecallEfficacy` for categorical targets.
    - `LinearRegression`, `MLPRegressor` for numerical targets.
  - Each metric object stores:
    - `mean`: average score across K synthetic runs.
    - `std`: standard deviation across K runs.
    - `scores`: list of per-run scores.

- **`privacy_eval.json`**
  - Top-level keys: subsample names.
  - Each subsample includes metrics such as:
    - `DCRBaselineProtection`: mean±std across runs; may include `median_DCR_to_real_data`.
    - `DCROverfittingProtection`: mean±std across runs; may include `synthetic_data_percentages` describing how many synthetic rows are close to training vs validation data.
    - `DisclosureProtection` (if configured): mean±std, and optional fields such as `cap_protection` and `baseline_protection`.

- **`quality_eval.json`**
  - Top-level keys: subsample names.
  - Each subsample includes metrics such as:
    - `KSComplement`: summary over numerical marginals with per-column means and counts.
    - `TVComplement`: summary over categorical marginals with per-column means and counts.
    - `ContingencySimilarity`: pairwise categorical/mixed association quality (with `num_pairs`, `total_pairs`, and optional `pair_means`).
    - `CorrelationSimilarity`: pairwise numerical correlation quality (with `num_pairs`, `total_pairs`, `pair_means`, and the correlation `coefficient` used).
  - For all metrics, only summary statistics (mean/std, counts, and per-column/pair means) are stored; raw per-run scores are not persisted to keep the files compact.

---

## Custom Data

To use your own CSV instead of an SDV demo:

1. Edit the script and set:
   ```bash
   DATA_SOURCE="--data-path /path/to/your/data.csv"
   # DATA_SOURCE="--sdv-demo"   # comment out
   ```
2. Set `STRATIFY_COLUMN`, `PREDICTION_COLUMN`, and `MINORITY_CLASS` to match your schema.
3. For many categorical columns, add `ML_LABEL_ENCODE="--ml-label-encode"`.

---

## Quick Reference: Which Script to Use?

| Your data type | Script |
|----------------|--------|
| Medical/health, mostly categoricals | `run_tabular_evaluation_cat_only.sh` |
| News/text-derived, single label | `run_tabular_evaluation_num_only.sh` |
| Census-/Adult-like, many categoricals + numeric | `run_tabular_evaluation_cat_and_num.sh` |
| Custom CSV | Copy any script, switch to `--data-path` |

---

## License

See repository license file.
