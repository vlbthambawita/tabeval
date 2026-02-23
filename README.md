# Tabular Data Evaluation

A Python pipeline for evaluating synthetic tabular data generation. It loads tabular datasets, creates stratified train/test splits, trains SDV (Synthetic Data Vault) synthesizers on subsamples, generates synthetic data, and evaluates quality via visualizations and ML augmentation metrics.

## Features

- **Data loading**: CSV/Excel files or [SDV demo datasets](https://docs.sdv.dev/sdv/datasets/demos)
- **Stratified splitting**: Train/test split and subsamples for low-data scenarios
- **Synthesizers**: Gaussian Copula (fast), CTGAN, TVAE
- **Visualizations**: Bar plots, comparative plots (subsamples + synthetic), real vs synthetic distribution plots
- **ML augmentation metrics**: BinaryClassifierPrecisionEfficacy and BinaryClassifierRecallEfficacy (via SDMetrics)

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

The main Python script that powers all three shell wrappers. It provides:

| Option | Description |
|--------|-------------|
| `--data-path PATH` | Load from CSV/Excel file |
| `--sdv-demo` | Use SDV demo dataset |
| `--sdv-modality` | SDV modality (e.g. `single_table`) |
| `--sdv-dataset` | Demo dataset name: `child`, `census`, `news`, etc. |
| `--stratify-column` | Column for stratified sampling |
| `--test-size` | Size of held-out test set |
| `--subsample-sizes` | Comma-separated sizes (e.g. `400,200`) |
| `--train-synthesizer` | `gaussian_copula`, `ctgan`, or `tvae` |
| `--save-synthetic` | Save synthetic CSV files |
| `--eval-visualizations` | Generate real vs synthetic distribution plots |
| `--eval-ml-augmentation` | Compute BinaryClassifierPrecision/RecallEfficacy |
| `--eval-k-runs` | K synthetic runs per subsample (for mean±std) |
| `--prediction-column` | Target column for ML metrics |
| `--minority-class-label` | Positive class label for binary metrics |
| `--ml-label-encode` | Label-encode categoricals (fixes XGBoost with high-cardinality) |
| `--comparative-plots` | All-in-one comparative plots per column |

---

## Shell Scripts: Use Cases

Three preconfigured shell scripts run `tabular_evaluation.py` with different datasets and settings. Edit the variables at the top of each script, then run:

```bash
./run_tabular_evaluation.sh
./run_tabular_evaluation_num_only.sh
./run_tabular_evaluation_num_both.sh
```

### 1. `run_tabular_evaluation.sh` — Categorical Medical/Health Data

| Setting | Value |
|---------|-------|
| **Dataset** | `child` (SDV demo) |
| **Stratify** | `Disease` |
| **Prediction** | `Sick` (binary: yes/no) |
| **Minority class** | `yes` |

**Use case**: Datasets with categorical targets and moderate numbers of categorical features (e.g., medical, health outcomes). No label encoding; suitable when categorical cardinality is low.

```bash
./run_tabular_evaluation.sh
```

---

### 2. `run_tabular_evaluation_num_only.sh` — News / Text-Derived Data

| Setting | Value |
|---------|-------|
| **Dataset** | `news` (SDV demo) |
| **Stratify** | `label` |
| **Prediction** | `label` |
| **Minority class** | `50000+` (adjust for your label semantics) |

**Use case**: Datasets from news or similar text-derived sources where the prediction target is a single categorical label column. Configure `MINORITY_CLASS` to match your positive class.

```bash
./run_tabular_evaluation_num_only.sh
```

---

### 3. `run_tabular_evaluation_num_both.sh` — Census / High-Cardinality Categoricals

| Setting | Value |
|---------|-------|
| **Dataset** | `census` (SDV demo) |
| **Stratify** | `label` (income) |
| **Prediction** | `label` |
| **Minority class** | `50000+` |
| **ML label encode** | `--ml-label-encode` ✅ |

**Use case**: Datasets with many categorical columns (e.g., demographics, occupation, education). Enables `--ml-label-encode` to avoid XGBoost `enable_categorical` issues during ML augmentation evaluation.

```bash
./run_tabular_evaluation_num_both.sh
```

---

## Output Structure

```
output/
├── <dataset>/                    # e.g. child, census, news
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── subsample_400.csv
│   ├── subsample_200.csv
│   └── plots/
│       ├── subsample_*.png       # Bar plots per subsample
│       └── comparative/          # Comparative plots per column
│           └── *.png
└── synthetic/<dataset>/<synthesizer>/
    ├── subsample_*_synthetic_run0.csv
    ├── subsample_*_metadata.json
    ├── eval_plots/               # Real vs synthetic (if --eval-visualizations)
    │   └── subsample_*/
    │       └── column_*.pdf
    ├── ml_augmentation_eval.json # If --eval-ml-augmentation
    └── plots/<dataset>/comparative/
```

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
| Medical/health, few categoricals | `run_tabular_evaluation.sh` |
| News/text-derived, single label | `run_tabular_evaluation_num_only.sh` |
| Census-like, many categoricals | `run_tabular_evaluation_num_both.sh` |
| Custom CSV | Copy any script, switch to `--data-path` |

---

## License

See repository license file.
