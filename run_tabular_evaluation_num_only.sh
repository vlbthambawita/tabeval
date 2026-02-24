#!/bin/bash
# Run tabular data evaluation script
# Edit the parameters below, then run: ./run_tabular_evaluation.sh

# Data source: use --sdv-demo OR --data-path PATH (comment out the one you don't use)
DATA_SOURCE="--sdv-demo"
# DATA_SOURCE="--data-path /path/to/data.csv"

# SDV options (when using --sdv-demo)
SDV_MODALITY="single_table"
SDV_DATASET="news"

# Stratification & split (test set is held out and used as ML validation for --eval-ml-augmentation)
STRATIFY_COLUMN="label"
TEST_SIZE=1000
SUBSAMPLE_SIZES="400,200"
RANDOM_STATE=42

# Output
OUTPUT_DIR="output"
SAVE_DATASETS="--save-datasets"
NO_PLOTS=""           # add "--no-plots" to skip bar plots
COMPARATIVE_PLOTS="--comparative-plots"  # add to generate all-in-one comparative plots per column
# Synthesizer: gaussian_copula (fast), ctgan, tvae (see https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers)
TRAIN_SYNTHESIZER="--train-synthesizer gaussian_copula"  # set to "--train-synthesizer gaussian_copula" or "ctgan" or "tvae" to enable
SAVE_SYNTHETIC="--save-synthetic"   # add to save synthetic datasets to output/synthetic/<dataset>/<synthesizer>/
EVAL_VISUALIZATIONS="--eval-visualizations"  # add to generate SDV eval plots (column + pair) per subsample
EVAL_PLOT_FORMAT="--eval-plot-format pdf"   # pdf or png
EVAL_ML_AUGMENTATION="--eval-ml-augmentation"   # add "--eval-ml-augmentation" to evaluate BinaryClassifierPrecision/RecallEfficacy
EVAL_PRIVACY="--eval-privacy"   # add "--eval-privacy" to evaluate privacy metrics
EVAL_PRIVACY_SUBSAMPLE=""      # optional: set to "--eval-privacy-subsample 500" for faster computation
EVAL_QUALITY="--eval-quality"   # add "--eval-quality" to evaluate ContingencySimilarity for column pairs
EVAL_QUALITY_SUBSAMPLE=""      # optional: set to "--eval-quality-subsample 500" for faster computation
# DisclosureProtection (optional): set EVAL_PRIVACY_DISCLOSURE="--eval-privacy-disclosure" to enable
EVAL_PRIVACY_DISCLOSURE=""     # set to "--eval-privacy-disclosure" to evaluate DisclosureProtection
EVAL_PRIVACY_DISCLOSURE_KNOWN=""       # e.g. "--eval-privacy-disclosure-known col1,col2" (required if EVAL_PRIVACY_DISCLOSURE is set)
EVAL_PRIVACY_DISCLOSURE_SENSITIVE=""   # e.g. "--eval-privacy-disclosure-sensitive label" (required if EVAL_PRIVACY_DISCLOSURE is set)
EVAL_PRIVACY_DISCLOSURE_CONTINUOUS=""  # optional: continuous cols needing discretization
EVAL_K_RUNS="--eval-k-runs 5"   # K training runs per subsample (saves *_synthetic_run0.csv, run1.csv, etc.); use with EVAL_ML_AUGMENTATION for mean±std
PREDICTION_COLUMN="--prediction-column label"   # add "--prediction-column Disease" for ML augmentation (default: stratify column)
MINORITY_CLASS="--minority-class-label 50000+"   # add "--minority-class-label Fallot" (or other class from prediction column)
SYNTHESIZER_EPOCHS="--synthesizer-epochs 100"           # for CTGAN/TVAE only
QUIET=""              # add "-q" to reduce verbosity

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 tabular_evaluation.py \
  $DATA_SOURCE \
  --sdv-modality "$SDV_MODALITY" \
  --sdv-dataset "$SDV_DATASET" \
  --stratify-column "$STRATIFY_COLUMN" \
  --test-size "$TEST_SIZE" \
  --subsample-sizes "$SUBSAMPLE_SIZES" \
  --random-state "$RANDOM_STATE" \
  -o "$OUTPUT_DIR" \
  $SAVE_DATASETS \
  $NO_PLOTS \
  $COMPARATIVE_PLOTS \
  $TRAIN_SYNTHESIZER \
  $SAVE_SYNTHETIC \
  $EVAL_VISUALIZATIONS \
  $EVAL_PLOT_FORMAT \
  $EVAL_ML_AUGMENTATION \
  $EVAL_PRIVACY \
  $EVAL_PRIVACY_SUBSAMPLE \
  $EVAL_QUALITY \
  $EVAL_QUALITY_SUBSAMPLE \
  $EVAL_PRIVACY_DISCLOSURE \
  $EVAL_PRIVACY_DISCLOSURE_KNOWN \
  $EVAL_PRIVACY_DISCLOSURE_SENSITIVE \
  $EVAL_PRIVACY_DISCLOSURE_CONTINUOUS \
  $EVAL_K_RUNS \
  $PREDICTION_COLUMN \
  $MINORITY_CLASS \
  $SYNTHESIZER_EPOCHS \
  $QUIET
