#!/bin/bash
# Run tabular data evaluation script
# Edit the parameters below, then run: ./run_tabular_evaluation.sh

# Data source: use --sdv-demo OR --data-path PATH (comment out the one you don't use)
DATA_SOURCE="--sdv-demo"
# DATA_SOURCE="--data-path /path/to/data.csv"

# SDV options (when using --sdv-demo)
SDV_MODALITY="single_table"
SDV_DATASET="child"

# Stratification & split
STRATIFY_COLUMN="Disease"
TEST_SIZE=1000
SUBSAMPLE_SIZES="1000,800,600,400,200"
RANDOM_STATE=42

# Output
OUTPUT_DIR="output"
SAVE_DATASETS="--save-datasets"
NO_PLOTS=""           # add "--no-plots" to skip bar plots
COMPARATIVE_PLOTS="--comparative-plots"  # add to generate all-in-one comparative plots per column
# Synthesizer: gaussian_copula (fast), ctgan, tvae (see https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers)
TRAIN_SYNTHESIZER="--train-synthesizer gaussian_copula"  # set to "--train-synthesizer gaussian_copula" or "ctgan" or "tvae" to enable
SAVE_SYNTHETIC="--save-synthetic"   # add to save synthetic datasets to output/synthetic/<synthesizer>/
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
  $SYNTHESIZER_EPOCHS \
  $QUIET
