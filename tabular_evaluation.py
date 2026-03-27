#!/usr/bin/env python3
"""
Tabular Data Evaluation Script

Creates stratified train/test split and subsamples from tabular data,
then optionally trains SDV synthesizers and evaluates ML, privacy, and quality metrics.

Configuration can be provided via a YAML file (--config) and/or individual CLI flags.
CLI flags always override values from the config file.

Usage examples:
    python tabular_evaluation.py --config config.yaml
    python tabular_evaluation.py --config config.yaml --quiet
    python tabular_evaluation.py --sdv-demo --stratify-column Disease
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from te_data import load_data
from te_splitter import (
    create_stratified_split,
    create_subsamples,
    drop_bin_column,
    save_datasets,
)
from te_synthesizer import train_synthesizers
from te_visualization import generate_comparative_plots, generate_subsample_plots


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

def _load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML configuration file and return its contents as a dict."""
    try:
        import yaml
    except ImportError:
        raise ImportError("Install PyYAML to use --config: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _apply_config_to_namespace(config: dict[str, Any], namespace: argparse.Namespace) -> None:
    """Set attributes on namespace from config, only when the CLI left them at their default."""
    # Map YAML keys (snake_case) to argparse dest names (also snake_case).
    # Entries in config are applied only when the namespace attribute is still None
    # or matches the argparse default (i.e. the user did not supply it on the CLI).
    for key, value in config.items():
        dest = key.replace("-", "_")
        if hasattr(namespace, dest):
            # Only override if the attribute is still None (or was not set by CLI).
            # For boolean flags we check against the specific defaults.
            current = getattr(namespace, dest)
            if current is None:
                setattr(namespace, dest, value)
            elif isinstance(current, bool) and not current:
                # bool flags default to False; apply config value only when still False
                if isinstance(value, bool):
                    setattr(namespace, dest, value)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stratified train/test split and subsamples from tabular data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        metavar="FILE",
        help="Path to a YAML configuration file. CLI flags override config values.",
    )

    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to CSV/Excel file to load (alternative to SDV demo).",
    )
    data_group.add_argument(
        "--sdv-demo",
        action="store_true",
        help="Use SDV demo dataset instead of file.",
    )

    # SDV demo options
    parser.add_argument("--sdv-modality", default=None, help="SDV demo modality (e.g. single_table, multi_table).")
    parser.add_argument("--sdv-dataset", default=None, help="SDV demo dataset name.")

    # Stratification & split
    parser.add_argument("--stratify-column", default=None, help="Column to use for stratified sampling.")
    parser.add_argument("--test-size", type=int, default=None, help="Number of samples in the held-out test set.")
    parser.add_argument(
        "--subsample-sizes",
        type=str,
        default=None,
        help="Comma-separated subsample sizes to create from training data.",
    )
    parser.add_argument("--random-state", type=int, default=None, help="Random seed for reproducibility.")

    # Output
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Directory to save output files.")
    parser.add_argument("--save-datasets", action="store_true", help="Save train, test, and subsample CSVs.")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating bar plots.")
    parser.add_argument("--comparative-plots", action="store_true", help="Generate comparative bar plots per column.")

    # Synthesizer training
    parser.add_argument(
        "--train-synthesizer",
        choices=["gaussian_copula", "ctgan", "tvae"],
        default=None,
        help="Train selected SDV synthesizer on each subsample.",
    )
    parser.add_argument("--save-synthetic", action="store_true", help="Save generated synthetic data to CSV files.")
    parser.add_argument("--synthesizer-epochs", type=int, default=None, help="Training epochs for CTGAN/TVAE.")

    # Eval: visualizations
    parser.add_argument("--eval-visualizations", action="store_true", help="Generate distribution comparison plots.")
    parser.add_argument("--eval-plot-format", choices=["pdf", "png"], default=None, help="Format for eval plots.")

    # Eval: ML augmentation
    parser.add_argument("--eval-ml-augmentation", action="store_true", help="Evaluate ML augmentation metrics.")
    parser.add_argument("--eval-k-runs", type=int, default=None, help="Number of synthesizer runs per subsample.")
    parser.add_argument("--prediction-column", default=None, help="Column to predict for ML augmentation metrics.")
    parser.add_argument("--minority-class-label", default=None, help="Positive/minority class value for binary metrics.")
    parser.add_argument("--ml-label-encode", action="store_true", help="Label-encode categorical columns before ML evaluation.")
    parser.add_argument("--eval-ml-max-epochs", type=int, default=None, metavar="N", help="Maximum epochs for ML evaluation.")

    # Eval: privacy
    parser.add_argument("--eval-privacy", action="store_true", help="Evaluate privacy metrics.")
    parser.add_argument("--eval-privacy-subsample", type=int, default=None, metavar="N", help="Subsample N rows for DCR computation.")
    parser.add_argument("--eval-privacy-disclosure", action="store_true", help="Enable DisclosureProtection evaluation.")
    parser.add_argument("--eval-privacy-disclosure-known", type=str, default=None, metavar="COL1,COL2,...", help="Columns the attacker knows.")
    parser.add_argument("--eval-privacy-disclosure-sensitive", type=str, default=None, metavar="COL1,COL2,...", help="Columns the attacker wants to guess.")
    parser.add_argument("--eval-privacy-disclosure-continuous", type=str, default=None, metavar="COL1,COL2,...", help="Continuous columns for DisclosureProtection.")
    parser.add_argument(
        "--eval-privacy-disclosure-computation",
        choices=["cap", "generalized_cap", "zero_cap"],
        default=None,
        help="CAP computation method for DisclosureProtection.",
    )

    # Eval: quality
    parser.add_argument("--eval-quality", action="store_true", help="Evaluate quality metrics.")
    parser.add_argument("--eval-quality-subsample", type=int, default=None, metavar="N", help="Subsample N rows for quality computation.")
    parser.add_argument("--eval-quality-threshold", type=float, default=None, metavar="FLOAT", help="Real association threshold for ContingencySimilarity.")
    parser.add_argument(
        "--eval-quality-correlation-coefficient",
        choices=["Pearson", "Spearman"],
        default=None,
        help="Correlation coefficient for CorrelationSimilarity.",
    )
    parser.add_argument("--eval-quality-correlation-threshold", type=float, default=None, metavar="FLOAT", help="Real correlation threshold for CorrelationSimilarity.")

    # Misc
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce output verbosity.")

    args = parser.parse_args()

    # Load YAML config and fill in any arguments not provided on the CLI
    if args.config:
        config = _load_yaml_config(args.config)
        _apply_config_to_namespace(config, args)

    # Apply built-in defaults for anything still None after config merge
    _apply_defaults(args)

    # Validate: data source must be specified (either via CLI or config)
    if not args.data_path and not args.sdv_demo:
        parser.error("One of --data-path or --sdv-demo is required (or set in config.yaml).")

    return args


def _apply_defaults(args: argparse.Namespace) -> None:
    """Set fallback defaults for fields not supplied by CLI or config."""
    defaults = {
        "sdv_modality": "single_table",
        "sdv_dataset": "child",
        "stratify_column": "Disease",
        "test_size": 1000,
        "subsample_sizes": "1000,800,600,400,200",
        "random_state": 42,
        "output_dir": Path("output"),
        "synthesizer_epochs": 100,
        "eval_plot_format": "pdf",
        "eval_k_runs": 1,
        "eval_privacy_disclosure_computation": "cap",
        "eval_quality_correlation_coefficient": "Pearson",
    }
    for dest, value in defaults.items():
        if getattr(args, dest, None) is None:
            setattr(args, dest, value)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    """Run the full pipeline and return datasets."""
    data, base_metadata = load_data(args)
    clean_data = data.dropna()

    subsample_sizes = [int(s.strip()) for s in str(args.subsample_sizes).split(",") if s.strip()]

    train_data, test_data, drop_bins = create_stratified_split(
        clean_data=clean_data,
        stratify_column=args.stratify_column,
        test_size=args.test_size,
        random_state=args.random_state,
        quiet=args.quiet,
    )

    stratified_subsamples = create_subsamples(
        train_data=train_data,
        subsample_sizes=subsample_sizes,
        stratify_column=args.stratify_column,
        random_state=args.random_state,
        drop_bins=drop_bins,
        quiet=args.quiet,
    )

    if drop_bins:
        clean_data, train_data, test_data, stratified_subsamples = drop_bin_column(
            clean_data, train_data, test_data, stratified_subsamples
        )

    result = {
        "clean_data": clean_data,
        "train_data": train_data,
        "test_data": test_data,
        "stratified_subsamples": stratified_subsamples,
    }

    dataset_name = args.data_path.stem if args.data_path else args.sdv_dataset

    if args.save_datasets:
        save_datasets(
            train_data=train_data,
            test_data=test_data,
            stratified_subsamples=stratified_subsamples,
            output_dir=args.output_dir,
            dataset_name=dataset_name,
            quiet=args.quiet,
        )

    if not args.no_plots:
        generate_subsample_plots(
            stratified_subsamples=stratified_subsamples,
            output_dir=args.output_dir,
            dataset_name=dataset_name,
            quiet=args.quiet,
        )

    if args.comparative_plots and not args.no_plots and not args.train_synthesizer:
        generate_comparative_plots(
            clean_data=clean_data,
            stratified_subsamples=stratified_subsamples,
            synthetic_by_subsample={},
            output_dir=args.output_dir,
            dataset_name=dataset_name,
            quiet=args.quiet,
        )

    if args.train_synthesizer:
        pred_col = args.prediction_column or args.stratify_column

        if args.eval_ml_augmentation:
            if pred_col not in clean_data.columns:
                raise ValueError(
                    f"--eval-ml-augmentation target '{pred_col}' not found in data columns."
                )
            target_is_numeric = pd.api.types.is_numeric_dtype(clean_data[pred_col])
            if not target_is_numeric and not args.minority_class_label:
                raise ValueError(
                    "--eval-ml-augmentation for categorical targets requires "
                    "--minority-class-label (e.g. one value from the prediction column)."
                )

        if args.eval_privacy_disclosure and (
            not args.eval_privacy_disclosure_known or not args.eval_privacy_disclosure_sensitive
        ):
            raise ValueError(
                "--eval-privacy-disclosure requires --eval-privacy-disclosure-known "
                "and --eval-privacy-disclosure-sensitive"
            )

        train_synthesizers(
            stratified_subsamples=stratified_subsamples,
            synthesizer_name=args.train_synthesizer,
            output_dir=args.output_dir,
            save_synthetic=args.save_synthetic,
            epochs=args.synthesizer_epochs,
            random_state=args.random_state,
            quiet=args.quiet,
            base_metadata=base_metadata,
            dataset_name=dataset_name,
            clean_data=clean_data,
            eval_visualizations=args.eval_visualizations,
            comparative_plots=args.comparative_plots and not args.no_plots,
            stratify_column=args.stratify_column,
            eval_plot_format=args.eval_plot_format,
            eval_ml_augmentation=args.eval_ml_augmentation,
            eval_ml_max_epochs=args.eval_ml_max_epochs,
            eval_k_runs=args.eval_k_runs,
            test_data=test_data,
            prediction_column=pred_col,
            minority_class_label=args.minority_class_label,
            eval_ml_label_encode=args.ml_label_encode,
            eval_privacy=args.eval_privacy,
            eval_privacy_subsample=args.eval_privacy_subsample,
            eval_privacy_disclosure=args.eval_privacy_disclosure,
            eval_privacy_disclosure_known=args.eval_privacy_disclosure_known,
            eval_privacy_disclosure_sensitive=args.eval_privacy_disclosure_sensitive,
            eval_privacy_disclosure_continuous=args.eval_privacy_disclosure_continuous,
            eval_privacy_disclosure_computation=args.eval_privacy_disclosure_computation,
            eval_quality=args.eval_quality,
            eval_quality_subsample=args.eval_quality_subsample,
            eval_quality_threshold=args.eval_quality_threshold,
            eval_quality_correlation_coefficient=args.eval_quality_correlation_coefficient,
            eval_quality_correlation_threshold=args.eval_quality_correlation_threshold,
        )

    return result


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
