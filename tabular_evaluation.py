#!/usr/bin/env python3
"""
Tabular Data Evaluation Script

Creates stratified train/test split and subsamples from tabular data.
All parameters can be set via command-line arguments.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate stratified train/test split and subsamples from tabular data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data-path",
        type=Path,
        help="Path to CSV/Excel file to load (alternative to SDV demo).",
    )
    data_group.add_argument(
        "--sdv-demo",
        action="store_true",
        help="Use SDV demo dataset instead of file.",
    )

    # SDV demo options (when --sdv-demo)
    parser.add_argument(
        "--sdv-modality",
        default="single_table",
        help="SDV demo modality (e.g. single_table, multi_table).",
    )
    parser.add_argument(
        "--sdv-dataset",
        default="child",
        help="SDV demo dataset name.",
    )

    # Stratification & split
    parser.add_argument(
        "--stratify-column",
        default="Disease",
        help="Column to use for stratified sampling.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Number of samples in the held-out test set.",
    )
    parser.add_argument(
        "--subsample-sizes",
        type=str,
        default="1000,800,600,400,200",
        help="Comma-separated subsample sizes to create from training data.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    # Output
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to save train, test, and subsample CSVs.",
    )
    parser.add_argument(
        "--save-datasets",
        action="store_true",
        help="Save train_data, test_data, and subsamples to CSV files.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating bar plots.",
    )
    parser.add_argument(
        "--comparative-plots",
        action="store_true",
        help="Generate all-in-one comparative bar plots per column (clean_data + subsamples).",
    )

    # Synthesizer training (SDV: https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers)
    parser.add_argument(
        "--train-synthesizer",
        choices=["gaussian_copula", "ctgan", "tvae"],
        help="Train selected SDV synthesizer on each subsample and generate synthetic data.",
    )
    parser.add_argument(
        "--save-synthetic",
        action="store_true",
        help="Save generated synthetic data to CSV files (use with --train-synthesizer when needed).",
    )
    parser.add_argument(
        "--synthesizer-epochs",
        type=int,
        default=100,
        help="Training epochs for CTGAN/TVAE (ignored for GaussianCopula).",
    )
    parser.add_argument(
        "--eval-visualizations",
        action="store_true",
        help="Generate Seaborn distribution comparison plots (real vs synthetic) for each subsample (use with --train-synthesizer).",
    )
    parser.add_argument(
        "--eval-plot-format",
        choices=["pdf", "png"],
        default="pdf",
        help="Format for eval plots: pdf or png.",
    )
    parser.add_argument(
        "--eval-ml-augmentation",
        action="store_true",
        help="Evaluate BinaryClassifierPrecisionEfficacy and BinaryClassifierRecallEfficacy (requires --train-synthesizer, --prediction-column, --minority-class-label).",
    )
    parser.add_argument(
        "--eval-k-runs",
        type=int,
        default=1,
        help="Train synthesizer K times per subsample and generate K synthetic datasets (saved as *_synthetic_run0.csv, etc.). Use with --eval-ml-augmentation for mean±std metrics.",
    )
    parser.add_argument(
        "--prediction-column",
        default=None,
        help="Column to predict for ML augmentation metrics (default: --stratify-column). Must be categorical/boolean. Validation uses the held-out test set.",
    )
    parser.add_argument(
        "--minority-class-label",
        default=None,
        help="Positive/minority class value for binary metrics (e.g. 'Fallot' for Disease column). Required with --eval-ml-augmentation.",
    )
    parser.add_argument(
        "--ml-label-encode",
        action="store_true",
        help="Label-encode categorical columns to int before ML augmentation (fixes XGBoost 'enable_categorical' error with many categorical features).",
    )
    parser.add_argument(
        "--eval-privacy",
        action="store_true",
        help="Evaluate privacy metrics: DCRBaselineProtection, DCROverfittingProtection, and optionally DisclosureProtection (requires --train-synthesizer). Use --eval-privacy-disclosure to enable DisclosureProtection.",
    )
    parser.add_argument(
        "--eval-privacy-subsample",
        type=int,
        default=None,
        metavar="N",
        help="Subsample N rows when computing DCRBaselineProtection to speed up on large datasets.",
    )
    parser.add_argument(
        "--eval-privacy-disclosure",
        action="store_true",
        help="Enable DisclosureProtection evaluation. Requires --eval-privacy-disclosure-known and --eval-privacy-disclosure-sensitive.",
    )
    parser.add_argument(
        "--eval-privacy-disclosure-known",
        type=str,
        default=None,
        metavar="COL1,COL2,...",
        help="Comma-separated column names the attacker knows (for DisclosureProtection). Use with --eval-privacy-disclosure.",
    )
    parser.add_argument(
        "--eval-privacy-disclosure-sensitive",
        type=str,
        default=None,
        metavar="COL1,COL2,...",
        help="Comma-separated column names the attacker wants to guess (for DisclosureProtection). Use with --eval-privacy-disclosure.",
    )
    parser.add_argument(
        "--eval-privacy-disclosure-continuous",
        type=str,
        default=None,
        metavar="COL1,COL2,...",
        help="Comma-separated continuous column names needing discretization (for DisclosureProtection).",
    )
    parser.add_argument(
        "--eval-privacy-disclosure-computation",
        choices=["cap", "generalized_cap", "zero_cap"],
        default="cap",
        help="CAP computation method for DisclosureProtection (default: cap).",
    )
    parser.add_argument(
        "--eval-quality",
        action="store_true",
        help="Evaluate ContingencySimilarity quality metric for all column pairs (real vs synthetic 2D distributions). Requires --train-synthesizer.",
    )
    parser.add_argument(
        "--eval-quality-subsample",
        type=int,
        default=None,
        metavar="N",
        help="Subsample N rows when computing ContingencySimilarity to speed up on large datasets.",
    )
    parser.add_argument(
        "--eval-quality-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Real association threshold (Cramer's V) for ContingencySimilarity. Pairs below this get NaN. Recommended: 0.3 or above.",
    )
    parser.add_argument(
        "--eval-quality-correlation-coefficient",
        choices=["Pearson", "Spearman"],
        default="Pearson",
        help="Correlation coefficient for CorrelationSimilarity (numerical pairs). Default: Pearson.",
    )
    parser.add_argument(
        "--eval-quality-correlation-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Real correlation threshold for CorrelationSimilarity. Pairs below |r| get NaN. Recommended: 0.4 or above.",
    )

    # Misc
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Reduce output verbosity.",
    )

    return parser.parse_args()


def load_data(args) -> tuple[pd.DataFrame, object]:
    """Load data from file or SDV demo.

    Returns:
        (data, metadata): data is always a DataFrame. metadata is Metadata from
        download_demo when using --sdv-demo, else None (file loads have no metadata).
    """
    if args.data_path:
        path = args.data_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            data = pd.read_csv(path)
        elif suffix in (".xlsx", ".xls"):
            data = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .xlsx")

        if args.quiet is False:
            print(f"Loaded {len(data)} rows from {path}")
        return data, None

    # SDV demo: download_demo returns (data, metadata)
    try:
        from sdv.datasets.demo import download_demo
    except ImportError:
        raise ImportError("Install SDV to use --sdv-demo: pip install sdv")

    data, metadata = download_demo(
        modality=args.sdv_modality,
        dataset_name=args.sdv_dataset,
    )
    #print(metadata)
    if args.quiet is False:
        print(f"Loaded SDV demo '{args.sdv_dataset}': {len(data)} rows (metadata included)")
    return data, metadata


def run(args) -> dict:
    """Run the full pipeline and return datasets."""
    data, base_metadata = load_data(args)
    clean_data = data.dropna()

    if args.stratify_column not in clean_data.columns:
        raise ValueError(
            f"Stratify column '{args.stratify_column}' not in data. "
            f"Available: {list(clean_data.columns)}"
        )

    subsample_sizes = [int(s.strip()) for s in args.subsample_sizes.split(",") if s.strip()]

    if args.test_size > len(clean_data):
        raise ValueError(
            f"clean_data has {len(clean_data)} rows; cannot create test set of {args.test_size}."
        )

    # For numerical stratify column: bin first (train_test_split requires ≥2 per class)
    stratify_col = args.stratify_column
    if pd.api.types.is_numeric_dtype(clean_data[stratify_col]):
        clean_data = clean_data.copy()
        n_bins = min(5, clean_data[stratify_col].nunique(), len(clean_data) // 2)
        for attempt in range(n_bins, 1, -1):
            try:
                clean_data["_stratify_bins"] = pd.qcut(
                    clean_data[stratify_col], q=attempt, labels=False, duplicates="drop"
                )
                min_per_bin = clean_data["_stratify_bins"].value_counts().min()
                if min_per_bin >= 2:
                    break
            except (ValueError, TypeError):
                continue
        else:
            raise ValueError(
                "Cannot stratify on numerical column: too few samples per bin "
                "(try fewer subsample sizes or a different stratify column)."
            )
        stratify_vals = clean_data["_stratify_bins"]
        drop_bins = True
    else:
        stratify_vals = clean_data[stratify_col]
        drop_bins = False

    train_data, test_data = train_test_split(
        clean_data,
        test_size=args.test_size,
        stratify=stratify_vals,
        random_state=args.random_state,
    )

    if args.quiet is False:
        print(f"Created test set: {len(test_data)} samples")

    stratify_for_subsample = (
        train_data["_stratify_bins"] if drop_bins else train_data[stratify_col]
    )

    stratified_subsamples = {}
    for size in subsample_sizes:
        if size > len(train_data):
            if args.quiet is False:
                print(f"Warning: subsample size {size} > train_data ({len(train_data)}). Skipping.")
            continue
        _, subsample = train_test_split(
            train_data,
            test_size=size / len(train_data),
            stratify=stratify_for_subsample,
            random_state=args.random_state,
        )
        stratified_subsamples[f"subsample_{size}"] = subsample
        if args.quiet is False:
            print(f"Generated subsample_{size}: {len(subsample)} rows")

    # Cleanup: remove bin column if we used numerical stratification
    if drop_bins:
        clean_data = clean_data.drop(columns=["_stratify_bins"])
        train_data = train_data.drop(columns=["_stratify_bins"])
        test_data = test_data.drop(columns=["_stratify_bins"])
        for name in list(stratified_subsamples.keys()):
            stratified_subsamples[name] = stratified_subsamples[name].drop(
                columns=["_stratify_bins"]
            )

    result = {
        "clean_data": clean_data,
        "train_data": train_data,
        "test_data": test_data,
        "stratified_subsamples": stratified_subsamples,
    }

    if args.save_datasets:
        out = args.output_dir.resolve()
        dataset_name = args.data_path.stem if args.data_path else args.sdv_dataset
        out_ds = out / dataset_name
        out_ds.mkdir(parents=True, exist_ok=True)

        test_data.to_csv(out_ds / "test_data.csv", index=False)
        train_data.to_csv(out_ds / "train_data.csv", index=False)

        for name, df in stratified_subsamples.items():
            df.to_csv(out_ds / f"{name}.csv", index=False)

        if args.quiet is False:
            print(f"Saved datasets to {out_ds}")

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            dataset_name = args.data_path.stem if args.data_path else args.sdv_dataset
            out = args.output_dir.resolve()
            out.mkdir(parents=True, exist_ok=True)
            plots_dir = out / "plots" / dataset_name
            plots_dir.mkdir(parents=True, exist_ok=True)

            for name, subsample_df in stratified_subsamples.items():
                cols = list(subsample_df.columns)
                ncols = min(4, len(cols))
                nrows = (len(cols) + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
                axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

                for i, col in enumerate(cols):
                    ax = axes[i]
                    sns.countplot(data=subsample_df, x=col, hue=col, ax=ax, legend=False, palette="viridis")
                    ax.set_title(f"{col} in {name}")
                    ax.set_ylabel("Count")
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

                for j in range(i + 1, len(axes)):
                    fig.delaxes(axes[j])

                plt.tight_layout()
                plt.savefig(plots_dir / f"{name}.png", dpi=100)
                plt.close()

            if args.quiet is False:
                print(f"Saved plots to {plots_dir}")
        except ImportError:
            if args.quiet is False:
                print("Skipping plots: matplotlib/seaborn not installed")

    if args.comparative_plots and not args.no_plots and not args.train_synthesizer:
        dataset_name = args.data_path.stem if args.data_path else args.sdv_dataset
        _generate_comparative_plots(
            clean_data=clean_data,
            stratified_subsamples=stratified_subsamples,
            synthetic_by_subsample={},
            output_dir=args.output_dir,
            dataset_name=dataset_name,
            quiet=args.quiet,
        )

    if args.train_synthesizer:
        dataset_name = args.data_path.stem if args.data_path else args.sdv_dataset
        pred_col = args.prediction_column or args.stratify_column
        if args.eval_ml_augmentation and not args.minority_class_label:
            raise ValueError(
                "--eval-ml-augmentation requires --minority-class-label "
                f"(e.g. one value from {args.stratify_column} column)"
            )
        if args.eval_privacy_disclosure and (not args.eval_privacy_disclosure_known or not args.eval_privacy_disclosure_sensitive):
            raise ValueError(
                "--eval-privacy-disclosure requires --eval-privacy-disclosure-known and --eval-privacy-disclosure-sensitive"
            )
        _train_synthesizers(
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


def _is_numeric(series: pd.Series) -> bool:
    """Check if column is numeric (int, float)."""
    return pd.api.types.is_numeric_dtype(series)


def _generate_comparative_plots(
    clean_data: pd.DataFrame,
    stratified_subsamples: dict,
    synthetic_by_subsample: dict,
    output_dir: Path,
    dataset_name: str,
    quiet: bool,
) -> None:
    """Generate comparative plots per column: subsamples only. Bar (percentage) for categorical, box plot for numerical."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        plt.rcParams["figure.max_open_warning"] = 0
    except ImportError as e:
        if not quiet:
            print(f"Skipping comparative plots: {e}")
        return

    out = output_dir.resolve()
    comparative_dir = out / "plots" / dataset_name / "comparative"
    comparative_dir.mkdir(parents=True, exist_ok=True)

    # Comparative plots show only subsamples (no clean_data, no synthetic)
    real_sources = dict(stratified_subsamples)
    cols = list(clean_data.columns)

    for col in cols:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            is_num = _is_numeric(clean_data[col])

            if is_num:
                # Box plot for numerical (subsamples only) - compares median, quartiles, spread
                parts = []
                for name, df in real_sources.items():
                    vals = df[col].dropna()
                    if len(vals) > 0:
                        parts.append(pd.DataFrame({"Subsample": name, col: vals.values}))
                if not parts:
                    plt.close()
                    continue
                plot_df = pd.concat(parts, ignore_index=True)
                subsample_order = list(real_sources.keys())
                sns.boxplot(
                    data=plot_df,
                    x="Subsample",
                    y=col,
                    hue="Subsample",
                    order=subsample_order,
                    hue_order=subsample_order,
                    ax=ax,
                    palette="viridis",
                    legend=False,
                )
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_ylabel(col)
            else:
                # Bar plot for categorical (subsamples only)
                all_cats = pd.Index(clean_data[col].dropna().unique())
                for df in stratified_subsamples.values():
                    all_cats = all_cats.union(pd.Index(df[col].dropna().unique()))
                cat_order = list(all_cats)
                n_cats = len(cat_order)
                if n_cats == 0:
                    continue

                real_counts = {name: df[col].value_counts().reindex(cat_order, fill_value=0).fillna(0).values
                              for name, df in real_sources.items()}
                # Convert to percentages
                real_pct = {name: 100 * vals / vals.sum() if vals.sum() > 0 else vals
                            for name, vals in real_counts.items()}
                n_sources = len(real_sources)
                width = 0.8 / max(n_sources, 1)
                x = np.arange(n_cats)
                cmap = plt.get_cmap("viridis")
                colors = cmap(np.linspace(0, 1, n_sources))

                for idx, (name, pct) in enumerate(real_pct.items()):
                    offset = (idx - n_sources / 2 + 0.5) * width
                    ax.bar(x + offset, pct, width, label=name, alpha=0.9, color=colors[idx])
                    y_max = max(pct)
                    for i, v in enumerate(pct):
                        ax.text(x[i] + offset, v + 0.02 * max(y_max, 1), f"{v:.1f}%", ha="center", va="bottom", fontsize=7)

                ax.set_xticks(x)
                ax.set_xticklabels(cat_order, rotation=45, ha="right")
                ax.set_ylabel("Percentage (%)")
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

            ax.set_title(f"Comparative: {col}")
            plt.tight_layout()
            safe_name = str(col).replace(" ", "_").replace("/", "_")
            fig.savefig(comparative_dir / f"{safe_name}.png", dpi=100, bbox_inches="tight")
            plt.close()
        except Exception as e:
            if not quiet:
                print(f"  Skip comparative plot '{col}': {e}")

    if not quiet:
        print(f"Saved comparative plots to {comparative_dir}")


def _generate_eval_visualizations(
    real_data: pd.DataFrame,
    synthetic_list: list[pd.DataFrame],
    metadata: object,
    output_dir: Path,
    subsample_name: str,
    stratify_column: str | None,
    quiet: bool,
    eval_plot_format: str = "pdf",
) -> None:
    """Generate Seaborn distribution comparison plots (real vs synthetic) per subsample.
    When len(synthetic_list) > 1, synthetic bars show mean ± std across K runs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        plt.rcParams["figure.max_open_warning"] = 0
    except ImportError:
        if not quiet:
            print("  Skipping eval visualizations: matplotlib/seaborn not installed")
        return

    ext = eval_plot_format
    cols = list(real_data.columns)
    plot_dir = output_dir / "eval_plots" / subsample_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    k_synth = len(synthetic_list)
    show_std = k_synth > 1

    # Single-column: distribution comparison
    for col in cols:
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            if _is_numeric(real_data[col]):
                real_m = real_data.copy()
                real_m["_Source"] = "Real"
                synth_all = pd.concat([s.copy() for s in synthetic_list], ignore_index=True)
                synth_all["_Source"] = "Synthetic"
                combined_num = pd.concat([real_m, synth_all], ignore_index=True)
                sns.kdeplot(
                    data=combined_num, x=col, hue="_Source", common_norm=False,
                    ax=ax, alpha=0.6, fill=True, warn_singular=False
                )
            else:
                # Categorical: compute mean and std across K synthetic runs
                all_cats = pd.Index(real_data[col].dropna().unique())
                for s in synthetic_list:
                    all_cats = all_cats.union(pd.Index(s[col].dropna().unique()))
                cat_order = list(all_cats)
                real_counts = real_data[col].value_counts().reindex(cat_order, fill_value=0).fillna(0)
                synth_counts_list = [s[col].value_counts().reindex(cat_order, fill_value=0).fillna(0) for s in synthetic_list]
                synth_mean = np.array([c.values for c in synth_counts_list]).mean(axis=0)
                synth_std = np.array([c.values for c in synth_counts_list]).std(axis=0) if show_std else np.zeros(len(cat_order))

                x = np.arange(len(cat_order))
                width = 0.35
                real_vals = np.asarray(real_counts.values, dtype=float)
                ax.bar(x - width / 2, real_vals, width, label="Real", alpha=0.9, color="C0")
                bars = ax.bar(x + width / 2, synth_mean, width, label="Synthetic (mean)" + (" ± std" if show_std else ""), alpha=0.9, color="C1")
                if show_std and np.any(synth_std > 0):
                    ax.errorbar(x + width / 2, synth_mean, yerr=synth_std, fmt="none", color="black", capsize=2)
                # Labels on top of bars
                y_max = max(real_vals.max(), (synth_mean + synth_std).max()) if len(cat_order) else 1
                offset = max(0.5, 0.03 * y_max)
                for i, v in enumerate(real_vals):
                    ax.text(x[i] - width / 2, v + offset, f"{int(v)}", ha="center", va="bottom", fontsize=8, color="C0")
                for i, (m, s) in enumerate(zip(synth_mean, synth_std)):
                    label = f"{m:.1f}" + (f"\n±{s:.1f}" if show_std and s > 0 else "")
                    ax.text(x[i] + width / 2, m + s + offset, label, ha="center", va="bottom", fontsize=8, color="C1")
                ax.set_xticks(x)
                ax.set_xticklabels(cat_order, rotation=45, ha="right")
                ax.set_ylabel("Count")
                ax.set_ylim(0, y_max + offset * 3)  # headroom for labels
                ax.legend()
            ax.set_title(f"{col} – Real vs Synthetic" + (f" (K={k_synth} runs)" if show_std else ""))
            plt.tight_layout()
            fig.savefig(plot_dir / f"column_{col}.{ext}", dpi=100, bbox_inches="tight")
            plt.close()
        except Exception as e:
            if not quiet:
                print(f"    Skip column plot '{col}': {e}")


def _prepare_ml_augmentation_data(
    real_training_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    real_validation_data: pd.DataFrame,
    prediction_column: str | None = None,
    minority_class_label: str | int | None = None,
    ml_label_encode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | int]:
    """Filter validation for seen categories and convert object columns to category.
    XGBoost requires int/float/bool/category (not object).
    When ml_label_encode=True, label-encode categorical columns to int to avoid XGBoost enable_categorical error.
    Returns (train, synth, val, effective_minority_class_label)."""
    train_plus_synth = pd.concat([real_training_data, synthetic_data], ignore_index=True)
    mask = pd.Series(True, index=real_validation_data.index)
    for col in real_validation_data.columns:
        if not pd.api.types.is_numeric_dtype(real_validation_data[col]):
            seen = set(train_plus_synth[col].dropna().astype(str).unique())
            mask &= real_validation_data[col].apply(
                lambda v: pd.isna(v) or str(v) in seen
            )
    val_filtered = real_validation_data.loc[mask].copy()

    def _object_to_category(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if out[col].dtype == "object" or (
                hasattr(out[col].dtype, "name") and "str" in str(out[col].dtype)
            ):
                out[col] = out[col].astype("category")
        return out

    train_prep = _object_to_category(real_training_data)
    syn_prep = _object_to_category(synthetic_data)
    val_prep = _object_to_category(val_filtered)
    effective_minority = minority_class_label

    if ml_label_encode:
        try:
            from sklearn.preprocessing import OrdinalEncoder
        except ImportError:
            return (train_prep, syn_prep, val_prep, effective_minority)
        cat_cols = [
            c for c in train_prep.columns
            if not pd.api.types.is_numeric_dtype(train_prep[c])
        ]
        if not cat_cols:
            return (train_prep, syn_prep, val_prep, effective_minority)
        train_prep = train_prep.copy()
        syn_prep = syn_prep.copy()
        val_prep = val_prep.copy()
        pred_enc = None
        for c in cat_cols:
            combined = pd.concat([train_prep[c], syn_prep[c]], ignore_index=True)
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc.fit(combined.astype(str).fillna("__nan__").values.reshape(-1, 1))
            tr_vals = train_prep[c].astype(str).fillna("__nan__").values.reshape(-1, 1)
            sy_vals = syn_prep[c].astype(str).fillna("__nan__").values.reshape(-1, 1)
            va_vals = val_prep[c].astype(str).fillna("__nan__").values.reshape(-1, 1)
            train_prep[c] = enc.transform(tr_vals).ravel().astype(int)
            syn_prep[c] = enc.transform(sy_vals).ravel().astype(int)
            val_prep[c] = enc.transform(va_vals).ravel().astype(int)
            if c == prediction_column:
                pred_enc = enc
        if pred_enc is not None and minority_class_label is not None:
            lab_encoded = pred_enc.transform([[str(minority_class_label)]])
            effective_minority = int(lab_encoded[0, 0])
        # Ensure consistent dtypes: encoded categoricals as int64, numerics as float64
        # (XGBoost rejects float for category indices)
        for df in (train_prep, syn_prep, val_prep):
            for c in cat_cols:
                df[c] = df[c].astype("int64")
            for c in df.columns:
                if c not in cat_cols:
                    df[c] = df[c].astype("float64")

    return (train_prep, syn_prep, val_prep, effective_minority)


def _metadata_for_label_encoded(meta_dict: dict, cat_cols: list, prediction_column: str) -> dict:
    """Return metadata copy with categorical columns (except target) as 'numerical'.
    SDMetrics then treats them as numeric, avoiding astype('category') that causes
    XGBoost 'index type must match' errors when train/synth/val have mixed int/float."""
    import copy
    out = copy.deepcopy(meta_dict)
    cols = out.get("columns", {})
    for c in cat_cols:
        if c != prediction_column and c in cols:
            cols[c] = {**cols[c], "sdtype": "numerical"}
    return out


def _compute_ml_augmentation_metrics(
    real_training_data: pd.DataFrame,
    synthetic_list: list[pd.DataFrame],
    real_validation_data: pd.DataFrame,
    metadata: object,
    prediction_column: str,
    minority_class_label: str | int,
    ml_label_encode: bool = False,
    quiet: bool = False,
) -> dict:
    """Compute BinaryClassifierPrecisionEfficacy and BinaryClassifierRecallEfficacy across K synthetic datasets. Returns mean±std."""
    try:
        from sdmetrics.single_table.data_augmentation import (
            BinaryClassifierPrecisionEfficacy,
            BinaryClassifierRecallEfficacy,
        )
    except ImportError as e:
        if not quiet:
            print(f"  Skipping ML augmentation eval: {e} (install xgboost: pip install xgboost)")
        return {}

    if pd.api.types.is_numeric_dtype(real_validation_data[prediction_column]):
        if not quiet:
            print("  Skipping ML augmentation eval: target variable is numerical (binary classifier requires categorical target).")
        return {}

    # Get metadata as dict (SingleTableMetadata format)
    if hasattr(metadata, "_convert_to_single_table"):
        meta_dict = metadata._convert_to_single_table().to_dict()
    else:
        meta_dict = metadata

    # When label-encoding, pass metadata with encoded categorical cols as "numerical"
    # so SDMetrics does not astype('category'), avoiding XGBoost dtype mismatch.
    if ml_label_encode:
        cat_cols = [c for c in real_training_data.columns
                    if not pd.api.types.is_numeric_dtype(real_training_data[c])]
        meta_dict = _metadata_for_label_encoded(meta_dict, cat_cols, prediction_column)

    precision_scores = []
    recall_scores = []
    for syn in synthetic_list:
        try:
            train_prep, syn_prep, val_prep, effective_minority = _prepare_ml_augmentation_data(
                real_training_data, syn, real_validation_data,
                prediction_column=prediction_column,
                minority_class_label=minority_class_label,
                ml_label_encode=ml_label_encode,
            )
            if len(val_prep) < 10:
                if not quiet:
                    print("    Metric skipped: validation set too small after filtering unseen categories")
                continue
            prec = BinaryClassifierPrecisionEfficacy.compute(
                real_training_data=train_prep,
                synthetic_data=syn_prep,
                real_validation_data=val_prep,
                metadata=meta_dict,
                prediction_column_name=prediction_column,
                minority_class_label=effective_minority,
                classifier="XGBoost",
                fixed_recall_value=0.9,
            )
            rec = BinaryClassifierRecallEfficacy.compute(
                real_training_data=train_prep,
                synthetic_data=syn_prep,
                real_validation_data=val_prep,
                metadata=meta_dict,
                prediction_column_name=prediction_column,
                minority_class_label=effective_minority,
                classifier="XGBoost",
                fixed_precision_value=0.9,
            )
            precision_scores.append(float(prec))
            recall_scores.append(float(rec))
        except Exception as e:
            if not quiet:
                print(f"    Metric computation failed for one run: {e}")

    import numpy as np
    result = {}
    if precision_scores:
        result["BinaryClassifierPrecisionEfficacy"] = {
            "mean": float(np.mean(precision_scores)),
            "std": float(np.std(precision_scores)),
            "scores": precision_scores,
        }
    if recall_scores:
        result["BinaryClassifierRecallEfficacy"] = {
            "mean": float(np.mean(recall_scores)),
            "std": float(np.std(recall_scores)),
            "scores": recall_scores,
        }
    return result


def _compute_privacy_metrics(
    real_data: pd.DataFrame,
    synthetic_list: list[pd.DataFrame],
    metadata: object,
    num_rows_subsample: int | None = None,
    real_validation_data: pd.DataFrame | None = None,
    disclosure_known_columns: list[str] | None = None,
    disclosure_sensitive_columns: list[str] | None = None,
    disclosure_continuous_columns: list[str] | None = None,
    disclosure_computation: str = "cap",
    quiet: bool = False,
) -> dict:
    """Compute privacy metrics across K synthetic datasets.
    - DCRBaselineProtection: https://docs.sdv.dev/sdmetrics/data-metrics/privacy/dcrbaselineprotection
    - DCROverfittingProtection: https://docs.sdv.dev/sdmetrics/data-metrics/privacy/dcroverfittingprotection
    - DisclosureProtection: https://docs.sdv.dev/sdmetrics/data-metrics/privacy/disclosureprotection
      (requires disclosure_known_columns and disclosure_sensitive_columns)
    Returns mean±std and per-run scores for each metric."""
    try:
        from sdmetrics.single_table import DCRBaselineProtection, DCROverfittingProtection, DisclosureProtection
    except ImportError as e:
        if not quiet:
            print(f"  Skipping privacy eval: {e}")
        return {}

    meta_dict = metadata._convert_to_single_table().to_dict() if hasattr(metadata, "_convert_to_single_table") else metadata

    import numpy as np
    out = {}

    # DCRBaselineProtection
    scores = []
    breakdowns = []
    for syn in synthetic_list:
        try:
            result = DCRBaselineProtection.compute_breakdown(
                real_data=real_data,
                synthetic_data=syn,
                metadata=meta_dict,
                num_rows_subsample=num_rows_subsample,
            )
            if isinstance(result, dict):
                score_val = result.get("score")
            else:
                score_val = float(result)
            if score_val is not None and not (isinstance(score_val, float) and pd.isna(score_val)):
                scores.append(float(score_val))
                breakdowns.append(result)
        except Exception as e:
            if not quiet:
                print(f"    DCRBaselineProtection failed for one run: {e}")

    if scores:
        out["DCRBaselineProtection"] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores,
        }
        if breakdowns and isinstance(breakdowns[0], dict) and "median_DCR_to_real_data" in breakdowns[0]:
            out["DCRBaselineProtection"]["median_DCR_to_real_data"] = breakdowns[0]["median_DCR_to_real_data"]

    # DCROverfittingProtection (requires validation holdout)
    if real_validation_data is not None and len(real_validation_data) > 0:
        scores_overfit = []
        breakdowns_overfit = []
        for syn in synthetic_list:
            try:
                result = DCROverfittingProtection.compute_breakdown(
                    real_training_data=real_data,
                    synthetic_data=syn,
                    real_validation_data=real_validation_data,
                    metadata=meta_dict,
                    num_rows_subsample=num_rows_subsample,
                )
                if isinstance(result, dict):
                    score_val = result.get("score")
                else:
                    score_val = float(result)
                if score_val is not None and not (isinstance(score_val, float) and pd.isna(score_val)):
                    scores_overfit.append(float(score_val))
                    breakdowns_overfit.append(result)
            except Exception as e:
                if not quiet:
                    print(f"    DCROverfittingProtection failed for one run: {e}")

        if scores_overfit:
            out["DCROverfittingProtection"] = {
                "mean": float(np.mean(scores_overfit)),
                "std": float(np.std(scores_overfit)),
                "scores": scores_overfit,
            }
            if breakdowns_overfit and isinstance(breakdowns_overfit[0], dict) and "synthetic_data_percentages" in breakdowns_overfit[0]:
                out["DCROverfittingProtection"]["synthetic_data_percentages"] = breakdowns_overfit[0]["synthetic_data_percentages"]

    # DisclosureProtection (requires known_columns and sensitive_columns)
    if (
        disclosure_known_columns
        and disclosure_sensitive_columns
        and set(disclosure_known_columns + disclosure_sensitive_columns).issubset(real_data.columns)
    ):
        scores_disc = []
        breakdowns_disc = []
        for syn in synthetic_list:
            try:
                result = DisclosureProtection.compute_breakdown(
                    real_data=real_data,
                    synthetic_data=syn,
                    known_column_names=disclosure_known_columns,
                    sensitive_column_names=disclosure_sensitive_columns,
                    continuous_column_names=disclosure_continuous_columns,
                    computation_method=disclosure_computation,
                )
                if isinstance(result, dict):
                    score_val = result.get("score")
                else:
                    score_val = float(result)
                if score_val is not None and not (isinstance(score_val, float) and pd.isna(score_val)):
                    scores_disc.append(float(score_val))
                    breakdowns_disc.append(result)
            except Exception as e:
                if not quiet:
                    print(f"    DisclosureProtection failed for one run: {e}")

        if scores_disc:
            out["DisclosureProtection"] = {
                "mean": float(np.mean(scores_disc)),
                "std": float(np.std(scores_disc)),
                "scores": scores_disc,
            }
            if breakdowns_disc and isinstance(breakdowns_disc[0], dict):
                b = breakdowns_disc[0]
                if "cap_protection" in b:
                    out["DisclosureProtection"]["cap_protection"] = b["cap_protection"]
                if "baseline_protection" in b:
                    out["DisclosureProtection"]["baseline_protection"] = b["baseline_protection"]

    return out


def _compute_quality_metrics(
    real_data: pd.DataFrame,
    synthetic_list: list[pd.DataFrame],
    num_rows_subsample: int | None = None,
    real_association_threshold: float | None = None,
    correlation_coefficient: str = "Pearson",
    correlation_threshold: float | None = None,
    quiet: bool = False,
) -> dict:
    """Compute KSComplement/TVComplement (marginals) + pairwise quality metrics.

    KSComplement: numerical marginals
    TVComplement: categorical/boolean marginals
    ContingencySimilarity: categorical/mixed column pairs (2D distributions)
    CorrelationSimilarity: numerical column pairs (correlations)

    See:
      https://docs.sdv.dev/sdmetrics/data-metrics/quality/kscomplement
      https://docs.sdv.dev/sdmetrics/data-metrics/quality/tvcomplement
      https://docs.sdv.dev/sdmetrics/data-metrics/quality/contingencysimilarity
      https://docs.sdv.dev/sdmetrics/data-metrics/quality/correlationsimilarity
    """
    try:
        from sdmetrics.column_pairs import ContingencySimilarity, CorrelationSimilarity
        from sdmetrics.single_column import KSComplement, TVComplement
        import itertools
        import numpy as np
    except ImportError as e:
        if not quiet:
            print(f"  Skipping quality eval: {e}")
        return {}

    cols = list(real_data.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(real_data[c])]
    out = {}

    # KSComplement: numerical columns only (marginal distribution similarity)
    if numeric_cols:
        ks_scores_by_col = {c: [] for c in numeric_cols}
        for syn in synthetic_list:
            for c in numeric_cols:
                try:
                    real_vals = real_data[c].dropna()
                    syn_vals = syn[c].dropna()
                    if len(real_vals) < 2 or len(syn_vals) < 2:
                        continue
                    if num_rows_subsample and len(real_vals) > num_rows_subsample:
                        real_vals = real_vals.sample(n=num_rows_subsample, random_state=42)
                    if num_rows_subsample and len(syn_vals) > num_rows_subsample:
                        syn_vals = syn_vals.sample(n=num_rows_subsample, random_state=42)
                    score = KSComplement.compute(
                        real_data=real_vals,
                        synthetic_data=syn_vals,
                    )
                    if score is not None and not (isinstance(score, float) and pd.isna(score)):
                        ks_scores_by_col[c].append(float(score))
                except Exception:
                    pass

        valid_ks_cols = {c: s for c, s in ks_scores_by_col.items() if s}
        if valid_ks_cols:
            all_ks_scores = []
            col_means = {}
            for col, run_scores in valid_ks_cols.items():
                col_means[col] = float(np.mean(run_scores))
                all_ks_scores.extend(run_scores)
            out["KSComplement"] = {
                "mean": float(np.mean(all_ks_scores)),
                "std": float(np.std(all_ks_scores)) if len(all_ks_scores) > 1 else 0.0,
                "scores": [float(s) for s in all_ks_scores],
                "num_columns": len(valid_ks_cols),
                "total_columns": len(numeric_cols),
                "column_means": col_means,
            }

    # TVComplement: categorical/boolean columns only (marginal distribution similarity)
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(real_data[c])]
    if cat_cols:
        tv_scores_by_col = {c: [] for c in cat_cols}
        for syn in synthetic_list:
            for c in cat_cols:
                try:
                    real_vals = real_data[c].dropna()
                    syn_vals = syn[c].dropna()
                    if len(real_vals) == 0 or len(syn_vals) == 0:
                        continue
                    if num_rows_subsample and len(real_vals) > num_rows_subsample:
                        real_vals = real_vals.sample(n=num_rows_subsample, random_state=42)
                    if num_rows_subsample and len(syn_vals) > num_rows_subsample:
                        syn_vals = syn_vals.sample(n=num_rows_subsample, random_state=42)
                    score = TVComplement.compute(
                        real_data=real_vals,
                        synthetic_data=syn_vals,
                    )
                    if score is not None and not (isinstance(score, float) and pd.isna(score)):
                        tv_scores_by_col[c].append(float(score))
                except Exception:
                    pass

        valid_tv_cols = {c: s for c, s in tv_scores_by_col.items() if s}
        if valid_tv_cols:
            all_tv_scores = []
            col_means = {}
            for col, run_scores in valid_tv_cols.items():
                col_means[col] = float(np.mean(run_scores))
                all_tv_scores.extend(run_scores)
            out["TVComplement"] = {
                "mean": float(np.mean(all_tv_scores)),
                "std": float(np.std(all_tv_scores)) if len(all_tv_scores) > 1 else 0.0,
                "scores": [float(s) for s in all_tv_scores],
                "num_columns": len(valid_tv_cols),
                "total_columns": len(cat_cols),
                "column_means": col_means,
            }

    if len(cols) < 2:
        return out

    # ContingencySimilarity: categorical/mixed pairs
    continuous_cols = [
        c for c in cols
        if pd.api.types.is_numeric_dtype(real_data[c]) and real_data[c].nunique() > 10
    ]
    pairs = list(itertools.combinations(cols, 2))
    scores_by_pair = {pair: [] for pair in pairs}

    for syn in synthetic_list:
        for c1, c2 in pairs:
            pair_continuous = [c for c in (c1, c2) if c in continuous_cols]
            try:
                real_sub = real_data[[c1, c2]].copy()
                syn_sub = syn[[c1, c2]].copy()
                if num_rows_subsample and len(real_sub) > num_rows_subsample:
                    real_sub = real_sub.sample(n=num_rows_subsample, random_state=42)
                if num_rows_subsample and len(syn_sub) > num_rows_subsample:
                    syn_sub = syn_sub.sample(n=num_rows_subsample, random_state=42)
                kwargs = {}
                if pair_continuous:
                    kwargs["continuous_column_names"] = pair_continuous
                if real_association_threshold is not None:
                    kwargs["real_association_threshold"] = real_association_threshold
                score = ContingencySimilarity.compute(
                    real_data=real_sub,
                    synthetic_data=syn_sub,
                    num_rows_subsample=None,
                    **kwargs,
                )
                if score is not None and not (isinstance(score, float) and pd.isna(score)):
                    scores_by_pair[(c1, c2)].append(float(score))
            except Exception:
                pass

    valid_pairs = {p: s for p, s in scores_by_pair.items() if s}
    if valid_pairs:
        all_scores = []
        pair_means = {}
        for pair, run_scores in valid_pairs.items():
            pair_means[pair] = float(np.mean(run_scores))
            all_scores.extend(run_scores)
        out["ContingencySimilarity"] = {
            "mean": float(np.mean(all_scores)),
            "std": float(np.std(all_scores)) if len(all_scores) > 1 else 0.0,
            "scores": [float(s) for s in all_scores],
            "num_pairs": len(valid_pairs),
            "total_pairs": len(pairs),
            "pair_means": {f"{p[0]}|{p[1]}": v for p, v in pair_means.items()},
        }

    # CorrelationSimilarity: numerical pairs only
    num_pairs = list(itertools.combinations(numeric_cols, 2))
    if not num_pairs:
        return out

    corr_scores_by_pair = {pair: [] for pair in num_pairs}
    for syn in synthetic_list:
        for c1, c2 in num_pairs:
            try:
                real_sub = real_data[[c1, c2]].dropna()
                syn_sub = syn[[c1, c2]].dropna()
                if len(real_sub) < 2 or len(syn_sub) < 2:
                    continue
                if num_rows_subsample and len(real_sub) > num_rows_subsample:
                    real_sub = real_sub.sample(n=num_rows_subsample, random_state=42)
                if num_rows_subsample and len(syn_sub) > num_rows_subsample:
                    syn_sub = syn_sub.sample(n=num_rows_subsample, random_state=42)
                kwargs = {"coefficient": correlation_coefficient}
                if correlation_threshold is not None:
                    kwargs["real_correlation_threshold"] = correlation_threshold
                score = CorrelationSimilarity.compute(
                    real_data=real_sub,
                    synthetic_data=syn_sub,
                    **kwargs,
                )
                if score is not None and not (isinstance(score, float) and pd.isna(score)):
                    corr_scores_by_pair[(c1, c2)].append(float(score))
            except Exception:
                pass

    valid_num_pairs = {p: s for p, s in corr_scores_by_pair.items() if s}
    if valid_num_pairs:
        all_corr_scores = []
        corr_pair_means = {}
        for pair, run_scores in valid_num_pairs.items():
            corr_pair_means[pair] = float(np.mean(run_scores))
            all_corr_scores.extend(run_scores)
        out["CorrelationSimilarity"] = {
            "mean": float(np.mean(all_corr_scores)),
            "std": float(np.std(all_corr_scores)) if len(all_corr_scores) > 1 else 0.0,
            "scores": [float(s) for s in all_corr_scores],
            "num_pairs": len(valid_num_pairs),
            "total_pairs": len(num_pairs),
            "pair_means": {f"{p[0]}|{p[1]}": v for p, v in corr_pair_means.items()},
            "coefficient": correlation_coefficient,
        }

    return out


def _train_synthesizers(
    stratified_subsamples: dict,
    synthesizer_name: str,
    output_dir: Path,
    save_synthetic: bool,
    epochs: int,
    random_state: int,
    quiet: bool,
    base_metadata: object = None,
    dataset_name: str = "data",
    clean_data: pd.DataFrame | None = None,
    eval_visualizations: bool = False,
    comparative_plots: bool = False,
    stratify_column: str | None = None,
    eval_plot_format: str = "pdf",
    eval_ml_augmentation: bool = False,
    eval_k_runs: int = 5,
    test_data: pd.DataFrame | None = None,
    prediction_column: str | None = None,
    minority_class_label: str | int | None = None,
    eval_ml_label_encode: bool = False,
    eval_privacy: bool = False,
    eval_privacy_subsample: int | None = None,
    eval_privacy_disclosure: bool = False,
    eval_privacy_disclosure_known: str | None = None,
    eval_privacy_disclosure_sensitive: str | None = None,
    eval_privacy_disclosure_continuous: str | None = None,
    eval_privacy_disclosure_computation: str = "cap",
    eval_quality: bool = False,
    eval_quality_subsample: int | None = None,
    eval_quality_threshold: float | None = None,
    eval_quality_correlation_coefficient: str = "Pearson",
    eval_quality_correlation_threshold: float | None = None,
) -> None:
    """Train SDV synthesizer on each subsample and generate synthetic data of same size.
    When eval_ml_augmentation=True, trains K times per subsample and evaluates with mean±std."""
    try:
        from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
        from sdv.metadata import Metadata
    except ImportError as e:
        raise ImportError(
            "Install SDV to use synthesizers: pip install sdv "
            "(GaussianCopula, CTGAN, TVAE require sdv)"
        ) from e

    def _make_synthesizer(metadata):
        if synthesizer_name == "gaussian_copula":
            return GaussianCopulaSynthesizer(metadata)
        if synthesizer_name == "ctgan":
            return CTGANSynthesizer(metadata, epochs=epochs, verbose=not quiet)
        return TVAESynthesizer(metadata, epochs=epochs, verbose=not quiet)

    out = output_dir.resolve()
    synthetic_dir = out / "synthetic" / dataset_name / synthesizer_name
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    if quiet is False:
        print(f"Training {synthesizer_name} on {len(stratified_subsamples)} subsamples...")

    k_runs = max(1, eval_k_runs)
    ml_results = {}
    privacy_results = {}
    quality_results = {}
    synthetic_by_subsample = {}

    # Suppress SDV's "save metadata" UserWarning since we save it ourselves below
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*save_to_json.*replicability",
            category=UserWarning,
            module="sdv",
        )
        for name, subsample_df in stratified_subsamples.items():
            n_rows = len(subsample_df)
            if base_metadata is not None:
                metadata = base_metadata
            else:
                metadata = Metadata.detect_from_dataframe(data=subsample_df, table_name=name)

            synthetic_list = []
            for k in range(k_runs):
                run_seed = random_state + k * 10000
                synthesizer = _make_synthesizer(metadata)
                synthesizer.fit(subsample_df)
                if hasattr(synthesizer, "_set_random_state"):
                    synthesizer._set_random_state(run_seed)
                synthetic_data = synthesizer.sample(num_rows=n_rows)
                synthetic_list.append(synthetic_data)

                if save_synthetic:
                    csv_path = synthetic_dir / f"{name}_synthetic_run{k}.csv"
                    synthetic_data.to_csv(csv_path, index=False)
                    if k == 0:
                        metadata_path = synthetic_dir / f"{name}_metadata.json"
                        metadata.save_to_json(filepath=metadata_path, mode="overwrite")
                    if quiet is False:
                        print(f"  {name}: run {k+1}/{k_runs} -> {csv_path}")

            if eval_visualizations and synthetic_list:
                if quiet is False:
                    print(f"  {name}: generating eval visualizations...")
                _generate_eval_visualizations(
                    real_data=subsample_df,
                    synthetic_list=synthetic_list,
                    metadata=metadata,
                    output_dir=synthetic_dir,
                    subsample_name=name,
                    stratify_column=stratify_column,
                    quiet=quiet,
                    eval_plot_format=eval_plot_format,
                )

            if eval_ml_augmentation and synthetic_list and test_data is not None and prediction_column and minority_class_label is not None:
                if quiet is False:
                    print(f"  {name}: evaluating ML augmentation (K={k_runs})...")
                metrics = _compute_ml_augmentation_metrics(
                    real_training_data=subsample_df,
                    synthetic_list=synthetic_list,
                    real_validation_data=test_data.copy(),
                    metadata=metadata,
                    prediction_column=prediction_column,
                    minority_class_label=minority_class_label,
                    ml_label_encode=eval_ml_label_encode,
                    quiet=quiet,
                )
                if metrics:
                    ml_results[name] = metrics
                    if quiet is False:
                        for m, v in metrics.items():
                            print(f"    {m}: {v['mean']:.4f} ± {v['std']:.4f}")

            if eval_privacy and synthetic_list:
                if quiet is False:
                    msg = "  {name}: evaluating privacy (DCRBaselineProtection, DCROverfittingProtection"
                    if eval_privacy_disclosure and eval_privacy_disclosure_known and eval_privacy_disclosure_sensitive:
                        msg += ", DisclosureProtection"
                    msg += f") (K={k_runs})..."
                    print(msg.format(name=name))
                _parse_csv = lambda s: [c.strip() for c in s.split(",") if c.strip()] if s else None
                privacy_metrics = _compute_privacy_metrics(
                    real_data=subsample_df,
                    synthetic_list=synthetic_list,
                    metadata=metadata,
                    num_rows_subsample=eval_privacy_subsample,
                    real_validation_data=test_data.copy() if test_data is not None else None,
                    disclosure_known_columns=_parse_csv(eval_privacy_disclosure_known) if eval_privacy_disclosure else None,
                    disclosure_sensitive_columns=_parse_csv(eval_privacy_disclosure_sensitive) if eval_privacy_disclosure else None,
                    disclosure_continuous_columns=_parse_csv(eval_privacy_disclosure_continuous) if eval_privacy_disclosure else None,
                    disclosure_computation=eval_privacy_disclosure_computation if eval_privacy_disclosure else "cap",
                    quiet=quiet,
                )
                if privacy_metrics:
                    privacy_results[name] = privacy_metrics
                    if quiet is False:
                        for m, v in privacy_metrics.items():
                            if "mean" in v and "std" in v:
                                print(f"    {m}: {v['mean']:.4f} ± {v['std']:.4f}")

            if eval_quality and synthetic_list:
                if quiet is False:
                    print(f"  {name}: evaluating quality (KSComplement, ContingencySimilarity, CorrelationSimilarity) (K={k_runs})...")
                quality_metrics = _compute_quality_metrics(
                    real_data=subsample_df,
                    synthetic_list=synthetic_list,
                    num_rows_subsample=eval_quality_subsample,
                    real_association_threshold=eval_quality_threshold,
                    correlation_coefficient=eval_quality_correlation_coefficient,
                    correlation_threshold=eval_quality_correlation_threshold,
                    quiet=quiet,
                )
                if quality_metrics:
                    quality_results[name] = quality_metrics
                    if quiet is False:
                        for m, v in quality_metrics.items():
                            if "mean" in v and "std" in v:
                                print(f"    {m}: {v['mean']:.4f} ± {v['std']:.4f}")

            if comparative_plots and synthetic_list:
                synthetic_by_subsample[name] = synthetic_list

    if comparative_plots and (clean_data is not None or stratified_subsamples):
        _generate_comparative_plots(
            clean_data=clean_data if clean_data is not None else next(iter(stratified_subsamples.values())),
            stratified_subsamples=stratified_subsamples,
            synthetic_by_subsample=synthetic_by_subsample,
            output_dir=output_dir,
            dataset_name=dataset_name,
            quiet=quiet,
        )

    if ml_results:
        import json
        eval_path = synthetic_dir / "ml_augmentation_eval.json"
        # Convert for JSON (handle numpy types)
        dumpable = {}
        for name, m in ml_results.items():
            dumpable[name] = {
                k: {"mean": v["mean"], "std": v["std"], "scores": v["scores"]}
                for k, v in m.items()
            }
        with open(eval_path, "w") as f:
            json.dump(dumpable, f, indent=2)
        if quiet is False:
            print(f"ML augmentation results saved to {eval_path}")

    if privacy_results:
        import json
        privacy_path = synthetic_dir / "privacy_eval.json"
        dumpable = {}
        for name, m in privacy_results.items():
            dumpable[name] = {}
            for k, v in m.items():
                entry = {"mean": v["mean"], "std": v["std"], "scores": v["scores"]}
                if "median_DCR_to_real_data" in v:
                    entry["median_DCR_to_real_data"] = v["median_DCR_to_real_data"]
                if "synthetic_data_percentages" in v:
                    entry["synthetic_data_percentages"] = v["synthetic_data_percentages"]
                if "cap_protection" in v:
                    entry["cap_protection"] = v["cap_protection"]
                if "baseline_protection" in v:
                    entry["baseline_protection"] = v["baseline_protection"]
                dumpable[name][k] = entry
        with open(privacy_path, "w") as f:
            json.dump(dumpable, f, indent=2)
        if quiet is False:
            print(f"Privacy evaluation results saved to {privacy_path}")

    if quality_results:
        import json
        quality_path = synthetic_dir / "quality_eval.json"
        dumpable = {}
        for name, m in quality_results.items():
            dumpable[name] = {}
            for k, v in m.items():
                # Do not store raw per-run scores; keep only summary statistics.
                entry = {"mean": v["mean"], "std": v["std"]}
                if "num_pairs" in v:
                    entry["num_pairs"] = v["num_pairs"]
                if "total_pairs" in v:
                    entry["total_pairs"] = v["total_pairs"]
                if "pair_means" in v:
                    entry["pair_means"] = v["pair_means"]
                if "num_columns" in v:
                    entry["num_columns"] = v["num_columns"]
                if "total_columns" in v:
                    entry["total_columns"] = v["total_columns"]
                if "column_means" in v:
                    entry["column_means"] = v["column_means"]
                if "coefficient" in v:
                    entry["coefficient"] = v["coefficient"]
                dumpable[name][k] = entry
        with open(quality_path, "w") as f:
            json.dump(dumpable, f, indent=2)
        if quiet is False:
            print(f"Quality evaluation results saved to {quality_path}")

    if quiet is False:
        print(f"Synthetic data saved to {synthetic_dir}")


def main():
    args = parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
