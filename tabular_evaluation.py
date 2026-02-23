#!/usr/bin/env python3
"""
Tabular Data Evaluation Script

Creates stratified train/test split and subsamples from tabular data.
All parameters can be set via command-line arguments.
"""

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

    train_data, test_data = train_test_split(
        clean_data,
        test_size=args.test_size,
        stratify=clean_data[args.stratify_column],
        random_state=args.random_state,
    )

    if args.quiet is False:
        print(f"Created test set: {len(test_data)} samples")

    stratified_subsamples = {}
    for size in subsample_sizes:
        if size > len(train_data):
            if args.quiet is False:
                print(f"Warning: subsample size {size} > train_data ({len(train_data)}). Skipping.")
            continue
        _, subsample = train_test_split(
            train_data,
            test_size=size / len(train_data),
            stratify=train_data[args.stratify_column],
            random_state=args.random_state,
        )
        stratified_subsamples[f"subsample_{size}"] = subsample
        if args.quiet is False:
            print(f"Generated subsample_{size}: {len(subsample)} rows")

    result = {
        "clean_data": clean_data,
        "train_data": train_data,
        "test_data": test_data,
        "stratified_subsamples": stratified_subsamples,
    }

    if args.save_datasets:
        out = args.output_dir.resolve()
        out.mkdir(parents=True, exist_ok=True)

        test_data.to_csv(out / "test_data.csv", index=False)
        train_data.to_csv(out / "train_data.csv", index=False)

        for name, df in stratified_subsamples.items():
            df.to_csv(out / f"{name}.csv", index=False)

        if args.quiet is False:
            print(f"Saved datasets to {out}")

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

    if args.comparative_plots and not args.no_plots:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            dataset_name = args.data_path.stem if args.data_path else args.sdv_dataset
            out = args.output_dir.resolve()
            out.mkdir(parents=True, exist_ok=True)
            comparative_dir = out / "plots" / dataset_name / "comparative"
            comparative_dir.mkdir(parents=True, exist_ok=True)

            all_datasets = {"clean_data": clean_data, **stratified_subsamples}

            for col in clean_data.columns:
                plot_data_list = []
                for name, df in all_datasets.items():
                    value_counts = df[col].value_counts(normalize=True).reset_index()
                    value_counts.columns = ["Category", "Proportion"]
                    value_counts["Source"] = name
                    plot_data_list.append(value_counts)

                combined_df = pd.concat(plot_data_list, ignore_index=True)

                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(
                    data=combined_df,
                    x="Category",
                    y="Proportion",
                    hue="Source",
                    palette="viridis",
                    ax=ax,
                )
                ax.set_title(f"Comparative Distribution of {col} Across Datasets", fontsize=16)
                ax.set_xlabel("Category", fontsize=12)
                ax.set_ylabel("Proportion", fontsize=12)
                plt.xticks(rotation=45, ha="right")
                plt.legend(title="Dataset Source")
                plt.tight_layout()
                plt.savefig(comparative_dir / f"{col}.png", dpi=100)
                plt.close()

            if args.quiet is False:
                print(f"Saved comparative plots to {comparative_dir}")
        except ImportError:
            if args.quiet is False:
                print("Skipping comparative plots: matplotlib/seaborn not installed")

    if args.train_synthesizer:
        dataset_name = args.data_path.stem if args.data_path else args.sdv_dataset
        pred_col = args.prediction_column or args.stratify_column
        if args.eval_ml_augmentation and not args.minority_class_label:
            raise ValueError(
                "--eval-ml-augmentation requires --minority-class-label "
                f"(e.g. one value from {args.stratify_column} column)"
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
            eval_visualizations=args.eval_visualizations,
            stratify_column=args.stratify_column,
            eval_plot_format=args.eval_plot_format,
            eval_ml_augmentation=args.eval_ml_augmentation,
            eval_k_runs=args.eval_k_runs,
            test_data=test_data,
            prediction_column=pred_col,
            minority_class_label=args.minority_class_label,
        )

    return result


def _is_numeric(series: pd.Series) -> bool:
    """Check if column is numeric (int, float)."""
    return pd.api.types.is_numeric_dtype(series)


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

    # Combined dataframe for pair plots (real + first synthetic to avoid K× inflation)
    real_m = real_data.copy()
    real_m["_Source"] = "Real"
    synth_first = synthetic_list[0].copy()
    synth_first["_Source"] = "Synthetic"
    combined = pd.concat([real_m, synth_first], ignore_index=True)

    # Single-column: distribution comparison
    for col in cols:
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            if _is_numeric(real_data[col]):
                real_m = real_data.copy()
                real_m["_Source"] = "Real"
                synth_all = pd.concat([s.copy() for s in synthetic_list], ignore_index=True)
                synth_all["_Source"] = "Synthetic"
                combined = pd.concat([real_m, synth_all], ignore_index=True)
                sns.histplot(
                    data=combined, x=col, hue="_Source", multiple="layer", kde=True,
                    ax=ax, alpha=0.6, common_norm=False
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

    # Pair plots: stratify_column vs each other column
    if stratify_column and stratify_column in cols:
        others = [c for c in cols if c != stratify_column]
        sample_size = min(500, len(combined))
        plot_df = combined.sample(n=sample_size, random_state=42) if len(combined) > sample_size else combined
        for other in others:
            try:
                if _is_numeric(real_data[other]):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.boxplot(
                        data=plot_df, x=stratify_column, y=other, hue="_Source",
                        ax=ax
                    )
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                    ax.set_title(f"{stratify_column} vs {other} – Real vs Synthetic")
                    plt.tight_layout()
                else:
                    g = sns.catplot(
                        data=plot_df, x=stratify_column, hue=other, col="_Source",
                        kind="count", height=4, aspect=1.2, legend_out=True
                    )
                    g.fig.suptitle(f"{stratify_column} vs {other} – Real vs Synthetic", y=1.02)
                    for ax in g.axes.flat:
                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                    fig = g.fig
                    plt.tight_layout()
                safe_name = f"pair_{stratify_column}_vs_{other}".replace(" ", "_")
                fig.savefig(plot_dir / f"{safe_name}.{ext}", dpi=100, bbox_inches="tight")
                plt.close()
            except Exception as e:
                if not quiet:
                    print(f"    Skip pair plot ({stratify_column}, {other}): {e}")
            finally:
                plt.close("all")


def _compute_ml_augmentation_metrics(
    real_training_data: pd.DataFrame,
    synthetic_list: list[pd.DataFrame],
    real_validation_data: pd.DataFrame,
    metadata: object,
    prediction_column: str,
    minority_class_label: str | int,
    quiet: bool,
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

    # Get metadata as dict (SingleTableMetadata format)
    if hasattr(metadata, "_convert_to_single_table"):
        meta_dict = metadata._convert_to_single_table().to_dict()
    else:
        meta_dict = metadata

    precision_scores = []
    recall_scores = []
    for syn in synthetic_list:
        try:
            prec = BinaryClassifierPrecisionEfficacy.compute(
                real_training_data=real_training_data,
                synthetic_data=syn,
                real_validation_data=real_validation_data,
                metadata=meta_dict,
                prediction_column_name=prediction_column,
                minority_class_label=minority_class_label,
                classifier="XGBoost",
                fixed_recall_value=0.9,
            )
            rec = BinaryClassifierRecallEfficacy.compute(
                real_training_data=real_training_data,
                synthetic_data=syn,
                real_validation_data=real_validation_data,
                metadata=meta_dict,
                prediction_column_name=prediction_column,
                minority_class_label=minority_class_label,
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
    eval_visualizations: bool = False,
    stratify_column: str | None = None,
    eval_plot_format: str = "pdf",
    eval_ml_augmentation: bool = False,
    eval_k_runs: int = 5,
    test_data: pd.DataFrame | None = None,
    prediction_column: str | None = None,
    minority_class_label: str | int | None = None,
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
                    quiet=quiet,
                )
                if metrics:
                    ml_results[name] = metrics
                    if quiet is False:
                        for m, v in metrics.items():
                            print(f"    {m}: {v['mean']:.4f} ± {v['std']:.4f}")

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
