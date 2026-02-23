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

            out = args.output_dir.resolve()
            out.mkdir(parents=True, exist_ok=True)
            plots_dir = out / "plots"
            plots_dir.mkdir(exist_ok=True)

            for name, subsample_df in stratified_subsamples.items():
                cols = list(subsample_df.columns)
                ncols = min(4, len(cols))
                nrows = (len(cols) + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
                axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

                for i, col in enumerate(cols):
                    ax = axes[i]
                    sns.countplot(data=subsample_df, x=col, ax=ax, palette="viridis")
                    ax.set_title(f"{col} in {name}")
                    ax.set_ylabel("Count")
                    ax.tick_params(axis="x", rotation=45)

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

            out = args.output_dir.resolve()
            out.mkdir(parents=True, exist_ok=True)
            comparative_dir = out / "plots" / "comparative"
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
        _train_synthesizers(
            stratified_subsamples=stratified_subsamples,
            synthesizer_name=args.train_synthesizer,
            output_dir=args.output_dir,
            save_synthetic=args.save_synthetic,
            epochs=args.synthesizer_epochs,
            random_state=args.random_state,
            quiet=args.quiet,
            base_metadata=base_metadata,
        )

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
) -> None:
    """Train SDV synthesizer on each subsample and generate synthetic data of same size."""
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
    synthetic_dir = out / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    if save_synthetic:
        (synthetic_dir / synthesizer_name).mkdir(exist_ok=True)

    if quiet is False:
        print(f"Training {synthesizer_name} on {len(stratified_subsamples)} subsamples...")

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
            synthesizer = _make_synthesizer(metadata)

            synthesizer.fit(subsample_df)
            synthetic_data = synthesizer.sample(num_rows=n_rows)

            if save_synthetic:
                synth_subdir = synthetic_dir / synthesizer_name
                csv_path = synth_subdir / f"{name}_synthetic.csv"
                synthetic_data.to_csv(csv_path, index=False)
                metadata_path = synth_subdir / f"{name}_metadata.json"
                metadata.save_to_json(filepath=metadata_path, mode="overwrite")
                if quiet is False:
                    print(f"  {name}: trained, sampled {n_rows} rows -> {csv_path}")

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
