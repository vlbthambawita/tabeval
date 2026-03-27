"""Stratified train/test split and subsampling utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_stratified_split(
    clean_data: pd.DataFrame,
    stratify_column: str,
    test_size: int,
    random_state: int,
    quiet: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Split clean_data into train and test sets using stratified sampling.

    For numerical stratify columns, quantile-binning is applied automatically.

    Returns:
        (train_data, test_data, drop_bins): drop_bins is True if a temporary
        '_stratify_bins' column was added and must be removed by the caller.
    """
    if stratify_column not in clean_data.columns:
        raise ValueError(
            f"Stratify column '{stratify_column}' not in data. "
            f"Available: {list(clean_data.columns)}"
        )

    if test_size > len(clean_data):
        raise ValueError(
            f"clean_data has {len(clean_data)} rows; cannot create test set of {test_size}."
        )

    # For numerical stratify column: bin first (train_test_split requires ≥2 per class)
    if pd.api.types.is_numeric_dtype(clean_data[stratify_column]):
        clean_data = clean_data.copy()
        n_bins = min(5, clean_data[stratify_column].nunique(), len(clean_data) // 2)
        for attempt in range(n_bins, 1, -1):
            try:
                clean_data["_stratify_bins"] = pd.qcut(
                    clean_data[stratify_column], q=attempt, labels=False, duplicates="drop"
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
        stratify_vals = clean_data[stratify_column]
        drop_bins = False

    train_data, test_data = train_test_split(
        clean_data,
        test_size=test_size,
        stratify=stratify_vals,
        random_state=random_state,
    )

    if not quiet:
        print(f"Created test set: {len(test_data)} samples")

    return train_data, test_data, drop_bins


def create_subsamples(
    train_data: pd.DataFrame,
    subsample_sizes: list[int],
    stratify_column: str,
    random_state: int,
    drop_bins: bool = False,
    quiet: bool = False,
) -> dict[str, pd.DataFrame]:
    """Create stratified subsamples from train_data.

    Args:
        drop_bins: If True, train_data contains '_stratify_bins' column used for stratification.
    """
    stratify_for_subsample = (
        train_data["_stratify_bins"] if drop_bins else train_data[stratify_column]
    )

    stratified_subsamples: dict[str, pd.DataFrame] = {}
    for size in subsample_sizes:
        if size > len(train_data):
            if not quiet:
                print(f"Warning: subsample size {size} > train_data ({len(train_data)}). Skipping.")
            continue
        _, subsample = train_test_split(
            train_data,
            test_size=size / len(train_data),
            stratify=stratify_for_subsample,
            random_state=random_state,
        )
        stratified_subsamples[f"subsample_{size}"] = subsample
        if not quiet:
            print(f"Generated subsample_{size}: {len(subsample)} rows")

    return stratified_subsamples


def drop_bin_column(
    clean_data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    stratified_subsamples: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Remove the temporary '_stratify_bins' column from all DataFrames."""
    clean_data = clean_data.drop(columns=["_stratify_bins"])
    train_data = train_data.drop(columns=["_stratify_bins"])
    test_data = test_data.drop(columns=["_stratify_bins"])
    stratified_subsamples = {
        name: df.drop(columns=["_stratify_bins"])
        for name, df in stratified_subsamples.items()
    }
    return clean_data, train_data, test_data, stratified_subsamples


def save_datasets(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    stratified_subsamples: dict[str, pd.DataFrame],
    output_dir: Path,
    dataset_name: str,
    quiet: bool = False,
) -> None:
    """Save train, test, and subsample DataFrames to CSV files."""
    out_ds = Path(output_dir).resolve() / dataset_name
    out_ds.mkdir(parents=True, exist_ok=True)

    test_data.to_csv(out_ds / "test_data.csv", index=False)
    train_data.to_csv(out_ds / "train_data.csv", index=False)

    for name, df in stratified_subsamples.items():
        df.to_csv(out_ds / f"{name}.csv", index=False)

    if not quiet:
        print(f"Saved datasets to {out_ds}")
