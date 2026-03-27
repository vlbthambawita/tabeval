"""Data loading utilities for tabular evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(args) -> tuple[pd.DataFrame, object]:
    """Load data from file or SDV demo.

    Returns:
        (data, metadata): data is always a DataFrame. metadata is Metadata from
        download_demo when using --sdv-demo, else None (file loads have no metadata).
    """
    if args.data_path:
        path = Path(args.data_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            data = pd.read_csv(path)
        elif suffix in (".xlsx", ".xls"):
            data = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .xlsx")

        if not args.quiet:
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
    if not args.quiet:
        print(f"Loaded SDV demo '{args.sdv_dataset}': {len(data)} rows (metadata included)")
    return data, metadata
