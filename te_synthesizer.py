"""SDV synthesizer training, evaluation, and result persistence."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from te_metrics import (
    compute_ml_augmentation_metrics,
    compute_ml_regression_metrics,
    compute_privacy_metrics,
    compute_quality_metrics,
)
from te_visualization import generate_comparative_plots, generate_eval_visualizations


def _parse_csv_columns(value: str | None) -> list[str] | None:
    """Parse a comma-separated column string into a list, or return None."""
    if not value:
        return None
    return [c.strip() for c in value.split(",") if c.strip()]


def train_synthesizers(
    stratified_subsamples: dict[str, pd.DataFrame],
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
    eval_ml_max_epochs: int | None = None,
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
    """Train an SDV synthesizer on each subsample and optionally evaluate it.

    When eval_k_runs > 1, trains K times per subsample and aggregates metrics as mean ± std.
    """
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

    out = Path(output_dir).resolve()
    synthetic_dir = out / "synthetic" / dataset_name / synthesizer_name
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"Training {synthesizer_name} on {len(stratified_subsamples)} subsamples...")

    k_runs = max(1, eval_k_runs)
    ml_results: dict = {}
    privacy_results: dict = {}
    quality_results: dict = {}
    synthetic_by_subsample: dict = {}

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*save_to_json.*replicability",
            category=UserWarning,
            module="sdv",
        )
        for name, subsample_df in stratified_subsamples.items():
            n_rows = len(subsample_df)
            metadata = (
                base_metadata
                if base_metadata is not None
                else Metadata.detect_from_dataframe(data=subsample_df, table_name=name)
            )

            synthetic_list: list[pd.DataFrame] = []
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
                    if not quiet:
                        print(f"  {name}: run {k+1}/{k_runs} -> {csv_path}")

            if eval_visualizations and synthetic_list:
                if not quiet:
                    print(f"  {name}: generating eval visualizations...")
                generate_eval_visualizations(
                    real_data=subsample_df,
                    synthetic_list=synthetic_list,
                    metadata=metadata,
                    output_dir=synthetic_dir,
                    subsample_name=name,
                    stratify_column=stratify_column,
                    quiet=quiet,
                    eval_plot_format=eval_plot_format,
                )

            if eval_ml_augmentation and synthetic_list and test_data is not None and prediction_column:
                target_is_numeric = pd.api.types.is_numeric_dtype(test_data[prediction_column])
                metrics = {}
                if target_is_numeric:
                    if not quiet:
                        print(f"  {name}: evaluating ML augmentation (regression, K={k_runs})...")
                    metrics = compute_ml_regression_metrics(
                        real_training_data=subsample_df,
                        synthetic_list=synthetic_list,
                        real_validation_data=test_data.copy(),
                        metadata=metadata,
                        prediction_column=prediction_column,
                        ml_max_epochs=eval_ml_max_epochs,
                        quiet=quiet,
                    )
                elif minority_class_label is not None:
                    if not quiet:
                        print(f"  {name}: evaluating ML augmentation (binary classification, K={k_runs})...")
                    metrics = compute_ml_augmentation_metrics(
                        real_training_data=subsample_df,
                        synthetic_list=synthetic_list,
                        real_validation_data=test_data.copy(),
                        metadata=metadata,
                        prediction_column=prediction_column,
                        minority_class_label=minority_class_label,
                        ml_label_encode=eval_ml_label_encode,
                        ml_max_epochs=eval_ml_max_epochs,
                        quiet=quiet,
                    )
                if metrics:
                    ml_results[name] = metrics
                    if not quiet:
                        for m, v in metrics.items():
                            print(f"    {m}: {v['mean']:.4f} ± {v['std']:.4f}")

            if eval_privacy and synthetic_list:
                if not quiet:
                    msg = f"  {name}: evaluating privacy (DCRBaselineProtection, DCROverfittingProtection"
                    if eval_privacy_disclosure and eval_privacy_disclosure_known and eval_privacy_disclosure_sensitive:
                        msg += ", DisclosureProtection"
                    msg += f") (K={k_runs})..."
                    print(msg)
                privacy_metrics = compute_privacy_metrics(
                    real_data=subsample_df,
                    synthetic_list=synthetic_list,
                    metadata=metadata,
                    num_rows_subsample=eval_privacy_subsample,
                    real_validation_data=test_data.copy() if test_data is not None else None,
                    disclosure_known_columns=_parse_csv_columns(eval_privacy_disclosure_known) if eval_privacy_disclosure else None,
                    disclosure_sensitive_columns=_parse_csv_columns(eval_privacy_disclosure_sensitive) if eval_privacy_disclosure else None,
                    disclosure_continuous_columns=_parse_csv_columns(eval_privacy_disclosure_continuous) if eval_privacy_disclosure else None,
                    disclosure_computation=eval_privacy_disclosure_computation if eval_privacy_disclosure else "cap",
                    quiet=quiet,
                )
                if privacy_metrics:
                    privacy_results[name] = privacy_metrics
                    if not quiet:
                        for m, v in privacy_metrics.items():
                            if "mean" in v and "std" in v:
                                print(f"    {m}: {v['mean']:.4f} ± {v['std']:.4f}")

            if eval_quality and synthetic_list:
                if not quiet:
                    print(f"  {name}: evaluating quality (KSComplement, ContingencySimilarity, CorrelationSimilarity) (K={k_runs})...")
                quality_metrics = compute_quality_metrics(
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
                    if not quiet:
                        for m, v in quality_metrics.items():
                            if "mean" in v and "std" in v:
                                print(f"    {m}: {v['mean']:.4f} ± {v['std']:.4f}")

            if comparative_plots and synthetic_list:
                synthetic_by_subsample[name] = synthetic_list

    if comparative_plots and (clean_data is not None or stratified_subsamples):
        generate_comparative_plots(
            clean_data=clean_data if clean_data is not None else next(iter(stratified_subsamples.values())),
            stratified_subsamples=stratified_subsamples,
            synthetic_by_subsample=synthetic_by_subsample,
            output_dir=output_dir,
            dataset_name=dataset_name,
            quiet=quiet,
        )

    _save_ml_results(ml_results, synthetic_dir, quiet)
    _save_privacy_results(privacy_results, synthetic_dir, quiet)
    _save_quality_results(quality_results, synthetic_dir, quiet)

    if not quiet:
        print(f"Synthetic data saved to {synthetic_dir}")


# ---------------------------------------------------------------------------
# Result persistence helpers
# ---------------------------------------------------------------------------

def _save_ml_results(ml_results: dict, synthetic_dir: Path, quiet: bool) -> None:
    if not ml_results:
        return
    eval_path = synthetic_dir / "ml_augmentation_eval.json"
    dumpable = {
        name: {k: {"mean": v["mean"], "std": v["std"], "scores": v["scores"]} for k, v in m.items()}
        for name, m in ml_results.items()
    }
    with open(eval_path, "w") as f:
        json.dump(dumpable, f, indent=2)
    if not quiet:
        print(f"ML augmentation results saved to {eval_path}")


def _save_privacy_results(privacy_results: dict, synthetic_dir: Path, quiet: bool) -> None:
    if not privacy_results:
        return
    privacy_path = synthetic_dir / "privacy_eval.json"
    dumpable = {}
    for name, m in privacy_results.items():
        dumpable[name] = {}
        for k, v in m.items():
            entry = {"mean": v["mean"], "std": v["std"], "scores": v["scores"]}
            for key in ("median_DCR_to_real_data", "synthetic_data_percentages", "cap_protection", "baseline_protection"):
                if key in v:
                    entry[key] = v[key]
            dumpable[name][k] = entry
    with open(privacy_path, "w") as f:
        json.dump(dumpable, f, indent=2)
    if not quiet:
        print(f"Privacy evaluation results saved to {privacy_path}")


def _save_quality_results(quality_results: dict, synthetic_dir: Path, quiet: bool) -> None:
    if not quality_results:
        return
    quality_path = synthetic_dir / "quality_eval.json"
    dumpable = {}
    for name, m in quality_results.items():
        dumpable[name] = {}
        for k, v in m.items():
            entry = {"mean": v["mean"], "std": v["std"]}
            for key in ("num_pairs", "total_pairs", "pair_means", "num_columns", "total_columns", "column_means", "coefficient"):
                if key in v:
                    entry[key] = v[key]
            dumpable[name][k] = entry
    with open(quality_path, "w") as f:
        json.dump(dumpable, f, indent=2)
    if not quiet:
        print(f"Quality evaluation results saved to {quality_path}")
