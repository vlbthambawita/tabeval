"""Metric computation: ML augmentation, privacy, and quality metrics."""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# ML augmentation helpers
# ---------------------------------------------------------------------------

def prepare_ml_augmentation_data(
    real_training_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    real_validation_data: pd.DataFrame,
    prediction_column: str | None = None,
    minority_class_label: str | int | None = None,
    ml_label_encode: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | int]:
    """Filter validation for seen categories and convert object columns to category.

    XGBoost requires int/float/bool/category (not object).
    When ml_label_encode=True, label-encode categorical columns to int to avoid
    XGBoost enable_categorical errors.

    Returns:
        (train, synth, val, effective_minority_class_label)
    """
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

        cat_cols = [c for c in train_prep.columns if not pd.api.types.is_numeric_dtype(train_prep[c])]
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
            train_prep[c] = enc.transform(train_prep[c].astype(str).fillna("__nan__").values.reshape(-1, 1)).ravel().astype(int)
            syn_prep[c] = enc.transform(syn_prep[c].astype(str).fillna("__nan__").values.reshape(-1, 1)).ravel().astype(int)
            val_prep[c] = enc.transform(val_prep[c].astype(str).fillna("__nan__").values.reshape(-1, 1)).ravel().astype(int)
            if c == prediction_column:
                pred_enc = enc

        if pred_enc is not None and minority_class_label is not None:
            lab_encoded = pred_enc.transform([[str(minority_class_label)]])
            effective_minority = int(lab_encoded[0, 0])

        for df in (train_prep, syn_prep, val_prep):
            for c in cat_cols:
                df[c] = df[c].astype("int64")
            for c in df.columns:
                if c not in cat_cols:
                    df[c] = df[c].astype("float64")

    return (train_prep, syn_prep, val_prep, effective_minority)


def metadata_for_label_encoded(meta_dict: dict, cat_cols: list, prediction_column: str) -> dict:
    """Return metadata copy with categorical columns (except target) as 'numerical'.

    SDMetrics then treats them as numeric, avoiding astype('category') that causes
    XGBoost 'index type must match' errors when train/synth/val have mixed int/float.
    """
    import copy
    out = copy.deepcopy(meta_dict)
    cols = out.get("columns", {})
    for c in cat_cols:
        if c != prediction_column and c in cols:
            cols[c] = {**cols[c], "sdtype": "numerical"}
    return out


# ---------------------------------------------------------------------------
# ML augmentation metrics
# ---------------------------------------------------------------------------

def compute_ml_augmentation_metrics(
    real_training_data: pd.DataFrame,
    synthetic_list: list[pd.DataFrame],
    real_validation_data: pd.DataFrame,
    metadata: object,
    prediction_column: str,
    minority_class_label: str | int,
    ml_label_encode: bool = False,
    ml_max_epochs: int | None = None,
    quiet: bool = False,
) -> dict:
    """Compute BinaryClassifierPrecisionEfficacy and BinaryClassifierRecallEfficacy.

    Runs across K synthetic datasets and returns mean ± std.
    ml_max_epochs: if set, patches XGBoost n_estimators in SDMetrics ClassifierTrainer.
    """
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

    da_base = None
    _orig_init = None
    if ml_max_epochs is not None:
        try:
            import sdmetrics.single_table.data_augmentation.base as _da_base
            da_base = _da_base
            _orig_init = _da_base.ClassifierTrainer.__init__

            def _patched_init(self, *args, **kwargs):
                _orig_init(self, *args, **kwargs)
                self._classifier.set_params(n_estimators=ml_max_epochs)

            _da_base.ClassifierTrainer.__init__ = _patched_init
        except Exception:
            da_base = None
            _orig_init = None

    try:
        if hasattr(metadata, "_convert_to_single_table"):
            meta_dict = metadata._convert_to_single_table().to_dict()
        else:
            meta_dict = metadata

        if ml_label_encode:
            cat_cols = [c for c in real_training_data.columns if not pd.api.types.is_numeric_dtype(real_training_data[c])]
            meta_dict = metadata_for_label_encoded(meta_dict, cat_cols, prediction_column)

        precision_scores = []
        recall_scores = []
        for syn in synthetic_list:
            try:
                train_prep, syn_prep, val_prep, effective_minority = prepare_ml_augmentation_data(
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
    finally:
        if da_base is not None and _orig_init is not None:
            da_base.ClassifierTrainer.__init__ = _orig_init


def compute_ml_regression_metrics(
    real_training_data: pd.DataFrame,
    synthetic_list: list[pd.DataFrame],
    real_validation_data: pd.DataFrame,
    metadata: object,
    prediction_column: str,
    ml_max_epochs: int | None = None,
    quiet: bool = False,
) -> dict:
    """Compute regression ML efficacy metrics (LinearRegression, MLPRegressor).

    TSTR setup: train on synthetic, test on real validation data.
    ml_max_epochs: if set, overrides MLPRegressor max_iter.
    """
    try:
        from sdmetrics.single_table import LinearRegression, MLPRegressor
        from sdmetrics.single_table.efficacy import regression as _regression_module
    except ImportError as e:
        if not quiet:
            print(f"  Skipping ML regression eval: {e}")
        return {}

    if not pd.api.types.is_numeric_dtype(real_validation_data[prediction_column]):
        if not quiet:
            print("  Skipping ML regression eval: target variable is not numerical.")
        return {}

    if hasattr(metadata, "_convert_to_single_table"):
        meta_dict = metadata._convert_to_single_table().to_dict()
    else:
        meta_dict = metadata

    _orig_mlp_kwargs = None
    if ml_max_epochs is not None:
        _orig_mlp_kwargs = _regression_module.MLPRegressor.MODEL_KWARGS
        _regression_module.MLPRegressor.MODEL_KWARGS = {
            **(dict(_orig_mlp_kwargs) if _orig_mlp_kwargs else {}),
            "max_iter": ml_max_epochs,
        }

    try:
        lr_scores: list[float] = []
        mlp_scores: list[float] = []

        for syn in synthetic_list:
            try:
                lr = LinearRegression.compute(
                    test_data=real_validation_data,
                    train_data=syn,
                    target=prediction_column,
                    metadata=meta_dict,
                )
                lr_scores.append(float(lr))
            except Exception as e:
                if not quiet:
                    print(f"    LinearRegression efficacy failed for one run: {e}")
            try:
                mlp = MLPRegressor.compute(
                    test_data=real_validation_data,
                    train_data=syn,
                    target=prediction_column,
                    metadata=meta_dict,
                )
                mlp_scores.append(float(mlp))
            except Exception as e:
                if not quiet:
                    print(f"    MLPRegressor efficacy failed for one run: {e}")

        import numpy as np
        result: dict = {}
        if lr_scores:
            result["LinearRegression"] = {
                "mean": float(np.mean(lr_scores)),
                "std": float(np.std(lr_scores)),
                "scores": lr_scores,
            }
        if mlp_scores:
            result["MLPRegressor"] = {
                "mean": float(np.mean(mlp_scores)),
                "std": float(np.std(mlp_scores)),
                "scores": mlp_scores,
            }
        return result
    finally:
        if _orig_mlp_kwargs is not None:
            _regression_module.MLPRegressor.MODEL_KWARGS = _orig_mlp_kwargs


# ---------------------------------------------------------------------------
# Privacy metrics
# ---------------------------------------------------------------------------

def compute_privacy_metrics(
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

    Metrics:
      - DCRBaselineProtection
      - DCROverfittingProtection (requires real_validation_data)
      - DisclosureProtection (requires known/sensitive columns)

    Returns mean ± std and per-run scores for each metric.
    """
    try:
        from sdmetrics.single_table import DCRBaselineProtection, DCROverfittingProtection, DisclosureProtection
    except ImportError as e:
        if not quiet:
            print(f"  Skipping privacy eval: {e}")
        return {}

    meta_dict = (
        metadata._convert_to_single_table().to_dict()
        if hasattr(metadata, "_convert_to_single_table")
        else metadata
    )

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
            score_val = result.get("score") if isinstance(result, dict) else float(result)
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

    # DCROverfittingProtection
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
                score_val = result.get("score") if isinstance(result, dict) else float(result)
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

    # DisclosureProtection
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
                score_val = result.get("score") if isinstance(result, dict) else float(result)
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


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_quality_metrics(
    real_data: pd.DataFrame,
    synthetic_list: list[pd.DataFrame],
    num_rows_subsample: int | None = None,
    real_association_threshold: float | None = None,
    correlation_coefficient: str = "Pearson",
    correlation_threshold: float | None = None,
    quiet: bool = False,
) -> dict:
    """Compute marginal and pairwise quality metrics.

    Metrics:
      - KSComplement: numerical marginals
      - TVComplement: categorical/boolean marginals
      - ContingencySimilarity: categorical/mixed column pairs
      - CorrelationSimilarity: numerical column pairs
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

    # KSComplement
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
                    score = KSComplement.compute(real_data=real_vals, synthetic_data=syn_vals)
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

    # TVComplement
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
                    score = TVComplement.compute(real_data=real_vals, synthetic_data=syn_vals)
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

    # ContingencySimilarity
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

    # CorrelationSimilarity
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
