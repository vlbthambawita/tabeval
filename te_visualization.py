"""Visualization utilities: subsample bar plots, comparative plots, eval plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _is_numeric(series: pd.Series) -> bool:
    """Check if a column is numeric (int or float)."""
    return pd.api.types.is_numeric_dtype(series)


def generate_subsample_plots(
    stratified_subsamples: dict[str, pd.DataFrame],
    output_dir: Path,
    dataset_name: str,
    quiet: bool = False,
) -> None:
    """Generate per-column count bar plots for each subsample."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        if not quiet:
            print("Skipping plots: matplotlib/seaborn not installed")
        return

    plots_dir = Path(output_dir).resolve() / "plots" / dataset_name
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

    if not quiet:
        print(f"Saved plots to {plots_dir}")


def generate_comparative_plots(
    clean_data: pd.DataFrame,
    stratified_subsamples: dict[str, pd.DataFrame],
    synthetic_by_subsample: dict,
    output_dir: Path,
    dataset_name: str,
    quiet: bool = False,
) -> None:
    """Generate comparative plots per column: subsamples only.

    Bar (percentage) for categorical columns, box plot for numerical columns.
    """
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

    comparative_dir = Path(output_dir).resolve() / "plots" / dataset_name / "comparative"
    comparative_dir.mkdir(parents=True, exist_ok=True)

    real_sources = dict(stratified_subsamples)
    cols = list(clean_data.columns)

    for col in cols:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            is_num = _is_numeric(clean_data[col])

            if is_num:
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
                all_cats = pd.Index(clean_data[col].dropna().unique())
                for df in stratified_subsamples.values():
                    all_cats = all_cats.union(pd.Index(df[col].dropna().unique()))
                cat_order = list(all_cats)
                n_cats = len(cat_order)
                if n_cats == 0:
                    continue

                real_counts = {
                    name: df[col].value_counts().reindex(cat_order, fill_value=0).fillna(0).values
                    for name, df in real_sources.items()
                }
                real_pct = {
                    name: 100 * vals / vals.sum() if vals.sum() > 0 else vals
                    for name, vals in real_counts.items()
                }
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
                        ax.text(
                            x[i] + offset, v + 0.02 * max(y_max, 1),
                            f"{v:.1f}%", ha="center", va="bottom", fontsize=7,
                        )

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


def generate_eval_visualizations(
    real_data: pd.DataFrame,
    synthetic_list: list[pd.DataFrame],
    metadata: object,
    output_dir: Path,
    subsample_name: str,
    stratify_column: str | None,
    quiet: bool = False,
    eval_plot_format: str = "pdf",
) -> None:
    """Generate Seaborn distribution comparison plots (real vs synthetic) per subsample.

    When len(synthetic_list) > 1, synthetic bars show mean ± std across K runs.
    """
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
    plot_dir = Path(output_dir) / "eval_plots" / subsample_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    k_synth = len(synthetic_list)
    show_std = k_synth > 1

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
                    ax=ax, alpha=0.6, fill=True, warn_singular=False,
                )
            else:
                all_cats = pd.Index(real_data[col].dropna().unique())
                for s in synthetic_list:
                    all_cats = all_cats.union(pd.Index(s[col].dropna().unique()))
                cat_order = list(all_cats)
                real_counts = real_data[col].value_counts().reindex(cat_order, fill_value=0).fillna(0)
                synth_counts_list = [
                    s[col].value_counts().reindex(cat_order, fill_value=0).fillna(0)
                    for s in synthetic_list
                ]
                synth_mean = np.array([c.values for c in synth_counts_list]).mean(axis=0)
                synth_std = (
                    np.array([c.values for c in synth_counts_list]).std(axis=0)
                    if show_std else np.zeros(len(cat_order))
                )

                x = np.arange(len(cat_order))
                width = 0.35
                real_vals = np.asarray(real_counts.values, dtype=float)
                ax.bar(x - width / 2, real_vals, width, label="Real", alpha=0.9, color="C0")
                ax.bar(
                    x + width / 2, synth_mean, width,
                    label="Synthetic (mean)" + (" ± std" if show_std else ""),
                    alpha=0.9, color="C1",
                )
                if show_std and np.any(synth_std > 0):
                    ax.errorbar(x + width / 2, synth_mean, yerr=synth_std, fmt="none", color="black", capsize=2)

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
                ax.set_ylim(0, y_max + offset * 3)
                ax.legend()

            ax.set_title(f"{col} – Real vs Synthetic" + (f" (K={k_synth} runs)" if show_std else ""))
            plt.tight_layout()
            fig.savefig(plot_dir / f"column_{col}.{ext}", dpi=100, bbox_inches="tight")
            plt.close()
        except Exception as e:
            if not quiet:
                print(f"    Skip column plot '{col}': {e}")
