"""
Visualization module for BDD100K analysis.

Generates research-grade charts covering class distribution, spatial density,
weather breakdowns, occlusion analysis, train-vs-val comparison, size
distribution, and per-class box geometry.  All plots are saved as high-DPI
PNGs suitable for a technical report.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

_PALETTE_SPLIT = {"train": "#4C72B0", "val": "#DD8452"}
_PALETTE_WEATHER = "Set2"
_PALETTE_CLASS = "Blues_d"
_FIG_DPI = 150


def _save(fig: plt.Figure, output_dir: str, filename: str) -> str:
    """Save *fig* to *output_dir/filename* and close it.

    Returns:
        Absolute path of the saved file.
    """
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main visualizer class
# ---------------------------------------------------------------------------


class BDDVisualizer:
    """Generates all visualizations for the BDD100K data-analysis step.

    Args:
        df: Combined object-level DataFrame (with a ``split`` column when
            train/val data are included together).
        output_dir: Directory where PNG files will be written.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str) -> None:
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", font_scale=1.05)

    # ------------------------------------------------------------------ #
    # 1. Class distribution (single split or combined)
    # ------------------------------------------------------------------ #

    def plot_class_distribution(self, title: str = "Class Distribution") -> str:
        """Bar chart of absolute object counts per class.

        Args:
            title: Plot title string.

        Returns:
            Path to saved PNG.
        """
        order = self.df["class"].value_counts().index.tolist()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(
            data=self.df,
            x="class",
            order=order,
            palette=_PALETTE_CLASS,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=35)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        # Annotate bars
        for bar in ax.patches:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max(h * 0.01, 200),
                    f"{int(h):,}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        fig.tight_layout()
        return _save(fig, self.output_dir, "class_dist.png")

    # ------------------------------------------------------------------ #
    # 2. Spatial heatmap (vanishing-point analysis)
    # ------------------------------------------------------------------ #

    def plot_spatial_heatmap(self) -> str:
        """KDE density map of bounding-box centres across the image plane.

        Returns:
            Path to saved PNG.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(
            data=self.df,
            x="cx",
            y="cy",
            fill=True,
            cmap="rocket",
            ax=ax,
        )
        ax.set_title(
            "Spatial Object Density (Vanishing Point Analysis)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlim(0, 1280)
        ax.set_ylim(720, 0)  # invert Y for image coordinates
        ax.set_xlabel("cx  (pixels)")
        ax.set_ylabel("cy  (pixels)")
        fig.tight_layout()
        return _save(fig, self.output_dir, "spatial_heatmap.png")

    # ------------------------------------------------------------------ #
    # 3. Train vs Val class distribution (side-by-side %)
    # ------------------------------------------------------------------ #

    def plot_train_val_class_comparison(self) -> str:
        """Grouped bar chart comparing normalised class frequencies.

        Requires the DataFrame to have a ``split`` column.

        Returns:
            Path to saved PNG.
        """
        if "split" not in self.df.columns:
            print("  [Visualizer] Skipping train/val comparison – no 'split' column.")
            return ""

        freq = (
            self.df.groupby(["split", "class"])
            .size()
            .reset_index(name="count")
        )
        totals = self.df.groupby("split").size().rename("total")
        freq = freq.join(totals, on="split")
        freq["pct"] = freq["count"] / freq["total"] * 100

        order = (
            self.df[self.df["split"] == "train"]["class"]
            .value_counts()
            .index.tolist()
        )

        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(
            data=freq,
            x="class",
            y="pct",
            hue="split",
            order=order,
            palette=_PALETTE_SPLIT,
            ax=ax,
        )
        ax.set_title(
            "Class Distribution: Train vs Val (% of split total)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Percentage (%)")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Split")
        fig.tight_layout()
        return _save(fig, self.output_dir, "train_val_class_comparison.png")

    # ------------------------------------------------------------------ #
    # 4. Weather distribution per split
    # ------------------------------------------------------------------ #

    def plot_weather_distribution(self) -> str:
        """Stacked proportional bar chart of weather conditions per split.

        Returns:
            Path to saved PNG.
        """
        col = "split" if "split" in self.df.columns else None

        if col:
            weather_freq = (
                self.df.groupby([col, "weather"])
                .size()
                .reset_index(name="count")
            )
            totals = self.df.groupby(col).size().rename("total")
            weather_freq = weather_freq.join(totals, on=col)
            weather_freq["pct"] = weather_freq["count"] / weather_freq["total"] * 100
            pivot = weather_freq.pivot(index=col, columns="weather", values="pct").fillna(0)
        else:
            pivot = (
                self.df["weather"]
                .value_counts(normalize=True)
                .mul(100)
                .to_frame("all")
                .T
            )

        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            colormap=_PALETTE_WEATHER,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title(
            "Weather Condition Distribution per Split",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Split")
        ax.set_ylabel("Percentage (%)")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Weather", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        return _save(fig, self.output_dir, "weather_distribution.png")

    # ------------------------------------------------------------------ #
    # 5. Occlusion rate per class (split-aware)
    # ------------------------------------------------------------------ #

    def plot_occlusion_per_class(self) -> str:
        """Grouped bar chart of occlusion rate (%) per class, split-aware.

        Returns:
            Path to saved PNG.
        """
        hue_col = "split" if "split" in self.df.columns else None

        occ = (
            self.df.groupby(
                ["class"] + ([hue_col] if hue_col else [])
            )["occluded"]
            .mean()
            .mul(100)
            .reset_index(name="occlusion_rate_pct")
        )

        order = (
            occ.groupby("class")["occlusion_rate_pct"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=occ,
            x="class",
            y="occlusion_rate_pct",
            hue=hue_col,
            order=order,
            palette=_PALETTE_SPLIT if hue_col else "Reds_d",
            ax=ax,
        )
        ax.set_title(
            "Occlusion Rate per Class (Train vs Val)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Occlusion Rate (%)")
        ax.tick_params(axis="x", rotation=35)
        if hue_col:
            ax.legend(title="Split")
        ax.axhline(
            self.df["occluded"].mean() * 100,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Overall mean ({self.df['occluded'].mean()*100:.1f}%)",
        )
        ax.legend()
        fig.tight_layout()
        return _save(fig, self.output_dir, "occlusion_per_class.png")

    # ------------------------------------------------------------------ #
    # 6. Bounding-box area distribution (log scale box-plots)
    # ------------------------------------------------------------------ #

    def plot_bbox_area_distribution(self) -> str:
        """Box-plots of bounding-box area per class on a log scale.

        Returns:
            Path to saved PNG.
        """
        order = (
            self.df.groupby("class")["area"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

        fig, ax = plt.subplots(figsize=(13, 6))
        sns.boxplot(
            data=self.df[self.df["area"] > 0],
            x="class",
            y="area",
            order=order,
            palette="Set3",
            fliersize=2,
            linewidth=0.8,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_title(
            "Bounding-Box Area Distribution per Class (log scale)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Area (px²) — log scale")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        return _save(fig, self.output_dir, "bbox_area_distribution.png")

    # ------------------------------------------------------------------ #
    # 7. Size-bucket distribution (small / medium / large)
    # ------------------------------------------------------------------ #

    def plot_size_bucket_distribution(self) -> str:
        """Stacked proportional bars showing object sizes per class.

        Returns:
            Path to saved PNG.
        """
        if "size_bucket" not in self.df.columns:
            print("  [Visualizer] Skipping size-bucket plot – column missing.")
            return ""

        pivot = (
            self.df.groupby(["class", "size_bucket"])
            .size()
            .reset_index(name="count")
        )
        totals = self.df.groupby("class").size().rename("total")
        pivot = pivot.join(totals, on="class")
        pivot["pct"] = pivot["count"] / pivot["total"] * 100
        pivot_wide = pivot.pivot(
            index="class", columns="size_bucket", values="pct"
        ).fillna(0)

        # Ensure consistent column order
        for col in ["small", "medium", "large"]:
            if col not in pivot_wide.columns:
                pivot_wide[col] = 0.0
        pivot_wide = pivot_wide[["small", "medium", "large"]]
        pivot_wide = pivot_wide.loc[
            self.df["class"].value_counts().index
        ]

        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_wide.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=["#2196F3", "#FF9800", "#F44336"],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title(
            "Object Size Distribution per Class (small / medium / large)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Percentage (%)")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Size bucket")
        fig.tight_layout()
        return _save(fig, self.output_dir, "size_bucket_distribution.png")

    # ------------------------------------------------------------------ #
    # 8. Time-of-day distribution
    # ------------------------------------------------------------------ #

    def plot_timeofday_distribution(self) -> str:
        """Grouped bar chart of time-of-day conditions per split.

        Returns:
            Path to saved PNG.
        """
        col = "split" if "split" in self.df.columns else None

        tod = (
            self.df.groupby(
                (["split", "timeofday"] if col else ["timeofday"])
            )
            .size()
            .reset_index(name="count")
        )

        fig, ax = plt.subplots(figsize=(11, 5))
        if col:
            sns.barplot(
                data=tod,
                x="timeofday",
                y="count",
                hue="split",
                palette=_PALETTE_SPLIT,
                ax=ax,
            )
            ax.legend(title="Split")
        else:
            sns.barplot(data=tod, x="timeofday", y="count", palette="Purples_d", ax=ax)

        ax.set_title(
            "Time-of-Day Distribution per Split",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Time of Day")
        ax.set_ylabel("Object Count")
        ax.tick_params(axis="x", rotation=20)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
        fig.tight_layout()
        return _save(fig, self.output_dir, "timeofday_distribution.png")

    # ------------------------------------------------------------------ #
    # 9. Per-class occlusion under different weather conditions
    # ------------------------------------------------------------------ #

    def plot_occlusion_by_weather(self) -> str:
        """Heatmap of occlusion rate by class × weather.

        Returns:
            Path to saved PNG.
        """
        pivot = (
            self.df.groupby(["class", "weather"])["occluded"]
            .mean()
            .mul(100)
            .unstack(fill_value=0)
            .round(1)
        )

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            linewidths=0.4,
            cbar_kws={"label": "Occlusion rate (%)"},
            ax=ax,
        )
        ax.set_title(
            "Occlusion Rate (%) — Class × Weather Condition",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Weather")
        ax.set_ylabel("Class")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return _save(fig, self.output_dir, "occlusion_by_weather_heatmap.png")

    # ------------------------------------------------------------------ #
    # 10. Aspect-ratio distribution per class
    # ------------------------------------------------------------------ #

    def plot_aspect_ratio_distribution(self) -> str:
        """Violin plot of bounding-box aspect ratios per class.

        Aspect ratio = width / height.  Values > 1 indicate wider-than-tall
        objects (e.g. cars seen from the side).

        Returns:
            Path to saved PNG.
        """
        if "aspect_ratio" not in self.df.columns:
            print("  [Visualizer] Skipping aspect-ratio plot – column missing.")
            return ""

        # Clip extreme outliers for readability
        sub = self.df[self.df["aspect_ratio"].between(0.1, 10)].copy()

        order = (
            sub.groupby("class")["aspect_ratio"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

        fig, ax = plt.subplots(figsize=(13, 6))
        sns.violinplot(
            data=sub,
            x="class",
            y="aspect_ratio",
            order=order,
            palette="muted",
            inner="quartile",
            linewidth=0.8,
            ax=ax,
        )
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Aspect ratio = 1")
        ax.set_title(
            "Bounding-Box Aspect Ratio Distribution per Class",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Aspect Ratio (width / height)")
        ax.tick_params(axis="x", rotation=35)
        ax.legend()
        fig.tight_layout()
        return _save(fig, self.output_dir, "aspect_ratio_distribution.png")

    # ------------------------------------------------------------------ #
    # 11. Train vs Val: imbalance ratio comparison
    # ------------------------------------------------------------------ #

    def plot_train_val_abs_count_comparison(self) -> str:
        """Side-by-side absolute count bars for every class in each split.

        Returns:
            Path to saved PNG.
        """
        if "split" not in self.df.columns:
            print("  [Visualizer] Skipping abs-count comparison – no 'split' column.")
            return ""

        counts = (
            self.df.groupby(["split", "class"])
            .size()
            .reset_index(name="count")
        )
        order = (
            self.df[self.df["split"] == "train"]["class"]
            .value_counts()
            .index.tolist()
        )

        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(
            data=counts,
            x="class",
            y="count",
            hue="split",
            order=order,
            palette=_PALETTE_SPLIT,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_title(
            "Absolute Object Counts per Class: Train vs Val (log scale)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Count (log scale)")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Split")
        fig.tight_layout()
        return _save(fig, self.output_dir, "train_val_abs_count_comparison.png")

    # ------------------------------------------------------------------ #
    # 12. Spatial heatmap split comparison (train vs val side by side)
    # ------------------------------------------------------------------ #

    def plot_spatial_heatmap_split_comparison(self) -> str:
        """Side-by-side KDE density maps for train and val.

        Returns:
            Path to saved PNG.
        """
        if "split" not in self.df.columns:
            print("  [Visualizer] Skipping split spatial comparison – no 'split' column.")
            return ""

        splits = [s for s in ["train", "val"] if s in self.df["split"].values]
        if len(splits) < 2:
            print("  [Visualizer] Skipping split spatial comparison – need both train and val.")
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        for ax, split_name in zip(axes, splits):
            sub = self.df[self.df["split"] == split_name]
            sns.kdeplot(
                data=sub,
                x="cx",
                y="cy",
                fill=True,
                cmap="rocket",
                ax=ax,
            )
            ax.set_title(
                f"Spatial Density — {split_name.capitalize()}",
                fontsize=13,
                fontweight="bold",
            )
            ax.set_xlim(0, 1280)
            ax.set_ylim(720, 0)
            ax.set_xlabel("cx (px)")
            ax.set_ylabel("cy (px)")

        fig.suptitle(
            "Spatial Object Density: Train vs Val",
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout()
        return _save(fig, self.output_dir, "spatial_heatmap_split_comparison.png")

    # ------------------------------------------------------------------ #
    # Convenience: run all plots at once
    # ------------------------------------------------------------------ #

    def generate_all(self) -> List[str]:
        """Call every plot method and return the list of saved file paths.

        Returns:
            List of absolute PNG file paths (skipped plots return empty
            strings and are excluded).
        """
        methods = [
            self.plot_class_distribution,
            self.plot_spatial_heatmap,
            self.plot_train_val_class_comparison,
            self.plot_weather_distribution,
            self.plot_occlusion_per_class,
            self.plot_bbox_area_distribution,
            self.plot_size_bucket_distribution,
            self.plot_timeofday_distribution,
            self.plot_occlusion_by_weather,
            self.plot_aspect_ratio_distribution,
            self.plot_train_val_abs_count_comparison,
            self.plot_spatial_heatmap_split_comparison,
        ]
        paths: List[str] = []
        for method in methods:
            print(f"  [Visualizer] Generating: {method.__name__} …", flush=True)
            result = method()
            if result:
                paths.append(result)
        print(f"  [Visualizer] Done – {len(paths)} plots saved.", flush=True)
        return paths


# ---------------------------------------------------------------------------
# Module-level convenience wrapper (used by main.py)
# ---------------------------------------------------------------------------


def create_visualizations(df: pd.DataFrame, output_dir: str) -> List[str]:
    """Generate the full suite of visualizations and return file paths.

    Args:
        df: Combined object-level DataFrame (with ``split`` column when
            multiple partitions are included).
        output_dir: Directory to write PNGs into.

    Returns:
        List of paths to the generated PNG files.
    """
    visualizer = BDDVisualizer(df, output_dir)
    return visualizer.generate_all()