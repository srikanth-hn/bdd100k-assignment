"""
Statistical Analyzer for BDD100K Dataset.

Identifies patterns, anomalies, and split-level distribution shifts.
All methods are pure functions (static) so they can be called without
instantiation, but the class groups related logic for clarity.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class BDDAnalyzer:
    """Performs statistical comparisons and anomaly detection on BDD100K data.

    All public methods accept a DataFrame produced by
    :class:`~src.parser.BDDDataParser` and return plain Python dicts or
    DataFrames – no side-effects, no I/O.
    """

    # Threshold: flag a class if its val share deviates > this amount from train
    DISTRIBUTION_SHIFT_THRESHOLD: float = 0.03  # 3 percentage points

    # Minimum absolute count to consider a class "present" in a split
    MIN_CLASS_COUNT: int = 5

    # ------------------------------------------------------------------ #
    # Core statistics
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_split_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for a single DataFrame (one split).

        Args:
            df: Object-level DataFrame from the parser.

        Returns:
            Dictionary with class counts, weather distribution, average
            bounding-box area per class, and overall occlusion rate.
        """
        stats: Dict[str, Any] = {
            "total_objects": int(len(df)),
            "total_images": int(df["image"].nunique()),
            "class_counts": df["class"].value_counts().to_dict(),
            "weather_dist": (
                df["weather"].value_counts(normalize=True).round(6).to_dict()
            ),
            "timeofday_dist": (
                df["timeofday"].value_counts(normalize=True).round(6).to_dict()
            ),
            "scene_dist": (
                df["scene"].value_counts(normalize=True).round(6).to_dict()
                if "scene" in df.columns
                else {}
            ),
            "avg_area_per_class": (
                df.groupby("class")["area"].mean().round(2).to_dict()
            ),
            "median_area_per_class": (
                df.groupby("class")["area"].median().round(2).to_dict()
            ),
            "occlusion_rate": float(round(df["occluded"].mean(), 6)),
            "truncation_rate": float(
                round(df["truncated"].mean(), 6)
                if "truncated" in df.columns
                else 0.0
            ),
            "size_bucket_dist": (
                df["size_bucket"].value_counts(normalize=True).round(6).to_dict()
                if "size_bucket" in df.columns
                else {}
            ),
            "occlusion_per_class": (
                df.groupby("class")["occluded"]
                .mean()
                .round(4)
                .to_dict()
            ),
        }
        return stats

    @staticmethod
    def get_combined_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compute per-split statistics from a combined DataFrame.

        Args:
            df: Combined DataFrame with a ``split`` column
                (e.g. produced by :py:meth:`BDDDataParser.parse_all_splits`).

        Returns:
            Mapping of split-name → stats dict from
            :py:meth:`get_split_stats`.
        """
        if "split" not in df.columns:
            return {"all": BDDAnalyzer.get_split_stats(df)}

        result: Dict[str, Dict[str, Any]] = {}
        for split_name, group in df.groupby("split"):
            result[str(split_name)] = BDDAnalyzer.get_split_stats(group)
        return result

    # ------------------------------------------------------------------ #
    # Train / Val comparison
    # ------------------------------------------------------------------ #

    @staticmethod
    def compare_splits(
        train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Build a side-by-side comparison table for train vs val.

        Computes the normalised class frequency in each split and the
        absolute difference, allowing quick identification of distribution
        shifts.

        Args:
            train_df: Training-split DataFrame.
            val_df:   Validation-split DataFrame.

        Returns:
            DataFrame indexed by class with columns
            ``train_pct``, ``val_pct``, ``abs_diff``, ``flag``.
        """
        def _freq(df: pd.DataFrame) -> pd.Series:
            return df["class"].value_counts(normalize=True).rename("pct")

        train_freq = _freq(train_df)
        val_freq = _freq(val_df)

        cmp = pd.concat(
            [train_freq.rename("train_pct"), val_freq.rename("val_pct")],
            axis=1,
        ).fillna(0.0)
        cmp["abs_diff"] = (cmp["train_pct"] - cmp["val_pct"]).abs()
        cmp["flag"] = (
            cmp["abs_diff"] > BDDAnalyzer.DISTRIBUTION_SHIFT_THRESHOLD
        )
        cmp = cmp.sort_values("train_pct", ascending=False)
        cmp["train_pct"] = cmp["train_pct"].round(4)
        cmp["val_pct"] = cmp["val_pct"].round(4)
        cmp["abs_diff"] = cmp["abs_diff"].round(4)
        return cmp

    @staticmethod
    def compare_weather_splits(
        train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compare weather distributions between train and val splits.

        Args:
            train_df: Training-split DataFrame.
            val_df:   Validation-split DataFrame.

        Returns:
            DataFrame with weather conditions as index and
            ``train_pct`` / ``val_pct`` / ``abs_diff`` columns.
        """
        def _wfreq(df: pd.DataFrame) -> pd.Series:
            return df["weather"].value_counts(normalize=True)

        cmp = pd.concat(
            [
                _wfreq(train_df).rename("train_pct"),
                _wfreq(val_df).rename("val_pct"),
            ],
            axis=1,
        ).fillna(0.0)
        cmp["abs_diff"] = (cmp["train_pct"] - cmp["val_pct"]).abs().round(4)
        return cmp.sort_values("train_pct", ascending=False)

    # ------------------------------------------------------------------ #
    # Anomaly detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_anomalies(
        train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> List[str]:
        """Check for notable anomalies between train and val splits.

        Checks performed:

        1. Classes present in train but absent in val (or vice-versa).
        2. Classes whose relative frequency shifts by more than
           :pyattr:`DISTRIBUTION_SHIFT_THRESHOLD`.
        3. Extreme class imbalance (< 0.1 % share) in train.
        4. Occlusion-rate difference between splits > 5 pp.
        5. Average bounding-box area difference > 20 % per class.

        Args:
            train_df: Training-split DataFrame.
            val_df:   Validation-split DataFrame.

        Returns:
            List of human-readable anomaly description strings.
        """
        anomalies: List[str] = []

        # 1. Class presence mismatch
        train_cls = set(train_df["class"].unique())
        val_cls = set(val_df["class"].unique())
        only_train = train_cls - val_cls
        only_val = val_cls - train_cls
        if only_train:
            anomalies.append(
                f"Classes in TRAIN but missing in VAL: {sorted(only_train)}"
            )
        if only_val:
            anomalies.append(
                f"Classes in VAL but missing in TRAIN: {sorted(only_val)}"
            )

        # 2. Distribution shift
        cmp = BDDAnalyzer.compare_splits(train_df, val_df)
        shifted = cmp[cmp["flag"]].index.tolist()
        if shifted:
            anomalies.append(
                f"Significant distribution shift (>{BDDAnalyzer.DISTRIBUTION_SHIFT_THRESHOLD*100:.0f}pp) "
                f"for classes: {shifted}"
            )

        # 3. Extreme imbalance in train
        total_train = len(train_df)
        for cls, cnt in train_df["class"].value_counts().items():
            share = cnt / total_train
            if share < 0.001:
                anomalies.append(
                    f"Severe class imbalance: '{cls}' is only "
                    f"{share*100:.3f}% of train ({cnt:,} objects)"
                )

        # 4. Occlusion rate gap
        occ_train = train_df["occluded"].mean()
        occ_val = val_df["occluded"].mean()
        if abs(occ_train - occ_val) > 0.05:
            anomalies.append(
                f"Occlusion-rate gap: train={occ_train:.2%}  val={occ_val:.2%}"
            )

        # 5. Box-area drift per class
        train_area = train_df.groupby("class")["area"].mean()
        val_area = val_df.groupby("class")["area"].mean()
        for cls in train_area.index.intersection(val_area.index):
            t_a, v_a = train_area[cls], val_area[cls]
            if t_a > 0 and abs(t_a - v_a) / t_a > 0.20:
                anomalies.append(
                    f"Avg bbox area drift >20% for '{cls}': "
                    f"train={t_a:.0f}  val={v_a:.0f}"
                )

        if not anomalies:
            anomalies.append("No significant anomalies detected.")

        for msg in anomalies:
            print(f"  [Anomaly] {msg}")

        return anomalies

    # ------------------------------------------------------------------ #
    # Per-class detail
    # ------------------------------------------------------------------ #

    @staticmethod
    def class_detail(
        df: pd.DataFrame, class_name: str
    ) -> Dict[str, Any]:
        """Return detailed statistics for a single class.

        Args:
            df: Combined or single-split DataFrame.
            class_name: The target category string.

        Returns:
            Dict with area statistics, occlusion breakdown, and
            weather co-occurrence.
        """
        sub = df[df["class"] == class_name]
        if sub.empty:
            return {"error": f"Class '{class_name}' not found in DataFrame."}

        area_stats = sub["area"].describe().round(2).to_dict()
        occ_by_weather = (
            sub.groupby("weather")["occluded"].mean().round(4).to_dict()
        )
        count_by_weather = sub["weather"].value_counts().to_dict()
        size_dist = (
            sub["size_bucket"].value_counts(normalize=True).round(4).to_dict()
            if "size_bucket" in sub.columns
            else {}
        )

        return {
            "class": class_name,
            "total_count": int(len(sub)),
            "area_stats": area_stats,
            "occlusion_rate": float(round(sub["occluded"].mean(), 4)),
            "truncation_rate": float(
                round(sub["truncated"].mean(), 4)
                if "truncated" in sub.columns
                else 0.0
            ),
            "occlusion_by_weather": occ_by_weather,
            "count_by_weather": count_by_weather,
            "size_distribution": size_dist,
        }

    # ------------------------------------------------------------------ #
    # Imbalance ratio helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def imbalance_ratio(df: pd.DataFrame) -> Tuple[float, str, str]:
        """Return the ratio of the most-frequent to least-frequent class.

        Args:
            df: DataFrame (single or combined split).

        Returns:
            Tuple of (ratio, majority_class, minority_class).
        """
        counts = df["class"].value_counts()
        majority_cls = str(counts.idxmax())
        minority_cls = str(counts.idxmin())
        ratio = float(counts.max() / max(counts.min(), 1))
        return round(ratio, 1), majority_cls, minority_cls