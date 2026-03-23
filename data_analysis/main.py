"""CLI entry point for the BDD100K data analysis project.

Usage examples
--------------
Analyse training split only::

    python main.py /data/BDD100k/labels/train

Analyse both train AND val splits (enables all comparison plots)::

    python main.py /data/BDD100k/labels/train \\
                   --val-dir /data/BDD100k/labels/val \\
                   --output-dir /results

Inside Docker (paths are mounted at /data and /results)::

    docker run --rm \\
        -v /host/BDD100k:/data \\
        -v /host/results:/results \\
        bdd-analysis \\
        /data/labels/train --val-dir /data/labels/val --output-dir /results
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd

from src.analyzer import BDDAnalyzer
from src.parser import BDDDataParser
from src.visualizer import create_visualizations


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse BDD100K label directories for object detection.\n"
            "Produces class-distribution charts, spatial heatmaps, "
            "train/val comparison plots, and a JSON summary."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "labels_dir",
        help="Path to directory containing .json label files (train split).",
    )
    parser.add_argument(
        "--val-dir",
        default=None,
        metavar="VAL_DIR",
        help=(
            "Optional path to the validation label directory.  "
            "When provided, train-vs-val comparison plots are generated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="/results",
        metavar="OUTPUT_DIR",
        help="Directory where reports and visualisations will be saved. "
             "Created automatically if it does not exist.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901  (complexity OK for a pipeline entry-point)
    """Run the full BDD100K data-analysis pipeline."""
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ----------------------------------------------------------------
        # 1. Parsing
        # ----------------------------------------------------------------
        print("=" * 60, flush=True)
        print("BDD100K Data Analysis Pipeline", flush=True)
        print("=" * 60, flush=True)
        print(f"Train labels : {args.labels_dir}", flush=True)
        print(f"Val labels   : {args.val_dir or '(not provided)'}", flush=True)
        print(f"Output dir   : {output_dir}", flush=True)
        print("-" * 60, flush=True)

        has_val = args.val_dir is not None

        if has_val:
            split_dirs = {
                "train": args.labels_dir,
                "val": args.val_dir,
            }
            data_parser = BDDDataParser(split_dirs)
            print("\n[Step 1/4] Parsing train + val splits …", flush=True)
            combined_df: pd.DataFrame = data_parser.parse_all_splits()
            train_df: pd.DataFrame = combined_df[combined_df["split"] == "train"].copy()
            val_df: pd.DataFrame = combined_df[combined_df["split"] == "val"].copy()
        else:
            data_parser = BDDDataParser(args.labels_dir)
            print("\n[Step 1/4] Parsing single split …", flush=True)
            combined_df = data_parser.parse_directory(split="train")
            train_df = combined_df.copy()
            val_df = pd.DataFrame()

        print(f"  Combined DataFrame shape: {combined_df.shape}", flush=True)
        print(
            f"  Train objects : {len(train_df):,}  |  "
            f"Val objects : {len(val_df):,}",
            flush=True,
        )

        # ----------------------------------------------------------------
        # 2. Statistical analysis
        # ----------------------------------------------------------------
        print("\n[Step 2/4] Computing statistics …", flush=True)

        train_stats = BDDAnalyzer.get_split_stats(train_df)
        print(f"  Train metrics computed: {len(train_stats)} keys", flush=True)

        val_stats: dict = {}
        if has_val and not val_df.empty:
            val_stats = BDDAnalyzer.get_split_stats(val_df)
            print(f"  Val metrics computed  : {len(val_stats)} keys", flush=True)

        # Anomaly detection (needs both splits)
        anomalies: list[str] = []
        if has_val and not val_df.empty:
            print("\n  Running anomaly detection …", flush=True)
            anomalies = BDDAnalyzer.detect_anomalies(train_df, val_df)

        # Imbalance ratio
        ratio, majority, minority = BDDAnalyzer.imbalance_ratio(train_df)
        print(
            f"  Train imbalance ratio: {ratio}×  "
            f"('{majority}' vs '{minority}')",
            flush=True,
        )

        # ----------------------------------------------------------------
        # 3. Visualisations
        # ----------------------------------------------------------------
        print("\n[Step 3/4] Generating visualisations …", flush=True)
        saved_plots = create_visualizations(combined_df, str(output_dir))
        print(f"  Saved {len(saved_plots)} visualisation(s).", flush=True)

        # ----------------------------------------------------------------
        # 4. Persist results
        # ----------------------------------------------------------------
        print("\n[Step 4/4] Saving summary …", flush=True)

        summary = {
            "train": train_stats,
            "val": val_stats if has_val else None,
            "anomalies": anomalies,
            "imbalance": {
                "ratio": ratio,
                "majority_class": majority,
                "minority_class": minority,
            },
            "saved_plots": [str(p) for p in saved_plots],
        }

        # Also keep the flat legacy key for backward compatibility
        summary["class_counts"] = train_stats["class_counts"]
        summary["weather_dist"] = train_stats["weather_dist"]
        summary["avg_area_per_class"] = train_stats["avg_area_per_class"]
        summary["occlusion_rate"] = train_stats["occlusion_rate"]

        summary_path = output_dir / "analysis_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"  Summary saved → {summary_path}", flush=True)

        # Print anomalies to stdout for CI / Docker log visibility
        if anomalies:
            print("\n  Detected anomalies:", flush=True)
            for msg in anomalies:
                print(f"    • {msg}", flush=True)

        # ----------------------------------------------------------------
        # Done
        # ----------------------------------------------------------------
        print("\n" + "=" * 60, flush=True)
        print("✓  Analysis complete!", flush=True)
        print(f"   Total objects parsed : {len(combined_df):,}", flush=True)
        print(f"   Results directory    : {output_dir}", flush=True)
        print("=" * 60, flush=True)

    except Exception as exc:  # noqa: BLE001
        print(f"\nFATAL ERROR: {exc}", flush=True)
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()