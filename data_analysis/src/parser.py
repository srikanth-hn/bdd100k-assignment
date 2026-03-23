"""
BDD100K Data Parser for Object Detection.

Extracts 2D bounding boxes and global attributes from individual JSON files.
Supports split-aware parsing (train / val / test) for comparative analysis.
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List

import pandas as pd


class BDDDataParser:
    """Parses BDD100K individual JSON label files into a structured DataFrame.

    Each JSON file corresponds to one image and may contain multiple labelled
    objects.  Only the 10 official detection classes with bounding boxes are
    retained; all other categories (lane, drivable area, etc.) are ignored.

    Args:
        labels_dir: Path to a directory that contains per-image ``.json``
            files **or** a mapping of split-name → directory path for
            multi-split ingestion.

    Example – single split::

        parser = BDDDataParser("/data/BDD100k/labels/train")
        df = parser.parse_directory()

    Example – multi-split (returns one combined DataFrame with a ``split``
    column)::

        parser = BDDDataParser({
            "train": "/data/BDD100k/labels/train",
            "val":   "/data/BDD100k/labels/val",
        })
        df = parser.parse_all_splits()
    """

    TARGET_CLASSES: List[str] = [
        "person",
        "bike",
        "car",
        "truck",
        "bus",
        "train",
        "motor",
        "rider",
        "traffic light",
        "traffic sign",
    ]

    # BDD100K canonical image dimensions (used to derive normalised coords)
    IMG_W: int = 1280
    IMG_H: int = 720

    def __init__(self, labels_dir: str | Dict[str, str]) -> None:
        if isinstance(labels_dir, dict):
            self._split_dirs: Dict[str, str] = labels_dir
            self.labels_dir: str = next(iter(labels_dir.values()))
        else:
            self._split_dirs = {"default": labels_dir}
            self.labels_dir = labels_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_directory(self, split: str = "default") -> pd.DataFrame:
        """Parse all JSON files in a single directory.

        Args:
            split: Label to store in the ``split`` column.  Defaults to
                ``"default"`` when :pyattr:`labels_dir` was provided as a
                plain string.

        Returns:
            DataFrame where every row is one annotated bounding-box object.
        """
        directory = self._split_dirs.get(split, self.labels_dir)
        return self._parse_one_dir(directory, split_label=split)

    def parse_all_splits(self) -> pd.DataFrame:
        """Parse every registered split and concatenate into one DataFrame.

        Returns:
            Combined DataFrame with a ``split`` column identifying the source
            partition (e.g. ``"train"`` / ``"val"``).
        """
        frames: List[pd.DataFrame] = []
        for split_name, directory in self._split_dirs.items():
            print(f"\n[Parser] Processing split: '{split_name}'  →  {directory}")
            df = self._parse_one_dir(directory, split_label=split_name)
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        print(
            f"\n[Parser] All splits done.  Total objects: {len(combined):,}  "
            f"| Splits: {combined['split'].unique().tolist()}"
        )
        return combined

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_one_dir(self, directory: str, split_label: str) -> pd.DataFrame:
        """Walk *directory* and parse each JSON file found there.

        Args:
            directory: Filesystem path to scan.
            split_label: String written into every row's ``split`` column.

        Returns:
            DataFrame for this single split.
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Labels directory not found: '{directory}'"
            )

        json_files = sorted(
            f for f in os.listdir(directory) if f.endswith(".json")
        )
        if not json_files:
            raise ValueError(f"No .json files found in: '{directory}'")

        print(
            f"  Found {len(json_files):,} JSON files in '{directory}'",
            flush=True,
        )

        all_data: List[Dict[str, Any]] = []
        for idx, file_name in enumerate(json_files):
            file_path = os.path.join(directory, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    content = json.load(fh)
                objects = self._extract_objects(content, split_label)
                all_data.extend(objects)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR parsing '{file_name}': {exc}", flush=True)

            if (idx + 1) % 1000 == 0 or idx == 0:
                print(
                    f"  Processed {idx + 1:,}/{len(json_files):,} files "
                    f"({len(all_data):,} objects so far)",
                    flush=True,
                )

        print(
            f"  Split '{split_label}' done → {len(all_data):,} objects.",
            flush=True,
        )
        return pd.DataFrame(all_data)

    def _extract_objects(
        self, data: Dict[str, Any], split_label: str
    ) -> List[Dict[str, Any]]:
        """Extract detection-relevant fields from one parsed JSON dict.

        Args:
            data: Parsed JSON content of a single BDD100K label file.
            split_label: Propagated to each extracted row.

        Returns:
            List of flat dicts – one per valid bounding-box annotation.
        """
        extracted: List[Dict[str, Any]] = []

        img_name: str = data.get("name", "unknown")
        attr: Dict[str, Any] = data.get("attributes", {})
        weather: str = attr.get("weather", "undefined")
        timeofday: str = attr.get("timeofday", "undefined")
        scene: str = attr.get("scene", "undefined")

        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                category: str = obj.get("category", "")
                if category not in self.TARGET_CLASSES:
                    continue
                if "box2d" not in obj:
                    continue

                box: Dict[str, float] = obj["box2d"]
                x1, y1, x2, y2 = (
                    box["x1"],
                    box["y1"],
                    box["x2"],
                    box["y2"],
                )
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                area = w * h

                obj_attrs: Dict[str, Any] = obj.get("attributes", {})
                occluded: bool = bool(obj_attrs.get("occluded", False))
                truncated: bool = bool(obj_attrs.get("truncated", False))

                # Aspect ratio (guard against zero-height boxes)
                aspect_ratio: float = (w / h) if h > 0 else 0.0

                # Relative size bucket (useful for small-object analysis)
                total_pixels = self.IMG_W * self.IMG_H  # 921 600
                rel_area = area / total_pixels
                if rel_area < 0.005:
                    size_bucket = "small"
                elif rel_area < 0.02:
                    size_bucket = "medium"
                else:
                    size_bucket = "large"

                extracted.append(
                    {
                        "split": split_label,
                        "image": img_name,
                        "class": category,
                        "weather": weather,
                        "timeofday": timeofday,
                        "scene": scene,
                        "occluded": occluded,
                        "truncated": truncated,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": w,
                        "height": h,
                        "area": area,
                        "aspect_ratio": aspect_ratio,
                        "cx": (x1 + x2) / 2.0,
                        "cy": (y1 + y2) / 2.0,
                        "size_bucket": size_bucket,
                    }
                )

        return extracted