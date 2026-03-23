"""
BDD100K Label Converter — JSON to YOLO format.

Confirmed category strings from JSON:
    person, rider, car, truck, bus, train, motor, bike,
    traffic light, traffic sign

Skipped (not object detection):
    area/alternative, area/drivable,
    lane/crosswalk, lane/double white, lane/double yellow,
    lane/road curb, lane/single white, lane/single yellow

Output: BDD100k/labels_yolo/train/ and BDD100k/labels_yolo/val/
        (matches your existing file tree)

Usage:
    cd C:/Users/FNI3KOR/Desktop/BDD_100
    python model/convert_labels.py
"""
from __future__ import annotations

import json
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

BDD_ROOT = Path(r"C:/Users/FNI3KOR/Desktop/BDD_100/BDD100k")

IMG_W = 1280
IMG_H = 720

# Confirmed exact JSON strings → class index (matches bdd100k.yaml)
CLASS_MAP: dict[str, int] = {
    "car":           0,
    "traffic sign":  1,
    "traffic light": 2,
    "truck":         3,
    "bus":           4,
    "rider":         5,
    "train":         6,
    "person":        7,   # JSON uses "person" not "pedestrian"
    "motor":         8,   # JSON uses "motor" not "motorcycle"
    "bike":          9,   # JSON uses "bike" not "bicycle"
}

# These exist in JSON but are NOT object detection classes — skip them
SKIP_CATEGORIES = {
    "area/alternative", "area/drivable",
    "lane/crosswalk", "lane/double white", "lane/double yellow",
    "lane/road curb", "lane/single white", "lane/single yellow",
}

SPLITS = ["train", "val"]


# ── Converter ─────────────────────────────────────────────────────────────────

def convert_box(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float, float]:
    """Convert absolute pixel box to YOLO normalised cx/cy/w/h."""
    cx = (x1 + x2) / 2.0 / IMG_W
    cy = (y1 + y2) / 2.0 / IMG_H
    w  = (x2 - x1) / IMG_W
    h  = (y2 - y1) / IMG_H
    # Clamp to [0, 1] — guards against annotation edge cases
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    )


def convert_split(split: str) -> None:
    """Convert all JSON files for one split to YOLO .txt files."""
    json_dir       = BDD_ROOT / "labels"      / split
    yolo_label_dir = BDD_ROOT / "labels_yolo" / split
    yolo_label_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"  [WARNING] No JSON files in {json_dir}")
        return

    print(f"\n[{split.upper()}]  {len(json_files):,} files")
    print(f"  Input  : {json_dir}")
    print(f"  Output : {yolo_label_dir}")

    converted    = 0
    total_boxes  = 0
    class_counts: dict[str, int] = {k: 0 for k in CLASS_MAP}

    for idx, json_path in enumerate(json_files):
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)

            lines: list[str] = []
            for frame in data.get("frames", []):
                for obj in frame.get("objects", []):
                    category = obj.get("category", "")

                    # Skip lane/area segmentation categories
                    if category in SKIP_CATEGORIES:
                        continue
                    # Skip unknown categories
                    if category not in CLASS_MAP:
                        continue
                    if "box2d" not in obj:
                        continue

                    box = obj["box2d"]
                    cx, cy, w, h = convert_box(
                        box["x1"], box["y1"], box["x2"], box["y2"]
                    )
                    if w <= 0 or h <= 0:
                        continue

                    cid = CLASS_MAP[category]
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    class_counts[category] += 1

            # Write .txt — empty file is valid (image with no target objects)
            out_path = yolo_label_dir / f"{json_path.stem}.txt"
            out_path.write_text("\n".join(lines), encoding="utf-8")

            total_boxes += len(lines)
            converted   += 1

        except Exception as exc:
            print(f"  ERROR {json_path.name}: {exc}")

        if (idx + 1) % 10000 == 0:
            print(f"  Progress: {idx + 1:,}/{len(json_files):,} ...")

    print(f"\n  Converted : {converted:,} files")
    print(f"  Total boxes: {total_boxes:,}")
    print(f"\n  Class breakdown:")
    for name, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        cid = CLASS_MAP[name]
        bar = "#" * min(40, cnt // 5000 + 1)
        print(f"    [{cid}] {name:<15} {cnt:>10,}  {bar}")


def main() -> None:
    print("=" * 60)
    print("BDD100K JSON -> YOLO Label Converter")
    print("=" * 60)
    print(f"Root    : {BDD_ROOT}")
    print(f"Classes : {len(CLASS_MAP)} detection classes")
    print(f"Skipped : {len(SKIP_CATEGORIES)} lane/area categories")

    for split in SPLITS:
        convert_split(split)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Labels written to: {BDD_ROOT / 'labels_yolo'}")
    print("Next: python model/train.py --mode subset")
    print("=" * 60)


if __name__ == "__main__":
    main()