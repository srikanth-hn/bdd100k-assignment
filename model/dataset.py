"""
Custom BDD100K Dataset Loader for YOLO11.

Fulfils the assignment bonus requirement:
    'build the loader to load the dataset into a model and train for 1 epoch
     for a subset of the data by building the training pipeline.'

Confirmed class strings from JSON category check:
    car, traffic sign, traffic light, truck, bus, rider, train,
    person (not pedestrian), motor (not motorcycle), bike (not bicycle)

Usage (smoke test):
    python dataset.py

Usage (in custom training loop):
    from dataset import build_dataloader
    loader = build_dataloader(img_dir, label_dir, img_size=640, batch_size=16)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# ── Class mapping — confirmed from JSON category check ───────────────────────

CLASS_NAMES: dict[int, str] = {
    0: "car",
    1: "traffic sign",
    2: "traffic light",
    3: "truck",
    4: "bus",
    5: "rider",
    6: "train",
    7: "person",    # JSON string confirmed
    8: "motor",     # JSON string confirmed
    9: "bike",      # JSON string confirmed
}

NUM_CLASSES = len(CLASS_NAMES)  # 10


# ── Dataset ───────────────────────────────────────────────────────────────────

class BDD100KDataset(Dataset):
    """PyTorch Dataset for BDD100K in YOLO label format.

    Reads images from BDD100k/images/{split}/
    Reads labels from BDD100k/labels_yolo/{split}/

    Args:
        img_dir:     Directory containing .jpg images.
        label_dir:   Directory containing YOLO .txt files.
        img_size:    Resize to (img_size x img_size).
        max_samples: Cap total samples — used for subset/bonus runs.
        transform:   Optional custom transform pipeline.
    """

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        img_dir:     str,
        label_dir:   str,
        img_size:    int = 640,
        max_samples: Optional[int] = None,
        transform:   Optional[transforms.Compose] = None,
    ) -> None:
        self.img_dir   = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size  = img_size

        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225],
            ),
        ])

        # Only keep images that have a matching .txt label file
        all_imgs = sorted(
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTENSIONS
        )
        self.samples = [
            p for p in all_imgs
            if (self.label_dir / f"{p.stem}.txt").exists()
        ]

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(
            f"[BDD100KDataset] split='{self.img_dir.name}'  "
            f"images={len(self.samples):,}  "
            f"classes={NUM_CLASSES}  "
            f"img_size={img_size}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (image_tensor, labels_tensor).

        Returns:
            image:  Float tensor (3, H, W).
            labels: Float tensor (N, 5) — [class_id, cx, cy, w, h].
                    (0, 5) empty tensor if no objects in image.
        """
        img_path   = self.samples[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        labels: list[list[float]] = []
        if label_path.exists():
            with label_path.open("r") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append([float(p) for p in parts])

        labels_tensor = (
            torch.tensor(labels, dtype=torch.float32)
            if labels
            else torch.zeros((0, 5), dtype=torch.float32)
        )
        return image, labels_tensor

    @staticmethod
    def collate_fn(
        batch: list[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        """Handle variable object counts across images in a batch.

        Returns:
            images: (B, 3, H, W) stacked tensor.
            labels: List of per-image tensors, each (N_i, 5).
        """
        images, labels = zip(*batch)
        return torch.stack(images, dim=0), list(labels)


# ── Convenience factory ───────────────────────────────────────────────────────

def build_dataloader(
    img_dir:     str,
    label_dir:   str,
    img_size:    int  = 640,
    batch_size:  int  = 16,
    shuffle:     bool = True,
    num_workers: int  = 0,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Build a DataLoader for one BDD100K split."""
    dataset = BDD100KDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=img_size,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=BDD100KDataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

def main() -> None:
    """Verify loader works with your exact file structure."""

    BDD_ROOT  = Path(r"C:/Users/FNI3KOR/Desktop/BDD_100/BDD100k")
    IMG_DIR   = str(BDD_ROOT / "images"      / "train")
    LABEL_DIR = str(BDD_ROOT / "labels_yolo" / "train")

    print("=" * 50)
    print("BDD100KDataset smoke test — 10 classes")
    print("=" * 50)
    print(f"  Images : {IMG_DIR}")
    print(f"  Labels : {LABEL_DIR}")
    print()

    loader = build_dataloader(
        img_dir     = IMG_DIR,
        label_dir   = LABEL_DIR,
        img_size    = 640,
        batch_size  = 4,
        shuffle     = False,
        num_workers = 0,      # 0 = safe on Windows
        max_samples = 20,
    )

    images, labels = next(iter(loader))

    print(f"\n  Batch shape       : {images.shape}")
    print(f"  Labels per image  : {[len(l) for l in labels]}")
    print(f"  Image range       : [{images.min():.2f}, {images.max():.2f}]")

    # Show which classes appear in this mini-batch
    all_cls = [int(l[i, 0]) for l in labels for i in range(len(l))]
    if all_cls:
        print(f"\n  Classes in batch:")
        for cid in sorted(set(all_cls)):
            print(f"    [{cid}] {CLASS_NAMES[cid]}  x{all_cls.count(cid)}")

    print("\nDataset loader working correctly!")


if __name__ == "__main__":
    main()