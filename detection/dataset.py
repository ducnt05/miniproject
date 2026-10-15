from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class YOLODetectionDataset(Dataset):
    """Loads images and normalized YOLO labels for the merged dataset."""

    def __init__(self, root: str | Path, split: str, image_size: int = 256) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.image_dir = self.root / split / "images"
        self.label_dir = self.root / split / "labels"
        self.images = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        image_path = self.images[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        image = image.resize((self.image_size, self.image_size))
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        boxes: List[List[float]] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id, xc, yc, width, height = map(float, parts)
                boxes.append([class_id, xc, yc, width, height])
        target = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5)
        target[:, 1:] = target[:, 1:].clamp(0.0, 1.0)
        return image_tensor, target, str(image_path)


def detection_collate(batch):
    images, targets, paths = zip(*batch)
    return torch.stack(images), list(targets), list(paths)

