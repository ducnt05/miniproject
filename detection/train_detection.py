from pathlib import Path
import argparse
import json
import random
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataset import YOLODetectionDataset, detection_collate
from .model import EELANDetector


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_targets(targets, grid: int, num_classes: int, device: torch.device):
    result = torch.zeros((len(targets), 5 + num_classes, grid, grid), device=device)
    for batch_index, boxes in enumerate(targets):
        for box in boxes:
            class_id = int(box[0].item())
            xc, yc, width, height = box[1:].tolist()
            cell_x = min(grid - 1, int(xc * grid))
            cell_y = min(grid - 1, int(yc * grid))
            if result[batch_index, 4, cell_y, cell_x] > 0:
                continue
            result[batch_index, 0:4, cell_y, cell_x] = torch.tensor(
                [xc * grid - cell_x, yc * grid - cell_y, width, height], device=device
            )
            result[batch_index, 4, cell_y, cell_x] = 1.0
            result[batch_index, 5 + class_id, cell_y, cell_x] = 1.0
    return result


def detection_loss(prediction, target):
    object_mask = target[:, 4:5] > 0
    no_object_mask = ~object_mask
    xywh_loss = nn.functional.smooth_l1_loss(prediction[:, :4][object_mask.expand_as(prediction[:, :4])],
                                              target[:, :4][object_mask.expand_as(target[:, :4])])
    object_loss = nn.functional.binary_cross_entropy_with_logits(
        prediction[:, 4:5][object_mask], target[:, 4:5][object_mask]
    ) if object_mask.any() else prediction[:, 4:5].sum() * 0
    no_object_loss = nn.functional.binary_cross_entropy_with_logits(
        prediction[:, 4:5][no_object_mask], target[:, 4:5][no_object_mask]
    )
    class_loss = nn.functional.binary_cross_entropy_with_logits(
        prediction[:, 5:][object_mask.expand_as(prediction[:, 5:])],
        target[:, 5:][object_mask.expand_as(target[:, 5:])],
    ) if object_mask.any() else prediction[:, 5:].sum() * 0
    return xywh_loss + object_loss + no_object_loss + class_loss


def run_epoch(model, loader, optimizer, device, num_classes, train):
    model.train(train)
    total = 0.0
    for images, targets, _ in loader:
        images = images.to(device)
        prediction = model(images)
        encoded = build_targets(targets, prediction.shape[-1], num_classes, device)
        loss = detection_loss(prediction, encoded)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += loss.item() * images.size(0)
    return total / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="D:/SPCV/miniproject/datasets/detection_merged")
    parser.add_argument("--out", default="D:/SPCV/miniproject/detection/outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overfit", action="store_true")
    args = parser.parse_args()
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_ds = YOLODetectionDataset(args.data, "train", args.image_size)
    valid_ds = YOLODetectionDataset(args.data, "valid", args.image_size)
    if args.overfit:
        train_ds = Subset(train_ds, list(range(min(10, len(train_ds)))))
        args.epochs = max(args.epochs, 100)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=not args.overfit, collate_fn=detection_collate)
    valid_loader = DataLoader(valid_ds, args.batch_size, shuffle=False, collate_fn=detection_collate)
    model = EELANDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, 3, True)
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_loader, optimizer, device, 3, False)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        print(f"epoch {epoch:03d}: train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")
        if valid_loss < best:
            best = valid_loss
            torch.save({"model": model.state_dict(), "classes": ["cat", "dog", "panda"], "history": history}, out / "best_detection.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved outputs to {out}")


if __name__ == "__main__":
    main()
