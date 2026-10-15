# E-ELAN object detection

The merged dataset is already normalized to `0=cat`, `1=dog`, `2=panda`.

Run the 10-image overfit check first:

```powershell
python -m detection.train_detection --overfit --epochs 100
```

Then train the full dataset:

```powershell
python -m detection.train_detection --epochs 30
```

Outputs are written to `detection/outputs`. The model is a compact single-scale
YOLO-style detector intended for the mini project; it is not a drop-in YOLOv7
replacement. The first implementation assigns one object per grid cell, so
overlapping objects sharing a cell are intentionally skipped.
