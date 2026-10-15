from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, g: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, (k - 1) // 2, groups=g, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch, channels, height, width = x.shape
    if groups == 1:
        return x
    if channels % groups != 0:
        raise ValueError("channels must be divisible by groups")
    x = x.reshape(batch, groups, channels // groups, height, width)
    return x.transpose(1, 2).contiguous().reshape(batch, channels, height, width)


class EELANLite(nn.Module):
    """Compact E-ELAN block with grouped expansion, parallel paths and shuffle."""

    def __init__(self, channels: int, groups: int = 2) -> None:
        super().__init__()
        expanded = channels * 2
        self.expand = ConvBNAct(channels, expanded, 1, g=groups)
        branch_channels = expanded // 2
        self.branches = nn.ModuleList([
            nn.Identity(),
            ConvBNAct(branch_channels, branch_channels, 3, g=groups),
            nn.Sequential(ConvBNAct(branch_channels, branch_channels, 3, g=groups),
                          ConvBNAct(branch_channels, branch_channels, 3, g=groups)),
        ])
        self.fuse = ConvBNAct(branch_channels * 3, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.chunk(self.expand(x), 2, dim=1)
        outputs = [branch(part) for branch, part in zip(self.branches, (parts[0], parts[1], parts[1]))]
        y = self.fuse(torch.cat(outputs, dim=1))
        return y + x


class EELANBackbone(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.stem = ConvBNAct(3, base_channels, 3, 2)
        self.stage1 = nn.Sequential(
            ConvBNAct(base_channels, base_channels * 2, 3, 2),
            EELANLite(base_channels * 2),
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(base_channels * 2, base_channels * 4, 3, 2),
            EELANLite(base_channels * 4),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(base_channels * 4, base_channels * 8, 3, 2),
            EELANLite(base_channels * 8),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        return {"p3": p3, "p4": p4, "p5": p5}


class FPNNeck(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.p5_lateral = ConvBNAct(channels * 8, channels * 4, 1)
        self.p4_lateral = ConvBNAct(channels * 4, channels * 4, 1)
        self.fuse = ConvBNAct(channels * 4, channels * 4, 3)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        p5 = self.p5_lateral(features["p5"])
        p4 = self.p4_lateral(features["p4"])
        p5 = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        return self.fuse(p4 + p5)


class EELANDetector(nn.Module):
    """Single-scale YOLO-style detector. Output: B x (5+num_classes) x S x S."""

    def __init__(self, num_classes: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = EELANBackbone(base_channels)
        self.neck = FPNNeck(base_channels)
        self.head = nn.Conv2d(base_channels * 4, 5 + num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.neck(self.backbone(x)))
