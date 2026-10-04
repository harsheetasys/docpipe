"""A compact U-Net for text-region segmentation (binary mask: text vs background).

Why a segmenter in addition to the detector? Boxes tell us WHERE a field is; the mask
tells us HOW MUCH of the page is text, catches rotated/curved lines the axis-aligned
boxes miss, and its mean-confidence is a cheap health signal for drift monitoring.
~0.5M parameters so it trains on CPU in minutes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 16):
        super().__init__()
        self.e1 = _block(in_ch, base)
        self.e2 = _block(base, base * 2)
        self.e3 = _block(base * 2, base * 4)
        self.b = _block(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = _block(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = _block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = _block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.b(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)  # logits


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)).mean()


def iou_score(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> float:
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum()
    union = ((p + target) > 0).float().sum()
    return float(inter / union.clamp(min=1))
