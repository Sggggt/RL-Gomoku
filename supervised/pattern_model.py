from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gomoku_logic import BOARD_SIZE
from .patterns import PATTERN_CLASSES


class PatternRecognizer(nn.Module):
    """
    Input: (B, 2, 20, 20) canonical planes.
    Output: logits for tactical motifs.
    """

    def __init__(self, num_classes: int = len(PATTERN_CLASSES)):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

