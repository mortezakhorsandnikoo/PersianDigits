"""Model definition for the Persian handwritten digit CNN.

Kept in one place so both training (the notebook) and inference (predict.py)
use exactly the same architecture. See preprocessing.json for the input
convention and normalization constants that go with the released weights.
"""
import torch
import torch.nn as nn


class DigitCNN(nn.Module):
    """Small CNN for single-channel 16x16 Persian digit images.

    features:  Conv(1->32,3x3,pad1) -> BN -> ReLU -> MaxPool2
               Conv(32->64,3x3,pad1) -> BN -> ReLU -> MaxPool2
    classifier: Flatten -> Dropout -> Linear(->128) -> ReLU -> Dropout -> Linear(->10)
    """

    def __init__(self, image_size: int = 16, n_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, image_size, image_size)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
