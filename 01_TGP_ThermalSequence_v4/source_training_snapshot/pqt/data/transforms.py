from __future__ import annotations

import torch
from torchvision import transforms

from pqt.registry.transforms import TRANSFORM_REGISTRY


class CurveAugment:
    """Lightweight jitter for normalized [1, 1, L] curve tensors.

    strength in (0, 1] scales: amplitude jitter ±20%, additive gaussian noise
    up to sigma 0.02 in [-1, 1] units (~2.5 quantization levels), and a cyclic
    shift of up to 8 samples.
    """

    def __init__(self, strength: float = 1.0) -> None:
        self.strength = float(strength)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        s = self.strength
        mean = tensor.mean()
        scale = 1.0 + (torch.rand(()) * 2 - 1) * 0.2 * s
        tensor = mean + (tensor - mean) * scale
        tensor = tensor + torch.randn_like(tensor) * (0.02 * s)
        shift = int(torch.randint(-8, 9, ()) * s)
        if shift != 0:
            tensor = torch.roll(tensor, shifts=shift, dims=-1)
        return tensor.clamp(-1.0, 1.0)


def build_transform(
    input_mode: str,
    image_size: tuple[int, int],
    train_contrast_factor: float | None = None,
) -> transforms.Compose:
    builder = TRANSFORM_REGISTRY.get(input_mode)
    return builder(image_size, train_contrast_factor=train_contrast_factor)


@TRANSFORM_REGISTRY.register("grayscale")
def build_grayscale_transform(
    image_size: tuple[int, int],
    train_contrast_factor: float | None = None,
) -> transforms.Compose:
    augmentation = []
    if train_contrast_factor is not None:
        augmentation.append(transforms.ColorJitter(contrast=(train_contrast_factor, train_contrast_factor)))
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            *augmentation,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


@TRANSFORM_REGISTRY.register("curve")
def build_curve_transform(
    image_size: tuple[int, int],
    train_contrast_factor: float | None = None,
) -> transforms.Compose:
    return build_grayscale_transform(image_size, train_contrast_factor)


@TRANSFORM_REGISTRY.register("rgb")
def build_rgb_transform(
    image_size: tuple[int, int],
    train_contrast_factor: float | None = None,
) -> transforms.Compose:
    augmentation = []
    if train_contrast_factor is not None:
        augmentation.append(transforms.ColorJitter(contrast=(train_contrast_factor, train_contrast_factor)))
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            *augmentation,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
