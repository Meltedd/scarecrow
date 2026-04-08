"""Gradient-based adversarial frame pattern optimization."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import torch
import torch.nn.functional as F

from scarecrow import model as yolo
from scarecrow.mask import frame_mask

PATTERN_H, PATTERN_W = 32, 128
MIN_PLATE_WIDTH = 30


@dataclass
class PlateData:
    image: torch.Tensor  # (3, H, W) original resolution, float32 [0,1]
    mask: torch.Tensor  # (1, H, W) original resolution


@dataclass
class Config:
    steps: int = 1000
    lr: float = 0.01
    eot_samples: int = 4
    batch_size: int = 2
    imgsz: int = 640


def composite(images: torch.Tensor, masks: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
    """Blend pattern into images where mask is active. All (B, C/1, H, W)."""
    return images * (1 - masks) + pattern.expand_as(images) * masks


def _composite_letterbox(
    pattern: torch.Tensor, items: list[PlateData], imgsz: int
) -> torch.Tensor:
    """Composite at original resolution, letterbox for detection. Returns (B, 3, imgsz, imgsz)."""
    result = []
    for item in items:
        pat = F.interpolate(pattern, size=item.image.shape[1:], mode="nearest")
        comp = composite(item.image.unsqueeze(0), item.mask.unsqueeze(0), pat)
        result.append(yolo.letterbox(comp, imgsz).squeeze(0))
    return torch.stack(result)


def eot_transform(images: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Differentiable EoT augmentations: rotation, perspective, brightness, blur, noise, scale."""
    B, C, H, W = images.shape
    device = images.device

    # Geometric: rotation +/-10 deg, perspective tilt +/-20 deg, pan +/-25 deg
    # Combined into one grid_sample to avoid compounding interpolation blur
    rot = 20 * (torch.rand(B, device=device, generator=rng) - 0.5)
    tilt = 40 * (torch.rand(B, device=device, generator=rng) - 0.5)
    pan = 50 * (torch.rand(B, device=device, generator=rng) - 0.5)

    rad_r = torch.deg2rad(rot)
    cos_r = torch.cos(rad_r)
    sin_r = torch.sin(rad_r)
    px = torch.sin(torch.deg2rad(pan)) / 2.0
    py = torch.sin(torch.deg2rad(tilt)) / 2.0

    M = torch.zeros(B, 3, 3, device=device)
    M[:, 0, 0] = cos_r
    M[:, 0, 1] = -sin_r
    M[:, 1, 0] = sin_r
    M[:, 1, 1] = cos_r
    M[:, 2, 0] = px * cos_r + py * sin_r
    M[:, 2, 1] = -px * sin_r + py * cos_r
    M[:, 2, 2] = 1.0

    gy = torch.linspace(-1, 1, H, device=device)
    gx = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(gy, gx, indexing="ij")
    base = torch.stack([xx, yy, torch.ones_like(xx)], dim=0).reshape(3, -1)
    warped = torch.bmm(M, base.unsqueeze(0).expand(B, -1, -1))
    grid = (warped[:, :2] / warped[:, 2:3]).permute(0, 2, 1).reshape(B, H, W, 2)

    images = F.grid_sample(
        images, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )

    # Brightness +/-0.1
    brightness = 1.0 + 0.2 * (torch.rand(B, 1, 1, 1, device=device, generator=rng) - 0.5)
    images = images * brightness

    # Contrast 0.8-1.2
    contrast = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=device, generator=rng)
    images = (images - 0.5) * contrast + 0.5

    # Gaussian blur k=3
    sigma = 0.1 + 0.9 * torch.rand(1, device=device, generator=rng).item()
    ax = torch.arange(3, device=device, dtype=torch.float32) - 1
    kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel_2d = (kernel[:, None] * kernel[None, :]).view(1, 1, 3, 3).expand(C, -1, -1, -1)
    images = F.conv2d(F.pad(images, [1, 1, 1, 1], mode="reflect"), kernel_2d, groups=C)

    # Additive noise, random sigma 0.005-0.03
    noise_sigma = 0.005 + 0.025 * torch.rand(1, device=device, generator=rng).item()
    images = images + torch.randn(B, C, H, W, device=device, generator=rng) * noise_sigma

    # Scale jitter 0.5-1.2
    scale = 0.5 + 0.7 * torch.rand(1, device=device, generator=rng).item()
    nh, nw = int(H * scale), int(W * scale)
    images = F.interpolate(
        F.interpolate(images, size=(nh, nw), mode="bilinear", align_corners=False),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )

    return images.clamp(0, 1)


def _load_plates(
    path: Path, model: torch.nn.Module, device: torch.device | str
) -> tuple[list[PlateData], int]:
    """Detect plates in an image and return PlateData at original resolution."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)

    bboxes, _ = yolo.predict(model, rgb)
    plates: list[PlateData] = []
    skipped = 0
    for bbox in bboxes:
        if bbox[2] < MIN_PLATE_WIDTH:
            skipped += 1
            continue
        mask_t = torch.from_numpy(frame_mask(rgb.shape[:2], bbox)).float().unsqueeze(0).to(device)
        plates.append(PlateData(img_t, mask_t))

    return plates, skipped


def optimize(
    image_path: str | Path,
    weights: str,
    config: Config | None = None,
    on_step: Callable | None = None,
) -> torch.Tensor:
    """Optimize a grayscale frame pattern to suppress plate detection.

    Returns pattern values in [0, 1] of shape (PATTERN_H, PATTERN_W).
    """
    if config is None:
        config = Config()

    det_model = yolo.load(weights)
    device = next(det_model.parameters()).device

    dataset, skipped = _load_plates(Path(image_path), det_model, device)
    if not dataset:
        detail = f" (detected {skipped} < {MIN_PLATE_WIDTH}px)" if skipped else ""
        raise ValueError(f"No usable plates in {image_path}{detail}")

    msg = f"Loaded {len(dataset)} plate crops from {image_path}"
    if skipped:
        msg += f" (skipped {skipped} < {MIN_PLATE_WIDTH}px)"
    print(msg)

    pattern = torch.full(
        (1, 1, PATTERN_H, PATTERN_W), 0.5, device=device, requires_grad=True
    )
    opt = torch.optim.Adam([pattern], lr=config.lr)
    rng = torch.Generator(device=device)

    tau = 3.0
    for step in range(config.steps):
        idx = torch.randperm(len(dataset), device=device)[:config.batch_size]
        batch_items = [dataset[i] for i in idx.tolist()]

        composited = _composite_letterbox(pattern, batch_items, config.imgsz)
        det_losses = []
        for _ in range(config.eot_samples):
            augmented = eot_transform(composited, rng)
            _, scores = yolo.detect(det_model, augmented)
            det_losses.append(scores.max(dim=-1).values.mean())

        det_agg = torch.logsumexp(tau * torch.stack(det_losses), dim=0).div(tau)
        (grad_det,) = torch.autograd.grad(det_agg, pattern)
        pattern.grad = grad_det

        opt.step()
        pattern.data.clamp_(0, 1)

        if on_step:
            on_step(step, sum(v.item() for v in det_losses) / config.eot_samples)

    return pattern.detach().squeeze()
