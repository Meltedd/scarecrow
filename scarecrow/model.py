"""Detection model loading and inference."""

import warnings

import numpy as np
import torch
import torch.export.passes
import torch.nn as nn
import torch.nn.functional as F


def load(weights: str, device: str | None = None) -> nn.Module:
    """Load detection model with frozen weights."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("CUDA not available, running on CPU (slower)")
    warnings.filterwarnings("ignore", message=".*not writable.*")
    ep = torch.export.load(weights)
    ep = torch.export.passes.move_to_device_pass(ep, device)
    model = ep.module()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def detect(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable forward pass returning raw boxes and scores.

    Args:
        images: (B, 3, H, W) float32 in [0, 1], RGB.

    Returns:
        boxes: (B, N, 4) xywh
        scores: (B, N) class confidence
    """
    raw = model(images)[0]
    # (B, 4+nc, N) -> (B, N, 4+nc)
    preds = raw.permute(0, 2, 1)
    return preds[..., :4], preds[..., 4:].max(dim=-1).values


def letterbox(images: torch.Tensor, imgsz: int) -> torch.Tensor:
    """Pad to square imgsz preserving aspect ratio."""
    _, _, h, w = images.shape
    if h == imgsz and w == imgsz:
        return images
    scale = imgsz / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = F.interpolate(images, size=(nh, nw), mode="bilinear", align_corners=False)
    # 114 is YOLO's standard letterbox padding value
    padded = torch.full((images.shape[0], 3, imgsz, imgsz), 114 / 255, device=images.device)
    py, px = (imgsz - nh) // 2, (imgsz - nw) // 2
    padded[:, :, py : py + nh, px : px + nw] = resized
    return padded


def predict(
    model: nn.Module,
    img: np.ndarray,
    imgsz: int = 640,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.7,
) -> tuple[list[tuple[int, int, int, int]], float]:
    """Run detection with NMS. img: RGB uint8."""
    h, w = img.shape[:2]
    device = next(model.parameters()).device
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    padded = letterbox(tensor, imgsz)

    with torch.no_grad():
        boxes, scores = detect(model, padded)

    boxes, scores = boxes[0], scores[0]
    mask = scores > conf_thresh
    boxes, scores = boxes[mask], scores[mask]
    if len(scores) == 0:
        return [], 0.0

    # xywh -> xyxy
    x, y, bw, bh = boxes.unbind(-1)
    xyxy = torch.stack([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2], dim=-1)

    keep = _nms(xyxy, scores, iou_thresh)
    xyxy, scores = xyxy[keep], scores[keep]

    scale = imgsz / max(h, w)
    pad_x = (imgsz - int(w * scale)) // 2
    pad_y = (imgsz - int(h * scale)) // 2
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale

    max_conf = float(scores.max())
    bboxes = []
    for i in range(len(scores)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        bboxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
    return bboxes, max_conf


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """NMS for single-class detection. boxes: (N, 4) xyxy, scores: (N,)."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.max(boxes[i, 0], boxes[rest, 0])
        yy1 = torch.max(boxes[i, 1], boxes[rest, 1])
        xx2 = torch.min(boxes[i, 2], boxes[rest, 2])
        yy2 = torch.min(boxes[i, 3], boxes[rest, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter)
        order = rest[iou <= iou_thresh]
    return torch.stack(keep)
