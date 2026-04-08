import cv2
import numpy as np

THICKNESS = 0.035
OVERLAP = 0.12


def frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Generate a hollow rectangular frame mask around a bbox."""
    x, y, w, h = bbox
    t = max(1, int(w * THICKNESS))
    o = max(1, int(h * OVERLAP))
    m = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(m, (x - t, y - t), (x + w + t, y + h + t), 1, -1)
    cv2.rectangle(m, (x + o, y + o), (x + w - o, y + h - o), 0, -1)
    return m
