from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def image_paths(path: Path) -> list[Path]:
    """Resolve a file or directory to a list of image paths."""
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Failed to decode: {p}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save(img: np.ndarray, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 and img.shape[2] == 3 else img
    if not cv2.imwrite(str(p), out):
        raise OSError(f"Failed to write: {p}")


def save_pattern(pattern: np.ndarray, path: str | Path) -> None:
    """Save grayscale pattern (float [0,1] or uint8 [0,255]) as PNG."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if pattern.dtype in (np.float32, np.float64):
        pattern = np.round(pattern * 255).clip(0, 255).astype(np.uint8)
    if not cv2.imwrite(str(p), pattern):
        raise OSError(f"Failed to write: {p}")


def load_pattern(path: str | Path) -> np.ndarray:
    """Load pattern PNG as float32 array in [0, 1]."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pattern not found: {p}")
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Failed to decode pattern: {p}")
    return m.astype(np.float32) / 255.0
