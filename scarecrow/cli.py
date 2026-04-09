#!/usr/bin/env python3
"""Adversarial license plate frame generator."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from scarecrow.io import image_paths, load, load_pattern, save, save_pattern
from scarecrow.mask import frame_mask
from scarecrow.model import DEFAULT_WEIGHTS_FILENAME

OCR_PAD = 0.15
_PLATE_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"


def _apply_pattern(img, pattern, bboxes):
    """Overlay grayscale pattern onto frame regions. Modifies img in-place."""
    pr = cv2.resize(pattern, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    vals = (pr * 255).astype(np.uint8)
    for bbox in bboxes:
        mask = frame_mask(img.shape[:2], bbox)
        for c in range(3):
            img[:, :, c] = np.where(mask, vals, img[:, :, c])


def _crop_for_ocr(img, bbox, pad=OCR_PAD):
    """Crop plate region with padding."""
    x, y, w, h = bbox
    h_img, w_img = img.shape[:2]
    cx1 = max(0, x - int(w * pad))
    cy1 = max(0, y - int(h * pad))
    cx2 = min(w_img, x + w + int(w * pad))
    cy2 = min(h_img, y + h + int(h * pad))
    return img[cy1:cy2, cx1:cx2]


def _read_plate(reader, crop):
    """Read plate text from a crop. Selects the largest text region."""
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    result, _ = reader(bgr)
    if not result:
        return ""
    # result: list of (bbox, text, conf); bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    best = max(result, key=lambda r: (r[0][2][1] - r[0][0][1]) * (r[0][1][0] - r[0][0][0]))
    # Filter to plate characters only
    return "".join(c for c in best[1].strip().upper() if c in _PLATE_CHARS)


def _cmd_optimize(args) -> int:
    from scarecrow.optimize import Config, optimize

    config = Config(steps=args.steps)

    def on_step(step, loss):
        if step % 25 == 0 or step == config.steps - 1:
            print(f"step {step:4d}  det={loss:.4f}")

    pattern = optimize(args.input, args.weights, config, on_step=on_step)
    pattern = pattern.cpu().numpy()
    out = args.output or "pattern.png"
    save_pattern(pattern, out)
    print(f"Saved pattern to {out}")
    return 0


def _cmd_apply(args) -> int:
    from scarecrow import model as yolo

    img = load(args.input)
    pattern = load_pattern(args.pattern)

    model = yolo.load(args.weights)
    bboxes, _ = yolo.predict(model, img)
    if not bboxes:
        print("No plate detected", file=sys.stderr)
        return 1

    _apply_pattern(img, pattern, bboxes)
    out = args.output or str(Path(args.input).with_stem(Path(args.input).stem + "_framed"))
    save(img, out)
    print(f"Saved to {out}")
    return 0


def _cmd_eval(args) -> int:
    from scarecrow import model as yolo
    from scarecrow.optimize import MIN_PLATE_WIDTH

    yolo_model = yolo.load(args.weights)
    pattern = load_pattern(args.pattern)
    paths = image_paths(Path(args.input))
    if not paths:
        print(f"No images in {args.input}", file=sys.stderr)
        return 1

    ocr_reader = None
    if args.ocr:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            print("rapidocr-onnxruntime required for --ocr. Install: uv sync --extra ocr", file=sys.stderr)
            return 1
        ocr_reader = RapidOCR()

    total, evaded, tiny_skipped = 0, 0, 0
    conf_clean_sum, conf_adv_sum = 0.0, 0.0
    ocr_total, ocr_corrupted = 0, 0

    for p in paths:
        img = load(p)
        bboxes, clean_conf = yolo.predict(yolo_model, img)
        n_before = len(bboxes)
        bboxes = [b for b in bboxes if b[2] >= MIN_PLATE_WIDTH]
        tiny_skipped += n_before - len(bboxes)
        if not bboxes:
            print(f"{p.name}  [no detection on clean]")
            continue

        total += 1
        adv = img.copy()
        _apply_pattern(adv, pattern, bboxes)
        adv_bboxes, adv_conf = yolo.predict(yolo_model, adv)
        adv_bboxes = [b for b in adv_bboxes if b[2] >= MIN_PLATE_WIDTH]
        was_evaded = len(adv_bboxes) == 0

        if was_evaded:
            evaded += 1
        conf_clean_sum += clean_conf
        conf_adv_sum += adv_conf

        status = "EVADED" if was_evaded else f"conf {clean_conf:.3f} -> {adv_conf:.3f}"

        if ocr_reader is not None:
            ocr_parts = []
            for bbox in bboxes:
                clean_crop = _crop_for_ocr(img, bbox)
                adv_crop = _crop_for_ocr(adv, bbox)
                clean_text = _read_plate(ocr_reader, clean_crop)
                adv_text = _read_plate(ocr_reader, adv_crop)
                if len(clean_text) >= 2:
                    ocr_total += 1
                    changed = clean_text != adv_text
                    if changed:
                        ocr_corrupted += 1
                    ocr_parts.append(
                        f'"{clean_text}" -> "{adv_text}"' if changed
                        else f'"{clean_text}" [unchanged]'
                    )
            if ocr_parts:
                status += "  OCR: " + ", ".join(ocr_parts)

        print(f"{p.name}  {status}")

    print("---")
    if total > 0:
        print(
            f"Evasion: {evaded}/{total} ({100 * evaded / total:.0f}%)"
            f" | Mean conf: {conf_clean_sum / total:.3f} -> {conf_adv_sum / total:.3f}"
        )
    if ocr_reader is not None and ocr_total > 0:
        print(f"OCR corrupted: {ocr_corrupted}/{ocr_total} ({100 * ocr_corrupted / ocr_total:.0f}%)")
    if tiny_skipped:
        print(f"(skipped {tiny_skipped} detections < {MIN_PLATE_WIDTH}px)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    opt = sub.add_parser("optimize", help="Optimize adversarial frame pattern")
    opt.add_argument("input", help="Plate image file")
    opt.add_argument("--weights", default=DEFAULT_WEIGHTS_FILENAME, help="Model weights file")
    opt.add_argument("--steps", type=int, default=1000)
    opt.add_argument("-o", "--output", help="Output pattern path (default: pattern.png)")

    ap = sub.add_parser("apply", help="Apply pattern to a plate image")
    ap.add_argument("input", help="Input image")
    ap.add_argument("--pattern", required=True, help="Pattern PNG")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_FILENAME, help="Model weights file")
    ap.add_argument("-o", "--output", help="Output image path")

    ev = sub.add_parser("eval", help="Evaluate pattern effectiveness")
    ev.add_argument("input", help="Image file or directory")
    ev.add_argument("--pattern", required=True, help="Pattern PNG")
    ev.add_argument("--weights", default=DEFAULT_WEIGHTS_FILENAME, help="Model weights file")
    ev.add_argument("--ocr", action="store_true", help="Evaluate OCR corruption (requires rapidocr)")

    args = p.parse_args()
    cmd = {"optimize": _cmd_optimize, "apply": _cmd_apply, "eval": _cmd_eval}
    return cmd[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
