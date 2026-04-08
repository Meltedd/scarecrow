import numpy as np
import torch

from scarecrow.io import load_pattern, save_pattern
from scarecrow.model import _nms, letterbox
from scarecrow.optimize import (
    PATTERN_H,
    PATTERN_W,
    PlateData,
    _composite_letterbox,
    composite,
    eot_transform,
)


class TestComposite:
    def test_blending(self):
        images = torch.full((1, 3, 8, 8), 0.5)
        masks = torch.zeros(1, 1, 8, 8)
        masks[:, :, 2:6, 2:6] = 1.0
        pattern = torch.ones(1, 1, 8, 8)
        result = composite(images, masks, pattern)
        assert (result[0, :, 3, 3] == 1.0).all()
        assert (result[0, :, 0, 0] == 0.5).all()

    def test_gradient_masked_only(self):
        """Gradients flow only through masked pixels."""
        images = torch.rand(1, 3, 8, 8)
        masks = torch.zeros(1, 1, 8, 8)
        masks[:, :, 2:6, 2:6] = 1.0
        pattern = torch.rand(1, 1, 8, 8, requires_grad=True)
        composite(images, masks, pattern).sum().backward()
        grad = pattern.grad[0, 0]
        assert grad[0, 0] == 0.0
        assert grad[3, 3] != 0.0


class TestCompositeLetterbox:
    def test_output_shape(self):
        items = [PlateData(
            image=torch.rand(3, 480, 640),
            mask=torch.zeros(1, 480, 640),
        )]
        pattern = torch.full((1, 1, PATTERN_H, PATTERN_W), 0.5)
        result = _composite_letterbox(pattern, items, 640)
        assert result.shape == (1, 3, 640, 640)

    def test_differentiable(self):
        items = [PlateData(
            image=torch.rand(3, 64, 64),
            mask=torch.ones(1, 64, 64),
        )]
        pattern = torch.full((1, 1, PATTERN_H, PATTERN_W), 0.5, requires_grad=True)
        _composite_letterbox(pattern, items, 64).sum().backward()
        assert pattern.grad is not None
        assert pattern.grad.abs().sum() > 0


class TestEoT:
    def test_preserves_shape(self):
        rng = torch.Generator().manual_seed(0)
        images = torch.rand(2, 3, 64, 64)
        assert eot_transform(images, rng).shape == (2, 3, 64, 64)

    def test_output_clamped(self):
        rng = torch.Generator().manual_seed(0)
        out = eot_transform(torch.rand(2, 3, 64, 64), rng)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_modifies_input(self):
        rng = torch.Generator().manual_seed(42)
        images = torch.rand(2, 3, 64, 64)
        out = eot_transform(images, rng)
        assert not torch.equal(out, images)

    def test_differentiable(self):
        rng = torch.Generator().manual_seed(0)
        images = torch.rand(1, 3, 32, 32, requires_grad=True)
        eot_transform(images, rng).sum().backward()
        assert images.grad is not None
        assert images.grad.abs().sum() > 0

    def test_perspective_keystone(self):
        """Perspective homography produces non-uniform scaling (keystone)."""
        py = 0.1
        M = torch.eye(3).unsqueeze(0)
        M[0, 2, 1] = py
        coords = torch.tensor([[[0.0, 0.0], [-1.0, 1.0], [1.0, 1.0]]])
        result = torch.bmm(M, coords)
        src = result[:, :2] / result[:, 2:3]
        top_y = src[0, 1, 0].item()  # output y=-1 (top)
        bot_y = src[0, 1, 1].item()  # output y=+1 (bottom)
        # Top maps farther from center than bottom (asymmetric)
        assert abs(top_y) > abs(bot_y)
        assert abs(top_y - (-1 / 0.9)) < 0.001
        assert abs(bot_y - (1 / 1.1)) < 0.001


class TestLetterbox:
    def test_square_passthrough(self):
        images = torch.rand(2, 3, 64, 64)
        assert letterbox(images, 64) is images

    def test_pads_nonsquare(self):
        images = torch.rand(1, 3, 32, 64)
        out = letterbox(images, 64)
        assert out.shape == (1, 3, 64, 64)
        assert torch.isclose(out[0, 0, 0, 0], torch.tensor(114 / 255), atol=1e-5)


class TestNMS:
    def test_suppresses_overlapping(self):
        boxes = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # high overlap with first
            [50.0, 50.0, 60.0, 60.0],  # no overlap
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = _nms(boxes, scores, iou_thresh=0.5)
        assert len(keep) == 2
        assert 0 in keep.tolist()
        assert 2 in keep.tolist()

    def test_keeps_all_non_overlapping(self):
        boxes = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ])
        scores = torch.tensor([0.9, 0.8])
        keep = _nms(boxes, scores, iou_thresh=0.5)
        assert len(keep) == 2

    def test_single_box(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        scores = torch.tensor([0.9])
        keep = _nms(boxes, scores, iou_thresh=0.5)
        assert len(keep) == 1


class TestPatternRoundTrip:
    def test_save_load(self, tmp_path):
        pattern = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
        path = tmp_path / "pattern.png"
        save_pattern(pattern, path)
        loaded = load_pattern(path)
        np.testing.assert_allclose(loaded, pattern, atol=1 / 255 + 1e-6)
