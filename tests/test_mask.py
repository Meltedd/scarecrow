from scarecrow.mask import OVERLAP, THICKNESS, frame_mask


class TestFrameMask:
    def test_interior_clear(self):
        """Plate interior inside the overlap margin is fully clear."""
        bbox = (100, 50, 200, 100)
        x, y, w, h = bbox
        o = max(1, int(h * OVERLAP))
        m = frame_mask((300, 500), bbox)
        assert m[y + o : y + h - o, x + o : x + w - o].sum() == 0

    def test_frame_filled(self):
        """Frame region between outer edge and inner cutout is set."""
        bbox = (100, 50, 200, 100)
        x, y, w, h = bbox
        t = max(1, int(w * THICKNESS))
        m = frame_mask((300, 500), bbox)
        assert m[y - 1, x + w // 2] == 1
        assert m[y - t, x + w // 2] == 1

    def test_outside_clear(self):
        """Pixels beyond the outer frame edge are clear."""
        bbox = (100, 50, 200, 100)
        x, y, w, h = bbox
        t = max(1, int(w * THICKNESS))
        m = frame_mask((300, 500), bbox)
        assert m[y - t - 1, x + w // 2] == 0

    def test_small_bbox(self):
        """Small bboxes clamp thickness and overlap to 1px minimum."""
        m = frame_mask((50, 50), bbox=(20, 15, 10, 10))
        assert m.any()
        assert m[20, 25] == 0

    def test_bbox_at_edge(self):
        """Bbox touching image edge clips frame without crashing."""
        m = frame_mask((100, 100), bbox=(0, 0, 50, 50))
        assert m.any()
        assert m.shape == (100, 100)
