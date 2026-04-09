from pathlib import Path

import pytest

from scarecrow.model import _verify_bundled_weights, load


class TestVerifyBundledWeights:
    def test_rejects_mismatched_bytes(self, tmp_path):
        """Mismatched file bytes raise RuntimeError."""
        bad = tmp_path / "weights.pt2"
        bad.write_bytes(b"not the real weights")
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            _verify_bundled_weights(str(bad))

    def test_bundled_file_matches_pinned_hash(self):
        """The committed bundled weights file matches BUNDLED_WEIGHTS_SHA256."""
        weights = Path(__file__).parent.parent / "license-plate-finetune-v1n.pt2"
        _verify_bundled_weights(str(weights))


class TestLoad:
    def test_raises_before_torch_export_load_for_bundled_name(self, tmp_path, monkeypatch):
        """Hash verification runs before torch.export.load for the bundled filename."""
        tmp_file = tmp_path / "license-plate-finetune-v1n.pt2"
        tmp_file.write_bytes(b"bogus")
        calls = []
        monkeypatch.setattr("torch.export.load", lambda *a, **k: calls.append(a))
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            load(str(tmp_file))
        assert calls == []

    def test_skips_verification_for_custom_filename(self, tmp_path, monkeypatch):
        """Non-bundled filenames skip verification and reach torch.export.load."""
        tmp_file = tmp_path / "custom.pt2"
        tmp_file.write_bytes(b"arbitrary")

        class Marker(Exception):
            pass

        def fake_load(*args, **kwargs):
            raise Marker

        monkeypatch.setattr("torch.export.load", fake_load)
        with pytest.raises(Marker):
            load(str(tmp_file), device="cuda")
