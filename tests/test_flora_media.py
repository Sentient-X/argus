"""PNG compression changes transport bytes, never native decoded pixels."""
from io import BytesIO
import threading

import numpy as np
from PIL import Image
import pytest

from events.flora_media import lossless_png_serialization


@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_lossless_png_keeps_actual_decoded_pixels_and_restores_save(level, mode):
    shape = {"RGB": (96, 128, 3), "RGBA": (96, 128, 4), "L": (96, 128)}[mode]
    array = np.random.default_rng(7).integers(0, 256, shape, dtype=np.uint8)
    image = Image.fromarray(array)
    original = Image.Image.save
    baseline, accelerated = BytesIO(), BytesIO()
    image.save(baseline, format="PNG")
    with lossless_png_serialization(level) as metrics:
        image.save(accelerated, format="PNG")
    assert Image.Image.save is original
    assert metrics["png_saves"] == 1 and metrics["encode_wall_s"] >= 0
    for payload in (baseline.getvalue(), accelerated.getvalue()):
        with Image.open(BytesIO(payload)) as decoded:
            assert decoded.mode == mode and decoded.size == (128, 96)
            assert np.array_equal(np.asarray(decoded), array)
    assert np.array_equal(np.asarray(image), array)


def test_options_formats_and_other_threads_are_untouched(tmp_path):
    image = Image.new("RGB", (64, 48), (20, 40, 60))
    baseline = {}
    cases = [("JPEG", {"quality": 88}), ("PNG", {"compress_level": 9}),
             ("PNG", {"optimize": True}), ("PNG", {"dpi": (72, 72)})]
    for index, (fmt, options) in enumerate(cases):
        output = BytesIO()
        image.save(output, format=fmt, **options)
        baseline[index] = output.getvalue()
    plain = BytesIO()
    image.save(plain, format="PNG")
    with lossless_png_serialization() as metrics:
        for index, (fmt, options) in enumerate(cases):
            output = BytesIO()
            image.save(output, format=fmt, **options)
            assert output.getvalue() == baseline[index]
        image.save(tmp_path / "inferred.png")
        assert (tmp_path / "inferred.png").read_bytes() == plain.getvalue()
        image.encoderinfo = {"compress_level": 9}
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        assert encoded.getvalue() == baseline[1]
        image.encoderinfo = {}
        other = BytesIO()
        thread = threading.Thread(target=lambda: image.save(other, format="PNG"))
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert other.getvalue() == plain.getvalue()
    assert metrics["png_saves"] == 0


def test_original_save_restored_after_encoder_failure(tmp_path):
    original = Image.Image.save
    with pytest.raises(FileNotFoundError):
        with lossless_png_serialization():
            Image.new("RGB", (8, 8)).save(tmp_path / "missing/frame.png", format="PNG")
    assert Image.Image.save is original


@pytest.mark.parametrize("level", [-1, 2, 9, True, 0.0])
def test_unknown_compression_is_refused_before_patching(level):
    original = Image.Image.save
    with pytest.raises(ValueError, match="level 0 or 1"):
        with lossless_png_serialization(level):
            pytest.fail("invalid level admitted")
    assert Image.Image.save is original
