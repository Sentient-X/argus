"""Scoped lossless serialization for Flora's temporary native RGB PNG frames.

This changes PNG container compression only. It does not resize, convert or
sample images. Callers must retain decoded-input parity checks: compressed file
hashes intentionally change even though every decoded pixel remains identical.
"""
from contextlib import contextmanager
import threading
import time

_PNG_SAVE_LOCK = threading.RLock()


@contextmanager
def lossless_png_serialization(compress_level: int = 0):
    """Accelerate bare explicit PNG saves on this thread during CPU admission.

    Only Flora's ``image.save(path, format='PNG')`` call shape is intercepted.
    Explicit encoder options, preexisting encoderinfo, inferred formats and
    other threads are untouched. The original method is restored on all exits.
    """
    if type(compress_level) is not int or compress_level not in (0, 1):
        raise ValueError("Lossless PNG acceleration requires compression level 0 or 1")
    from PIL import Image

    metrics = {"compress_level": compress_level, "png_saves": 0, "encode_wall_s": 0.0}
    with _PNG_SAVE_LOCK:
        original = Image.Image.save
        owner = threading.get_ident()

        def save(image, fp, format=None, **params):
            accelerate = (threading.get_ident() == owner and format == "PNG" and not params
                          and not getattr(image, "encoderinfo", {}))
            if not accelerate:
                return original(image, fp, format=format, **params)
            started = time.monotonic()
            result = original(image, fp, format=format, compress_level=compress_level)
            metrics["png_saves"] += 1
            metrics["encode_wall_s"] += time.monotonic() - started
            return result

        Image.Image.save = save
        try:
            yield metrics
        finally:
            Image.Image.save = original
