"""Durable generation receipts. An unsettled intent never authorizes a retry."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import sys


def _normalized(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Generation input mappings require string keys")
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"tensor": str(tensor.dtype), "shape": list(tensor.shape),
                "sha256": hashlib.sha256(raw).hexdigest()}
    numpy = sys.modules.get("numpy")
    if numpy is not None and isinstance(value, numpy.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot bind generation inputs")
        return {"array": value.dtype.str, "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    if numpy is not None and isinstance(value, numpy.generic):
        return _normalized(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return _normalized(asdict(value))
    raise TypeError(f"Unsupported generation input type: {type(value).__name__}")


def input_digest(value) -> str:
    encoded = json.dumps(_normalized(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _save(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


class GenerationJournal:
    def __init__(self, root: Path, request_sha256: str, durable_commit=lambda: None):
        self.root, self.request_sha256, self.index = root, request_sha256, 0
        self.durable_commit = durable_commit
        root.mkdir(parents=True, exist_ok=True)
        _sync_directory(root.parent)

    def run(self, inputs, effect, encode=lambda value: value, decode=lambda value: value):
        index = self.index
        self.index += 1
        binding = {"request_sha256": self.request_sha256, "input_sha256": input_digest(inputs)}
        intent = self.root / f"{index:06d}.intent.json"
        result = self.root / f"{index:06d}.result.json"
        if intent.exists():
            if json.loads(intent.read_text()) != binding:
                raise RuntimeError(f"Generation {index} input or request changed; replay refused")
            if not result.exists():
                raise RuntimeError(f"Generation {index} has an unsettled intent; automatic retry refused")
            stored = json.loads(result.read_text())
            if stored["binding"] != binding:
                raise RuntimeError(f"Generation {index} result binding differs")
            return decode(stored["value"])
        if result.exists():
            raise RuntimeError(f"Generation {index} result has no intent")
        _save(intent, binding)
        self.durable_commit()
        value = effect()
        _save(result, {"binding": binding, "value": encode(value)})
        self.durable_commit()
        return value
