"""Bounded, read-only warming of admitted Flora caches on remote volumes.

Four readers overlap object fetch latency. This is never admission: the normal
loader still verifies every byte, checkpoint and native processor certificate.
A failed future cache is recorded without interrupting a paid current batch.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import stat
import threading
import time

MAX_ARRAY_BYTES = 4 * 1024 ** 3
MAX_MANIFEST_BYTES = 64 * 1024 ** 2
_BLOCK_BYTES = 256 * 1024


@dataclass(frozen=True)
class _Read:
    batch: int
    path: Path
    sha256: str
    size: int


def _cache_files(descriptor: dict, allowed_root: Path, index: int) -> list[_Read]:
    manifest = Path(descriptor["path"])
    if (manifest.name != "cache.json" or manifest.is_symlink()
            or not manifest.resolve().is_relative_to(allowed_root.resolve())):
        raise ValueError("Prefetch cache is outside admitted evidence")
    if manifest.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Prefetch manifest exceeds the memory limit")
    with manifest.open("rb") as source:
        raw = source.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
        raise ValueError("Prefetch manifest differs from admitted evidence")
    files = json.loads(raw)["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Prefetch cache has no array files")
    result, total = [], 0
    for name, digest in files.items():
        path = manifest.parent / name
        if (Path(name).name != name or not name.endswith(".npy") or path.is_symlink()
                or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError("Invalid prefetch array identity")
        stamp = path.stat()
        if not stat.S_ISREG(stamp.st_mode):
            raise ValueError("Prefetch array is not a regular file")
        total += stamp.st_size
        if total > MAX_ARRAY_BYTES:
            raise ValueError("Prefetch arrays exceed the memory limit")
        result.append(_Read(index, path, digest, stamp.st_size))
    return result


def _verify_file(item: _Read, cancelled: threading.Event) -> bool:
    if cancelled.is_set():
        return False
    descriptor = os.open(item.path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        stamp = os.fstat(source.fileno())
        if not stat.S_ISREG(stamp.st_mode) or stamp.st_size != item.size:
            raise ValueError("Prefetch array changed before reading")
        digest, count = hashlib.sha256(), 0
        while not cancelled.is_set():
            block = source.read(_BLOCK_BYTES)
            if not block:
                if count != item.size or digest.hexdigest() != item.sha256:
                    raise ValueError("Prefetch array SHA differs from admitted evidence")
                return True
            count += len(block)
            if count > item.size:
                raise ValueError("Prefetch array changed while reading")
            digest.update(block)
    return False


class CaptionCachePrefetch:
    """Warm current + successor only; advance after the caller commits a batch."""

    def __init__(self, descriptors: list[dict], allowed_root: Path):
        if not 1 <= len(descriptors) <= 32:
            raise ValueError("Prefetch needs 1..32 bounded caches")
        self._descriptors, self._allowed_root = descriptors, allowed_root
        self._cancelled = threading.Event()
        self._condition = threading.Condition()
        self._work = queue.Queue(maxsize=4)
        self._finished = [threading.Event() for _ in descriptors]
        self._current = 0
        self._threads = []
        self._states = [{"batch_index": i, "status": "pending", "verified_files": 0,
                         "verified_bytes": 0, "error": None} for i in range(len(descriptors))]
        self._remaining = [0 for _ in descriptors]

    def __enter__(self):
        if self._threads:
            raise ValueError("Prefetch lifetime cannot be reused")
        self._threads = [threading.Thread(target=self._reader, name=f"flora-read-{i}", daemon=True)
                         for i in range(4)]
        self._threads.append(threading.Thread(target=self._coordinate, name="flora-read-plan", daemon=True))
        for thread in self._threads:
            thread.start()
        return self

    def completed(self, index: int) -> None:
        with self._condition:
            if index != self._current or index >= len(self._descriptors):
                raise ValueError("Prefetch advancement must follow committed batch order")
            self._current += 1
            self._condition.notify_all()

    def snapshot(self) -> dict:
        with self._condition:
            return {"streaming_readers": 4, "maximum_caches_ahead": 2,
                    "batches": [dict(state) for state in self._states]}

    def _coordinate(self):
        for index, descriptor in enumerate(self._descriptors):
            with self._condition:
                self._condition.wait_for(lambda: self._cancelled.is_set() or index <= self._current + 1)
            if self._cancelled.is_set():
                return
            started = time.monotonic()
            try:
                items = _cache_files(descriptor, self._allowed_root, index)
                with self._condition:
                    if self._cancelled.is_set():
                        return
                    self._remaining[index] = len(items)
                    self._states[index].update(status="warming", total_files=len(items))
                for item in items:
                    while not self._cancelled.is_set():
                        try:
                            self._work.put(item, timeout=0.1)
                            break
                        except queue.Full:
                            pass
                    if self._cancelled.is_set():
                        return
                while not self._finished[index].wait(0.1):
                    if self._cancelled.is_set():
                        return
            except Exception as error:
                with self._condition:
                    self._states[index].update(status="failed", error=str(error))
            finally:
                with self._condition:
                    self._states[index]["elapsed_s"] = time.monotonic() - started

    def _reader(self):
        while not self._cancelled.is_set():
            try:
                item = self._work.get(timeout=0.1)
            except queue.Empty:
                continue
            valid, error = False, None
            try:
                valid = _verify_file(item, self._cancelled)
            except Exception as exc:
                error = str(exc)
            finally:
                with self._condition:
                    state = self._states[item.batch]
                    if error is not None:
                        state.update(status="failed", error=error)
                    if valid:
                        state["verified_files"] += 1
                        state["verified_bytes"] += item.size
                    self._remaining[item.batch] -= 1
                    if not self._remaining[item.batch]:
                        if state["status"] == "warming":
                            state["status"] = "complete" if not self._cancelled.is_set() else "cancelled"
                        self._finished[item.batch].set()
                self._work.task_done()

    def __exit__(self, *exc):
        self._cancelled.set()
        with self._condition:
            self._condition.notify_all()
            for state in self._states:
                if state["status"] in {"pending", "warming"}:
                    state["status"] = "cancelled"
        # Storage reads can block in the filesystem. Daemon readers own no GPU,
        # writes or model calls; cancellation stops them between bounded reads.
        # Never hold a completed batch hostage to a stalled future cache fetch.
        deadline = time.monotonic() + 0.5
        for thread in self._threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
