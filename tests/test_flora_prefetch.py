"""Remote-cache warming is bounded and cannot replace loader admission."""
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from events import flora_prefetch as prefetch


def cache(root, name, count=8):
    directory = root / name
    directory.mkdir(parents=True)
    files = {}
    for index in range(count):
        path = directory / f"{index:06d}.npy"
        np.save(path, np.full((4, 4, 3), index, dtype=np.uint8), allow_pickle=False)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = directory / "cache.json"
    manifest.write_text(json.dumps({"files": files}))
    return {"path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("prefetch condition did not complete")
        time.sleep(0.005)


def test_four_readers_overlap_and_horizon_waits_for_committed_batch(tmp_path, monkeypatch):
    descriptors = [cache(tmp_path, str(i)) for i in range(3)]
    entered, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    active = maximum = 0
    original = prefetch._verify_file

    def verify(item, cancelled):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            if active == 4:
                entered.set()
        try:
            assert release.wait(5), "all four readers must be dispatched concurrently"
            return original(item, cancelled)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(prefetch, "_verify_file", verify)
    with prefetch.CaptionCachePrefetch(descriptors, tmp_path) as warmer:
        assert entered.wait(5)
        release.set()
        wait_for(lambda: warmer.snapshot()["batches"][1]["status"] == "complete")
        assert maximum == 4
        assert warmer.snapshot()["batches"][2]["status"] == "pending"
        warmer.completed(0)
        wait_for(lambda: warmer.snapshot()["batches"][2]["status"] == "complete")
        assert all(row["verified_files"] == 8 for row in warmer.snapshot()["batches"])
        with pytest.raises(ValueError, match="committed batch order"):
            warmer.completed(0)
    for descriptor in descriptors:
        manifest = Path(descriptor["path"])
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == descriptor["sha256"]
        for name, digest in json.loads(manifest.read_text())["files"].items():
            assert hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest() == digest


def test_bad_future_is_reported_without_raising_or_losing_current_result(tmp_path):
    descriptors = [cache(tmp_path, str(i)) for i in range(2)]
    (tmp_path / "1/000000.npy").write_bytes(b"changed")
    paid = tmp_path / "paid-result.json"
    original = b'{"caption":"Move the cup."}'
    paid.write_bytes(original)
    with prefetch.CaptionCachePrefetch(descriptors, tmp_path) as warmer:
        wait_for(lambda: warmer.snapshot()["batches"][1]["status"] == "failed")
        report = warmer.snapshot()["batches"]
        assert report[0]["status"] == "complete"
        assert "SHA differs" in report[1]["error"]
        warmer.completed(0)
    assert paid.read_bytes() == original


@pytest.mark.parametrize("failure", ["manifest", "escape", "symlink", "oversize"])
def test_prefetch_rejects_invalid_evidence_before_streaming(tmp_path, monkeypatch, failure):
    descriptor = cache(tmp_path / "prepared", "batch", count=1)
    path = Path(descriptor["path"])
    record = json.loads(path.read_text())
    if failure == "manifest":
        descriptor["sha256"] = "0" * 64
    elif failure == "escape":
        record["files"] = {"../outside.npy": "0" * 64}
        path.write_text(json.dumps(record))
        descriptor["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif failure == "symlink":
        array = path.parent / "000000.npy"
        target = tmp_path / "outside.npy"
        target.write_bytes(array.read_bytes())
        array.unlink()
        array.symlink_to(target)
    else:
        monkeypatch.setattr(prefetch, "MAX_ARRAY_BYTES", 1)
    monkeypatch.setattr(prefetch, "_verify_file", lambda *args: pytest.fail("invalid file was streamed"))
    with prefetch.CaptionCachePrefetch([descriptor], tmp_path / "prepared") as warmer:
        wait_for(lambda: warmer.snapshot()["batches"][0]["status"] == "failed")
        assert warmer.snapshot()["batches"][0]["verified_files"] == 0


def test_cancellation_does_not_hold_completed_work_hostage_to_storage(tmp_path, monkeypatch):
    descriptors = [cache(tmp_path, "batch", count=1)]
    entered, release = threading.Event(), threading.Event()

    def blocked(item, cancelled):
        entered.set()
        release.wait(5)
        return not cancelled.is_set()

    monkeypatch.setattr(prefetch, "_verify_file", blocked)
    warmer = prefetch.CaptionCachePrefetch(descriptors, tmp_path)
    warmer.__enter__()
    assert entered.wait(5)
    started = time.monotonic()
    try:
        warmer.__exit__(None, None, None)
        assert time.monotonic() - started < 1
        assert warmer.snapshot()["batches"][0]["status"] == "cancelled"
    finally:
        release.set()
        for thread in warmer._threads:
            thread.join(timeout=1)
    assert not any(thread.is_alive() for thread in warmer._threads)
