"""Resident lifecycle and durable boundaries; not GPU/model qualification."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from events import flora_vllm as flora
from events.flora_journal import GenerationJournal


@pytest.fixture
def resident_work(tmp_path, monkeypatch):
    root = tmp_path / "captioner"
    root.mkdir()
    monkeypatch.setenv("FLORA_MODEL_CACHE", str(tmp_path))
    weights = root / "weights.safetensors"
    weights.write_bytes(b"verified weights")
    (root / "SOURCE_MANIFEST.sha256").write_text(f"{flora._file_sha(weights)}  {weights.name}\n")
    monkeypatch.setattr(flora, "_processor_binding", lambda root: {"runtime": "pinned"})
    tokenizer = SimpleNamespace(eos_token_id=2, decode=lambda ids, **kwargs: "Move the cup.")
    processor = SimpleNamespace(tokenizer=tokenizer)
    prepared = [{"request": {"prompt": "video", "multi_modal_data": {
        "video": [(np.zeros((2, 2, 2, 3), dtype=np.uint8), {"fps": 8})]}},
        "prompt_token_ids": [5, 6], "max_tokens": 512}]
    monkeypatch.setattr(flora, "prepare", lambda request: (
        root, processor, prepared, {"id": request["id"], "segments": [
            {"start_s": 0, "end_s": 1, "status": "complete"}]},
    ))
    monkeypatch.setattr(flora, "check_processor_parity", lambda root, tokenizer, items: {
        "status": "PASS_NATIVE_PROCESSOR_PARITY", "requests": len(items),
        "checked_precisions": ["float32", "bfloat16"], "elapsed_s": 0,
        "binding": flora._parity_binding(root, items),
    })
    calls = {"engine": 0, "processor": 0, "generate": 0, "checkpoint_hash": 0}
    verify = flora._verify_model_files

    def verify_checkpoint(*args):
        calls["checkpoint_hash"] += 1
        return verify(*args)

    monkeypatch.setattr(flora, "_verify_model_files", verify_checkpoint)

    def load_processor(*args, **kwargs):
        calls["processor"] += 1
        return processor

    def generate(requests, params, **kwargs):
        calls["generate"] += 1
        return [SimpleNamespace(prompt_token_ids=[5, 6], outputs=[SimpleNamespace(
            token_ids=[1, 2], text="Move the cup.", finish_reason="stop", stop_reason=2,
        )]) for _ in requests]

    def load_engine(path, max_num_seqs):
        calls["engine"] += 1
        return SimpleNamespace(model_config=SimpleNamespace(model=str(path)), generate=generate,
                               collective_rpc=lambda method: [{"verified": True}])

    monkeypatch.setattr(flora, "load_engine", load_engine)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=load_processor)))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))

    def cache(name):
        requests = [{"id": name, "models": {"captioner": {"revision": "pinned"}}}]
        directory = tmp_path / name
        flora.prepare_many_cache(requests, directory)
        return requests, directory

    return tmp_path, calls, cache


def test_two_admitted_chunks_share_one_engine_processor_and_full_weight_verification(resident_work,
                                                                                   monkeypatch):
    root, calls, cache = resident_work
    first, second = cache("first"), cache("second")
    # CPU certificates already exist. Any GPU-side preprocessing rerun is a defect.
    monkeypatch.setattr(flora, "check_processor_parity", lambda *args, **kwargs: pytest.fail("reprocessed"))
    session = flora.ResidentCaptionSession(max_chunks=2)
    for index, (requests, directory) in enumerate((first, second)):
        journal = GenerationJournal(root / f"journal{index}", f"chunk{index}")
        result = flora.caption_many(requests, root / f"out{index}", journal,
                                    prepared_cache=directory, session=session)
        assert result[0]["timing"]["resident_engine_reused"] is bool(index)
        assert result[0]["timing"]["processor_parity_reused"] is True
        assert (journal.root / "000000.intent.json").exists()
        saved = json.loads((journal.root / "000000.result.json").read_text())
        assert saved["value"]["completions"][0]["token_ids"] == [1, 2]
    assert calls == {"engine": 1, "processor": 1, "generate": 2, "checkpoint_hash": 1}
    with pytest.raises(ValueError, match="chunk limit"):
        flora.caption_many(second[0], root / "excess", GenerationJournal(root / "excessj", "x"),
                           prepared_cache=second[1], session=session)


def test_invalid_certificate_fails_without_poisoning_resident_session(resident_work):
    root, calls, cache = resident_work
    first, invalid, valid = cache("first"), cache("invalid"), cache("valid")
    session = flora.ResidentCaptionSession(max_chunks=2)
    flora.caption_many(first[0], root / "out1", GenerationJournal(root / "j1", "a"),
                       prepared_cache=first[1], session=session)
    manifest = invalid[1] / "cache.json"
    value = json.loads(manifest.read_text())
    value["episodes"][0]["processor_parity"]["binding"]["prepared_sha256"] = "changed"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="parity certificate"):
        flora.caption_many(invalid[0], root / "badout", GenerationJournal(root / "badj", "b"),
                           prepared_cache=invalid[1], session=session)
    assert calls["generate"] == 1 and not (root / "badj" / "000000.intent.json").exists()
    flora.caption_many(valid[0], root / "out2", GenerationJournal(root / "j2", "c"),
                       prepared_cache=valid[1], session=session)
    assert calls["engine"] == 1 and calls["generate"] == 2


def test_resident_rejects_runtime_change_and_preserves_completed_reply(resident_work, monkeypatch):
    root, calls, cache = resident_work
    first, second = cache("first"), cache("second")
    session = flora.ResidentCaptionSession(max_chunks=2)
    flora.caption_many(first[0], root / "out", GenerationJournal(root / "journal", "a"),
                       prepared_cache=first[1], session=session)
    paid = (root / "journal/000000.result.json").read_bytes()
    monkeypatch.setattr(flora, "_processor_binding", lambda root: {"runtime": "changed"})
    with pytest.raises(ValueError, match="binding changed"):
        flora.caption_many(second[0], root / "out2", GenerationJournal(root / "j2", "b"),
                           prepared_cache=second[1], session=session)
    assert calls["engine"] == calls["generate"] == 1
    assert (root / "journal/000000.result.json").read_bytes() == paid


def test_cpu_cache_reuse_rejects_changed_arrays_and_missing_weights_without_full_scan(
        resident_work, monkeypatch):
    from events import flora_hosted
    root, calls, cache = resident_work
    requests, directory = cache('prepared/batch')
    monkeypatch.setattr(flora_hosted, 'EVIDENCE_ROOT', root)
    descriptor = {'path': str(directory / 'cache.json'),
                  'sha256': flora._file_sha(directory / 'cache.json')}
    before = calls['checkpoint_hash']
    flora_hosted.verify_prepared_cache(requests, descriptor)
    assert calls['checkpoint_hash'] == before
    with pytest.raises(ValueError, match='request or implementation'):
        flora_hosted.verify_prepared_cache([{**requests[0], 'id': 'changed'}], descriptor)
    array = next(directory.glob('*.npy'))
    original = array.read_bytes()
    array.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='array hash'):
        flora_hosted.verify_prepared_cache(requests, descriptor)
    array.write_bytes(original)
    weights = root / 'captioner/weights.safetensors'
    weights.rename(weights.with_suffix('.saved'))
    with pytest.raises((ValueError, FileNotFoundError)):
        flora_hosted.verify_prepared_cache(requests, descriptor)
    assert calls['checkpoint_hash'] == before


def _chunks(root, caches):
    return [flora.CaptionChunk(requests, root / f"out{index}",
                               GenerationJournal(root / f"journal{index}", f"batch{index}"), directory)
            for index, (requests, directory) in enumerate(caches)]


def test_stream_prepares_next_episode_during_generation_with_only_one_lookahead(
        resident_work, monkeypatch):
    import threading

    root, calls, cache = resident_work
    chunks = _chunks(root, [cache(name) for name in ("first", "second", "third")])
    generating, next_prepared = threading.Event(), threading.Event()
    original_load, original_engine = flora._load_prepared_cache, flora.load_engine
    loaded = []

    def load(requests, *args, **kwargs):
        if requests[0]["id"] == "second":
            assert generating.wait(timeout=5), "lookahead must run alongside generation"
        result = original_load(requests, *args, **kwargs)
        loaded.append(requests[0]["id"])
        if requests[0]["id"] == "second":
            next_prepared.set()
        return result

    def engine(*args):
        llm = original_engine(*args)
        original_generate = llm.generate

        def generate(*args, **kwargs):
            if calls["generate"] == 0:
                generating.set()
                assert next_prepared.wait(timeout=5), "GPU must not block CPU lookahead"
            return original_generate(*args, **kwargs)
        llm.generate = generate
        return llm

    monkeypatch.setattr(flora, "_load_prepared_cache", load)
    monkeypatch.setattr(flora, "load_engine", engine)
    session = flora.ResidentCaptionSession(max_chunks=3)
    stream = flora.caption_stream(chunks, session)
    first = next(stream)
    assert first[0]["segments"][0]["caption"] == "Move the cup."
    assert loaded == ["first", "second"]  # Third cannot allocate ahead of the bound.
    rest = list(stream)
    assert len(rest) == 2
    assert calls == {"engine": 1, "processor": 1, "generate": 3, "checkpoint_hash": 1}
    assert all(receipt[0]["timing"]["preparation_lookahead"] for receipt in (first, *rest))
    assert all(receipt[0]["timing"]["preparation_wait_wall_s"] >= 0 for receipt in (first, *rest))
    assert not session._lock.locked()


def test_invalid_lookahead_preserves_prior_paid_output_and_never_generates_it(resident_work):
    root, calls, cache = resident_work
    first, second = cache("first"), cache("second")
    manifest = second[1] / "cache.json"
    record = json.loads(manifest.read_text())
    record["episodes"][0]["processor_parity"]["binding"]["prepared_sha256"] = "changed"
    manifest.write_text(json.dumps(record))
    chunks = _chunks(root, [first, second])
    session = flora.ResidentCaptionSession(max_chunks=2)
    stream = flora.caption_stream(chunks, session)
    assert next(stream)[0]["segments"][0]["caption"] == "Move the cup."
    paid = (chunks[0].journal.root / "000000.result.json").read_bytes()
    raw = (chunks[0].output_dir / "first.caption.raw.json").read_bytes()
    with pytest.raises(ValueError, match="parity certificate"):
        next(stream)
    assert calls["generate"] == 1
    assert (chunks[0].journal.root / "000000.result.json").read_bytes() == paid
    assert (chunks[0].output_dir / "first.caption.raw.json").read_bytes() == raw
    assert not (chunks[1].journal.root / "000000.intent.json").exists()
    assert not session._lock.locked()


def test_stream_refuses_oversize_array_cache_before_engine_or_allocation(resident_work, monkeypatch):
    root, calls, cache = resident_work
    chunks = _chunks(root, [cache("first")])
    monkeypatch.setattr(flora, "MAX_PREPARED_ARRAY_BYTES", 1)
    with pytest.raises(ValueError, match="memory limit"):
        list(flora.caption_stream(chunks, flora.ResidentCaptionSession()))
    assert calls["engine"] == calls["generate"] == 0
    assert not (chunks[0].journal.root / "000000.intent.json").exists()
