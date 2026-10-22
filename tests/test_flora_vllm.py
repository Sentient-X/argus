"""CPU checks of caption acceptance boundaries, not GPU/model qualification."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from events.flora_journal import GenerationJournal
from events.flora_vllm import _decode, _vllm_prompt, caption_many


class Tokens:
    eos_token_id = 2

    def decode(self, ids, *, skip_special_tokens):
        return "".join({1: "Move the cup.", 2: "", 3: " "}[token] for token in ids)


def test_caption_requires_native_eos_and_nonempty_instruction():
    assert _decode({"token_ids": [1, 2], "finish_reason": "stop"}, Tokens()) == "Move the cup."
    for completion in (
        {"token_ids": [1], "finish_reason": "length"},
        {"token_ids": [1], "finish_reason": "stop"},
        {"token_ids": [3, 2], "finish_reason": "stop"},
        {"token_ids": [], "finish_reason": "stop"},
    ):
        with pytest.raises(ValueError, match="complete native caption"):
            _decode(completion, Tokens())


def test_caption_batch_refuses_duplicate_or_escaping_outputs_before_loading(tmp_path: Path):
    journal = GenerationJournal(tmp_path / "journal", "request")
    for requests in ([{"id": "episode"}, {"id": "episode"}], [{"id": "../escape"}]):
        with pytest.raises(ValueError, match="unique local filenames"):
            caption_many(requests, tmp_path / "outputs", journal)


def test_caption_batch_preserves_completed_outputs_before_loading(tmp_path: Path):
    output = tmp_path / "episode.caption.raw.json"
    output.write_text("retained paid reply")
    with pytest.raises(ValueError, match="already exist"):
        caption_many([{"id": "episode"}], tmp_path,
                     GenerationJournal(tmp_path / "journal", "request"))
    assert output.read_text() == "retained paid reply"


def test_vllm_transport_wraps_bare_native_video_markers_without_changing_text():
    native = "Before:\n<|video_pad|>Target:\n<|video_pad|>Move the cup."
    expected = ("Before:\n<|vision_start|><|video_pad|><|vision_end|>Target:\n"
                "<|vision_start|><|video_pad|><|vision_end|>Move the cup.")
    assert _vllm_prompt(native, "<|video_pad|>", 2) == expected
    for text, count in ((native, 1), (expected, 2)):
        with pytest.raises(ValueError, match="placeholder contract"):
            _vllm_prompt(text, "<|video_pad|>", count)


@pytest.fixture
def preparation_cache(tmp_path, monkeypatch):
    import numpy as np
    from events import flora_vllm

    root = tmp_path / "captioner"
    root.mkdir()
    monkeypatch.setenv("FLORA_MODEL_CACHE", str(tmp_path))
    weights = root / "model-00001.safetensors"
    weights.write_bytes(b"pinned test weights")
    (root / "SOURCE_MANIFEST.sha256").write_text(
        f"{flora_vllm._file_sha(weights)}  {weights.name}\n")
    frames = np.arange(48, dtype=np.uint8).reshape(2, 2, 4, 3)
    prepared = [{"request": {"multi_modal_data": {"video": [
        (frames, {"fps": 8.0, "frames_indices": [0, 1]})
    ]}}, "prompt_token_ids": [1, 2], "max_tokens": 512}]
    requests = [{"id": "episode", "models": {"captioner": {"revision": "pinned"}}}]
    monkeypatch.setattr(flora_vllm, "prepare", lambda request: (
        root, SimpleNamespace(tokenizer=object()), prepared,
        {"id": request["id"], "segments": [{"status": "complete"}]},
    ))
    monkeypatch.setattr(flora_vllm, "_processor_binding", lambda root: {"test_runtime": "pinned"})
    monkeypatch.setattr(flora_vllm, "check_processor_parity", lambda root, tokenizer, items: {
        "status": "PASS_NATIVE_PROCESSOR_PARITY", "requests": len(items), "elapsed_s": 0.0,
        "checked_precisions": ["float32", "bfloat16"],
        "binding": flora_vllm._parity_binding(root, items),
    })
    directory = tmp_path / "prepared"
    descriptor = flora_vllm.prepare_many_cache(requests, directory)
    return flora_vllm, requests, directory, descriptor, prepared


def test_preparation_cache_preserves_exact_native_arrays_and_container_types(preparation_cache):
    import numpy as np
    from events.flora_journal import input_digest

    module, requests, directory, descriptor, original = preparation_cache
    cached = module._load_prepared_cache(requests, directory)
    assert descriptor["sha256"] == module._file_sha(directory / "cache.json")
    assert descriptor["episodes"] == descriptor["intervals"] == 1
    assert input_digest(cached[0][1]) == input_digest(original)
    video = cached[0][1][0]["request"]["multi_modal_data"]["video"]
    assert isinstance(video, list) and isinstance(video[0], tuple)
    assert video[0][0].dtype == np.uint8
    assert video[0][0].shape == (2, 2, 4, 3)
    with pytest.raises(FileExistsError):
        module.prepare_many_cache(requests, directory)


def test_preparation_cache_rejects_array_tampering_before_model_load(preparation_cache):
    module, requests, directory, descriptor, _ = preparation_cache
    array = directory / next(iter(descriptor["files"]))
    with array.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="array hash differs"):
        module._load_prepared_cache(requests, directory)


def test_preparation_cache_rejects_request_and_checkpoint_changes(preparation_cache):
    module, requests, directory, _, _ = preparation_cache
    with pytest.raises(ValueError, match="request or implementation binding"):
        module._load_prepared_cache([{**requests[0], "id": "different"}], directory)
    (directory.parent / "captioner" / "SOURCE_MANIFEST.sha256").write_text("changed")
    with pytest.raises(ValueError, match="model manifest differs"):
        module._load_prepared_cache(requests, directory)


def test_preparation_cache_rejects_changed_weights_without_rehashing(preparation_cache):
    import os

    module, requests, directory, _, _ = preparation_cache
    weights = directory.parent / "captioner" / "model-00001.safetensors"
    stamp = weights.stat()
    # Same length still fails because the saved modification time has changed.
    weights.write_bytes(b"mutant test weights")
    os.utime(weights, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000))
    with pytest.raises(ValueError, match="checkpoint files changed"):
        module._load_prepared_cache(requests, directory)


def test_preparation_cache_rejects_different_model_cache_root(preparation_cache, monkeypatch):
    module, requests, directory, _, _ = preparation_cache
    monkeypatch.setenv("FLORA_MODEL_CACHE", str(directory / "other-cache"))
    with pytest.raises(ValueError, match="checkpoint root differs"):
        module._load_prepared_cache(requests, directory)


def test_preparation_cache_rehashes_checkpoint_even_when_stat_is_unchanged(preparation_cache):
    import os

    module, requests, directory, _, _ = preparation_cache
    weights = directory.parent / "captioner" / "model-00001.safetensors"
    stamp = weights.stat()
    weights.write_bytes(b"mutant test weights")
    os.utime(weights, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert weights.stat().st_size == stamp.st_size
    with pytest.raises(ValueError, match="checkpoint content hash differs"):
        module._load_prepared_cache(requests, directory)
