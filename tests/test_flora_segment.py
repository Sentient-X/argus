"""TempSeg admission and provenance boundaries; no substitute GPU model."""
import copy
from fractions import Fraction
import json

import pytest

from events import flora
from events import flora_segment as segment
from events.flora_journal import GenerationJournal


def test_native_intervals_preserve_concurrent_events_and_provenance():
    rows = [{"start_s": 0, "end_s": 3, "metadata": {"segmentation_provenance": "fine"}},
            {"start_s": 1, "end_s": 4, "metadata": {"windows": frozenset({2, 1})}}]
    original = copy.deepcopy(rows)
    segment._validate_segments(rows, 4)
    assert rows == original
    retained = json.loads(json.dumps(segment._json_value(rows), allow_nan=False))
    assert retained[1]["start_s"] == 1
    assert retained[1]["metadata"]["windows"] == [1, 2]
    assert "label" not in retained[0]
    assert segment._json_value(Fraction(1, 3)) == {"numerator": 1, "denominator": 3}


@pytest.mark.parametrize("start,end", [(True, 2), (float("nan"), 2), (0, float("inf")),
                                       (-1, 2), (1, 1), (0, 5)])
def test_native_interval_rejection(start, end):
    with pytest.raises(flora.FloraError):
        segment._validate_segments([{"start_s": start, "end_s": end, "metadata": {}}], 4)


def test_native_segments_require_order_and_provenance():
    with pytest.raises(flora.FloraError, match="ordered"):
        segment._validate_segments([{"start_s": 2, "end_s": 3, "metadata": {}},
                                    {"start_s": 1, "end_s": 4, "metadata": {}}], 4)
    with pytest.raises(flora.FloraError, match="provenance"):
        segment._validate_segments([{"start_s": 0, "end_s": 1}], 4)
    segment._validate_segments([], 4)


def test_segmentation_admission_rejects_changed_pins_and_bytes_before_runtime(tmp_path):
    path = tmp_path / "input.mp4"
    path.write_bytes(b"changed native bytes")
    request = {"id": "episode", "use": "noncommercial_research", "models": flora.MODELS,
               "tempseg_models": copy.deepcopy(segment.TEMPSEG_MODELS),
               "videos": [str(path)], "video_sha256": ["0" * 64], "duration_s": 4}
    with pytest.raises(flora.FloraError, match="Native video changed"):
        segment.preflight([request])
    request["tempseg_models"]["tempseg"]["revision"] = "0" * 40
    with pytest.raises(flora.FloraError, match="pins"):
        segment.preflight([request])
    request["id"] = "../escape"
    with pytest.raises(flora.FloraError, match="IDs"):
        segment.preflight([request])


def test_generation_limit_refuses_before_native_or_torch_load(tmp_path):
    journal = GenerationJournal(tmp_path / "journal", "bound")
    journal.index = 64
    def forbidden(*args):
        pytest.fail("generation ran after its limit")
    generate = segment._bounded_generate(forbidden, journal)
    with pytest.raises(flora.FloraError, match="64-generation"):
        generate(None, None, {}, 1024)
    assert list(journal.root.iterdir()) == []


def test_partial_segmentation_evidence_is_not_overwritten(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    retained = output / "episode.segment.raw.json"
    retained.write_text("paid output")
    with pytest.raises(flora.FloraError, match="contains evidence"):
        segment.run([], output, GenerationJournal(tmp_path / "journal", "bound"))
    assert retained.read_text() == "paid output"


def test_native_session_reuses_loaded_model_across_episodes():
    class LoadCounter:
        def __init__(self):
            self.model = self.proc = None
            self.loads = 0

        def load(self, kind):
            self.loads += 1
            self.model, self.proc = object(), object()
            return self.model, self.proc

    session = segment._cached_session(LoadCounter)
    first = session.load("tempseg")
    assert session.load("tempseg") == first
    assert session.loads == 1
    with pytest.raises(flora.FloraError, match="only loads TempSeg"):
        session.load("captioner")
    assert session.loads == 1


def test_duplicate_semantic_inputs_refused_before_native_model_loading(tmp_path, monkeypatch):
    path = tmp_path / 'native.mp4'
    path.write_bytes(b'video')
    request = {'id': 'a', 'use': 'noncommercial_research', 'models': flora.MODELS,
               'tempseg_models': segment.TEMPSEG_MODELS, 'videos': [str(path)],
               'video_sha256': [flora.digest(path)], 'duration_s': 4,
               'task': 'Move the cup.', 'profile': 'human'}
    def forbidden(*args):
        pytest.fail('duplicate admission reached native runtime')
    monkeypatch.setattr(segment, '_runtime', forbidden)
    with pytest.raises(flora.FloraError, match='Duplicate TempSeg'):
        segment.preflight([request, {**request, 'id': 'b', 'yam32': False}])


@pytest.fixture
def replay_source(tmp_path):
    from events.flora_journal import input_digest
    root = tmp_path / "prior"
    root.mkdir()
    request = {"id": "episode", "use": "noncommercial_research", "models": flora.MODELS,
               "tempseg_models": segment.TEMPSEG_MODELS, "videos": ["/old/native.mp4"],
               "video_sha256": ["a" * 64], "duration_s": 4, "task": "Move cup", "profile": "human"}
    manifest = {"run_id": "old-run", "stage": "segment-native", "use": "noncommercial_research",
                "requests": [request], "implementation_sha256": {"flora_segment.py": "old-code"}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    kwargs = {"input_ids": [[1, 2]], "max_new_tokens": 1024}
    old = GenerationJournal(root / "generations", input_digest(manifest))
    old.run(kwargs, lambda: {"dtype": "int64", "device": "cuda:0", "shape": [1, 3],
                             "tokens": [[1, 2, 3]]})
    entry = {"index": 0, **{f"{kind}_sha256": flora.digest(root / "generations" / f"000000.{kind}.json")
                            for kind in ("intent", "result")}}
    descriptor = {"run_id": "old-run", "manifest_sha256": flora.digest(root / "manifest.json"),
                  "episodes": {"episode": [entry]}}
    proof = root / "terminal.log"
    proof.write_text("Operator checked source stopped and zero live containers")
    reconciliation = {"run_id": "old-run", "manifest_sha256": descriptor["manifest_sha256"],
                      "confirmed_stopped": True, "observed_at_utc": "2026-10-22T08:00:00+00:00",
                      "reason": "Operator verified canceled source and zero live containers",
                      "evidence": [{"path": proof.name, "sha256": flora.digest(proof)}]}
    reconciliation_path = root / "continuation-reconciliation.json"
    reconciliation_path.write_text(json.dumps(reconciliation))
    descriptor["reconciliation_sha256"] = flora.digest(reconciliation_path)
    return root, {**request, "videos": ["/new/native.mp4"]}, descriptor, kwargs


def test_replay_validates_exact_source_and_pins_without_models(replay_source):
    root, request, descriptor, _ = replay_source
    receipt = segment.validate_replay([request], descriptor, root)
    assert receipt["source_implementation_sha256"] == {"flora_segment.py": "old-code"}
    assert json.loads(json.dumps(receipt)) == receipt
    assert segment.validate_replay([request], None, None) is None
    for key, value in (("task", "changed"), ("duration_s", 3), ("video_sha256", ["b" * 64]),
                       ("tempseg_models", {}), ("models", {})):
        with pytest.raises(flora.FloraError, match="source, task, clocks or model pins"):
            segment.validate_replay([{**request, key: value}], descriptor, root)


@pytest.mark.parametrize("damage", ["unsettled", "orphan", "changed", "manifest", "selection", "escape"])
def test_replay_refuses_uncertain_or_changed_paid_evidence(replay_source, damage):
    root, request, descriptor, _ = replay_source
    if damage == "unsettled":
        (root / "generations/000001.intent.json").write_text("{}")
    elif damage == "orphan":
        (root / "generations/000001.result.json").write_text("{}")
    elif damage == "changed":
        path = root / "generations/000000.result.json"
        value = json.loads(path.read_text())
        value["value"]["tokens"][0][-1] = 4
        path.write_text(json.dumps(value))
    elif damage == "manifest":
        descriptor["manifest_sha256"] = "0" * 64
    elif damage == "selection":
        descriptor["episodes"]["episode"][0]["index"] = True
    else:
        path = root / "generations/000000.result.json"
        outside = root.parent / "outside.json"
        path.rename(outside)
        path.symlink_to(outside)
    with pytest.raises(flora.FloraError):
        segment.validate_replay([request], descriptor, root)


def test_replay_retains_new_journal_and_provenance_before_decode(replay_source, tmp_path):
    root, request, descriptor, kwargs = replay_source
    receipt = segment.validate_replay([request], descriptor, root)
    output = tmp_path / "new-results"
    output.mkdir()
    commits = []
    journal = GenerationJournal(tmp_path / "new-journal", "new-binding",
                                lambda: commits.append(sorted(path.name for path in output.iterdir())))
    replayer = segment._Replay(receipt, root, output, journal)
    replayer.start("episode")
    saved = journal.run(kwargs, lambda: replayer.take(kwargs))
    assert saved["tokens"] == [[1, 2, 3]]
    replayer.finish()
    assert journal.index == 1
    assert (journal.root / "000000.result.json").exists()
    provenance = json.loads((output / "000000.replay.json").read_text())
    assert provenance["used"]["source_run_id"] == "old-run"
    assert provenance["used"]["generation_index"] == 0
    assert commits[-1] == ["000000.replay.json"]
    with pytest.raises(segment._ReplayAbort, match="already paid"):
        replayer.take(kwargs)
    assert replayer.take({**kwargs, "max_new_tokens": 99}) is None
    assert len(list((root / "generations").iterdir())) == 2


def test_replay_mismatch_and_unused_prefix_cannot_fall_back_to_inference(replay_source, tmp_path):
    root, request, descriptor, kwargs = replay_source
    receipt = segment.validate_replay([request], descriptor, root)
    journal = GenerationJournal(tmp_path / "new-journal", "new-binding")
    replay = segment._Replay(receipt, root, tmp_path, journal)
    replay.start("episode")
    with pytest.raises(segment._ReplayAbort, match="every declared"):
        replay.finish()
    with pytest.raises(segment._ReplayAbort, match="input digest differs"):
        replay.take({**kwargs, "max_new_tokens": 99})
    assert not issubclass(segment._ReplayAbort, Exception)
    assert list(journal.root.iterdir()) == []
    (root / "generations/000000.result.json").write_text("changed after admission")
    with pytest.raises(segment._ReplayAbort, match="changed after admission"):
        replay.take(kwargs)


def test_multiple_replayed_generations_have_distinct_durable_provenance(replay_source, tmp_path):
    from events.flora_journal import input_digest
    root, request, descriptor, kwargs = replay_source
    old = GenerationJournal(root / "generations", input_digest(json.loads((root / "manifest.json").read_text())))
    old.index = 1
    second = {**kwargs, "input_ids": [[4, 5]]}
    old.run(second, lambda: {"dtype": "int64", "device": "cuda:0", "shape": [1, 3], "tokens": [[4, 5, 6]]})
    descriptor["episodes"]["episode"].append({"index": 1, **{
        f"{kind}_sha256": flora.digest(root / "generations" / f"000001.{kind}.json")
        for kind in ("intent", "result")}})
    receipt = segment.validate_replay([request], descriptor, root)
    output = tmp_path / "output"
    output.mkdir()
    journal = GenerationJournal(tmp_path / "new-journal", "new-binding")
    replay = segment._Replay(receipt, root, output, journal)
    replay.start("episode")
    for inputs in (kwargs, second):
        journal.run(inputs, lambda: replay.take(inputs))
    replay.finish()
    assert sorted(path.name for path in output.iterdir()) == ["000000.replay.json", "000001.replay.json"]
    assert len(list(journal.root.iterdir())) == 4


def test_replay_model_proxy_uses_retained_tokens_without_calling_model(replay_source, tmp_path):
    torch = pytest.importorskip("torch")
    from events.flora_journal import input_digest
    root, request, descriptor, _ = replay_source
    kwargs = {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1024}
    intent_path, result_path = (root / "generations" / f"000000.{kind}.json" for kind in ("intent", "result"))
    intent = json.loads(intent_path.read_text())
    intent["input_sha256"] = input_digest(kwargs)
    intent_path.write_text(json.dumps(intent))
    result = json.loads(result_path.read_text())
    result["binding"] = intent
    result_path.write_text(json.dumps(result))
    descriptor["episodes"]["episode"][0].update(intent_sha256=flora.digest(intent_path), result_sha256=flora.digest(result_path))
    receipt = segment.validate_replay([request], descriptor, root)
    output = tmp_path / "output"
    output.mkdir()
    journal = GenerationJournal(tmp_path / "new-journal", "new-binding")
    replay = segment._Replay(receipt, root, output, journal)
    replay.start("episode")

    class NeverGenerate:
        def generate(self, **kwargs):
            pytest.fail("retained generation was repurchased")

    def native_generate(model, processor, inputs, budget):
        return model.generate(**inputs)

    tokens = segment._bounded_generate(native_generate, journal, replay)(NeverGenerate(), None, kwargs, 1024)
    replay.finish()
    assert tokens.tolist() == [[1, 2, 3]]
    saved = json.loads((journal.root / "000000.result.json").read_text())
    assert saved["value"]["tokens"] == [[1, 2, 3]]
    assert saved["binding"]["request_sha256"] == "new-binding"


def test_source_journal_extension_refuses_new_inference(replay_source, tmp_path):
    from events.flora_journal import input_digest
    root, request, descriptor, kwargs = replay_source
    receipt = segment.validate_replay([request], descriptor, root)
    output = tmp_path / "output"
    output.mkdir()
    journal = GenerationJournal(tmp_path / "new-journal", "new-binding")
    replay = segment._Replay(receipt, root, output, journal)
    replay.start("episode")
    journal.run(kwargs, lambda: replay.take(kwargs))
    old = GenerationJournal(root / "generations", input_digest(json.loads((root / "manifest.json").read_text())))
    old.index = 1
    next_inputs = {**kwargs, "input_ids": [[4, 5]]}
    old.run(next_inputs, lambda: {"dtype": "int64", "device": "cuda:0", "shape": [1, 3], "tokens": [[4, 5, 6]]})
    with pytest.raises(segment._ReplayAbort, match="journal changed"):
        replay.take(next_inputs)


@pytest.mark.parametrize("damage", ["not_stopped", "no_proof", "changed_proof", "outside", "bad_time"])
def test_continuation_requires_bound_operator_stop_evidence(replay_source, damage):
    root, request, descriptor, _ = replay_source
    path = root / "continuation-reconciliation.json"
    value = json.loads(path.read_text())
    if damage == "not_stopped":
        value["confirmed_stopped"] = False
    elif damage == "no_proof":
        value["evidence"] = []
    elif damage == "changed_proof":
        (root / "terminal.log").write_text("different observation")
    elif damage == "outside":
        value["evidence"][0]["path"] = "../outside.log"
    else:
        value["observed_at_utc"] = "yesterday"
    path.write_text(json.dumps(value))
    descriptor["reconciliation_sha256"] = flora.digest(path)
    with pytest.raises(flora.FloraError):
        segment.validate_replay([request], descriptor, root)
