"""Offline Flora contracts: native media, retained evidence and fail-closed parsing.

These are not model-quality tests and do not simulate the NVIDIA model runtime.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys

import av
import numpy as np
import pytest

from events import flora


@pytest.fixture
def planned(tmp_path):
    video = tmp_path / "episode.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(32):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 5, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    events = tmp_path / "events.json"
    events.write_text(json.dumps({"events": [{"start_s": 0, "end_s": 2, "label": "old label"},
                                           {"start_s": 2, "end_s": 4, "label": "old label"}]}))
    out = tmp_path / "run"
    request = flora.prepare([video], events, "Put cup on tray.", "human", out)
    return request, out, video, events


def outputs(request):
    return {
        "caption": {"segments": [{**event, "caption": f"Move item {i}.", "status": "complete"}
                                  for i, event in enumerate(request["events"])]},
        "outcome": {"id": "episode", "task": request["task"], "duration_s": 4,
                    "score": 3, "expected_score": 3, "progress": 0.5,
                    "probabilities": {"1": 0, "2": 0.25, "3": 0.5, "4": 0.25, "5": 0}},
        "dense": {"id": "episode", "task": request["task"], "duration_s": 4,
                  "invalid_clips": 0, "worst_score": 2,
                  "clips": [{"valid": True, "quality": "2" * 16 + "3" * 16}],
                  "segments": [{"start_s": 0, "end_s": 2, "score": 2},
                               {"start_s": 2, "end_s": 4, "score": 3}]},
    }


def test_plan_decodes_native_clock_and_reuses_identical_request(planned):
    request, out, video, events = planned
    assert request["duration_s"] == 4
    assert flora.video_clock(video).frames == 32
    assert flora.prepare([video], events, request["task"], "human", out) == request
    assert list(out.iterdir()) == [out / "request.json"]
    with pytest.raises(flora.FloraError, match="changed"):
        flora.prepare([video], events, "Another task", "human", out)


@pytest.mark.parametrize("event", [
    {"start_s": -1, "end_s": 2}, {"start_s": 2, "end_s": 1},
    {"start_s": 0, "end_s": 5}, {"start_s": True, "end_s": 2},
    {"start_s": 0, "end_s": float("nan")}, {"start_s": 0, "end_s": float("inf")},
])
def test_invalid_intervals_rejected_before_execution(planned, event):
    request, out, video, events = planned
    events.write_text(json.dumps({"events": [event]}))
    with pytest.raises(flora.FloraError):
        flora.prepare([video], events, request["task"], "human", out.parent / "bad")


def test_overlapping_events_preserved(planned):
    request, out, video, events = planned
    events.write_text(json.dumps({"events": [{"start_s": 0, "end_s": 3}, {"start_s": 1, "end_s": 4}]}))
    other = flora.prepare([video], events, request["task"], "human", out.parent / "overlap")
    assert other["events"] == [{"start_s": 0, "end_s": 3}, {"start_s": 1, "end_s": 4}]


def test_retained_outputs_join_without_launching_models(planned):
    request, out, _, _ = planned
    for stage, raw in outputs(request).items():
        (out / f"{stage}.json").write_text(json.dumps(raw) + "\n")
        (out / f"{stage}.pending.json").write_text(json.dumps({"request_sha256": request["request_sha256"], "stage": stage}))
    result_path = flora.execute(request, out, "/no/gpu/python", "/no/gpu/python")
    result = json.loads(result_path.read_text())
    assert [(r["start_s"], r["end_s"]) for r in result["events"]] == [(0, 2), (2, 4)]
    assert "skill" not in result["events"][0]
    assert "summary" not in result
    assert result["outcome_reward"]["score"] == 3
    assert result["raw_sha256"]["dense"] == flora.digest(out / "dense.json")
    assert flora.execute(request, out, "/no/gpu/python", "/no/gpu/python") == result_path


def test_unsettled_stage_never_repeats_compute(planned):
    request, out, _, _ = planned
    (out / "caption.pending.json").write_text(json.dumps({"request_sha256": request["request_sha256"], "stage": "caption"}))
    with pytest.raises(flora.FloraError, match="interrupted"):
        flora.execute(request, out, "/no/gpu/python", "/no/gpu/python")
    assert not (out / "result.json").exists()
    assert not (out / ".running").exists()


def test_live_owner_is_not_removed(planned):
    request, out, _, _ = planned
    (out / ".running").mkdir()
    with pytest.raises(flora.FloraError, match="still own"):
        flora.execute(request, out, "/no/gpu/python", "/no/gpu/python")
    assert (out / ".running").is_dir()


@pytest.mark.parametrize("stage,change", [
    ("caption", lambda x: x["segments"].pop()),
    ("caption", lambda x: x["segments"][0].update(end_s=2.1)),
    ("caption", lambda x: x["segments"][0].update(caption=" ")),
    ("caption", lambda x: x["segments"][0].update(status="unsupported_target_frames")),
    ("outcome", lambda x: x.update(score=True)),
    ("outcome", lambda x: x.update(score=6)),
    ("outcome", lambda x: x.update(expected_score=4)),
    ("outcome", lambda x: x.update(task="different task")),
    ("outcome", lambda x: x["probabilities"].update({"3": -0.5})),
    ("dense", lambda x: x.update(invalid_clips=1)),
    ("dense", lambda x: x["clips"][0].update(valid=False)),
    ("dense", lambda x: x["clips"][0].update(quality="3" * 31)),
    ("dense", lambda x: x["segments"][0].update(score=0)),
    ("dense", lambda x: x["segments"][1].update(start_s=2.5)),
    ("dense", lambda x: x["segments"][1].update(end_s=3.5)),
])
def test_bad_outputs_fail_closed(planned, stage, change):
    request, _, _, _ = planned
    raw = copy.deepcopy(outputs(request)[stage])
    flora.validate(stage, raw, request)
    change(raw)
    with pytest.raises(flora.FloraError):
        flora.validate(stage, raw, request)


def test_invalid_dense_raw_is_kept_without_a_completed_result(planned):
    request, out, _, _ = planned
    raw = outputs(request)
    raw["dense"]["invalid_clips"] = 1
    for stage, value in raw.items():
        (out / f"{stage}.json").write_text(json.dumps(value))
        (out / f"{stage}.pending.json").write_text(json.dumps({"request_sha256": request["request_sha256"], "stage": stage}))
    with pytest.raises(flora.FloraError, match="unparsed"):
        flora.execute(request, out, "/no/gpu/python", "/no/gpu/python")
    assert json.loads((out / "dense.json").read_text())["invalid_clips"] == 1
    assert not (out / "result.json").exists()


def test_execute_requires_explicit_research_environments(planned):
    _, out, video, events = planned
    result = subprocess.run([sys.executable, "-m", "events.flora", "--video", str(video),
                             "--events", str(events), "--task", "task", "--profile", "human",
                             "--out", str(out), "--execute"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "--noncommercial-research" in result.stderr
    assert not list(out.glob("*.pending.json"))


def test_unbound_result_is_rejected(planned):
    request, out, _, _ = planned
    (out / "caption.json").write_text(json.dumps(outputs(request)["caption"]))
    with pytest.raises(flora.FloraError, match="no matching request"):
        flora.execute(request, out, "/no/gpu/python", "/no/gpu/python")


def test_generation_saved_before_downstream_parser_failure(tmp_path):
    from events.flora_journal import GenerationJournal
    root = tmp_path / "generations"
    calls = []

    def generate():
        calls.append(1)
        return {"tokens": [10, 20, 30], "raw": "not a valid score"}

    journal = GenerationJournal(root, "request-a")
    reply = journal.run({"prompt": "score", "pixels": [1, 2, 3]}, generate)
    with pytest.raises(ValueError):
        int(reply["raw"])
    replay = GenerationJournal(root, "request-a").run({"prompt": "score", "pixels": [1, 2, 3]}, generate)
    assert replay == reply and calls == [1]


def test_generation_uncertainty_and_changed_inputs_never_retry(tmp_path):
    from events.flora_journal import GenerationJournal
    calls = []

    def interrupted():
        calls.append(1)
        raise RuntimeError("lost GPU response")

    root = tmp_path / "generations"
    with pytest.raises(RuntimeError, match="lost GPU response"):
        GenerationJournal(root, "request-a").run({"prompt": "score"}, interrupted)
    with pytest.raises((RuntimeError, ValueError)):
        GenerationJournal(root, "request-a").run({"prompt": "score"}, interrupted)
    assert calls == [1]
    other = tmp_path / "completed"
    GenerationJournal(other, "request-a").run({"prompt": "a"}, lambda: "retained")
    with pytest.raises((RuntimeError, ValueError)):
        GenerationJournal(other, "request-a").run({"prompt": "b"}, lambda: pytest.fail("changed input executed"))
    with pytest.raises((RuntimeError, ValueError)):
        GenerationJournal(other, "request-b").run({"prompt": "a"}, lambda: pytest.fail("changed owner executed"))


def test_generation_array_identity_includes_content_and_shape():
    from events.flora_journal import input_digest
    array = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    assert input_digest(array) == input_digest(array.copy())
    assert input_digest(array) != input_digest(array.reshape(4))
    assert input_digest(array) != input_digest(array + 1)


def test_completed_raw_tampering_is_rejected(planned):
    request, out, _, _ = planned
    for stage, raw in outputs(request).items():
        (out / f"{stage}.json").write_text(json.dumps(raw))
        (out / f"{stage}.pending.json").write_text(json.dumps({"request_sha256": request["request_sha256"], "stage": stage}))
    flora.execute(request, out, "/no/gpu/python", "/no/gpu/python")
    raw = json.loads((out / "caption.json").read_text())
    raw["segments"][0]["caption"] = "Changed caption."
    (out / "caption.json").write_text(json.dumps(raw))
    with pytest.raises(flora.FloraError, match="retained evidence changed"):
        flora.execute(request, out, "/no/gpu/python", "/no/gpu/python")


def test_reward_completion_journal_preserves_raw_logprobs_before_parse(tmp_path):
    from types import SimpleNamespace as Record
    from events.flora_journal import GenerationJournal
    from events.flora_worker import _reward_generate
    calls = []
    completion = Record(text="bad score", token_ids=[42], logprobs=[{
        42: Record(logprob=-0.5), 43: Record(logprob=float("-inf")),
    }])

    class Engine:
        def generate(self, *args, **kwargs):
            calls.append(1)
            return [Record(outputs=[completion])]

    params = Record(temperature=0, max_tokens=64, logprobs=5, n=1, top_p=1,
                    top_k=-1, min_p=0, seed=None, repetition_penalty=1,
                    presence_penalty=0, frequency_penalty=0,
                    structured_outputs=Record(regex="[1-5]"))
    request = [{"prompt": "score", "frames": np.zeros((2, 3, 4), dtype=np.uint8)}]
    root = tmp_path / "reward-generations"
    live = _reward_generate(Engine(), GenerationJournal(root, "identity")).generate(request, params)
    with pytest.raises(ValueError):
        int(live[0].outputs[0].text)
    replay = _reward_generate(Engine(), GenerationJournal(root, "identity")).generate(request, params)
    assert calls == [1]
    assert replay[0].outputs[0].token_ids == [42]
    assert replay[0].outputs[0].logprobs[0][43].logprob == float("-inf")
