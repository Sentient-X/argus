"""Batch admission and retained reward evidence; no model runtime substitute."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from events import flora, flora_reward_batch as batch
from events.flora_journal import GenerationJournal


@pytest.fixture
def planned_request(tmp_path):
    video = tmp_path / "native.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(16):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    events = tmp_path / "events.json"
    events.write_text(json.dumps({"events": [{"start_s": 0, "end_s": 2}]}))
    return flora.prepare([video], events, "Lift cup.", "human", tmp_path / "plan")


def test_reward_batch_verifies_media_and_preserves_native_camera_order(planned_request):
    main = Path(planned_request["videos"][0])
    left, right = main.with_name("left.mp4"), main.with_name("right.mp4")
    left.write_bytes(main.read_bytes())
    right.write_bytes(main.read_bytes())
    planned_request["videos"] = [str(main), str(left), str(right)]
    planned_request["video_sha256"] *= 3
    jobs = batch.preflight([planned_request])
    assert jobs == [{"id": "episode", "task": "Lift cup.", "videos": [str(left), str(main), str(right)]}]
    planned_request["video_sha256"][1] = "0" * 64
    with pytest.raises(flora.FloraError, match="Native video changed"):
        batch.preflight([planned_request])


@pytest.mark.parametrize("change,message", [
    (lambda value: value.update(duration_s=3), "source clocks"),
    (lambda value: value["models"]["rewarder"].update(revision="0" * 40), "immutable release"),
    (lambda value: value.update(use="commercial"), "noncommercial"),
])
def test_reward_batch_refuses_changed_admission(planned_request, change, message):
    changed = copy.deepcopy(planned_request)
    change(changed)
    with pytest.raises(flora.FloraError, match=message):
        batch.preflight([changed])


def test_reward_batch_refuses_duplicate_episode_ids(planned_request):
    with pytest.raises(flora.FloraError, match="unique"):
        batch.preflight([planned_request, planned_request])


def test_batch_result_requires_every_episode_and_preserves_rows(planned_request, tmp_path):
    second = {**planned_request, "id": "another"}
    row = {"id": "episode", "task": "Lift cup.", "duration_s": 2,
           "score": 3, "expected_score": 3, "progress": 0.5,
           "probabilities": {"1": 0, "2": 0, "3": 1, "4": 0, "5": 0}}
    output = tmp_path / "outcome.jsonl"
    output.write_text(json.dumps(row) + "\n")
    with pytest.raises(flora.FloraError, match="omitted"):
        batch.read_results(output, "outcome", [planned_request, second])
    with output.open("a") as stream:
        stream.write(json.dumps({**row, "id": "another"}) + "\n")
    records = batch.read_results(output, "outcome", [planned_request, second])
    assert [record["id"] for record in records] == ["episode", "another"]
    with output.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    with pytest.raises(flora.FloraError, match="duplicate"):
        batch.read_results(output, "outcome", [planned_request, second])


def test_batch_result_rejects_native_invalid_clip_substitution(planned_request, tmp_path):
    output = tmp_path / "dense.jsonl"
    output.write_text(json.dumps({"id": "episode", "task": "Lift cup.", "duration_s": 2,
                                  "invalid_clips": 1, "worst_score": 3}) + "\n")
    with pytest.raises(flora.FloraError, match="unparsed clips"):
        batch.read_results(output, "dense", [planned_request])


def test_publication_copies_without_hardlinks_and_refuses_replacement(tmp_path):
    source, target = tmp_path / 'attempt.jsonl', tmp_path / 'outcome.jsonl'
    source.write_bytes(b'{"retained": true}\n')
    batch._publish(source, target)
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_ino != source.stat().st_ino
    batch._publish(source, target)
    target.write_bytes(b'partial')
    with pytest.raises(flora.FloraError, match='differs'):
        batch._publish(source, target)
    assert target.read_bytes() == b'partial'
    assert source.read_bytes() == b'{"retained": true}\n'


@pytest.fixture
def terminal_recovery(planned_request, tmp_path):
    output_dir, journal_dir = tmp_path / 'retained', tmp_path / 'journal'
    output_dir.mkdir()
    journal_dir.mkdir()
    binding = {'requests_sha256': batch.input_digest([planned_request]), 'stages': ['outcome', 'dense'],
               'max_num_seqs': 1, 'journal_sha256': 'original-manifest'}
    (output_dir / 'batch.json').write_text(json.dumps(binding))
    intent = {'request_sha256': 'original-manifest', 'input_sha256': 'native-input'}
    (journal_dir / '000000.intent.json').write_text(json.dumps(intent))
    completion = 'Rationale: Lifted cup.\n\nScore: 3'
    (journal_dir / '000000.result.json').write_text(json.dumps({
        'binding': intent, 'value': [[{'text': completion, 'token_ids': [1, 2], 'logprobs': []}]]}))
    row = {'id': 'episode', 'task': planned_request['task'], 'duration_s': 2,
           'score': 3, 'expected_score': 3, 'progress': 0.5, 'output': completion,
           'probabilities': {'1': 0, '2': 0, '3': 1, '4': 0, '5': 0}}
    (output_dir / 'outcome.attempt-complete.jsonl').write_text(json.dumps(row) + '\n')
    (output_dir / 'outcome.attempt-partial.jsonl').write_text('{')
    return planned_request, output_dir, journal_dir


def test_terminal_recovery_retains_paid_results_without_loading_models(terminal_recovery):
    planned_request, output_dir, journal_dir = terminal_recovery
    assert batch.recover_outcome([planned_request], output_dir, journal_dir)[0]['score'] == 3
    receipt = json.loads((output_dir / 'outcome.recovery.json').read_text())
    assert receipt['model_calls'] == 0
    assert (output_dir / 'outcome.attempt-partial.jsonl').read_text() == '{'
    assert batch.recover_outcome([planned_request], output_dir, journal_dir)[0]['score'] == 3


def test_terminal_recovery_refuses_changed_generation(terminal_recovery):
    planned_request, output_dir, journal_dir = terminal_recovery
    result_path = journal_dir / '000000.result.json'
    value = json.loads(result_path.read_text())
    value['value'][0][0]['text'] = 'different paid response'
    result_path.write_text(json.dumps(value))
    with pytest.raises(flora.FloraError, match='differs from durable'):
        batch.recover_outcome([planned_request], output_dir, journal_dir)
    assert not (output_dir / 'outcome.jsonl').exists()


@pytest.mark.parametrize('value', [None, 0, 1, 'false'])
def test_reward_mode_requires_boolean_before_media_admission(tmp_path, value):
    journal = GenerationJournal(tmp_path / 'journal', 'request')
    with pytest.raises(flora.FloraError, match='with_rationale must be a boolean'):
        batch.run([], tmp_path / 'result', journal, with_rationale=value)
    assert not (tmp_path / 'result').exists()


@pytest.mark.parametrize('with_rationale', [False, True])
def test_retained_outcome_is_bound_to_rationale_mode(planned_request, tmp_path, with_rationale):
    output = tmp_path / 'result'
    output.mkdir()
    journal = GenerationJournal(tmp_path / 'journal', 'request')
    binding = {'requests_sha256': batch.input_digest([planned_request]), 'stages': ['outcome'],
               'max_num_seqs': 8, 'journal_sha256': 'request', 'with_rationale': with_rationale,
               'adapter_sha256': flora.digest(Path(batch.__file__))}
    (output / 'batch.json').write_text(json.dumps(binding))
    row = {'id': 'episode', 'task': planned_request['task'], 'duration_s': 2,
           'score': 3, 'expected_score': 3, 'progress': 0.5,
           'probabilities': {'1': 0, '2': 0, '3': 1, '4': 0, '5': 0}}
    (output / 'outcome.jsonl').write_text(json.dumps(row) + '\n')
    (output / 'receipt.json').write_text(json.dumps({
        'binding': binding, 'raw_sha256': {'outcome': flora.digest(output / 'outcome.jsonl')}}))
    kwargs = {} if with_rationale else {'with_rationale': False}
    assert batch.run([planned_request], output, journal, stage='outcome', **kwargs) == {'outcome': [row]}
    with pytest.raises(flora.FloraError, match='inputs or implementation changed'):
        batch.run([planned_request], output, journal, stage='outcome', with_rationale=not with_rationale)
    assert not (output / '.running').exists()


@pytest.mark.parametrize('stage,expected', [
    ('outcome', [[0, 19]]), ('dense', [(0, 8), (8, 19)]),
    ('both', [[0, 19], (0, 8), (8, 19)]),
])
def test_native_media_admission_reads_every_selected_plan(stage, expected):
    calls = []
    collage = SimpleNamespace(duration_s=2, num_frames=20, fps=10,
                              frames=lambda indices: calls.append(indices), release=lambda: calls.append('released'))
    core = SimpleNamespace(Collage=lambda paths: collage, terminal_frame_indices=lambda n, fps: [0, 19],
                           dense_windows=lambda n, fps: [SimpleNamespace(frame_indices=p) for p in ((0, 8), (8, 19))])
    batch._native_media_preflight([{'id': 'episode', 'videos': ['native.mp4']}],
                                  [{'id': 'episode', 'duration_s': 2}], core, stage)
    assert calls == expected + ['released']


@pytest.mark.parametrize('clock_mismatch', [False, True])
def test_native_media_failure_releases_buffers_and_names_episode(clock_mismatch):
    released = []
    def fail_read(indices):
        raise ValueError('decoded fewer frames than its container reports')
    collage = SimpleNamespace(duration_s=2.1 if clock_mismatch else 2, num_frames=21, fps=10,
                              frames=fail_read, release=lambda: released.append(True))
    core = SimpleNamespace(Collage=lambda paths: collage, terminal_frame_indices=lambda n, fps: [0, 20])
    with pytest.raises(flora.FloraError, match='Native reward media rejected episode:'):
        batch._native_media_preflight([{'id': 'episode', 'videos': ['native.mp4']}],
                                      [{'id': 'episode', 'duration_s': 2}], core, 'outcome')
    assert released == [True]


@pytest.mark.parametrize('packets', [16, 17])
def test_decoded_probe_preserves_geometry_and_restores_on_failure(planned_request, packets):
    path = planned_request['videos'][0]
    original = lambda name: SimpleNamespace(path=name, fps=8, num_frames=packets, width=64, height=48)
    core = SimpleNamespace(probe_video=original, VideoInfo=lambda path, fps, num_frames, width, height:
                           SimpleNamespace(path=path, fps=fps, num_frames=num_frames, width=width, height=height))
    with pytest.raises(RuntimeError, match='abort native stage'):
        with batch._decoded_video_probe(core, [{'videos': [path]}]) as audit:
            assert vars(core.probe_video(path)) == {'path': path, 'fps': 8., 'num_frames': 16, 'width': 64, 'height': 48}
            assert audit[0]['packet_count_corrected'] is (packets != 16)
            assert audit[0]['native_packet_count'] == packets
            with pytest.raises(flora.FloraError, match='unadmitted'):
                core.probe_video('another.mp4')
            raise RuntimeError('abort native stage')
    assert core.probe_video is original
