"""Retained hosted evidence is checked before it can stand in for inference."""
import hashlib
import json
from pathlib import Path

import pytest

from events import flora_hosted
from events.flora_journal import input_digest


def test_hosted_replay_verifies_saved_bytes_and_never_reenters_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(flora_hosted, 'EVIDENCE_ROOT', tmp_path)
    manifest = {'run_id': 'retained', 'stage': 'caption', 'use': 'noncommercial_research',
                'max_num_seqs': 1, 'requests': [], 'implementation_sha256': {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(flora_hosted.__file__).parent.glob('flora*.py')}}
    root = tmp_path / 'runs' / 'retained'
    root.mkdir(parents=True)
    raw = root / 'raw.json'
    raw.write_text('{"caption": "Move the cup."}')
    receipt = {'manifest_sha256': input_digest(manifest), 'status': 'complete',
               'files': {'raw.json': hashlib.sha256(raw.read_bytes()).hexdigest()}}
    (root / 'receipt.json').write_text(json.dumps(receipt))
    def forbidden():
        pytest.fail('replay must not write or generate')
    assert flora_hosted.execute(manifest, forbidden)['status'] == 'complete'
    raw.write_text('{"caption": "corrupted"}')
    with pytest.raises(ValueError, match='evidence changed'):
        flora_hosted.execute(manifest, forbidden)
    raw.unlink()
    with pytest.raises(ValueError, match='evidence changed'):
        flora_hosted.execute(manifest, forbidden)


def test_modal_volume_alias_admits_native_input_but_rejects_escape(tmp_path, monkeypatch):
    from events.flora import MODELS
    actual = tmp_path / 'mounted-volume'
    (actual / 'inputs').mkdir(parents=True)
    alias = tmp_path / 'evidence'
    alias.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(flora_hosted, 'EVIDENCE_ROOT', alias)
    video = alias / 'inputs' / 'native.mp4'
    video.write_bytes(b'original native bytes')
    manifest = {'run_id': 'alias', 'stage': 'caption', 'use': 'noncommercial_research',
                'max_num_seqs': 1, 'implementation_sha256': {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(flora_hosted.__file__).parent.glob('flora*.py')},
                'requests': [{'models': MODELS, 'videos': [str(video)],
                              'video_sha256': [hashlib.sha256(video.read_bytes()).hexdigest()]}]}
    class Admitted(Exception):
        pass
    def stop_before_dispatch(*args):
        raise Admitted
    monkeypatch.setattr(flora_hosted, 'save', stop_before_dispatch)
    with pytest.raises(Admitted):
        flora_hosted.execute(manifest, lambda: None)
    outside = tmp_path / 'outside.mp4'
    outside.write_bytes(video.read_bytes())
    escape = alias / 'inputs' / 'escape.mp4'
    escape.symlink_to(outside)
    manifest['requests'][0]['videos'] = [str(escape)]
    with pytest.raises(ValueError, match='under the evidence volume'):
        flora_hosted.execute(manifest, lambda: None)


def test_resident_backlog_binds_order_and_refuses_duplicate_ids():
    request = {'id': 'same', 'video_sha256': ['sha'], 'events': [],
               'task': 'task', 'profile': 'human', 'models': {}}
    batches = [{'requests': [request]}, {'requests': [{**request, 'task': 'different'}]}]
    manifest = {'batches': batches, 'requests': [b['requests'][0] for b in batches]}
    assert flora_hosted.caption_batches(manifest) == batches
    manifest['requests'] = [{'id': 'changed'}, {'id': 'same'}]
    with pytest.raises(ValueError, match='ordered batches'):
        flora_hosted.caption_batches(manifest)
    batches[0]['requests'].append({'id': 'same'})
    with pytest.raises(ValueError, match='unique within'):
        flora_hosted.caption_batches(manifest)


def test_resident_backlog_refuses_duplicate_paid_inputs_even_with_different_ids():
    request = {'id': 'a', 'video_sha256': ['sha'], 'events': [],
               'task': 'task', 'profile': 'human', 'models': {}}
    batches = [{'requests': [request]}, {'requests': [{**request, 'id': 'b'}]}]
    with pytest.raises(ValueError, match='Duplicate caption inputs'):
        flora_hosted.caption_batches({'batches': batches,
            'requests': [request, {**request, 'id': 'b'}]})


def test_resident_backlog_is_bounded_before_gpu_dispatch():
    for batches in ([], [{'requests': [{'id': 'a'}]}] * 9,
                    [{'requests': [{'id': str(i)} for i in range(17)]}]):
        with pytest.raises(ValueError):
            flora_hosted.caption_batches({'batches': batches, 'requests': []})


def test_admitted_cache_rejects_modified_bytes_and_path_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(flora_hosted, 'EVIDENCE_ROOT', tmp_path)
    cache = tmp_path / 'prepared' / 'batch' / 'cache.json'
    cache.parent.mkdir(parents=True)
    cache.write_text('{}')
    descriptor = {'path': str(cache), 'sha256': hashlib.sha256(cache.read_bytes()).hexdigest()}
    assert flora_hosted.admitted_cache(descriptor) == cache.parent
    cache.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='differs'):
        flora_hosted.admitted_cache(descriptor)
    outside = tmp_path / 'outside' / 'cache.json'
    outside.parent.mkdir()
    outside.write_text('{}')
    descriptor['path'] = str(outside)
    with pytest.raises(ValueError, match='differs'):
        flora_hosted.admitted_cache(descriptor)


def test_collect_existing_call_is_idempotent_and_resumes_evidence_transfer(tmp_path):
    source = tmp_path / 'run.json'
    source.write_text('{}')
    ledger = tmp_path / 'modal-budget.jsonl'
    ledger.write_text(json.dumps({'run_id': 'run', 'reserved_usd': 4}) + '\n')
    payload = b'paid response'
    result = {'run_id': 'run', 'status': 'complete', 'error': None, 'elapsed_s': 10,
              'files': {'result.json': hashlib.sha256(payload).hexdigest()}}
    def interrupted(path):
        yield payload[:3]
        raise ConnectionError('offline')
    with pytest.raises(ConnectionError):
        flora_hosted.retain_result(source, result, interrupted)
    assert not (tmp_path / 'run.artifacts/result.json').exists()
    flora_hosted.retain_result(source, result, lambda path: iter([payload]))
    def forbidden(path):
        pytest.fail('verified artifact must not be downloaded again')
    flora_hosted.retain_result(source, result, forbidden)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[1]['released_usd'] < 4
    assert (tmp_path / 'run.artifacts/result.json').read_bytes() == payload
    with pytest.raises(ValueError, match='receipt changed'):
        flora_hosted.retain_result(source, {**result, 'status': 'failed'}, forbidden)
