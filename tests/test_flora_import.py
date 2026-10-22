"""Import adopts only mutually bound, complete and untampered offline evidence."""
import hashlib
import json
from pathlib import Path

import pytest

from events.flora import FloraError
from events.flora_import import assemble
from events.flora_journal import input_digest


def seal(root, stage, requests, files):
    folder = root / f'{stage}.artifacts'
    folder.mkdir(exist_ok=True)
    manifest = {'run_id': stage, 'stage': stage, 'requests': requests}
    files = {'manifest.json': json.dumps(manifest), **files}
    for name, content in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    receipt = {'status': 'complete', 'run_id': stage, 'manifest_sha256': input_digest(manifest),
               'files': {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in files}}
    path = root / f'{stage}.receipt.json'
    path.write_text(json.dumps(receipt))
    return path


def replace_artifact(receipt_path, name, content):
    receipt = json.loads(receipt_path.read_text())
    folder = receipt_path.with_suffix('').with_suffix('.artifacts')
    (folder / name).write_text(content)
    receipt['files'][name] = hashlib.sha256(content.encode()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))


@pytest.fixture
def batch(tmp_path):
    requests, captions, outcomes, dense = [], {}, [], []
    for identifier in ('alpha', 'beta'):
        request = {'id': identifier, 'task': f'Move {identifier}.', 'duration_s': 4,
                   'videos': [f'/evidence/inputs/{identifier}.mp4'],
                   'events': [{'start_s': 0, 'end_s': 4}]}
        requests.append(request)
        captions[f'results/{identifier}.caption.raw.json'] = json.dumps({
            'id': identifier, 'segments': [{'start_s': 0, 'end_s': 4,
                                          'caption': f' Move {identifier}. ', 'status': 'complete'}]})
        outcomes.append({'id': identifier, 'task': request['task'], 'duration_s': 4, 'score': 3,
                         'expected_score': 3, 'progress': 0.5,
                         'probabilities': {str(i): int(i == 3) for i in range(1, 6)}})
        dense.append({'id': identifier, 'task': request['task'], 'duration_s': 4,
                      'invalid_clips': 0, 'worst_score': 3,
                      'clips': [{'valid': True, 'quality': '3' * 32}],
                      'segments': [{'start_s': 0, 'end_s': 4, 'score': 3}]})
    cap = seal(tmp_path, 'caption', requests, {**captions, 'worker.log': 'finished'})
    rew = seal(tmp_path, 'reward', requests, {
        'results/outcome.jsonl': '\n'.join(map(json.dumps, reversed(outcomes))),
        'results/dense.jsonl': '\n'.join(map(json.dumps, dense))})
    return cap, rew, tmp_path / 'imported', requests


def test_complete_import_matches_schema_and_maps_reward_ids_without_model_calls(batch):
    cap, rew, out, requests = batch
    paths = assemble(cap, rew, out)
    assert len(paths) == 2
    for path, request in zip(paths, requests, strict=True):
        result = json.loads(path.read_text())
        assert set(result) == {'schema', 'request', 'events', 'outcome_reward', 'dense_reward',
                               'caption_evidence', 'raw_sha256'}
        assert result['schema'] == 'argus.flora.research/v1'
        assert result['request'] == request
        assert result['events'] == [{'start_s': 0, 'end_s': 4, 'label': request['task']}]
        assert result['outcome_reward']['id'] == request['id']
        assert result['raw_sha256']['dense'] == hashlib.sha256(
            (rew.parent / 'reward.artifacts/results/dense.jsonl').read_bytes()).hexdigest()
    original = paths[0].read_bytes()
    with pytest.raises(FloraError, match='already exists'):
        assemble(cap, rew, out)
    assert paths[0].read_bytes() == original


@pytest.mark.parametrize('mutation', ['unconsumed_file', 'failed', 'manifest', 'request_list', 'traversal', 'symlink'])
def test_untrusted_receipt_or_evidence_is_refused_before_publication(batch, mutation):
    cap, rew, out, requests = batch
    receipt = json.loads(cap.read_text())
    folder = cap.parent / 'caption.artifacts'
    if mutation == 'unconsumed_file':
        (folder / 'worker.log').write_text('changed')
    elif mutation == 'failed':
        receipt['status'] = 'failed'
        cap.write_text(json.dumps(receipt))
    elif mutation == 'manifest':
        manifest = json.loads((folder / 'manifest.json').read_text())
        manifest['run_id'] = 'another'
        replace_artifact(cap, 'manifest.json', json.dumps(manifest))
    elif mutation == 'request_list':
        manifest = json.loads((folder / 'manifest.json').read_text())
        manifest['requests'][0]['videos'] = ['/different/source.mp4']
        replace_artifact(cap, 'manifest.json', json.dumps(manifest))
        receipt = json.loads(cap.read_text())
        receipt['manifest_sha256'] = input_digest(manifest)
        cap.write_text(json.dumps(receipt))
    else:
        outside = cap.parent / 'outside.log'
        outside.write_text('finished')
        if mutation == 'traversal':
            receipt['files']['../outside.log'] = hashlib.sha256(outside.read_bytes()).hexdigest()
            cap.write_text(json.dumps(receipt))
        else:
            (folder / 'worker.log').unlink()
            (folder / 'worker.log').symlink_to(outside)
    with pytest.raises(FloraError):
        assemble(cap, rew, out)
    assert not out.exists()


@pytest.mark.parametrize('mutation', ['duplicate_reward', 'late_invalid_dense', 'changed_caption_interval', 'missing_dense'])
def test_valid_hashes_do_not_bypass_record_validation(batch, mutation):
    cap, rew, out, _ = batch
    folder = rew.parent / 'reward.artifacts'
    if mutation == 'missing_dense':
        receipt = json.loads(rew.read_text())
        del receipt['files']['results/dense.jsonl']
        rew.write_text(json.dumps(receipt))
        (folder / 'results/dense.jsonl').unlink()
    elif mutation == 'changed_caption_interval':
        name = 'results/beta.caption.raw.json'
        record = json.loads((cap.parent / 'caption.artifacts' / name).read_text())
        record['segments'][0]['end_s'] = 3
        replace_artifact(cap, name, json.dumps(record))
    else:
        rows = [json.loads(line) for line in (folder / 'results/dense.jsonl').read_text().splitlines()]
        if mutation == 'duplicate_reward':
            rows[1] = rows[0]
        else:
            rows[1]['invalid_clips'] = 1
        replace_artifact(rew, 'results/dense.jsonl', '\n'.join(map(json.dumps, rows)))
    with pytest.raises(FloraError):
        assemble(cap, rew, out)
    assert not out.exists()
