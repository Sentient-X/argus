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
