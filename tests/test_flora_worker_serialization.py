"""Native model audit serialization must not strand completed caption replies."""
import json

import pytest

from events.flora_worker import _caption_load_audit, _write_json


def test_native_loading_key_sets_are_stable_without_mutating_audit(tmp_path):
    original = {'captioner': {'gdn': {'all_values_equal_saved_checkpoint': True},
                             'loading_info': {'missing_keys': set(), 'unexpected_keys': {'z.extra', 'a.extra'},
                                              'mismatched_keys': set(), 'error_msgs': []}}}
    path = tmp_path / 'caption.json'
    _write_json(path, {'segments': [{'caption': 'Dice the onion.'}],
                       'model_load_audit': _caption_load_audit(original)})
    result = json.loads(path.read_text())
    assert result['segments'][0]['caption'] == 'Dice the onion.'
    assert result['model_load_audit']['captioner']['loading_info'] == {
        'missing_keys': [], 'unexpected_keys': ['a.extra', 'z.extra'], 'mismatched_keys': [], 'error_msgs': []}
    assert original['captioner']['loading_info']['unexpected_keys'] == {'z.extra', 'a.extra'}


def test_unrecognized_audit_set_is_rejected_without_partial_output(tmp_path):
    path = tmp_path / 'caption.json'
    audit = {'captioner': {'gdn': {'unrecognized': {'x'}}, 'loading_info': {'missing_keys': set()}}}
    with pytest.raises(TypeError):
        _write_json(path, {'segments': [{'caption': 'Already generated.'}],
                           'model_load_audit': _caption_load_audit(audit)})
    assert not path.exists()


def test_invalid_numeric_output_is_rejected_before_creating_file(tmp_path):
    path = tmp_path / 'caption.json'
    with pytest.raises(ValueError):
        _write_json(path, {'duration_s': float('nan')})
    assert not path.exists()


def test_serialized_output_never_replaces_retained_bytes(tmp_path):
    path = tmp_path / 'caption.json'
    path.write_bytes(b'original partial evidence')
    with pytest.raises(FileExistsError):
        _write_json(path, {'segments': []})
    assert path.read_bytes() == b'original partial evidence'
