"""Retained evidence parsing and ordinal metrics, without a model/runtime mock."""
import copy
import json
from pathlib import Path

import pytest

from bench.flora_reward_eval import SOURCE, evaluate
from events import flora
from events.flora_journal import input_digest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def seal(root, receipt_path):
    manifest = json.loads((root / 'manifest.json').read_text())
    native = json.loads((root / 'results/receipt.json').read_text())
    native['raw_sha256']['outcome'] = flora.digest(root / 'results/outcome.jsonl')
    put(root / 'results/receipt.json', native)
    put(receipt_path, {'status': 'complete', 'run_id': manifest['run_id'],
                      'manifest_sha256': input_digest(manifest),
                      'implementation_sha256': manifest['implementation_sha256'],
                      'files': {str(p.relative_to(root)): flora.digest(p) for p in root.rglob('*') if p.is_file()}})


@pytest.fixture
def evidence(tmp_path):
    root, receipt = tmp_path / 'artifacts', tmp_path / 'receipt.json'
    plan_path, gold_path, admission_path = (tmp_path / name for name in ('plan.json', 'gold.json', 'admission.json'))
    source = dict(zip(('repo', 'revision', 'split'), SOURCE))
    frozen, gold_rows, media, requests, outputs = [], [], [], [], []
    for i, (label, prediction) in enumerate(((1, 2), (5, 5), (3, None))):
        eid, sha = str(i), str(i) * 64
        row = {'id': eid, 'ground_truth_score': label, 'source_row_index': i,
               'file_name': eid + '.mp4', 'video_url': 'https://example.org/' + eid + '.mp4', 'task': 'Lift cup.'}
        frozen.append(row)
        truth = {k: v for k, v in row.items() if k != 'task'}
        truth.update(video_sha256=sha, admission_status='admitted' if prediction else 'unsupported')
        clock = {'id': eid, 'sha256': sha, 'status': truth['admission_status'],
                 'duration_s': 2, 'frames': 20, 'fps': '10'}
        if prediction is None:
            truth['admission_error'] = clock['error'] = 'nonconstant clock'
        else:
            requests.append({'id': eid, 'task': row['task'], 'video_sha256': [sha],
                             'videos': ['/evidence/' + eid + '.mp4'], 'duration_s': 2,
                             'models': copy.deepcopy(flora.MODELS)})
            outputs.append({'id': eid, 'task': row['task'], 'duration_s': 2, 'score': prediction,
                            'expected_score': prediction, 'progress': (prediction - 1) / 4,
                            'probabilities': {str(k): int(k == prediction) for k in range(1, 6)},
                            'output': f'Score: {prediction}', 'rationale': None, 'sampled_frames': 20})
        gold_rows.append(truth)
        media.append(clock)
    put(plan_path, {'source': source, 'rows': frozen})
    put(admission_path, {'plan_sha256': flora.digest(plan_path), 'rows': media})
    coverage = {'frozen_count': 3, 'admitted_count': 2, 'admission_failures': 1, 'replacements': 0}
    put(gold_path, {'source': source, 'rows': gold_rows, **coverage,
                    'frozen_plan_sha256': flora.digest(plan_path),
                    'cpu_admission_sha256': flora.digest(admission_path)})
    provenance = {**coverage, 'frozen_plan_sha256': flora.digest(plan_path),
                  'evaluation_gold_sha256': flora.digest(gold_path), 'cpu_admission_sha256': flora.digest(admission_path)}
    manifest = {'run_id': 'test', 'stage': 'reward-outcome', 'reward_rationale': False,
                'use': 'noncommercial_research', 'requests': requests, 'max_num_seqs': 8,
                'implementation_sha256': {'flora_reward_batch.py': 'adapter'}, 'benchmark_provenance': provenance}
    put(root / 'manifest.json', manifest)
    binding = {'requests_sha256': input_digest(requests), 'journal_sha256': input_digest(manifest),
               'stages': ['outcome'], 'with_rationale': False, 'max_num_seqs': 8, 'adapter_sha256': 'adapter'}
    put(root / 'results/batch.json', binding)
    put(root / 'results/receipt.json', {'binding': binding, 'raw_sha256': {}})
    (root / 'results/outcome.jsonl').write_text('\n'.join(json.dumps(row) for row in outputs) + '\n')
    intent = {'request_sha256': input_digest(manifest), 'input_sha256': 'native-input'}
    put(root / 'generations/000000.intent.json', intent)
    put(root / 'generations/000000.result.json', {'binding': intent, 'value': [[{'text': row['output']}] for row in outputs]})
    seal(root, receipt)
    return root, receipt, gold_path, plan_path, admission_path


def test_ordinal_metrics_preserve_admission_denominator(evidence):
    report = evaluate(*evidence)
    assert report['coverage'] == {'frozen': 3, 'admitted': 2, 'answered': 2, 'admission_failures': 1, 'fraction': 2 / 3}
    assert report['answered_metrics'] == {'count': 2, 'mae': .5, 'accuracy': .5, 'mae_times_100': 50}
    assert report['per_label']['1']['mae'] == 1
    assert report['per_label']['3']['mae'] is None
    assert report['frozen_cohort_mae_bounds'] == [1 / 3, 1]
    assert report['confusion_matrix_gold_rows_predicted_columns'][0][1] == 1


def test_changed_gold_and_artifacts_are_refused(evidence):
    root, receipt, gold, *_ = evidence
    gold.write_text(gold.read_text() + ' ')
    with pytest.raises(ValueError, match='Changed benchmark evidence'):
        evaluate(*evidence)
    (root / 'results/outcome.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='Artifact changed'):
        evaluate(*evidence)


@pytest.mark.parametrize('change,message', [
    (lambda m: m.update(reward_rationale=True), 'score-only'),
    (lambda m: m['requests'][0]['models']['rewarder'].update(revision='wrong'), 'model revision'),
    (lambda m: m['requests'][0].update(task='different task'), 'Episode input'),
    (lambda m: m['requests'].pop(), 'omitted or added'),
    (lambda m: m['benchmark_provenance'].update(frozen_count=2), 'denominator'),
])
def test_resealed_wrong_protocol_or_population_is_refused(evidence, change, message):
    root, receipt, *_ = evidence
    manifest = json.loads((root / 'manifest.json').read_text())
    change(manifest)
    put(root / 'manifest.json', manifest)
    seal(root, receipt)
    with pytest.raises(ValueError, match=message):
        evaluate(*evidence)


def test_native_output_must_match_durable_generation(evidence):
    root, receipt, *_ = evidence
    path = root / 'generations/000000.result.json'
    journal = json.loads(path.read_text())
    journal['value'][0][0]['text'] = 'Score: 4'
    put(path, journal)
    seal(root, receipt)
    with pytest.raises(ValueError, match='paid generation'):
        evaluate(*evidence)
