"""Offline comparison must not turn absent evidence into quality scores."""
import json

import pytest

from bench.flora_compare import aggregate, artifact, evaluate, judge_key, prepare, tally


def put(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture
def cohort(tmp_path):
    qualification = tmp_path / "qualification"
    video = qualification / "homer_33" / "native.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"retained-source-video")
    event = {"start_s": 0, "end_s": 10, "label": "Slice onion"}
    events = put(video.parent / "events.json", {"id": "homer_33", "events": [event]})
    ep = {"id": "homer_33", "split": "dev", "task": "Prepare onion", "duration_s": 10,
          "video": str(video), "video_sha256": artifact(video)["sha256"],
          "events": str(events), "events_sha256": artifact(events)["sha256"], "argus_model": "test/argus"}
    put(qualification / "qualification-inputs.json", {"episodes": [ep]})
    flora = qualification / "caption-seq8-v3.artifacts" / "results" / "homer_33.caption.raw.json"
    put(flora, {"id": "homer_33", "input_sha256": {"main": ep["video_sha256"]},
                "segments": [{"start_s": 0, "end_s": 10, "caption": "Cut the onion", "status": "complete"}]})
    gold = put(tmp_path / "gold.json", {"episodes": [{"id": "homer_33", "instruction": "Prepare onion",
                "segments": [{"start_sec": 0, "end_sec": 10, "subtask": "Slice onion"}]}]})
    return qualification, gold, flora, video


def test_unknown_verdict_is_not_false_or_perfect(cohort):
    qualification, gold, _, _ = cohort
    manifest = prepare(qualification, {"videodb": gold}, {}, qualification)
    report, pending = evaluate(manifest)
    row = report["references"]["videodb"]["systems"]["argus_flora_vllm"]["snap@0.5"]["per_episode"]["homer_33"]
    assert row["segmentation"]["f1"] == 1
    assert row["semantic"] is None
    assert row["caption_accuracy_on_temporal_matches"] is None
    assert row["semantic_bounds"]["lower"]["f1"] == 0
    assert row["semantic_bounds"]["upper"]["f1"] == 1
    assert len(pending) == 2  # one per label pair, deduplicated over four IoU/snap protocols
    verdict = judge_key("openai/gpt-5.5", "Prepare onion", "Slice onion", "Cut the onion")
    assert verdict in {item["key"] for item in pending}
    for accepted in (False, True):
        scored, _ = evaluate(manifest, {verdict: accepted})
        row = scored["references"]["videodb"]["systems"]["argus_flora_vllm"]["snap@0.5"]["per_episode"]["homer_33"]
        assert row["semantic"]["f1"] == int(accepted)
        assert row["unknown"] == 0


def test_missing_system_and_reference_are_not_zero_scores(cohort):
    qualification, gold, _, _ = cohort
    manifest = prepare(qualification, {"videodb": gold}, {}, qualification)
    report, _ = evaluate(manifest)
    assert report["references"]["macrodata"]["status"] == "missing"
    assert any("does not evaluate Flora TempSeg" in note for note in report["limitations"])
    missing = report["references"]["videodb"]["systems"]["argus_astra"]["raw@0.75"]["aggregate"]
    assert missing["pooled_micro"] is None
    assert missing["observed_pooled_micro"] is None
    assert missing["unavailable"] == {"homer_33": "missing_prediction"}
    manifest["systems"]["argus_astra"]["episodes"] = manifest["systems"]["argus"]["episodes"].copy()
    path = qualification / "empty.json"
    put(path, {"id": "homer_33", "events": []})
    manifest["systems"]["argus_astra"]["episodes"]["homer_33"] = {"artifact": artifact(path)}
    report, _ = evaluate(manifest)
    empty = report["references"]["videodb"]["systems"]["argus_astra"]["raw@0.75"]["aggregate"]
    assert empty["complete_cohort"] is True
    assert empty["pooled_micro"]["segmentation"]["f1"] == 0
    assert empty["pooled_micro"]["semantic"]["f1"] == 0
    assert empty["pooled_micro"]["caption_accuracy_on_temporal_matches"] is None
    assert empty["unavailable"] == {}


def test_reference_drift_fails_before_scoring(cohort):
    qualification, gold, _, _ = cohort
    manifest = prepare(qualification, {"videodb": gold}, {}, qualification)
    gold.write_text("{}")
    with pytest.raises(ValueError, match="Artifact changed"):
        evaluate(manifest)


def test_captioner_cannot_claim_changed_segmentation(cohort):
    qualification, gold, flora, _ = cohort
    doc = json.loads(flora.read_text())
    doc["segments"][0]["start_s"] = 1
    put(flora, doc)
    manifest = prepare(qualification, {"videodb": gold}, {}, qualification)
    with pytest.raises(ValueError, match="fixed-interval contract"):
        evaluate(manifest)


def test_bench_run_import_binds_same_video_and_retains_model(cohort, tmp_path):
    qualification, gold, _, video = cohort
    run = tmp_path / "run"
    put(run / "run.json", {"episodes": ["homer_33"], "model": "test/astra"})
    put(run / "out" / "homer_33.json", {"id": "homer_33", "events": [
        {"start_s": 0, "end_s": 10, "label": "Slice onion"}]})
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "homer_33.mp4").write_bytes(b"other-video")
    with pytest.raises(ValueError, match="Baseline video differs"):
        prepare(qualification, {"videodb": gold}, {"argus_astra": run}, raw)
    (raw / "homer_33.mp4").write_bytes(video.read_bytes())
    manifest = prepare(qualification, {"videodb": gold}, {"argus_astra": run}, raw)
    assert manifest["systems"]["argus_astra"]["episodes"]["homer_33"]["model"] == "test/astra"
    report, _ = evaluate(manifest)
    assert report["video_bindings"]["argus_astra"]["homer_33"] == "retained_source_file"
    assert any("Historical outputs" in note and "argus_astra" in note for note in report["limitations"])
    assert report["references"]["videodb"]["systems"]["argus_astra"]["raw@0.5"]["aggregate"]["complete_cohort"]


def test_micro_episode_and_source_averages_stay_distinct():
    def row(predicted, gold, matched):
        return {"status": "available", **tally([{"predicted": predicted, "gold": gold, "matched": matched,
                                                "accepted": matched, "unknown": 0}])}
    rows = {"homer_1": row(10, 10, 0), "galaxea_1": row(1, 1, 1), "galaxea_2": row(1, 1, 1)}
    result = aggregate(rows, list(rows))
    assert result["pooled_micro"]["semantic"]["f1"] == 0.1667
    assert result["episode_macro"]["semantic"] == 0.6667
    assert result["source_equal_macro"]["semantic"] == 0.5
    rows["homer_1"] = {"status": "missing_prediction"}
    result = aggregate(rows, list(rows))
    assert result["pooled_micro"] is None
    assert result["source_equal_macro"] is None
    assert result["observed_pooled_micro"]["semantic"]["f1"] == 1


def test_native_tempseg_scores_only_boundaries_without_fake_labels(cohort, tmp_path):
    qualification, gold, _, video = cohort
    run = tmp_path / "tempseg"
    put(run / "run.json", {"episodes": ["homer_33"], "model": "flora/tempseg"})
    put(run / "out" / "homer_33.json", {"id": "homer_33", "video_sha256": artifact(video)["sha256"],
        "segments": [{"start_s": 0, "end_s": 10}]})
    manifest = prepare(qualification, {"videodb": gold}, {}, tmp_path / "absent-raw",
                       segmentation_only={"flora_tempseg": run})
    report, pending = evaluate(manifest)
    assert not any("does not evaluate Flora TempSeg" in note for note in report["limitations"])
    assert any("imported native segmentation" in note for note in report["limitations"])
    assert report["video_bindings"]["flora_tempseg"]["homer_33"] == "output_digest"
    assert report["video_bindings"]["argus"]["homer_33"] == "qualification_manifest"
    assert any("Output-recorded" in note and "flora_tempseg" in note for note in report["limitations"])
    scored = report["references"]["videodb"]["systems"]["flora_tempseg"]["raw@0.75"]
    row = scored["per_episode"]["homer_33"]
    assert row["segmentation"]["f1"] == 1
    assert row["semantic"] is None and row["semantic_bounds"] is None
    assert row["caption_available"] is False
    assert scored["aggregate"]["pooled_micro"]["semantic"] is None
    assert len(pending) == 2  # only the two labeled systems require judges
