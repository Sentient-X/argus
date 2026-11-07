"""Offline outcome evaluation of a frozen RoboReward cohort; never invokes models.

Supply retained hosted artifacts/receipt and separate frozen plan, gold and CPU
admission files. Only complete, score-only outcome runs can become quality evidence.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import re

from events import flora
from events.flora_journal import input_digest

SOURCE = ("teetone/RoboReward", "469b9af76e8539e8d2ac553b081307738fd92ca5", "test")


def read(path: Path):
    return json.loads(path.read_text())


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def indexed(rows: list[dict]) -> dict:
    result = {row["id"]: row for row in rows}
    require(bool(rows) and len(result) == len(rows), "Empty or duplicate episode IDs")
    return result


def metrics(pairs: list[tuple[int, int]]) -> dict:
    return {"count": len(pairs),
            "mae": sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else None,
            "accuracy": sum(a == b for a, b in pairs) / len(pairs) if pairs else None}


def evaluate(artifacts: Path, receipt_path: Path, gold_path: Path,
             plan_path: Path, admission_path: Path) -> dict:
    artifacts = artifacts.resolve()
    receipt, gold, plan, admission = map(read, (receipt_path, gold_path, plan_path, admission_path))
    require(receipt.get("status") == "complete", "Hosted run is not complete")
    files = receipt.get("files")
    require(isinstance(files, dict) and bool(files), "Receipt lacks file hashes")
    for name, expected in files.items():
        path = (artifacts / name).resolve()
        require(path.is_relative_to(artifacts) and path.is_file(), "Unsafe or missing artifact path")
        require(flora.digest(path) == expected, f"Artifact changed: {name}")

    def retained(name):
        require(name in files, f"Receipt does not cover {name}")
        return read(artifacts / name)

    manifest = retained("manifest.json")
    manifest_sha = input_digest(manifest)
    require(receipt.get("manifest_sha256") == manifest_sha
            and receipt.get("run_id") == manifest.get("run_id")
            and receipt.get("implementation_sha256") == manifest.get("implementation_sha256"),
            "Hosted manifest identity mismatch")
    require(manifest.get("stage") == "reward-outcome" and manifest.get("reward_rationale") is False
            and manifest.get("use") == "noncommercial_research", "Expected explicit score-only outcome run")
    provenance = manifest.get("benchmark_provenance", {})
    for key, path in (("frozen_plan_sha256", plan_path), ("evaluation_gold_sha256", gold_path),
                      ("cpu_admission_sha256", admission_path)):
        require(provenance.get(key) == flora.digest(path), f"Changed benchmark evidence: {key}")
    require(gold.get("source") == plan.get("source")
            and tuple(gold["source"].get(k) for k in ("repo", "revision", "split")) == SOURCE,
            "Unexpected reward source revision/split")
    require(gold.get("frozen_plan_sha256") == flora.digest(plan_path)
            and admission.get("plan_sha256") == flora.digest(plan_path)
            and gold.get("cpu_admission_sha256") == flora.digest(admission_path), "Broken source lineage")
    truth, frozen, media = (indexed(value["rows"]) for value in (gold, plan, admission))
    require(truth.keys() == frozen.keys() == media.keys(), "Frozen cohort membership changed")
    eligible = set()
    for eid, row in truth.items():
        flora.score(row["ground_truth_score"], 5)
        require(all(row[key] == frozen[eid][key] for key in
                    ("ground_truth_score", "source_row_index", "file_name", "video_url")), "Gold differs from frozen selection")
        require(row["video_sha256"] == media[eid]["sha256"]
                and row["admission_status"] == media[eid]["status"], "Source media/admission changed")
        require(row["admission_status"] in ("admitted", "unsupported"), "Unknown admission status")
        if row["admission_status"] == "admitted":
            eligible.add(eid)
        else:
            require(bool(row.get("admission_error")) and row["admission_error"] == media[eid].get("error"),
                    "Missing admission failure evidence")
    total = len(truth)
    require(gold.get("frozen_count") == provenance.get("frozen_count") == total
            and gold.get("admitted_count") == provenance.get("admitted_count") == len(eligible)
            and gold.get("admission_failures") == provenance.get("admission_failures") == total - len(eligible)
            and gold.get("replacements") == provenance.get("replacements") == 0, "Coverage denominator changed")
    requests = indexed(manifest["requests"])
    require(requests.keys() == eligible, "Run omitted or added admitted episodes")
    for eid, request in requests.items():
        require(request.get("models", {}).get("rewarder") == flora.MODELS["rewarder"], "Reward model revision changed")
        require(request.get("task") == frozen[eid]["task"]
                and request.get("video_sha256") == [truth[eid]["video_sha256"]]
                and len(request.get("videos", [])) == 1
                and request.get("duration_s") == media[eid]["duration_s"], "Episode input identity changed")
        require(not ({"ground_truth_score", "reward", "gpt5_mini_check"} & request.keys()), "Gold leaked into inference request")
    binding = retained("results/batch.json")
    native_receipt = retained("results/receipt.json")
    size = manifest["max_num_seqs"]
    require(type(size) is int and 1 <= size <= 64, "Invalid concurrency")
    require(binding == native_receipt.get("binding")
            and binding.get("with_rationale") is False and binding.get("stages") == ["outcome"]
            and binding.get("requests_sha256") == input_digest(manifest["requests"])
            and binding.get("journal_sha256") == manifest_sha and binding.get("max_num_seqs") == size
            and binding.get("adapter_sha256") == manifest["implementation_sha256"].get("flora_reward_batch.py"),
            "Native receipt mode/input/implementation mismatch")
    require("results/outcome.jsonl" in files
            and native_receipt.get("raw_sha256", {}).get("outcome") == files["results/outcome.jsonl"],
            "Native output hash mismatch")
    rows = [json.loads(line) for line in (artifacts / "results/outcome.jsonl").read_text().splitlines() if line.strip()]
    outputs = indexed(rows)
    require(outputs.keys() == eligible and list(outputs) == list(requests), "Outcome coverage/order changed")
    journal_texts = []
    for i in range(math.ceil(len(requests) / size)):
        intent = retained(f"generations/{i:06d}.intent.json")
        result = retained(f"generations/{i:06d}.result.json")
        require(intent.get("request_sha256") == manifest_sha and result.get("binding") == intent,
                "Generation journal identity mismatch")
        values = result["value"]
        require(len(values) == min(size, len(requests) - i * size)
                and all(len(value) == 1 for value in values), "Generation journal count mismatch")
        journal_texts.extend(value[0]["text"] for value in values)
    require(journal_texts == [row.get("output") for row in rows], "Output differs from paid generation journal")
    pairs = []
    for eid, row in outputs.items():
        flora.validate("outcome", row, requests[eid])
        match = re.fullmatch(r"Score:\s*([1-5])\s*", row["output"].strip())
        count = min(64, max(2, math.floor(media[eid]["frames"] / float(Fraction(media[eid]["fps"])) * 10)))
        require(match is not None and int(match[1]) == row["score"] and row.get("rationale") is None
                and row.get("sampled_frames") == max(2, count - count % 2), "Output violates native score-only protocol")
        pairs.append((truth[eid]["ground_truth_score"], row["score"]))
    result = metrics(pairs)
    error_sum = sum(abs(a - b) for a, b in pairs)
    missing_max_error = sum(max(row["ground_truth_score"] - 1, 5 - row["ground_truth_score"])
                            for eid, row in truth.items() if eid not in eligible)
    return {"schema": "argus.flora.reward-evaluation/v1", "run_id": manifest["run_id"],
            "provenance": {"receipt_sha256": flora.digest(receipt_path), **provenance,
                           "evaluator_sha256": flora.digest(Path(__file__)), "model": flora.MODELS["rewarder"],
                           "source": gold["source"]},
            "coverage": {"frozen": total, "admitted": len(eligible), "answered": len(pairs),
                         "admission_failures": total - len(eligible), "fraction": len(pairs) / total},
            "selection": {"kind": "score-stratified frozen cohort",
                          "frozen_per_label": {str(i): sum(r["ground_truth_score"] == i for r in truth.values())
                                               for i in range(1, 6)},
                          "admitted_per_label": {str(i): sum(truth[eid]["ground_truth_score"] == i for eid in eligible)
                                                 for i in range(1, 6)}},
            "answered_metrics": {**result, "mae_times_100": result["mae"] * 100},
            "per_label": {str(i): metrics([p for p in pairs if p[0] == i]) for i in range(1, 6)},
            "confusion_matrix_gold_rows_predicted_columns": [[sum(a == i and b == j for a, b in pairs)
                                                              for j in range(1, 6)] for i in range(1, 6)],
            "frozen_cohort_mae_bounds": [error_sum / total, (error_sum + missing_max_error) / total],
            "episodes": [{"id": eid, "gold": row["ground_truth_score"],
                          "prediction": outputs[eid]["score"] if eid in eligible else None,
                          "admission_status": row["admission_status"],
                          "admission_error": row.get("admission_error")} for eid, row in truth.items()],
            "limitations": ["Score-stratified selection estimates this cohort, not dataset prevalence or its official headline score.",
                            "Answered metrics are conditional on CPU admission; failed admissions remain in the frozen denominator.",
                            "Missing-result bounds are identification bounds, not confidence intervals.",
                            "No population confidence claim: tasks and original trajectories can be correlated.",
                            "Outcome labels do not validate dense rewards, policy ranking, or SOTA."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifacts", "receipt", "gold", "plan", "admission", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.artifacts, args.receipt, args.gold, args.plan, args.admission)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
