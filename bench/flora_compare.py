"""Offline, paired WGO evaluation of retained Argus and Flora outputs.

prepare --qualification DIR --gold videodb=FILE [--gold macrodata=FILE]
        [--baseline argus_astra=RUN_DIR] [--baseline-videos DIR] --out manifest.json
score manifest.json [--judge-cache FILE] --out report.json [--pending-judgments FILE]

No inference is performed. Missing judge verdicts are unknown, never rejected.
The three qualification clips are development evidence, not a SOTA benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from bench import wgo

SCHEMA = "argus.flora.comparison/v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": digest(path)}


def checked(ref: dict) -> Path:
    path = Path(ref["path"])
    if digest(path) != ref["sha256"]:
        raise ValueError(f"Artifact changed since manifest preparation: {path}")
    return path


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A comparison manifest/report is a retained snapshot, not a mutable latest pointer.
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def assignments(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or name in result:
            raise ValueError(f"Expected unique NAME=PATH, got {value!r}")
        result[name] = Path(path)
    return result


def gold_rows(path: Path) -> dict:
    doc = read(path)
    if isinstance(doc, list):
        return {r["id"]: r for r in doc}
    if "episodes" in doc:
        return {r["id"]: r for r in doc["episodes"]}
    return doc


def segments(doc: dict, duration: float, *, gold: bool = False, labels: bool = True) -> list[dict]:
    rows = doc["segments"] if gold or "segments" in doc else doc["events"]
    result = []
    for row in rows:
        a, b = (row["start_sec"], row["end_sec"]) if gold else (row["start_s"], row["end_s"])
        label = row["subtask"] if gold else row.get("caption", row.get("label"))
        if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float))
                or not isinstance(b, (int, float)) or not math.isfinite(a) or not math.isfinite(b)
                or not 0 <= a < b <= duration + 0.05):
            raise ValueError(f"Invalid segment bounds: {a}, {b}, duration={duration}")
        if labels and (not isinstance(label, str) or not label.strip()):
            raise ValueError("Missing segment label")
        if row.get("status", "complete") != "complete":
            raise ValueError("Incomplete Flora caption")
        if result and a < result[-1]["start_s"]:
            raise ValueError("Segments must be chronological")
        result.append({"start_s": a, "end_s": b, "label": label})
    return result


def prepare(qualification: Path, gold: dict[str, Path], baselines: dict[str, Path],
            baseline_videos: Path, flora_results: Path | None = None, judge: str | None = None,
            segmentation_only: dict[str, Path] | None = None) -> dict:
    segmentation_only = segmentation_only or {}
    if baselines.keys() & segmentation_only.keys():
        raise ValueError("A system cannot be both a labeled and segmentation-only import")
    baselines = {**baselines, **segmentation_only}
    inputs = qualification / "qualification-inputs.json"
    cohort = read(inputs)["episodes"]
    if len({ep["id"] for ep in cohort}) != len(cohort) or not cohort:
        raise ValueError("Cohort must contain unique episodes")
    refs = {name: {"status": "missing", "reason": "reference not supplied"}
            for name in ("videodb", "macrodata")}
    for name, path in gold.items():
        rows = gold_rows(path)
        document = read(path)
        refs[name] = {"status": "available", "artifact": artifact(path),
                      "total_episodes": len(rows),
                      "total_segments": sum(len(r["segments"]) for r in rows.values()),
                      "source_provenance": {key: document[key] for key in
                          ("dataset", "upstream_revision", "selection") if isinstance(document, dict) and key in document},
                      "judge": judge or wgo.JUDGES.get(name, wgo.JUDGES["macrodata"])}
    systems = {name: {"episodes": {}, "provenance": {}}
               for name in ("argus", "argus_flora_vllm", "argus_astra", "argus_sol61")}
    reserved = {"argus", "argus_flora_vllm"}
    if reserved & baselines.keys():
        raise ValueError("Baseline names cannot replace the retained Argus/Flora systems")
    for name, directory in baselines.items():
        systems[name] = {"episodes": {}, "segmentation_only": name in segmentation_only,
                         "provenance": {"run": artifact(directory / "run.json")}}
    episodes = []
    results = flora_results or qualification / "caption-seq8-v3.artifacts" / "results"
    for ep in cohort:
        eid = ep["id"]
        video = artifact(Path(ep["video"]))
        if video["sha256"] != ep["video_sha256"]:
            raise ValueError(f"Qualification video digest mismatch: {eid}")
        item = {key: ep[key] for key in ("id", "split", "task", "duration_s")}
        item["video"] = video
        episodes.append(item)
        argus_path = Path(ep["events"])
        if digest(argus_path) != ep["events_sha256"]:
            raise ValueError(f"Qualification events digest mismatch: {eid}")
        argus_doc = read(argus_path)
        if "video_sha256" in argus_doc and argus_doc["video_sha256"] != video["sha256"]:
            raise ValueError(f"Argus output video digest mismatch: {eid}")
        systems["argus"]["episodes"][eid] = {"artifact": artifact(argus_path), "model": ep["argus_model"],
            "video_binding": "output_digest" if "video_sha256" in argus_doc else "qualification_manifest"}
        flora_path = results / f"{eid}.caption.raw.json"
        if flora_path.exists():
            doc = read(flora_path)
            if doc.get("input_sha256", {}).get("main") != video["sha256"]:
                raise ValueError(f"Flora source video mismatch: {eid}")
            systems["argus_flora_vllm"]["episodes"][eid] = {"artifact": artifact(flora_path),
                "model": doc.get("model_revisions"), "interval_source": "argus", "video_binding": "output_digest"}
        for name, directory in baselines.items():
            path = directory / "out" / f"{eid}.json"
            if not path.exists():
                continue
            meta = read(directory / "run.json")
            if eid not in meta["episodes"]:
                raise ValueError(f"Baseline output not declared in run: {name}/{eid}")
            output = read(path)
            if "video_sha256" in output:
                if output["video_sha256"] != video["sha256"]:
                    raise ValueError(f"Baseline video differs from cohort: {name}/{eid}")
                source_video = video
            else:
                source_video = artifact(baseline_videos / f"{eid}.mp4")
                if source_video["sha256"] != video["sha256"]:
                    raise ValueError(f"Baseline video differs from cohort: {name}/{eid}")
            systems[name]["episodes"][eid] = {"artifact": artifact(path), "model": meta["model"],
                "source_video": source_video,
                "video_binding": "output_digest" if "video_sha256" in output else "retained_source_file"}
    binding_notes = []
    for binding, note in (
        ("output_digest", "Output-recorded video digests match the cohort for"),
        ("qualification_manifest", "Outputs without video digests rely on the pinned qualification manifest for"),
        ("retained_source_file", "Historical outputs without video digests rely on matching retained source files for"),
    ):
        names = sorted(name for name, system in systems.items()
                       if any(ep["video_binding"] == binding for ep in system["episodes"].values()))
        if names:
            binding_notes.append(f"{note}: {', '.join(names)}. See per-episode video_bindings.")
    return {"schema": SCHEMA, "qualification": artifact(inputs), "episodes": episodes,
            "references": refs, "systems": systems,
            "limitations": ["Development cohort; no held-out or SOTA claim.",
                             ("Fixed-interval Flora captions and imported native segmentation are separate systems."
                              if segmentation_only else "Flora relabels Argus intervals; this does not evaluate Flora TempSeg."),
                             "Semantic verdicts use the existing WGO text-only judge protocol.", *binding_notes]}


def judge_key(model: str, task: str, ref: str, pred: str) -> str:
    return hashlib.sha256(json.dumps([model, task, ref, pred]).encode()).hexdigest()


def tally(rows: list[dict]) -> dict:
    counts = {key: sum(row[key] for row in rows)
              for key in ("predicted", "gold", "matched", "accepted", "unknown")}
    p, g, m, a, u = (counts[k] for k in ("predicted", "gold", "matched", "accepted", "unknown"))
    captions = all(row.get("caption_available", True) for row in rows)
    return {**counts, "segmentation": wgo._f1(m, p, g),
            "caption_available": captions,
            "semantic": wgo._f1(a, p, g) if captions and not u else None,
            "semantic_bounds": {"lower": wgo._f1(a, p, g), "upper": wgo._f1(a + u, p, g)} if captions else None,
            "caption_accuracy_on_temporal_matches": round(a / m, 4) if captions and m and not u else None}


def aggregate(rows: dict[str, dict], cohort: list[str]) -> dict:
    present = {eid: row for eid, row in rows.items() if row["status"] == "available"}
    missing = {eid: row["status"] for eid, row in rows.items() if row["status"] != "available"}
    result = {"evaluated": len(present), "expected": len(cohort), "unavailable": missing,
              "complete_cohort": not missing, "pooled_micro": None,
              "observed_pooled_micro": None, "episode_macro": None, "source_equal_macro": None,
              "per_source": {}}
    if not present:
        return result
    pooled = tally(list(present.values()))
    result["observed_pooled_micro"] = pooled
    for source in sorted({wgo.source(eid) for eid in cohort}):
        subset = [row for eid, row in present.items() if wgo.source(eid) == source]
        result["per_source"][source] = tally(subset) if subset else None
    if missing:
        return result
    result["pooled_micro"] = pooled
    for name, values in (("episode_macro", list(present.values())),
                         ("source_equal_macro", list(result["per_source"].values()))):
        result[name] = {metric: round(sum(r[metric]["f1"] for r in values) / len(values), 4)
                        if all(r[metric] is not None for r in values) else None
                        for metric in ("segmentation", "semantic")}
    return result


def evaluate(manifest: dict, cache: dict | None = None) -> tuple[dict, list[dict]]:
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unsupported comparison manifest")
    cache = {} if cache is None else cache
    if any(type(value) is not bool for value in cache.values()):
        raise ValueError("Judge cache must contain boolean verdicts")
    checked(manifest["qualification"])
    episodes = manifest["episodes"]
    for ep in episodes:
        checked(ep["video"])
    loaded = {}
    for system, entry in manifest["systems"].items():
        for provenance in entry["provenance"].values():
            checked(provenance)
        loaded[system] = {}
        for ep in episodes:
            eid = ep["id"]
            item = entry["episodes"].get(eid)
            if item is None:
                continue
            if "source_video" in item:
                checked(item["source_video"])
            doc = read(checked(item["artifact"]))
            if doc.get("id") != eid:
                raise ValueError(f"Prediction episode identity mismatch: {system}/{eid}")
            loaded[system][eid] = segments(doc, ep["duration_s"], labels=not entry.get("segmentation_only", False))
    fixed = {}
    for ep in episodes:
        eid = ep["id"]
        a, b = loaded.get("argus", {}).get(eid), loaded.get("argus_flora_vllm", {}).get(eid)
        bounds = lambda rows: [(r["start_s"], r["end_s"]) for r in rows]
        fixed[eid] = bounds(a) == bounds(b) if a is not None and b is not None else None
        if b is not None and fixed[eid] is not True:
            raise ValueError(f"Flora fixed-interval contract differs from Argus: {eid}")
    report = {"schema": SCHEMA, "limitations": manifest["limitations"], "fixed_argus_flora_intervals": fixed,
              "video_bindings": {system: {eid: ep.get("video_binding", "not_recorded_in_manifest")
                  for eid, ep in entry["episodes"].items()} for system, entry in manifest["systems"].items()},
              "references": {}, "reward": {"status": "unmeasured", "reason": "No matched reward ground truth supplied"}}
    pending = {}
    for reference, info in manifest["references"].items():
        if info["status"] != "available":
            report["references"][reference] = info
            continue
        gold = gold_rows(checked(info["artifact"]))
        result = {"status": "available", "artifact": info["artifact"], "judge": info["judge"],
                  "total_reference_segments": info["total_segments"],
                  "source_provenance": info.get("source_provenance", {}), "systems": {}}
        for system, docs in loaded.items():
            captions = not manifest["systems"][system].get("segmentation_only", False)
            variants = {}
            for snap in (False, True):
                for threshold in (0.5, 0.75):
                    key = f"{'snap' if snap else 'raw'}@{threshold}"
                    rows = {}
                    for ep in episodes:
                        eid = ep["id"]
                        if eid not in gold or not gold[eid].get("segments"):
                            rows[eid] = {"status": "missing_or_empty_reference"}
                            continue
                        if eid not in docs:
                            rows[eid] = {"status": "missing_prediction"}
                            continue
                        prediction = docs[eid]
                        target = segments(gold[eid], ep["duration_s"], gold=True)
                        p = [(r["start_s"], r["end_s"]) for r in prediction]
                        g = [(r["start_s"], r["end_s"]) for r in target]
                        pairs = wgo.match_in_order(wgo.snapped(p, g) if snap else p, g, threshold)
                        accepted = unknown = 0
                        evidence = []
                        for i, j in pairs:
                            if not captions:
                                evidence.append({"prediction": i, "reference": j, "verdict": None,
                                                 "reason": "segmentation-only output has no caption"})
                                continue
                            args = [info["judge"], gold[eid]["instruction"], target[j]["label"], prediction[i]["label"]]
                            verdict_key = judge_key(*args)
                            verdict = cache.get(verdict_key)
                            accepted += verdict is True
                            unknown += verdict is None
                            evidence.append({"prediction": i, "reference": j, "judge_key": verdict_key, "verdict": verdict})
                            if verdict is None:
                                pending[verdict_key] = {"key": verdict_key, "model": args[0], "task": args[1],
                                    "reference": args[2], "prediction": args[3],
                                    "prompt": wgo.JUDGE_PROMPT.format(task=args[1], ref=args[2], pred=args[3])}
                        row = tally([{"predicted": len(p), "gold": len(g), "matched": len(pairs),
                                      "accepted": accepted, "unknown": unknown, "caption_available": captions}])
                        rows[eid] = {"status": "available", **row, "pairs": evidence}
                    variants[key] = {"per_episode": rows, "aggregate": aggregate(rows, [ep["id"] for ep in episodes])}
            result["systems"][system] = variants
        report["references"][reference] = result
    report["pending_judgments"] = len(pending)
    return report, list(pending.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--qualification", type=Path, required=True)
    prep.add_argument("--gold", action="append", default=[], help="Reference name=JSON file; repeat for revised/original")
    prep.add_argument("--baseline", action="append", default=[], help="System name=bench.wgo run directory")
    prep.add_argument("--segmentation-only", action="append", default=[], help="System name=run directory with unlabeled intervals")
    prep.add_argument("--baseline-videos", type=Path, default=wgo.RAW)
    prep.add_argument("--flora-results", type=Path)
    prep.add_argument("--judge", help="Override judge for all references; verdict cache remains model-keyed")
    prep.add_argument("--out", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("manifest", type=Path)
    score.add_argument("--judge-cache", type=Path)
    score.add_argument("--pending-judgments", type=Path)
    score.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        write(args.out, prepare(args.qualification, assignments(args.gold), assignments(args.baseline),
                                args.baseline_videos, args.flora_results, args.judge,
                                assignments(args.segmentation_only)))
    else:
        report, pending = evaluate(read(args.manifest), read(args.judge_cache) if args.judge_cache else None)
        report["manifest"] = artifact(args.manifest)
        report["judge_cache"] = artifact(args.judge_cache) if args.judge_cache else None
        write(args.out, report)
        if args.pending_judgments:
            write(args.pending_judgments, pending)
    print(args.out)


if __name__ == "__main__":
    main()
