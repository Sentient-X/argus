"""Native, unlabelled Flora TempSeg baseline with retained generation evidence.

Keep native overlapping events and stitching provenance. This module neither
names actions nor routes intervals through Argus's overlap-removing clean().
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from fractions import Fraction
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from . import flora
from .flora_journal import GenerationJournal, input_digest
from .flora_worker import _caption_generate, _snapshot, _write_json

TEMPSEG_MODELS = {
    "tempseg": {"repo": "nvidia/Flora-TempSeg",
                "revision": "114a1076f70eb53f4a58ac045505db3087e79347",
                "sha256": "1d80fe544e4cd1a7ff6b0ea4be71e988972408dab81d4c8af72e3a628739d6c3"},
    "tempseg_base": {"repo": "Qwen/Qwen3.5-9B",
                     "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"},
}
MAX_GENERATIONS = 64


def _runtime(request: dict):
    for name in ("FLORA_TEMPSEG_REPO", "FLORA_CAPTIONER_REPO"):
        if os.environ.get(name):
            raise flora.FloraError(f"Unset {name}; the request pins model repositories")
    existing = sys.modules.get("batch_utils")
    if existing is None:
        _snapshot(request, "captioner")
    elif Path(existing.__file__).resolve().parent.parent.name != flora.MODELS["captioner"]["revision"]:
        raise flora.FloraError("Another Flora runtime is already imported")
    api = importlib.import_module("batch_utils.api")
    config = importlib.import_module("batch_utils.config")
    for name, expected in TEMPSEG_MODELS.items():
        if {key: config.MODELS[name].get(key) for key in expected} != expected:
            raise flora.FloraError("Native TempSeg model pins differ from the audited release")
    return api, config


def _prepare_native(requests: list[dict]) -> list[dict]:
    if not isinstance(requests, list) or not 1 <= len(requests) <= 16:
        raise flora.FloraError("TempSeg requires 1..16 episodes")
    ids = set()
    identities = set()
    for request in requests:
        identifier = request.get("id")
        if (not isinstance(identifier, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identifier)
                or identifier in ids):
            raise flora.FloraError("TempSeg requires unique local episode IDs")
        ids.add(identifier)
        if request.get("use") != "noncommercial_research":
            raise flora.FloraError("TempSeg requires noncommercial research use")
        if request.get("models") != flora.MODELS or request.get("tempseg_models") != TEMPSEG_MODELS:
            raise flora.FloraError("TempSeg model pins must match the audited release")
        paths, hashes = request.get("videos"), request.get("video_sha256")
        if (not isinstance(paths, list) or len(paths) not in (1, 3)
                or not isinstance(hashes, list) or len(paths) != len(hashes)):
            raise flora.FloraError("TempSeg needs one or three native videos with hashes")
        for name, expected in zip(paths, hashes, strict=True):
            if not isinstance(name, str) or not Path(name).is_absolute():
                raise flora.FloraError("TempSeg videos must have absolute paths")
            if flora.digest(Path(name)) != expected:
                raise flora.FloraError("Native video changed since preparation")
        flora.number(request.get("duration_s"))
        identity = input_digest({**{key: request[key] for key in
                                 ("video_sha256", "task", "profile", "models", "tempseg_models")},
                                 "yam32": request.get("yam32", False)})
        if identity in identities:
            raise flora.FloraError("Duplicate TempSeg inputs; reuse completed evidence")
        identities.add(identity)
    api, config = _runtime(requests[0])
    prepared = []
    for request in requests:
        paths = request["videos"]
        yam32 = request.get("yam32", False)
        videos, profile = api.DataEnginePipeline._inputs(
            paths[0], request["profile"], paths[1] if len(paths) == 3 else None,
            paths[2] if len(paths) == 3 else None, yam32, request["task"],
        )
        if not math.isclose(videos["main"].duration_s, request["duration_s"], abs_tol=0.001, rel_tol=0):
            raise flora.FloraError("TempSeg source clock differs from the saved request")
        if [video.sha256 for video in videos.values()] != request["video_sha256"]:
            raise flora.FloraError("Native video changed during media validation")
        prepared.append({"videos": videos, "profile": profile,
                         "pool": "yam32" if yam32 else config.PROFILE_POOLS[profile]})
    return prepared


def preflight(requests: list[dict]) -> list[dict]:
    """Return JSON-only CPU media receipts; never load model weights.

    The pinned small runtime may be fetched on first use. The native media probe
    supports Human VFR and verifies exact synchronization of three-view inputs.
    """
    prepared = _prepare_native(requests)
    return [{"id": request["id"], "status": "inputs_validated", "new_model_calls": 0,
             "profile": item["profile"], "pool": item["pool"], "model_revisions": TEMPSEG_MODELS,
             "cameras": {role: {"sha256": video.sha256, "source_fps": str(video.rate),
                                 "frames": len(video.pts_us), "size": list(video.size),
                                 "constant_frame_rate": video.cfr, "duration_s": video.duration_s}
                         for role, video in item["videos"].items()}}
            for request, item in zip(requests, prepared, strict=True)]


def _cached_session(session_type):
    class CachedSession(session_type):
        def load(self, kind):
            if kind != "tempseg":
                raise flora.FloraError("This baseline only loads TempSeg")
            if self.model is None:
                return super().load(kind)
            return self.model, self.proc
    return CachedSession()


def _json_value(value):
    """Retain all native metadata; sets become sorted JSON arrays, never strings."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Native metadata keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value),
                      key=lambda item: json.dumps(item, sort_keys=True, allow_nan=False))
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    raise TypeError(f"Unsupported native metadata type: {type(value).__name__}")


def _validate_segments(segments, duration: float) -> None:
    if not isinstance(segments, list):
        raise flora.FloraError("TempSeg did not return an interval list")
    previous = -1.0
    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("metadata"), dict):
            raise flora.FloraError("TempSeg interval lacks native provenance")
        start, end = flora.number(segment.get("start_s")), flora.number(segment.get("end_s"))
        if not 0 <= start < end <= duration or start < previous:
            raise flora.FloraError("TempSeg intervals must be ordered and within the source clock")
        previous = start


def validate_replay(requests: list[dict], replay: dict | None, replay_root: Path | None) -> dict | None:
    """Verify retained tokens and source identity on CPU before allocating a GPU."""
    if replay is None and replay_root is None:
        return None
    if not isinstance(replay, dict) or replay_root is None:
        raise flora.FloraError("TempSeg replay needs both a descriptor and its source directory")
    if (set(replay) != {"run_id", "manifest_sha256", "reconciliation_sha256", "episodes"}
            or not isinstance(replay["run_id"], str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", replay["run_id"])):
        raise flora.FloraError("Invalid TempSeg replay descriptor")
    root = Path(replay_root).resolve()

    def read(relative):
        path = root / relative
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise flora.FloraError("TempSeg replay source is missing or escapes its directory")
        return json.loads(path.read_text()), flora.digest(path)

    manifest, manifest_hash = read("manifest.json")
    if (manifest_hash != replay["manifest_sha256"] or manifest.get("run_id") != replay["run_id"]
            or manifest.get("stage") != "segment-native"
            or manifest.get("use") != "noncommercial_research"):
        raise flora.FloraError("TempSeg replay manifest identity differs")
    reconciliation, reconciliation_hash = read("continuation-reconciliation.json")
    if (reconciliation_hash != replay["reconciliation_sha256"]
            or reconciliation.get("confirmed_stopped") is not True
            or reconciliation.get("run_id") != replay["run_id"]
            or reconciliation.get("manifest_sha256") != manifest_hash
            or not isinstance(reconciliation.get("reason"), str) or not reconciliation["reason"].strip()):
        raise flora.FloraError("TempSeg continuation requires explicit source stop reconciliation")
    try:
        observed = datetime.fromisoformat(reconciliation["observed_at_utc"])
        if observed.utcoffset() is None or observed.utcoffset().total_seconds() != 0:
            raise ValueError("Expected a UTC timestamp")
    except (KeyError, TypeError, ValueError) as exc:
        raise flora.FloraError("TempSeg source stop reconciliation needs a UTC observation time") from exc
    evidence = reconciliation.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise flora.FloraError("TempSeg source stop reconciliation needs retained terminal evidence")
    source_files = {"manifest.json": manifest_hash, "continuation-reconciliation.json": reconciliation_hash}
    for item in evidence:
        if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
                or not isinstance(item["path"], str) or Path(item["path"]).is_absolute()
                or ".." in Path(item["path"]).parts):
            raise flora.FloraError("Invalid TempSeg terminal evidence path")
        path = root / item["path"]
        if (not path.resolve().is_relative_to(root) or not path.is_file()
                or flora.digest(path) != item["sha256"]):
            raise flora.FloraError("TempSeg terminal evidence is missing or changed")
        source_files[item["path"]] = item["sha256"]
    binding = input_digest(manifest)
    source_requests = {request["id"]: request for request in manifest["requests"]}
    current_requests = {request["id"]: request for request in requests}
    episodes = replay["episodes"]
    if not isinstance(episodes, dict) or not episodes or not set(episodes) <= current_requests.keys():
        raise flora.FloraError("TempSeg replay episode selection differs")
    # Scan the whole journal: an uncertain paid attempt never permits continuation.
    intents = sorted((root / "generations").glob("*.intent.json"))
    results = sorted((root / "generations").glob("*.result.json"))
    if (not intents or [path.name for path in intents] != [f"{i:06d}.intent.json" for i in range(len(intents))]
            or [path.name for path in results] != [f"{i:06d}.result.json" for i in range(len(intents))]):
        raise flora.FloraError("TempSeg replay journal has an unsettled, orphaned or missing generation")
    generations = []
    for index in range(len(intents)):
        intent, intent_hash = read(f"generations/{index:06d}.intent.json")
        result, result_hash = read(f"generations/{index:06d}.result.json")
        if (set(intent) != {"request_sha256", "input_sha256"}
                or intent["request_sha256"] != binding or result.get("binding") != intent
                or not isinstance(intent["input_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", intent["input_sha256"])):
            raise flora.FloraError("TempSeg replay generation binding differs")
        value = result.get("value", {})
        tokens = value.get("tokens")
        if (value.get("dtype") != "int64" or not isinstance(tokens, list) or len(tokens) != 1
                or not isinstance(tokens[0], list) or not tokens[0]
                or value.get("shape") != [1, len(tokens[0])]
                or any(type(token) is not int or token < 0 or token >= 2**63 for token in tokens[0])):
            raise flora.FloraError("TempSeg replay token evidence is malformed")
        source_files[f"generations/{index:06d}.intent.json"] = intent_hash
        source_files[f"generations/{index:06d}.result.json"] = result_hash
        generations.append({"index": index, "input_sha256": intent["input_sha256"],
                            "intent_sha256": intent_hash, "result_sha256": result_hash})
    selected = set()
    for identifier, entries in episodes.items():
        current, source = current_requests[identifier], source_requests.get(identifier)
        if (source is None or current.get("models") != flora.MODELS
                or current.get("tempseg_models") != TEMPSEG_MODELS
                or {key: value for key, value in current.items() if key != "videos"}
                != {key: value for key, value in source.items() if key != "videos"}):
            raise flora.FloraError("TempSeg replay source, task, clocks or model pins differ")
        if not isinstance(entries, list) or not entries:
            raise flora.FloraError("TempSeg replay requires a nonempty generation prefix")
        previous = -1
        for entry in entries:
            if (not isinstance(entry, dict) or set(entry) != {"index", "intent_sha256", "result_sha256"}
                    or type(entry["index"]) is not int or not 0 <= entry["index"] < len(generations)
                    or entry["index"] <= previous or entry["index"] in selected):
                raise flora.FloraError("TempSeg replay generation selection is invalid")
            index = entry["index"]
            if any(entry[key] != generations[index][key] for key in entry):
                raise flora.FloraError("TempSeg replay generation hashes differ")
            previous = index
            selected.add(index)
    return {"descriptor": replay, "source_request_sha256": binding, "generations": generations,
            "source_implementation_sha256": manifest.get("implementation_sha256", {}),
            "source_files": source_files}


class _ReplayAbort(BaseException):
    """Bypass upstream adaptive fallback: a replay mismatch must stop the batch."""


class _Replay:
    def __init__(self, receipt, root, output, journal):
        self.receipt, self.root, self.output, self.journal = receipt, Path(root), output, journal
        self.identifier, self.cursor, self.used = None, 0, []

    def start(self, identifier):
        self.identifier, self.cursor = identifier, 0

    def finish(self):
        if self.cursor != len(self.receipt["descriptor"]["episodes"].get(self.identifier, [])):
            raise _ReplayAbort("Native TempSeg did not consume every declared replay generation")

    def _unchanged(self):
        expected = self.receipt["source_files"]
        names = {f"generations/{path.name}" for path in (self.root / "generations").glob("*.json")}
        if names != {name for name in expected if name.startswith("generations/")}:
            raise _ReplayAbort("TempSeg source journal changed after stop reconciliation")
        for name, digest in expected.items():
            path = self.root / name
            if (not path.resolve().is_relative_to(self.root.resolve()) or not path.is_file()
                    or flora.digest(path) != digest):
                raise _ReplayAbort("TempSeg replay evidence changed after admission")

    def take(self, kwargs):
        self._unchanged()
        digest = input_digest(kwargs)
        entries = self.receipt["descriptor"]["episodes"].get(self.identifier, [])
        if self.cursor == len(entries):
            if any(item["input_sha256"] == digest for item in self.receipt["generations"]):
                raise _ReplayAbort("This generation was already paid for; declare its replay instead")
            return None
        entry = entries[self.cursor]
        expected = self.receipt["generations"][entry["index"]]
        if digest != expected["input_sha256"]:
            raise _ReplayAbort("TempSeg replay input digest differs; refusing new inference")
        for kind in ("intent", "result"):
            path = self.root / "generations" / f"{entry['index']:06d}.{kind}.json"
            if (not path.resolve().is_relative_to(self.root.resolve()) or not path.is_file()
                    or flora.digest(path) != entry[f"{kind}_sha256"]):
                raise _ReplayAbort("TempSeg replay evidence changed after admission")
        value = json.loads(path.read_text())["value"]
        self.cursor += 1
        self.used.append({"episode_id": self.identifier, "generation_index": self.journal.index - 1,
                          "source_run_id": self.receipt["descriptor"]["run_id"], **expected})
        _write_json(self.output / f"{self.journal.index - 1:06d}.replay.json", {"admission": self.receipt, "used": self.used[-1]})
        self.journal.durable_commit()
        return value

    def model(self, original):
        replay = self

        class ModelProxy:
            def __getattr__(self, name):
                return getattr(original, name)

            def generate(self, **kwargs):
                saved = replay.take(kwargs)
                if saved is None:
                    return original.generate(**kwargs)
                import torch
                prompt = kwargs["input_ids"]
                tokens = torch.tensor(saved["tokens"], dtype=torch.int64, device=prompt.device)
                if not torch.equal(tokens[:, :prompt.shape[1]], prompt):
                    raise _ReplayAbort("Retained TempSeg tokens do not contain the exact input prompt")
                return tokens
        return ModelProxy()


def _bounded_generate(original, journal: GenerationJournal, replay=None):
    retained = _caption_generate(original, journal)

    def generate(model, processor, inputs, budget):
        if journal.index >= MAX_GENERATIONS:
            raise flora.FloraError("TempSeg reached its 64-generation batch limit; retained outputs remain available")
        if budget != 1024:
            raise flora.FloraError("Native TempSeg output budget changed")
        return retained(replay.model(model) if replay else model, processor, inputs, budget)
    return generate


def run(requests: list[dict], output_dir: Path, journal: GenerationJournal, *,
        replay: dict | None = None, replay_root: Path | None = None) -> list[dict]:
    """Run native adaptive segmentation with one verified model load per batch.

    A partial batch is retained and refused on redispatch. Reconcile its raw
    tokens and interval files explicitly; never overwrite paid evidence. The
    hosting process additionally enforces the whole-run wall-clock deadline.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise flora.FloraError("TempSeg output already contains evidence; reconcile it before redispatch")
    if journal.index != 0:
        raise flora.FloraError("TempSeg requires a fresh generation journal cursor")
    replay_receipt = validate_replay(requests, replay, replay_root)
    prepared = _prepare_native(requests)
    pipeline = importlib.import_module("batch_utils.pipeline")
    models = importlib.import_module("batch_utils.models")

    output_dir.mkdir(parents=True, exist_ok=True)
    binding = {"requests_sha256": input_digest(requests), "journal_sha256": journal.request_sha256,
               "adapter_sha256": flora.digest(Path(__file__)), "max_generations": MAX_GENERATIONS}
    if replay_receipt is not None:
        binding["replay"] = replay_receipt
    _write_json(output_dir / "batch.json", binding)
    journal.durable_commit()
    session = _cached_session(models.Session)
    original = pipeline.generate
    replayer = _Replay(replay_receipt, replay_root, output_dir, journal) if replay_receipt else None
    pipeline.generate = _bounded_generate(original, journal, replayer)
    results = []
    try:
        for request, item in zip(requests, prepared, strict=True):
            if replayer:
                replayer.start(request["id"])
            started = time.monotonic()
            first_generation = journal.index
            segments, audit = pipeline.segment(item["videos"], item["pool"], session,
                                               request["task"], lambda *args, **kwargs: None)
            if replayer:
                replayer.finish()
            record = _json_value({
                "schema": "argus.flora.segmentation/v1", "id": request["id"],
                "segments": segments, "duration_s": item["videos"]["main"].duration_s,
                "task": request["task"], "profile": item["profile"], "pool": item["pool"],
                "camera_roles": list(item["videos"]),
                "input_sha256": {role: video.sha256 for role, video in item["videos"].items()},
                "runtime": request["models"]["captioner"], "model_revisions": TEMPSEG_MODELS,
                "source_clock": "seconds relative to the first native source presentation timestamp",
                "pipeline": {"temporal_windows": audit}, "model_load_audit": session.audit,
                "generation_range": [first_generation, journal.index],
                "replayed_generations": [item for item in replayer.used if item["episode_id"] == request["id"]] if replayer else [],
                "elapsed_s": time.monotonic() - started,
                "metadata_encoding": "sets are sorted arrays; fractions have numerator and denominator",
            })
            # Keep native output before acceptance checks, including malformed intervals.
            _write_json(output_dir / f"{request['id']}.segment.raw.json", record)
            journal.durable_commit()
            _validate_segments(segments, record["duration_s"])
            results.append(record)
        _write_json(output_dir / "receipt.json", {
            "binding": binding, "status": "complete", "generations": journal.index,
            "raw_sha256": {request["id"]: flora.digest(output_dir / f"{request['id']}.segment.raw.json")
                           for request in requests},
        })
        journal.durable_commit()
        return results
    except _ReplayAbort as exc:
        raise flora.FloraError(str(exc)) from exc
    finally:
        pipeline.generate = original
        session.close()
