"""Execute and retain one immutable, bounded Flora batch on a mounted volume."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from events.flora_journal import GenerationJournal, input_digest

EVIDENCE_ROOT = Path("/evidence")


def caption_batches(manifest: dict) -> list[dict]:
    """A bounded ready backlog; no polling or unbounded GPU idle time."""
    batches = manifest.get("batches")
    if not isinstance(batches, list) or not 1 <= len(batches) <= 8:
        raise ValueError("Caption session needs 1..8 admitted batches")
    flattened = []
    identities = set()
    for batch in batches:
        if not isinstance(batch, dict):
            raise ValueError("Invalid caption batch")
        requests = batch.get("requests")
        if not isinstance(requests, list) or not 1 <= len(requests) <= 16:
            raise ValueError("Each caption batch needs 1..16 episodes")
        ids = [request.get("id") for request in requests]
        if any(not isinstance(eid, str) for eid in ids) or len(set(ids)) != len(ids):
            raise ValueError("Episode IDs must be unique within each batch")
        for request in requests:
            identity = input_digest({key: request[key] for key in
                                     ("video_sha256", "events", "task", "profile", "models")})
            if identity in identities:
                raise ValueError("Duplicate caption inputs in session; reuse completed evidence")
            identities.add(identity)
        flattened.extend(requests)
    if flattened != manifest.get("requests"):
        raise ValueError("Caption session requests differ from its ordered batches")
    return batches


def admitted_cache(descriptor: dict) -> Path:
    path = Path(descriptor["path"])
    if (not path.resolve().is_relative_to((EVIDENCE_ROOT / "prepared").resolve())
            or path.name != "cache.json"
            or hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]):
        raise ValueError("CPU preparation cache differs from admitted evidence")
    return path.parent


def verify_prepared_cache(requests: list[dict], descriptor: dict) -> None:
    """Re-admit retained CPU inputs without rereading unused model weights.

    The GPU worker independently verifies checkpoint contents before its first
    engine load, then validates decoded arrays and parity certificates per batch.
    CPU reuse checks the immutable preparation record and every retained array.
    """
    from events import flora_vllm
    directory = admitted_cache(descriptor)
    record = json.loads((directory / "cache.json").read_text())
    if (record["version"] != 1 or record["requests_sha256"] != input_digest(requests)
            or record["implementation_sha256"] != hashlib.sha256(Path(flora_vllm.__file__).read_bytes()).hexdigest()
            or len(record["episodes"]) != len(requests)):
        raise ValueError("Preparation cache request or implementation binding differs")
    for name, expected in record["files"].items():
        path = directory / name
        if (Path(name).name != name or not name.endswith(".npy") or path.is_symlink()
                or not path.is_file() or flora_vllm._file_sha(path) != expected):
            raise ValueError("Preparation cache array hash differs")
    for episode in record["episodes"]:
        root = Path(episode["root"])
        if (flora_vllm._file_sha(root / "SOURCE_MANIFEST.sha256") != episode["model_manifest_sha256"]
                or flora_vllm._model_file_stats(root) != episode["model_files"]):
            raise ValueError("Preparation cache checkpoint metadata changed")
        certificate = episode["processor_parity"]
        binding = {"prepared_sha256": episode["prepared_sha256"],
                   "implementation_sha256": record["implementation_sha256"],
                   "model_manifest_sha256": episode["model_manifest_sha256"],
                   "processor": flora_vllm._processor_binding(root)}
        if (certificate.get("status") != "PASS_NATIVE_PROCESSOR_PARITY"
                or certificate.get("requests") != len(episode["prepared"])
                or certificate.get("checked_precisions") != ["float32", "bfloat16"]
                or certificate.get("binding") != binding):
            raise ValueError("CPU processor parity certificate differs from admitted runtime")


def segment_replay_root(manifest: dict) -> Path | None:
    replay = manifest.get("segment_replay")
    if replay is None:
        return None
    if not isinstance(replay, dict) or not isinstance(replay.get("run_id"), str):
        raise ValueError("Invalid segment replay descriptor")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", replay["run_id"]):
        raise ValueError("Invalid replay run id")
    if replay["run_id"] == manifest["run_id"]:
        raise ValueError("Replay must reference a prior run")
    return EVIDENCE_ROOT / "runs" / replay["run_id"]


def save(path: Path, value, commit) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    commit()


def execute(manifest: dict, commit) -> dict:
    implementation = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in Path(__file__).parent.glob("flora*.py")}
    if implementation != manifest.get("implementation_sha256"):
        raise ValueError("Hosted implementation differs from the submitted batch")
    run_id = manifest["run_id"]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError("Invalid run id")
    if manifest["stage"] not in {"caption", "caption-session", "caption-native", "segment-native",
                                  "reward", "reward-dense", "reward-outcome"}:
        raise ValueError("Invalid stage")
    if type(manifest.get("reward_rationale", True)) is not bool:
        raise ValueError("reward_rationale must be a boolean")
    if manifest["use"] != "noncommercial_research":
        raise ValueError("Research use acknowledgement required")
    if manifest["max_num_seqs"] not in (1, 2, 4, 8, 16):
        raise ValueError("Unsupported concurrency")
    root = EVIDENCE_ROOT / "runs" / run_id
    root.mkdir(parents=True, exist_ok=True)
    binding = input_digest(manifest)
    receipt = root / "receipt.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["manifest_sha256"] != binding:
            raise ValueError("Run identity changed")
        files = saved.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("Retained receipt lacks evidence hashes")
        for name, expected in files.items():
            path = root / name
            if (not path.resolve().is_relative_to(root.resolve()) or not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
                raise ValueError("Retained evidence changed or is missing")
        return saved
    if (root / "manifest.json").exists():
        raise ValueError("Unsettled run: reconcile retained evidence before another allocation")
    requests = manifest["requests"]
    if manifest["stage"] == "caption-session":
        for batch in caption_batches(manifest):
            admitted_cache(batch["prepared_cache"])
    elif not isinstance(requests, list) or not 1 <= len(requests) <= (64 if manifest["stage"] == "reward-outcome" else 16):
        raise ValueError("Batch exceeds its bounded episode count")
    for request in requests:
        from events.flora import MODELS
        if request["models"] != MODELS:
            raise ValueError("Model code and weights must match the audited release")
        for name, expected in zip(request["videos"], request["video_sha256"], strict=True):
            path = Path(name)
            if not path.resolve().is_relative_to((EVIDENCE_ROOT / "inputs").resolve()):
                raise ValueError("Inputs must be under the evidence volume inputs directory")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("Input bytes changed")
    save(root / "manifest.json", manifest, commit)
    start = time.time()
    gpu_log = (root / "gpu.csv").open("x")
    monitor = subprocess.Popen([
        "nvidia-smi", "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits", "--loop-ms=1000"], stdout=gpu_log, stderr=subprocess.STDOUT)
    status, error = "failed", None
    try:
        with (root / "worker.log").open("x") as log:
            process = subprocess.Popen([sys.executable, "-m", "events.flora_hosted", str(root)],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=1650)
            except subprocess.TimeoutExpired:
                error = "GPU worker reached its 1650-second deadline"
                code = -1
            finally:
                # vLLM engine descendants inherit this dedicated process group.
                # Kill the group even if its parent exited first; no orphan owns CUDA.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        status = "complete" if code == 0 else "failed"
        if code:
            error = error or f"Worker exited {code}; inspect worker.log and generation receipts"
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        gpu_log.close()
        commit()
    elapsed = time.time() - start
    result = {"run_id": run_id, "status": status, "error": error,
              "implementation_sha256": implementation,
              "manifest_sha256": binding, "elapsed_s": elapsed,
              "gpu": "H200", "gpu_cost_estimate_usd": elapsed * .001261,
              "compute_cost_upper_estimate_usd": elapsed * (.001261 + 8*.0000131 + 64*.00000222),
              "cost_note": "List-price estimate for worker interval; excludes image build, startup, and storage.",
              "files": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in root.rglob("*") if path.is_file()},
              "log_tail": (root / "worker.log").read_text()[-12000:]}
    save(receipt, result, commit)
    return result


def worker(root: Path) -> None:
    import importlib.metadata
    import modal
    volume = modal.Volume.from_name("argus-flora-research-evidence")
    manifest = json.loads((root / "manifest.json").read_text())
    commit = volume.commit
    journal = GenerationJournal(root / "generations", input_digest(manifest), commit)
    save(root / "environment.json", {
        name: importlib.metadata.version(name) for name in
        ("torch", "vllm", "transformers", "av", "huggingface-hub", "peft")}, commit)
    stage = manifest["stage"]
    if stage == "segment-native":
        from events.flora_segment import run
        run(manifest["requests"], root / "results", journal,
            replay=manifest.get("segment_replay"), replay_root=segment_replay_root(manifest))
    elif stage in {"reward", "reward-dense", "reward-outcome"}:
        from events.flora_reward_batch import run
        run(manifest["requests"], root / "results", journal, manifest["max_num_seqs"],
            stage={"reward-dense": "dense", "reward-outcome": "outcome"}.get(stage, "both"),
            with_rationale=manifest.get("reward_rationale", True))
    elif stage in {"caption", "caption-session"}:
        from events.flora import validate
        from events.flora_vllm import caption_many, ResidentCaptionSession
        batches = caption_batches(manifest) if stage == "caption-session" else [manifest]
        session = ResidentCaptionSession(max_num_seqs=manifest["max_num_seqs"])
        for index, batch in enumerate(batches):
            batch_root = root / "batches" / f"{index:03d}" if stage == "caption-session" else root
            batch_journal = (GenerationJournal(batch_root / "generations",
                                              input_digest({"manifest": manifest, "batch_index": index}), commit)
                             if stage == "caption-session" else journal)
            results = caption_many(batch["requests"], batch_root / "results", batch_journal,
                                   max_num_seqs=manifest["max_num_seqs"],
                                   prepared_cache=admitted_cache(batch["prepared_cache"]), session=session)
            for request, result in zip(batch["requests"], results, strict=True):
                validate("caption", result, request)
            save(batch_root / "complete.json", {"batch_index": index, "episodes": len(results)}, commit)
    else:
        from events.flora import validate
        from events.flora_worker import _caption as caption
        if len(manifest["requests"]) != 1:
            raise ValueError("Native caption baseline runs one episode in its isolated process")
        for index, request in enumerate(manifest["requests"]):
            path = root / f"caption-{index:03d}.json"
            caption(request, path, journal)
            validate("caption", json.loads(path.read_text()), request)
            commit()
    commit()

def retain_result(source_manifest: Path, result: dict, read_file) -> None:
    """Collect one existing call; interrupted downloads can resume without inference."""
    import fcntl
    import hashlib
    import json
    import os
    from events.flora_journal import _save

    saved = source_manifest.with_suffix(".receipt.json")
    if saved.exists():
        if json.loads(saved.read_text()) != result:
            raise ValueError("Retained hosted receipt changed")
    else:
        _save(saved, result)
    upper = (result["elapsed_s"] + 605) * (.001261 + 8*.0000131 + 64*.00000222)
    ledger = source_manifest.parent / "modal-budget.jsonl"
    with ledger.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        rows = [json.loads(line) for line in stream if line.strip()]
        if not any(row["run_id"] == result["run_id"] and "released_usd" in row for row in rows):
            if not any(row["run_id"] == result["run_id"] and row.get("reserved_usd") == 4 for row in rows):
                raise ValueError("No matching allocation reservation")
            stream.write(json.dumps({"run_id": result["run_id"], "released_usd": max(0, 4-upper),
                                     "completed_upper_cost_usd": upper}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    artifacts = source_manifest.with_suffix(".artifacts")
    artifacts.mkdir(exist_ok=True)
    for name, expected in result["files"].items():
        path = artifacts / name
        if not path.resolve().is_relative_to(artifacts.resolve()):
            raise ValueError("Invalid retained artifact path")
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Retained artifact differs; preserve and reconcile it")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # An interrupted transfer stays distinct from accepted paid evidence.
        partial = path.with_name(path.name + ".download")
        with partial.open("wb") as stream:
            for chunk in read_file(f"/runs/{result['run_id']}/{name}"):
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
            raise ValueError("Downloaded evidence hash differs from hosted receipt")
        partial.rename(path)
    print(json.dumps({key: result[key] for key in ("run_id", "status", "elapsed_s", "error")}, indent=2))


if __name__ == "__main__":
    worker(Path(sys.argv[1]))
