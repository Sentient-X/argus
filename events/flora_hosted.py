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
    if manifest["stage"] not in {"caption", "caption-native", "reward", "reward-dense"}:
        raise ValueError("Invalid stage")
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
    if not isinstance(requests, list) or not 1 <= len(requests) <= 16:
        raise ValueError("Batch needs 1..16 episodes")
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
    if stage in {"reward", "reward-dense"}:
        from events.flora_reward_batch import run
        run(manifest["requests"], root / "results", journal, manifest["max_num_seqs"],
            stage="dense" if stage == "reward-dense" else "both")
    elif stage == "caption":
        from events.flora import validate
        from events.flora_vllm import caption_many
        descriptor = manifest["prepared_cache"]
        cache_file = Path(descriptor["path"])
        if (not cache_file.resolve().is_relative_to((EVIDENCE_ROOT / "prepared").resolve())
                or cache_file.name != "cache.json"
                or hashlib.sha256(cache_file.read_bytes()).hexdigest() != descriptor["sha256"]):
            raise ValueError("CPU preparation cache differs from admitted evidence")
        results = caption_many(manifest["requests"], root / "results", journal,
                               max_num_seqs=manifest["max_num_seqs"],
                               prepared_cache=cache_file.parent)
        for request, result in zip(manifest["requests"], results, strict=True):
            validate("caption", result, request)
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


if __name__ == "__main__":
    worker(Path(sys.argv[1]))
