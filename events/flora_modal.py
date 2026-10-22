"""Bounded, authenticated Modal hosting for the optional Flora research path.

Run with the pinned Modal CLI, separate from Argus's CPU environment.
Weights are prefetched on CPU; GPU calls have no automatic retries.
"""
from pathlib import Path
import modal

APP_NAME = "argus-flora-research"
app = modal.App(APP_NAME)
cache = modal.Volume.from_name(APP_NAME + "-models", create_if_missing=True)
evidence = modal.Volume.from_name(APP_NAME + "-evidence", create_if_missing=True)
ENV = {"HF_HOME": "/models/huggingface", "FLORA_MODEL_CACHE": "/models/flora",
       "FLORA_RUNTIME_CACHE": "/models/runtime",
       "VLLM_CACHE_ROOT": "/models/vllm", "TRITON_CACHE_DIR": "/models/triton",
       "TORCHINDUCTOR_CACHE_DIR": "/models/torchinductor",
       "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false",
       "VLLM_LOG_STATS_INTERVAL": "1", "PYTHONPATH": "/opt/argus"}
source = Path(__file__).resolve().parent
cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install("huggingface-hub==1.33.0")
             .env(ENV).add_local_dir(source, "/opt/argus/events"))
gpu_image = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
             .entrypoint([]).apt_install("ffmpeg", "libgl1", "libglib2.0-0")
             .uv_pip_install("vllm==0.24.0", "transformers==5.8.1", "peft==0.18.1",
                             "av==16.0.1", "decord==0.6.0", "qwen-vl-utils==0.0.14")
             .env(ENV).add_local_dir(source, "/opt/argus/events")
             .add_local_dir(source.parent / "label", "/opt/argus/label"))

@app.function(image=cpu_image, volumes={"/models": cache}, timeout=3600,
              cpu=(4, 4), memory=(16384, 16384), retries=0, max_containers=1, scaledown_window=2)
def prefetch():
    from huggingface_hub import snapshot_download
    models = {
        "captioner": ("nvidia/Flora-Captioner", "f736ca36925ae55d2f7915993ed1bc9f87288bb3",
                      "fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6"),
        "rewarder": ("nvidia/Flora-Rewarder", "7de68d22c4a9decec5ddb7f86ddcaee8f82d3367",
                     "7de68d22c4a9decec5ddb7f86ddcaee8f82d3367"),
    }
    result = {}
    for kind, (repo, code, weights) in models.items():
        runtime = snapshot_download(repo, revision=code,
                                    local_dir=f"/models/runtime/{kind}/{code}",
                                    allow_patterns=["batch_utils/*", "README.md", "LICENSE.pdf"])
        path = snapshot_download(repo, revision=weights)
        if kind == "captioner":
            snapshot_download(repo, revision=weights, local_dir="/models/flora/captioner")
        cache.commit()
        result[kind] = {"runtime": runtime, "weights": path, "revision": weights}
    return result

@app.function(image=gpu_image, cpu=(8, 8), memory=(65536, 65536),
              volumes={"/models": cache, "/evidence": evidence}, timeout=600,
              retries=0, max_containers=1, single_use_containers=True)
def inspect(manifest: dict):
    """Exercise the actual upstream media/processor contract without renting a GPU."""
    evidence.reload()
    import hashlib
    from events import flora_hosted
    implementation = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in Path(flora_hosted.__file__).parent.glob("flora*.py")}
    if implementation != manifest.get("implementation_sha256"):
        raise ValueError("CPU admission implementation differs from the submitted batch")
    if manifest["stage"] == "caption":
        import re
        from events.flora_vllm import prepare_many_cache, _load_prepared_cache
        run_id = manifest["run_id"]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
            raise ValueError("Invalid run id")
        if "prepared_cache" in manifest:
            result = manifest["prepared_cache"]
            path = Path(result["path"])
            if (not path.resolve().is_relative_to(Path("/evidence/prepared").resolve())
                    or path.name != "cache.json"
                    or hashlib.sha256(path.read_bytes()).hexdigest() != result["sha256"]):
                raise ValueError("Retained CPU preparation descriptor changed")
            _load_prepared_cache(manifest["requests"], path.parent)
        else:
            result = prepare_many_cache(manifest["requests"], Path("/evidence/prepared") / run_id)
            evidence.commit()
    elif manifest["stage"] == "caption-native":
        from events.flora_vllm import prepare
        result = []
        for request in manifest["requests"]:
            _, _, prepared, receipt = prepare(request)
            result.append({"id": request["id"], "requests": len(prepared),
                           "duration_s": receipt["duration_s"]})
    else:
        from events.flora_reward_batch import preflight
        result = preflight(manifest["requests"])
    cache.commit()
    return result

@app.function(image=gpu_image, gpu="H200", cpu=(8, 8), memory=(65536, 65536),
              volumes={"/models": cache, "/evidence": evidence},
              timeout=1800, startup_timeout=600, retries=0, max_containers=1,
              min_containers=0, scaledown_window=2, single_use_containers=True)
def qualify(manifest: dict):
    # Defined below: authenticated function calls, no publicly exposed web endpoint.
    from events.flora_hosted import execute
    evidence.reload()
    try:
        return execute(manifest, evidence.commit)
    finally:
        cache.commit()

@app.local_entrypoint()
def main(prefetch_only: bool = False, manifest: str = "", deployed: bool = False,
         admit_only: bool = False, prepared_cache: str = ""):
    import hashlib
    import json
    import os
    if prefetch_only:
        print(json.dumps(prefetch.remote(), indent=2))
    elif manifest:
        source_manifest = Path(manifest).resolve()
        request = json.loads(source_manifest.read_text())
        request["implementation_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source.glob("flora*.py")}
        # Reserve before dispatch, including a full startup + execution timeout.
        # Six worst-case GPU allocations plus CPU preparation stay below $30.
        ledger = source_manifest.parent / "modal-budget.jsonl"
        if not admit_only:
            with ledger.open("a+") as stream:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX)
                stream.seek(0)
                reservations = [json.loads(line) for line in stream if line.strip()]
                if any(row["run_id"] == request["run_id"] for row in reservations):
                    raise ValueError("Run already dispatched; inspect its Modal evidence, never retry blindly")
                if sum(row.get("reserved_usd", 0) - row.get("released_usd", 0)
                       for row in reservations) + 4 > 24:
                    raise ValueError("Qualification allocation budget exhausted; explicit budget review required")
                stream.write(json.dumps({"run_id": request["run_id"], "reserved_usd": 4}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        existing = {"/" + entry.path.lstrip("/") for entry in evidence.listdir("/", recursive=True)}
        with evidence.batch_upload() as upload:
            uploaded = set()
            for episode in request["requests"]:
                paths = []
                for name, digest in zip(episode["videos"], episode["video_sha256"], strict=True):
                    path = Path(name).resolve(strict=True)
                    with path.open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                            raise ValueError("Video changed since preparation")
                    target = f"/inputs/{digest}{path.suffix}"
                    if target not in uploaded and target not in existing:
                        upload.put_file(path, target)
                        uploaded.add(target)
                    paths.append("/evidence" + target)
                episode["videos"] = paths
        admission = modal.Function.from_name(APP_NAME, "inspect") if deployed else inspect
        inference = modal.Function.from_name(APP_NAME, "qualify") if deployed else qualify
        if prepared_cache:
            request["prepared_cache"] = json.loads(Path(prepared_cache).read_text())
        admitted = admission.remote(request)
        print("CPU admission:", admitted, flush=True)
        if request["stage"] == "caption":
            request["prepared_cache"] = admitted
        if admit_only:
            saved = source_manifest.with_suffix(".admission.json")
            with saved.open("x") as stream:
                json.dump(admitted, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            print(f"CPU preparation retained: {saved}", flush=True)
            return
        result = inference.remote(request)
        saved = source_manifest.with_suffix(".receipt.json")
        with saved.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        # Reconcile only after durable completion evidence. Keep the full startup
        # timeout reserved even when the real cold start was shorter.
        upper = (result["elapsed_s"] + 605) * (.001261 + 8*.0000131 + 64*.00000222)
        with ledger.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps({"run_id": request["run_id"],
                                     "released_usd": max(0, 4 - upper),
                                     "completed_upper_cost_usd": upper}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        artifacts = source_manifest.with_suffix(".artifacts")
        artifacts.mkdir()
        for name, expected in result["files"].items():
            path = artifacts / name
            if not path.resolve().is_relative_to(artifacts.resolve()):
                raise ValueError("Invalid retained artifact path")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                for chunk in evidence.read_file(f"/runs/{request['run_id']}/{name}"):
                    stream.write(chunk)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Downloaded evidence hash differs from the hosted receipt")
        print(json.dumps(result, indent=2))
    else:
        raise ValueError("Pass --prefetch-only or --manifest PATH")
