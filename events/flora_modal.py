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
             .env(ENV).add_local_dir(source, "/opt/argus/events")
             .add_local_dir(source.parent / "label", "/opt/argus/label"))
gpu_image = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
             .entrypoint([]).apt_install("ffmpeg", "libgl1", "libglib2.0-0")
             .uv_pip_install("vllm==0.24.0", "transformers==5.8.1", "peft==0.18.1",
                             "av==16.0.1", "decord==0.6.0", "qwen-vl-utils==0.0.14")
             .env(ENV).add_local_dir(source, "/opt/argus/events")
             .add_local_dir(source.parent / "label", "/opt/argus/label"))

@app.function(image=cpu_image, volumes={"/models": cache}, timeout=3600,
              cpu=(4, 4), memory=(16384, 16384), retries=0, max_containers=1, scaledown_window=2)
def prefetch(segmenter_only: bool = False):
    from huggingface_hub import snapshot_download
    if segmenter_only:
        from events.flora_segment import TEMPSEG_MODELS
        paths = {}
        for kind, spec in TEMPSEG_MODELS.items():
            options = {"local_dir": "/models/flora/tempseg"} if kind == "tempseg" else {}
            paths[kind] = snapshot_download(spec["repo"], revision=spec["revision"],
                allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "*.sha256"], **options)
        cache.commit()
        return paths
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

@app.function(image=gpu_image, cpu=(4, 4), memory=(16384, 16384),
              volumes={"/models": cache, "/evidence": evidence}, timeout=1200, startup_timeout=600,
              retries=0, max_containers=3, single_use_containers=True)
def inspect(manifest: dict):
    """Exercise the actual upstream media/processor contract without renting a GPU."""
    evidence.reload()
    import re
    import time
    from events.flora_journal import GenerationJournal, _save, input_digest

    run_id = manifest["run_id"]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError("Invalid admission run id")
    root = Path("/evidence/admissions") / run_id
    journal = GenerationJournal(root, input_digest(manifest), evidence.commit)

    def measured():
        started = time.time()
        try:
            result = _inspect(manifest)
        except BaseException as exc:
            _save(root / "failure.json", {"started_at_unix": started,
                "ended_at_unix": time.time(), "error": str(exc)})
            evidence.commit()
            raise
        ended = time.time()
        return {"result": result, "started_at_unix": started, "ended_at_unix": ended,
                "elapsed_s": ended - started,
                "resources": {"cpu": 4, "memory_gib": 16},
                "compute_cost_estimate_usd": (ended - started) * (4*.0000131 + 16*.00000222),
                "cost_note": "CPU and memory list-price estimate; startup and storage separate."}

    return journal.run({"stage": "cpu-admission", "manifest": manifest}, measured)["result"]


def _inspect(manifest: dict):
    import hashlib
    from events import flora_hosted
    implementation = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in Path(flora_hosted.__file__).parent.glob("flora*.py")}
    if implementation != manifest.get("implementation_sha256"):
        raise ValueError("CPU admission implementation differs from the submitted batch")
    from events.flora import require_stage
    for request in manifest["requests"]:
        require_stage(request, manifest["stage"])
    if type(manifest.get("reward_rationale", True)) is not bool:
        raise ValueError("reward_rationale must be a boolean")
    if manifest["stage"] in {"caption", "caption-session"}:
        import re
        from events.flora_journal import _save
        from events.flora_media import lossless_png_serialization
        from events.flora_vllm import prepare_many_cache
        run_id = manifest["run_id"]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
            raise ValueError("Invalid run id")
        batches = (flora_hosted.caption_batches(manifest)
                   if manifest["stage"] == "caption-session" else [manifest])
        descriptors = []
        for index, batch in enumerate(batches):
            if "prepared_cache" in batch:
                descriptor = batch["prepared_cache"]
                flora_hosted.verify_prepared_cache(batch["requests"], descriptor)
            else:
                name = f"{run_id}-batch-{index:03d}" if len(batches) > 1 else run_id
                with lossless_png_serialization() as serialization:
                    descriptor = prepare_many_cache(batch["requests"], Path("/evidence/prepared") / name)
                descriptor["png_serialization"] = serialization
                _save(Path(descriptor["path"]).with_name("serialization.json"), serialization)
                evidence.commit()
            descriptors.append(descriptor)
        result = descriptors if manifest["stage"] == "caption-session" else descriptors[0]
    elif manifest["stage"] == "segment-native":
        from events.flora_segment import preflight, validate_replay
        validate_replay(manifest["requests"], manifest.get("segment_replay"),
                        flora_hosted.segment_replay_root(manifest))
        result = preflight(manifest["requests"])
    elif manifest["stage"] == "caption-native":
        from events.flora_vllm import prepare
        result = []
        for request in manifest["requests"]:
            _, _, prepared, receipt = prepare(request)
            result.append({"id": request["id"], "requests": len(prepared),
                           "duration_s": receipt["duration_s"]})
    else:
        from events.flora_reward_batch import preflight
        result = preflight(manifest["requests"], stage={
            "reward-dense": "dense", "reward-outcome": "outcome"}.get(manifest["stage"], "both"))
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
         admit_only: bool = False, prepared_cache: str = "", segmenter_only: bool = False,
         submit_only: bool = False, collect_only: bool = False):
    import hashlib
    import json
    import os
    if prefetch_only:
        print(json.dumps(prefetch.remote(segmenter_only), indent=2))
    elif manifest:
        source_manifest = Path(manifest).resolve()
        request = json.loads(source_manifest.read_text())
        if collect_only:
            dispatch = json.loads(source_manifest.with_suffix(".dispatch.json").read_text())
            if dispatch["run_id"] != request["run_id"]:
                raise ValueError("Dispatch identity differs from the selected manifest")
            try:
                result = modal.FunctionCall.from_id(dispatch["call_id"]).get(timeout=0)
            except TimeoutError:
                print("Still running; collect the same call later. No new allocation.", flush=True)
                return
            if result["manifest_sha256"] != dispatch["manifest_sha256"]:
                raise ValueError("Returned result differs from the submitted manifest")
            _retain_result(source_manifest, result)
            return
        if submit_only and (not deployed or admit_only):
            raise ValueError("Submit-only requires deployed GPU inference")
        if request["stage"] == "caption-session":
            from events.flora_hosted import caption_batches
            caption_batches(request)
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
        if request["stage"] == "caption-session":
            # JSON decoding does not preserve aliases between the top-level upload
            # list and the per-batch requests. Rebind only after ordered equality passed.
            offset = 0
            for batch in request["batches"]:
                count = len(batch["requests"])
                batch["requests"] = request["requests"][offset:offset + count]
                offset += count
        admission = modal.Function.from_name(APP_NAME, "inspect") if deployed else inspect
        inference = modal.Function.from_name(APP_NAME, "qualify") if deployed else qualify
        if prepared_cache:
            retained = json.loads(Path(prepared_cache).read_text())
            if request["stage"] == "caption-session":
                for batch, descriptor in zip(request["batches"], retained, strict=True):
                    batch["prepared_cache"] = descriptor
            else:
                request["prepared_cache"] = retained
        admitted = admission.remote(request)
        print("CPU admission:", admitted, flush=True)
        if request["stage"] == "caption":
            request["prepared_cache"] = admitted
        elif request["stage"] == "caption-session":
            for batch, descriptor in zip(request["batches"], admitted, strict=True):
                batch["prepared_cache"] = descriptor
        if admit_only:
            saved = source_manifest.with_suffix(".admission.json")
            with saved.open("x") as stream:
                json.dump(admitted, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            print(f"CPU preparation retained: {saved}", flush=True)
            return
        if deployed:
            from events.flora_journal import _save, input_digest
            call = inference.spawn(request)
            _save(source_manifest.with_suffix(".dispatch.json"), {
                "run_id": request["run_id"], "call_id": call.object_id,
                "manifest_sha256": input_digest(request)})
            print(f"Submitted durable call {call.object_id}; collect with --collect-only.", flush=True)
            if submit_only:
                return
            result = call.get()
        else:
            result = inference.remote(request)
        _retain_result(source_manifest, result)
    else:
        raise ValueError("Pass --prefetch-only or --manifest PATH")


def _retain_result(source_manifest: Path, result: dict) -> None:
    from events.flora_hosted import retain_result
    retain_result(source_manifest, result, evidence.read_file)
