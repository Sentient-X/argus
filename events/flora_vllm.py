"""Batched FLORA captioning with native input preparation and verified FP32 GDN.

The upstream Transformers path remains the reference. This backend changes the
inference engine, retains its exact processor inputs, and checks prompt-token
parity before accepting a completion. GPU dependencies are imported lazily.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from .flora_journal import GenerationJournal, input_digest

_PREPARE_LOCK = threading.RLock()


def _verify_gdn(model, root: Path) -> dict:
    import torch
    from safetensors import safe_open

    mapping = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = {name for name in mapping if name.endswith((".A_log", ".dt_bias"))}
    actual = {name.replace("language_model.model.", "model.language_model.", 1): value
              for name, value in model.named_parameters()
              if name.endswith((".A_log", ".dt_bias"))}
    if not wanted or set(actual) != wanted:
        raise RuntimeError("vLLM GDN parameter names differ from the pinned checkpoint")
    groups = defaultdict(list)
    for name in wanted:
        groups[mapping[name]].append(name)
    records = []
    for shard, names in sorted(groups.items()):
        with safe_open(str(root / shard), framework="pt", device="cpu") as source:
            for name in sorted(names):
                parameter = actual[name]
                saved = source.get_tensor(name)
                value = parameter.detach().cpu()
                if (parameter.dtype != torch.float32 or parameter.device.type != "cuda"
                        or not torch.equal(value, saved.float())):
                    raise RuntimeError(f"vLLM did not preserve native FP32 GDN values: {name}")
                records.append({"name": name, "dtype": str(parameter.dtype),
                                "sha256": hashlib.sha256(value.contiguous().view(torch.uint8)
                                                         .numpy().tobytes()).hexdigest()})
    return {"status": "PASS_NATIVE_GDN_FP32_CHECKPOINT_VALUES", "parameters": records}


def __getattr__(name):
    # vLLM resolves this qualified worker class inside each spawned GPU process.
    if name != "FP32Worker":
        raise AttributeError(name)
    from vllm.v1.worker.gpu_worker import Worker

    class FP32Worker(Worker):
        def load_model(self, *, load_dummy_weights=False):
            from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
                QwenGatedDeltaNetAttention,
            )
            original = QwenGatedDeltaNetAttention.__init__

            def initialize(layer, *args, **kwargs):
                original(layer, *args, **kwargs)
                # Before checkpoint assignment: upcasting loaded BF16 is too late.
                layer.dt_bias.data = layer.dt_bias.data.float()

            QwenGatedDeltaNetAttention.__init__ = initialize
            try:
                super().load_model(load_dummy_weights=load_dummy_weights)
            finally:
                QwenGatedDeltaNetAttention.__init__ = original
            if load_dummy_weights:
                raise RuntimeError("FLORA requires checkpoint value verification")
            self.flora_gdn_receipt = _verify_gdn(
                self.model_runner.get_model(), Path(self.model_config.model),
            )

        def flora_precision_receipt(self):
            return self.flora_gdn_receipt

    return FP32Worker


def _upstream(request: dict):
    import sys
    from .flora_worker import _snapshot

    spec = request["models"]["captioner"]
    if os.environ.get("FLORA_CAPTIONER_REPO"):
        raise ValueError("Unset FLORA_CAPTIONER_REPO; the request pins the repository")
    existing = sys.modules.get("batch_utils")
    if existing is None:
        _snapshot(request, "captioner")
    elif Path(existing.__file__).absolute().parent.parent.name != spec["revision"]:
        raise RuntimeError("A different upstream batch_utils package is already loaded")
    from batch_utils import models, pipeline
    from batch_utils.api import DataEnginePipeline
    from batch_utils.config import MODELS

    if (MODELS["captioner"]["repo"].lower() != spec["repo"].lower()
            or MODELS["captioner"]["revision"] != spec["weights_revision"]):
        raise RuntimeError("Native captioner weight binding differs from the saved request")
    return models, pipeline, DataEnginePipeline


class _ProcessorRecorder:
    """Capture the exact RGB blocks and text passed to the native processor."""
    def __init__(self, processor):
        self.processor = processor
        self.request = None

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        import numpy as np

        if args or kwargs.get("do_sample_frames") is not False:
            raise RuntimeError("Unexpected native captioner processor contract")
        videos, metadata = kwargs["videos"], kwargs["video_metadata"]
        if len(kwargs["text"]) != 1 or len(videos) != len(metadata):
            raise RuntimeError("Native caption blocks and metadata differ")
        self.request = {
            "prompt": _vllm_prompt(kwargs["text"][0], self.processor.video_token, len(videos)),
            "multi_modal_data": {"video": [
                (np.stack([np.asarray(frame) for frame in frames]),
                 {"total_num_frames": meta.total_num_frames, "fps": meta.fps,
                  "frames_indices": list(meta.frames_indices)})
                for frames, meta in zip(videos, metadata, strict=True)
            ]},
            "mm_processor_kwargs": {"do_sample_frames": False},
        }
        return self.processor(**kwargs)


def _vllm_prompt(native: str, video_token: str, videos: int) -> str:
    # Native Transformers expands a bare video token. vLLM 0.24 replaces the
    # complete triplet, emitting the same frame timestamps and patch tokens.
    # Only transport markers change; admission checks every resulting token.
    if (video_token != "<|video_pad|>" or native.count(video_token) != videos
            or "<|vision_start|>" in native or "<|vision_end|>" in native):
        raise ValueError("Unexpected native caption video placeholder contract")
    return native.replace(video_token, "<|vision_start|>" + video_token + "<|vision_end|>")


def prepare(request: dict) -> tuple[Path, object, list[dict], dict]:
    """Prepare every interval through the released native loop, without a model."""
    import torch

    # Native Session.load sets four CPU threads before preprocessing. We bypass
    # that loader, so retain its bound here rather than using the host CPU count
    # inside an eight-core Modal container (which can severely oversubscribe).
    torch.set_num_threads(4)
    paths = request["videos"]
    if len(paths) not in (1, 3) or len(paths) != len(request["video_sha256"]):
        raise ValueError("Captioning needs one video or main/left/right synchronized views")
    for path, expected in zip(paths, request["video_sha256"], strict=True):
        with Path(path).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Native video changed after request preparation")
    models, pipeline, api = _upstream(request)
    videos, profile = api._inputs(paths[0], request["profile"],
                                 paths[1] if len(paths) == 3 else None,
                                 paths[2] if len(paths) == 3 else None,
                                 False, request["task"])
    if not request["events"] or any(
        not 0 <= event["start_s"] < event["end_s"] <= videos["main"].duration_s
        for event in request["events"]
    ):
        raise ValueError("Caption intervals must be inside the native video")
    root = models.snapshot("captioner")
    models.validate_manifest(root)
    processor = models.processor("captioner", root)
    recorder = _ProcessorRecorder(processor)
    prepared = []

    class PreparationSession:
        def load(self, kind):
            if kind != "captioner":
                raise RuntimeError("Only supplied-interval captioning is supported")
            return None, recorder

    def capture(model, proc, inputs, budget):
        if model is not None or recorder.request is None:
            raise RuntimeError("Unexpected native preparation call")
        tokens = inputs["input_ids"][0].tolist()
        if len(tokens) + budget > 32768:
            raise ValueError("Native caption request exceeds its context budget")
        prepared.append({"request": recorder.request, "prompt_token_ids": tokens,
                         "native_input_sha256": input_digest(dict(inputs)),
                         "native_vision_sha256": input_digest({
                             key: inputs[key] for key in ("pixel_values_videos", "video_grid_thw")
                         }), "native_vision_fields": _vision_description(inputs),
                         "native_vision_bf16_sha256": input_digest({
                             key: inputs[key].to(torch.bfloat16) if inputs[key].is_floating_point()
                             else inputs[key] for key in ("pixel_values_videos", "video_grid_thw")
                         }), "max_tokens": budget})
        recorder.request = None
        # The native loop needs a text return to continue. This preparation value
        # is discarded; only verified model completions can enter the output.
        return ""

    with _PREPARE_LOCK:
        original = pipeline.generate
        pipeline.generate = capture
        try:
            progress = lambda *args, **kwargs: None
            if len(paths) == 1:
                rows, audit = pipeline.caption_single(videos, request["events"], profile,
                                                       PreparationSession(), request["task"], progress)
            else:
                rows, audit = pipeline.caption_three(videos, request["events"], PreparationSession(),
                                                      request["task"], progress, profile=profile)
        finally:
            pipeline.generate = original
    if len(prepared) != len(rows) or any(row["status"] != "complete" for row in rows):
        raise ValueError("A supplied interval has unsupported native caption frames")
    receipt = {"id": request["id"], "segments": rows, "duration_s": videos["main"].duration_s,
               "profile": profile, "camera_roles": list(videos), "pipeline": {"caption_inputs": audit},
               "input_sha256": {role: video.sha256 for role, video in videos.items()},
               "model_revisions": request["models"], "backend": "vllm-0.24.0",
               "source_clock": "seconds relative to the first native source presentation timestamp"}
    return root, processor, prepared, receipt


def _processor_kwargs(model_path: Path) -> dict:
    config = json.loads((model_path / "processor_config.json").read_text())["video_processor"]
    return {"do_sample_frames": False, "fps": 8, "max_frames": 40,
            "size": config["size"]}


def _vision_description(inputs) -> dict:
    def describe(value):
        if isinstance(value, (list, tuple)):
            return {"type": type(value).__name__, "items": [describe(item) for item in value]}
        return {"type": type(value).__name__, "dtype": str(value.dtype),
                "shape": list(value.shape), "sha256": input_digest(value)}
    return {key: describe(inputs[key]) for key in ("pixel_values_videos", "video_grid_thw")}


def check_processor_parity(root: Path, tokenizer, prepared: list[dict], model_config=None) -> dict:
    """Execute vLLM's real CPU processor before admitting any GPU model load."""
    from copy import copy
    import torch
    import vllm
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY

    if vllm.__version__ != "0.24.0":
        raise RuntimeError("Caption processor admission requires vLLM 0.24.0")
    started = time.monotonic()
    processor_kwargs = _processor_kwargs(root)
    if model_config is None:
        model_config = ModelConfig(
            model=str(root), tokenizer=str(root), dtype="bfloat16", max_model_len=32768,
            limit_mm_per_prompt={"image": 0, "video": 3},
            mm_processor_kwargs=processor_kwargs,
        )
    if model_config.dtype != torch.bfloat16:
        raise ValueError("Caption inference must retain native BF16 model precision")
    # vLLM's InputProcessingContext casts processor floats to model dtype. Check
    # pre-cast FP32 exactly, then check its actual BF16 output against the same
    # cast performed by HF Qwen3_5VisionPatchEmbed.forward before its convolution.
    # Copy the config so neither the engine nor its processor cache is mutated.
    native_config = copy(model_config)
    native_config.dtype = torch.float32
    for config, digest_key in ((native_config, "native_vision_sha256"),
                               (model_config, "native_vision_bf16_sha256")):
        checker = MULTIMODAL_REGISTRY.create_processor(config, tokenizer=tokenizer, cache=None)
        for index, item in enumerate(prepared):
            request = item["request"]
            checked = checker._call_hf_processor(
                prompt=request["prompt"], mm_data={"videos": request["multi_modal_data"]["video"]},
                mm_kwargs=processor_kwargs | request["mm_processor_kwargs"], tok_kwargs={},
            )
            token_ids = checked["input_ids"]
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if token_ids[0] != item["prompt_token_ids"]:
                raise ValueError(f"vLLM processor prompt differs from native captioner at interval {index}")
            vision = {key: checked[key] for key in ("pixel_values_videos", "video_grid_thw")}
            if input_digest(vision) != item[digest_key]:
                raise ValueError(f"vLLM processor vision mismatch at interval {index}: " + json.dumps({
                    "comparison_dtype": str(config.dtype), "expected_sha256": item[digest_key],
                    "native": item["native_vision_fields"], "actual": _vision_description(vision),
                }, sort_keys=True))
    return {"status": "PASS_NATIVE_PROCESSOR_PARITY", "requests": len(prepared),
            "checked_precisions": ["float32", "bfloat16"],
            "binding": _parity_binding(root, prepared),
            "elapsed_s": time.monotonic() - started}


def _file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _processor_binding(root: Path) -> dict:
    from importlib.metadata import version

    return {"files": {name: _file_sha(root / name) for name in (
                "config.json", "processor_config.json", "tokenizer.json",
                "tokenizer_config.json", "chat_template.jinja")},
            "versions": {name: version(name) for name in (
                "vllm", "transformers", "torch", "torchvision", "numpy", "pillow", "tokenizers")},
            "kwargs": _processor_kwargs(root)}


def _parity_binding(root: Path, prepared: list[dict]) -> dict:
    return {"prepared_sha256": input_digest(prepared),
            "implementation_sha256": _file_sha(Path(__file__)),
            "model_manifest_sha256": _file_sha(root / "SOURCE_MANIFEST.sha256"),
            "processor": _processor_binding(root)}


def _validate_parity(certificate: dict, root: Path, prepared: list[dict]) -> None:
    if (certificate.get("status") != "PASS_NATIVE_PROCESSOR_PARITY"
            or certificate.get("requests") != len(prepared)
            or certificate.get("checked_precisions") != ["float32", "bfloat16"]
            or certificate.get("binding") != _parity_binding(root, prepared)):
        raise ValueError("CPU processor parity certificate differs from admitted inputs/runtime")


def _resident_binding(root: Path, model_spec: dict) -> dict:
    return {"root": str(root.resolve()), "model": dict(model_spec),
            "manifest_sha256": _file_sha(root / "SOURCE_MANIFEST.sha256"),
            "processor": _processor_binding(root),
            "implementation_sha256": _file_sha(Path(__file__))}


class ResidentCaptionSession:
    """One bounded, serial engine lifetime; the caller owns shutdown and deadline."""

    def __init__(self, max_num_seqs: int = 8, *, max_chunks: int = 16):
        if not 1 <= max_num_seqs <= 64 or max_chunks < 1:
            raise ValueError("Invalid resident caption session limits")
        self.max_num_seqs, self.max_chunks = max_num_seqs, max_chunks
        self._engine = self._processor = self._binding = None
        self._chunks = 0
        self._lock = threading.Lock()

    def _check_binding(self, binding: dict) -> bool:
        if self._engine is None:
            return False
        if self._binding != binding:
            raise ValueError("Resident caption engine model or processor binding changed")
        return True

    def _load(self, root: Path, model_spec: dict):
        binding = _resident_binding(root, model_spec)
        if not self._check_binding(binding):
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(root, local_files_only=True)
            engine = load_engine(root, self.max_num_seqs)
            self._processor, self._engine, self._binding = processor, engine, binding
        return self._engine, self._processor


def _model_file_stats(root: Path) -> dict:
    expected = (Path(os.environ.get("FLORA_MODEL_CACHE", Path.home() / ".cache/flora/models"))
                / "captioner").resolve()
    if root != root.resolve() or root != expected:
        raise ValueError("Preparation cache checkpoint root differs from the native cache")
    stats = {}
    for line in (root / "SOURCE_MANIFEST.sha256").read_text().splitlines():
        if not line.strip():
            continue
        _, name = line.split(maxsplit=1)
        relative = Path(name.lstrip("*"))
        source = root / relative
        if (relative.is_absolute() or ".." in relative.parts
                or not source.resolve().is_relative_to(root) or str(relative) in stats):
            raise ValueError("Invalid preparation cache model manifest path")
        stamp = source.stat()
        stats[str(relative)] = {"size": stamp.st_size, "mtime_ns": stamp.st_mtime_ns}
    if not stats:
        raise ValueError("Empty preparation cache model manifest")
    return stats


def _verify_model_files(root: Path, expected_stats: dict, manifest_sha256: str) -> None:
    manifest = (root / "SOURCE_MANIFEST.sha256").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != manifest_sha256:
        raise ValueError("Preparation cache model manifest differs")
    if _model_file_stats(root) != expected_stats:
        raise ValueError("Preparation cache checkpoint files changed after CPU verification")
    for line in manifest.decode().splitlines():
        if line.strip():
            expected, name = line.split(maxsplit=1)
            if _file_sha(root / name.lstrip("*")) != expected:
                raise ValueError("Preparation cache checkpoint content hash differs")
    if (_model_file_stats(root) != expected_stats
            or (root / "SOURCE_MANIFEST.sha256").read_bytes() != manifest):
        raise ValueError("Checkpoint files changed during content verification")


def prepare_many_cache(requests: list[dict], cache_dir: Path) -> dict:
    """Save verified CPU preparation; cache.json is the completion receipt.

    The caller must bind the returned manifest hash into its admitted run. Model
    weights remain in the same immutable, pinned shared cache verified by prepare.
    """
    import numpy as np
    from .flora_worker import _write_json

    if not requests:
        raise ValueError("Provide at least one captioning episode")
    started = time.monotonic()
    cache_dir.mkdir(parents=True, exist_ok=False)
    files = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("Object arrays cannot enter the preparation cache")
            name = f"{len(files):06d}.npy"
            path = cache_dir / name
            with path.open("xb") as stream:
                np.save(stream, value, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            files[name] = _file_sha(path)
            return {"__flora_array__": name}
        if isinstance(value, tuple):
            return {"__flora_tuple__": [encode(item) for item in value]}
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            if any(key.startswith("__flora_") for key in value):
                raise ValueError("Reserved preparation cache key")
            return {key: encode(item) for key, item in value.items()}
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise TypeError(f"Unsupported preparation cache value: {type(value).__name__}")

    episodes = []
    for request in requests:
        root, processor, prepared, receipt = prepare(request)
        parity = check_processor_parity(root, processor.tokenizer, prepared)
        root = root.resolve()
        episodes.append({"root": str(root), "prepared": encode(prepared),
                         "receipt": encode(receipt), "prepared_sha256": input_digest(prepared),
                         "model_manifest_sha256": _file_sha(root / "SOURCE_MANIFEST.sha256"),
                         "model_files": _model_file_stats(root), "processor_parity": parity})
    record = {"version": 1, "requests_sha256": input_digest(requests), "files": files,
              "implementation_sha256": _file_sha(Path(__file__)), "episodes": episodes,
              "cpu_prepare_wall_s": time.monotonic() - started}
    manifest = cache_dir / "cache.json"
    _write_json(manifest, record)
    return {"path": str(manifest.absolute()), "sha256": _file_sha(manifest),
            "requests_sha256": record["requests_sha256"], "files": files,
            "episodes": len(episodes),
            "cpu_prepare_wall_s": record["cpu_prepare_wall_s"],
            "intervals": sum(len(episode["prepared"]) for episode in episodes)}


def _load_prepared_cache(requests: list[dict], cache_dir: Path, *,
                         resident: ResidentCaptionSession | None = None) -> list[tuple]:
    import numpy as np

    record = json.loads((cache_dir / "cache.json").read_text())
    if (record["version"] != 1 or record["requests_sha256"] != input_digest(requests)
            or record["implementation_sha256"] != _file_sha(Path(__file__))
            or len(record["episodes"]) != len(requests)):
        raise ValueError("Preparation cache request or implementation binding differs")
    for name, digest in record["files"].items():
        path = cache_dir / name
        if (Path(name).name != name or not name.endswith(".npy") or path.is_symlink()
                or _file_sha(path) != digest):
            raise ValueError("Preparation cache array hash differs")

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"__flora_array__"}:
                if value["__flora_array__"] not in record["files"]:
                    raise ValueError("Unbound preparation cache array")
                return np.load(cache_dir / value["__flora_array__"], allow_pickle=False)
            if set(value) == {"__flora_tuple__"}:
                return tuple(decode(item) for item in value["__flora_tuple__"])
            return {key: decode(item) for key, item in value.items()}
        return [decode(item) for item in value] if isinstance(value, list) else value

    episodes, verified_models = [], set()
    for episode, request in zip(record["episodes"], requests, strict=True):
        root = Path(episode["root"])
        if (not root.is_absolute() or _file_sha(root / "SOURCE_MANIFEST.sha256")
                != episode["model_manifest_sha256"]):
            raise ValueError("Preparation cache model manifest differs")
        model_binding = (root, input_digest(episode["model_files"]), episode["model_manifest_sha256"])
        loaded = resident is not None and resident._check_binding(
            _resident_binding(root, request["models"]["captioner"]))
        if loaded:
            # The same engine already contains the SHA-verified weights. Never
            # skip the content scan before loading/replacing an engine.
            if _model_file_stats(root) != episode["model_files"]:
                raise ValueError("Resident checkpoint files changed after loading")
        elif model_binding not in verified_models:
            _verify_model_files(root, episode["model_files"], episode["model_manifest_sha256"])
            verified_models.add(model_binding)
        elif _model_file_stats(root) != episode["model_files"]:
            raise ValueError("Preparation cache checkpoint files changed after CPU verification")
        prepared = decode(episode["prepared"])
        if input_digest(prepared) != episode["prepared_sha256"]:
            raise ValueError("Preparation cache native inputs differ")
        _validate_parity(episode["processor_parity"], root, prepared)
        receipt = decode(episode["receipt"])
        receipt["processor_admission"] = episode["processor_parity"]
        episodes.append((root, prepared, receipt))
    return episodes


def load_engine(model_path: Path, max_num_seqs: int = 8):
    import vllm
    from vllm import LLM

    if vllm.__version__ != "0.24.0":
        raise RuntimeError("The FLORA native precision adapter requires vLLM 0.24.0")
    if not 1 <= max_num_seqs <= 64:
        raise ValueError("max_num_seqs must be between 1 and 64")
    return LLM(model=str(model_path), dtype="bfloat16", tensor_parallel_size=1,
               max_model_len=32768, max_num_seqs=max_num_seqs, gpu_memory_utilization=0.88,
               limit_mm_per_prompt={"image": 0, "video": 3},
               mm_processor_kwargs=_processor_kwargs(model_path),
               gdn_prefill_backend="triton", mamba_ssm_cache_dtype="float32",
               worker_cls="events.flora_vllm.FP32Worker", seed=0)


def _decode(completion: dict, tokenizer) -> str:
    eos = tokenizer.eos_token_id
    stops = [eos] if isinstance(eos, int) else eos
    ids = completion["token_ids"]
    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    if completion["finish_reason"] != "stop" or not ids or ids[-1] not in stops or not text:
        raise ValueError("vLLM did not return a complete native caption")
    return text


def _batch(prepared: list[dict], llm, journal: GenerationJournal, max_num_seqs: int,
           admitted_groups: list[tuple] | None = None):
    from vllm import SamplingParams

    if any(item["max_tokens"] != 512 for item in prepared):
        raise ValueError("Native captioner changed its output token budget")
    precision = llm.collective_rpc("flora_precision_receipt")
    sampling = SamplingParams(temperature=0.0, max_tokens=512, seed=0)
    # Independent CPU processor: no engine cache mutation and no paid generation.
    # Compare actual patches as well as token IDs; equal shapes alone cannot prove
    # matching RGB conversion, normalization, resizing, or temporal patch order.
    if admitted_groups is None:
        parity = check_processor_parity(Path(llm.model_config.model), llm.get_tokenizer(),
                                        prepared, model_config=llm.model_config)
        preflight_wall_s = parity["elapsed_s"]
    else:
        started = time.monotonic()
        if input_digest([item for group, _ in admitted_groups for item in group]) != input_digest(prepared):
            raise ValueError("CPU admission groups differ from the inference batch")
        for group, certificate in admitted_groups:
            _validate_parity(certificate, Path(llm.model_config.model), group)
        preflight_wall_s = time.monotonic() - started

    def generate():
        started = time.monotonic()
        started_at_unix = time.time()
        outputs = llm.generate([item["request"] for item in prepared], sampling, use_tqdm=False)
        completions = [{"prompt_token_ids": result.prompt_token_ids,
                        "token_ids": list(result.outputs[0].token_ids),
                        "text": result.outputs[0].text,
                        "finish_reason": result.outputs[0].finish_reason,
                        "stop_reason": result.outputs[0].stop_reason} for result in outputs]
        elapsed = time.monotonic() - started
        generated = sum(len(item["token_ids"]) for item in completions)
        return {"completions": completions, "metrics": {
            "inference_wall_s": elapsed, "requests": len(completions),
            "started_at_unix": started_at_unix, "ended_at_unix": time.time(),
            "prompt_tokens": sum(len(item["prompt_token_ids"]) for item in prepared),
            "generated_tokens": generated,
            "generated_tokens_per_second": generated / elapsed if elapsed else 0,
            "max_num_seqs": max_num_seqs,
        }}

    replayed = (journal.root / f"{journal.index:06d}.result.json").exists()
    saved = journal.run({"backend": "vllm-0.24.0", "prepared": prepared,
                         "max_num_seqs": max_num_seqs, "temperature": 0, "max_tokens": 512,
                         "precision": precision}, generate)
    if len(saved["completions"]) != len(prepared):
        raise ValueError("vLLM returned the wrong number of caption completions")
    for item, completion in zip(prepared, saved["completions"], strict=True):
        if completion["prompt_token_ids"] != item["prompt_token_ids"]:
            raise ValueError("vLLM preprocessing differs from the native captioner prompt tokens")
    return saved["completions"], precision, {**saved["metrics"], "generation_replayed": replayed,
                                           "processor_parity_reused": admitted_groups is not None,
                                           "processor_parity_wall_s": preflight_wall_s}


def caption(request: dict, output: Path, journal: GenerationJournal,
            max_num_seqs: int = 8, engine=None) -> dict:
    """Run all prepared intervals as one continuously scheduled vLLM batch."""
    from .flora_worker import _write_json

    if output.exists():
        raise ValueError("Caption output already exists; preserve the original result")
    started = time.monotonic()
    root, processor, prepared, receipt = prepare(request)
    prepared_at = time.monotonic()
    llm = engine if engine is not None else load_engine(root, max_num_seqs)
    loaded_at = time.monotonic()
    if Path(llm.model_config.model).resolve() != root.resolve():
        raise ValueError("Cached vLLM engine holds a different checkpoint")
    saved, precision, metrics = _batch(prepared, llm, journal, max_num_seqs)
    for completion, row in zip(saved, receipt["segments"], strict=True):
        row["caption"] = _decode(completion, processor.tokenizer)
    receipt["model_load_audit"] = precision
    receipt["generation_batch_size"] = len(prepared)
    receipt["timing"] = {**metrics, "prepare_wall_s": prepared_at - started,
                         "engine_load_wall_s": loaded_at - prepared_at,
                         "total_wall_s": time.monotonic() - started}
    _write_json(output, receipt)
    return receipt


def caption_many(requests: list[dict], output_dir: Path, journal: GenerationJournal,
                 max_num_seqs: int = 8, prepared_cache: Path | None = None,
                 session: ResidentCaptionSession | None = None) -> list[dict]:
    """Batch intervals across episodes while loading the captioner once."""
    session = session if session is not None else ResidentCaptionSession(max_num_seqs, max_chunks=1)
    if session.max_num_seqs != max_num_seqs:
        raise ValueError("Resident caption concurrency differs from the admitted chunk")
    if not session._lock.acquire(blocking=False):
        raise ValueError("Resident caption session already has an active chunk")
    try:
        if session._chunks >= session.max_chunks:
            raise ValueError("Resident caption session reached its admitted chunk limit")
        return _caption_many(requests, output_dir, journal, max_num_seqs, prepared_cache, session)
    finally:
        session._lock.release()


def _caption_many(requests: list[dict], output_dir: Path, journal: GenerationJournal,
                  max_num_seqs: int, prepared_cache: Path | None,
                  session: ResidentCaptionSession) -> list[dict]:
    from .flora_worker import _write_json

    if not requests:
        raise ValueError("Provide at least one captioning episode")
    ids = [request["id"] for request in requests]
    if len(set(ids)) != len(ids) or any(Path(value).name != value for value in ids):
        raise ValueError("Caption batch IDs must be unique local filenames")
    paths = [output_dir / f"{value}.caption.raw.json" for value in ids]
    if any(path.exists() for path in paths):
        raise ValueError("Caption outputs already exist; preserve original results")
    started = time.monotonic()
    reused = session._engine is not None
    if prepared_cache is None:
        episodes = [prepare(request) for request in requests]
        for root, processor, items, receipt in episodes:
            receipt["processor_admission"] = check_processor_parity(root, processor.tokenizer, items)
    else:
        cached = _load_prepared_cache(requests, prepared_cache, resident=session)
        episodes = [(root, None, items, receipt) for root, items, receipt in cached]
    roots = {root.resolve() for root, _, _, _ in episodes}
    if len(roots) != 1:
        raise ValueError("All episodes in one batch must use the same checkpoint")
    specs = [request["models"]["captioner"] for request in requests]
    if any(spec != specs[0] for spec in specs):
        raise ValueError("All episodes in one batch must use the same model binding")
    prepared = [item for _, _, items, _ in episodes for item in items]
    prepared_at = time.monotonic()
    llm, processor = session._load(episodes[0][0], specs[0])
    loaded_at = time.monotonic()
    session._chunks += 1
    saved, precision, metrics = _batch(prepared, llm, journal, max_num_seqs,
        admitted_groups=[(items, receipt["processor_admission"]) for _, _, items, receipt in episodes])
    offset, receipts = 0, []
    for (_, _, items, receipt), path in zip(episodes, paths, strict=True):
        for row, completion in zip(receipt["segments"], saved[offset:offset + len(items)], strict=True):
            row["caption"] = _decode(completion, processor.tokenizer)
        offset += len(items)
        receipt["model_load_audit"] = precision
        receipt["generation_batch_size"] = len(prepared)
        receipt["timing"] = {**metrics, "episodes": len(episodes),
                             "prepared_cache_used": prepared_cache is not None,
                             "resident_engine_reused": reused, "resident_chunk": session._chunks,
                             "prepare_wall_s": prepared_at - started,
                             "engine_load_wall_s": loaded_at - prepared_at,
                             "total_wall_s": time.monotonic() - started}
        receipts.append(receipt)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, receipt in zip(paths, receipts, strict=True):
        _write_json(path, receipt)
    return receipts
