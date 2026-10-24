"""Isolated NVIDIA FLORA inference worker; run with the upstream GPU environment.

This adapter imports the pinned upstream implementation at runtime. It does not
vendor FLORA code, replace its preprocessing, or run temporal segmentation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

if __package__:
    from .flora_journal import GenerationJournal
    from .flora import require_stage
else:
    # The isolated worker is also launched by absolute filename from outside the
    # checkout. Make its sibling packages available without requiring PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from events.flora_journal import GenerationJournal
    from events.flora import require_stage


def _caption_generate(original, journal: GenerationJournal):
    def generate(model, processor, inputs, budget):
        import torch

        class ModelProxy:
            def __getattr__(self, name):
                return getattr(model, name)

            def generate(self, **kwargs):
                # Capture returned tokens before the native decoder or stop checks.
                def encode(tokens):
                    return {"dtype": str(tokens.dtype).removeprefix("torch."),
                            "device": str(tokens.device), "shape": list(tokens.shape),
                            "tokens": tokens.detach().cpu().tolist()}

                def decode(value):
                    dtype = getattr(torch, value["dtype"])
                    if not isinstance(dtype, torch.dtype):
                        raise ValueError("Unsupported saved caption token dtype")
                    return torch.tensor(value["tokens"], dtype=dtype,
                                        device=value["device"]).reshape(value["shape"])

                return journal.run(kwargs, lambda: model.generate(**kwargs), encode, decode)

        return original(ModelProxy(), processor, inputs, budget)
    return generate


def _reward_generate(engine, journal: GenerationJournal):
    class EngineProxy:
        def __getattr__(self, name):
            return getattr(engine, name)

        def generate(self, requests, params, **kwargs):
            # These are every SamplingParams field configured by the pinned CLIs;
            # pinned code/dependencies bind their other default decoding settings.
            sampling = {name: getattr(params, name) for name in (
                "temperature", "max_tokens", "logprobs", "n", "top_p", "top_k",
                "min_p", "seed", "repetition_penalty", "presence_penalty", "frequency_penalty",
            )}
            sampling["structured_regex"] = params.structured_outputs.regex

            def encode(outputs):
                return [[{"text": completion.text, "token_ids": list(completion.token_ids),
                          "logprobs": None if completion.logprobs is None else [
                              {str(token): float(item.logprob).hex() for token, item in candidates.items()}
                              for candidates in completion.logprobs]}
                         for completion in output.outputs] for output in outputs]

            def decode(value):
                return [SimpleNamespace(outputs=[SimpleNamespace(
                    text=completion["text"], token_ids=completion["token_ids"],
                    logprobs=None if completion["logprobs"] is None else [
                        {int(token): SimpleNamespace(logprob=float.fromhex(logprob))
                         for token, logprob in candidates.items()}
                        for candidates in completion["logprobs"]],
                ) for completion in outputs]) for outputs in value]

            return journal.run({"requests": requests, "sampling": sampling, "kwargs": kwargs},
                               lambda: engine.generate(requests, params, **kwargs), encode, decode)
    return EngineProxy()


def _snapshot(request: dict, kind: str) -> tuple[Path, dict]:
    spec = request["models"][kind]
    expected_repo = f"nvidia/Flora-{kind.capitalize()}"
    if spec["repo"].lower() != expected_repo.lower():
        raise ValueError(f"Expected the released {expected_repo} repository")
    for name in ("revision", "weights_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", spec[name]):
            raise ValueError(f"{kind}.{name} must be a full immutable commit")
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(
        repo_id=spec["repo"],
        revision=spec["revision"],
        # Upstream resolves __file__ to find adjacent JSON/vendor files. Hub
        # snapshots contain symlinks into blobs/, so materialize this tiny runtime.
        local_dir=Path(os.environ.get("FLORA_RUNTIME_CACHE", Path.home() / ".cache" / "argus-flora"))
                  / kind / spec["revision"],
        allow_patterns=["batch_utils/*", "README.md", "LICENSE.pdf",
                        "MODEL_PROVENANCE.json", "PROVENANCE.json"],
    ))
    # Both releases use the same package name; one stage runs in one process.
    if any(name == "batch_utils" or name.startswith("batch_utils.") for name in sys.modules):
        raise RuntimeError("FLORA worker requires a fresh process for each stage")
    sys.path.insert(0, str(root))
    return root, spec


def _write_json(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
    # Exclusive creation protects paid results from accidental replacement.
    with path.open("x") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _caption_load_audit(audit: dict) -> dict:
    """Represent native loading-info key sets without coercing other audit values."""
    saved = dict(audit)
    if "captioner" in saved and "loading_info" in saved["captioner"]:
        report = dict(saved["captioner"])
        loading = dict(report["loading_info"])
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if isinstance(loading.get(field), set):
                loading[field] = sorted(loading[field])
        report["loading_info"] = loading
        saved["captioner"] = report
    return saved


def _caption(request: dict, output: Path, journal: GenerationJournal) -> None:
    if os.environ.get("FLORA_CAPTIONER_REPO"):
        raise ValueError("Unset FLORA_CAPTIONER_REPO; the request pins the repository")
    _, spec = _snapshot(request, "captioner")
    from batch_utils.api import DataEnginePipeline
    from batch_utils.config import MODELS
    from batch_utils import pipeline
    from batch_utils.models import Session

    pipeline.generate = _caption_generate(pipeline.generate, journal)

    native = MODELS["captioner"]
    if (native["repo"].lower() != spec["repo"].lower()
            or native["revision"] != spec["weights_revision"]):
        raise RuntimeError("Released captioner weight binding differs from the request")
    paths = request["videos"]
    if len(paths) not in (1, 3):
        raise ValueError("Captioning accepts one native video or three synchronized views")
    # Argus order is main, left wrist, right wrist; map to native named roles.
    main = paths[0]
    videos, profile = DataEnginePipeline._inputs(
        main, request["profile"],
        paths[1] if len(paths) == 3 else None,
        paths[2] if len(paths) == 3 else None,
        False, request["task"],
    )
    duration = videos["main"].duration_s
    segments = []
    for event in request["events"]:
        start, end = event["start_s"], event["end_s"]
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end)
                or not 0 <= start < end <= duration):
            raise ValueError("Every caption event must be finite and inside the native video")
        segments.append({"start_s": start, "end_s": end})
    if not segments:
        raise ValueError("Captioning requires supplied Argus event intervals")
    if len(paths) == 1:
        _, dispositions = pipeline.single_plans(videos, segments, profile)
        if any(row["status"] != "caption_input" for row in dispositions):
            raise ValueError("A caption interval has unsupported native target frames")
    session = Session()
    try:
        progress = lambda *args, **kwargs: None
        if len(paths) == 1:
            rows, audit = pipeline.caption_single(
                videos, segments, profile, session, request["task"], progress,
            )
        else:
            rows, audit = pipeline.caption_three(
                videos, segments, session, request["task"], progress, profile=profile,
            )
        _write_json(output, {
            "id": request["id"], "segments": rows, "duration_s": duration,
            "profile": profile, "camera_roles": list(videos),
            "input_sha256": {role: video.sha256 for role, video in videos.items()},
            "source_clock": "seconds relative to the first native source presentation timestamp",
            "model_revisions": {"captioner": native},
            "code_revision": spec["revision"],
            "pipeline": {"caption_inputs": audit},
            "model_load_audit": _caption_load_audit(session.audit),
        })
    finally:
        session.close()


def _reward(request: dict, output: Path, stage: str, journal: GenerationJournal) -> None:
    if os.environ.get("FLORA_REWARDER_REPO"):
        raise ValueError("Unset FLORA_REWARDER_REPO; the request pins the repository")
    _, spec = _snapshot(request, "rewarder")
    if spec["weights_revision"] != spec["revision"]:
        raise ValueError("Rewarder code and weight revisions must match")
    module = importlib.import_module(
        "batch_utils.terminal_reward" if stage == "outcome" else "batch_utils.dense_reward"
    )
    load_llm = module.load_llm
    module.load_llm = lambda *args, **kwargs: _reward_generate(load_llm(*args, **kwargs), journal)
    paths = request["videos"]
    ordered = [paths[1], paths[0], paths[2]] if len(paths) == 3 else paths
    # The dense CLI otherwise creates its GPU engine before checking the media.
    # Run its own native collage validation first, without loading model weights.
    core = importlib.import_module("batch_utils.core")
    collage = core.Collage(ordered)
    try:
        if stage == "outcome":
            core.terminal_frame_indices(collage.num_frames, collage.fps)
        else:
            core.dense_windows(collage.num_frames, collage.fps)
    finally:
        collage.release()
    args = ["--model", spec["repo"], "--revision", spec["weights_revision"],
            "--video", *ordered, "--task", request["task"],
            "--output", str(output)]
    if stage == "outcome":
        args.append("--with-rationale")
    result = module.main(args)
    if result != 0:
        raise RuntimeError(f"Released FLORA rewarder exited with status {result}")
    # Preserve the released JSONL record, including invalid_clips and raw answers.
    # The parent parses it and rejects malformed/invalid rewards before adoption.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("caption", "outcome", "dense"), required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.request.is_absolute() or not args.output.is_absolute():
        parser.error("--request and --output must be absolute paths")
    if args.output.exists():
        parser.error("Output already exists; preserve the original result")
    request_bytes = args.request.read_bytes()
    request = json.loads(request_bytes)
    require_stage(request, args.stage)
    journal_path = Path(__file__).with_name("flora_journal.py")
    if hashlib.sha256(journal_path.read_bytes()).hexdigest() != request["journal_sha256"]:
        raise ValueError("Generation journal code differs from the saved request")
    paths = request["videos"]
    if not isinstance(paths, list) or not paths or not all(
        isinstance(path, str) and Path(path).is_absolute() and Path(path).is_file()
        for path in paths
    ):
        parser.error("videos must be nonempty absolute local file paths")
    if not isinstance(request["task"], str) or not request["task"].strip():
        parser.error("task must be a nonempty episode instruction")
    expected = request["video_sha256"]
    if not isinstance(expected, list) or len(expected) != len(paths):
        parser.error("video_sha256 must bind every video in input order")
    for path, digest in zip(paths, expected, strict=True):
        with Path(path).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Native video changed since the request was saved: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal = GenerationJournal(args.output.parent / f"{args.stage}-generations",
                                hashlib.sha256(request_bytes).hexdigest())
    if args.stage == "caption":
        _caption(request, args.output, journal)
    else:
        _reward(request, args.output, args.stage, journal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
