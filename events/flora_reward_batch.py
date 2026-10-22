"""Pinned native Flora reward jobs sharing one continuously batching vLLM engine.

Terminal jobs batch across episodes. Dense jobs pool windows across episodes while
using the released frame planning, prompts, constrained decoding and stitching. Run in a fresh process, separate from the Captioner's batch_utils package.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import uuid

from . import flora
from .flora_journal import GenerationJournal, input_digest, _sync_directory
from .flora_worker import _reward_generate, _snapshot, _write_json


def preflight(requests: list[dict], *, stage: str | None = None) -> list[dict]:
    """Verify bytes/clocks; an explicit stage also exercises pinned native decoding.

    Hosted CPU admission must supply its stage. The default retains the lightweight
    local check without downloading upstream code; it is not GPU admission proof.
    """
    if stage is not None and stage not in ("outcome", "dense", "both"):
        raise flora.FloraError("unknown reward stage")
    if not requests:
        raise flora.FloraError("reward batch requires at least one episode")
    ids = set()
    jobs = []
    for request in requests:
        identifier = request.get("id")
        if (not isinstance(identifier, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identifier)
                or identifier in ids):
            raise flora.FloraError("reward batch requires unique native episode IDs")
        ids.add(identifier)
        if request.get("use") != "noncommercial_research":
            raise flora.FloraError("Flora requires explicit noncommercial research use")
        if request.get("models", {}).get("rewarder") != flora.MODELS["rewarder"]:
            raise flora.FloraError("rewarder model must match the audited immutable release")
        task = request.get("task")
        if not isinstance(task, str) or not task.strip() or len(task) > 2000:
            raise flora.FloraError("reward requires an explicit episode task")
        paths = request.get("videos")
        hashes = request.get("video_sha256")
        if (not isinstance(paths, list) or len(paths) not in (1, 3)
                or not isinstance(hashes, list) or len(hashes) != len(paths)):
            raise flora.FloraError("reward requires hashes for one or three native videos")
        clocks = []
        for value, expected in zip(paths, hashes, strict=True):
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise flora.FloraError("reward videos must have absolute local paths")
            path = Path(value)
            if flora.digest(path) != expected:
                raise flora.FloraError(f"Native video changed: {path}")
            clocks.append(flora.video_clock(path))
        duration = flora.number(request.get("duration_s"))
        if any(clock != clocks[0] for clock in clocks) or not math.isclose(
                clocks[0].duration, duration, abs_tol=0.001, rel_tol=0):
            raise flora.FloraError("reward source clocks differ from the saved request")
        ordered = [paths[1], paths[0], paths[2]] if len(paths) == 3 else paths
        jobs.append({"id": identifier, "task": task, "videos": ordered})
    if stage is not None:
        _snapshot(requests[0], "rewarder")
        core = importlib.import_module("batch_utils.core")
        with _decoded_video_probe(core, jobs):
            _native_media_preflight(jobs, requests, core, stage)
    return jobs


@contextmanager
def _decoded_video_probe(core, jobs: list[dict]):
    """Correct the pinned reader's packet count using fully verified native frames.

    Demux packets need not produce images. Keep upstream geometry and sampling;
    only the frame clock comes from the existing strict, full-decode admission.
    """
    original = core.probe_video
    verified, audit = {}, []
    for path in dict.fromkeys(path for job in jobs for path in job["videos"]):
        clock = flora.video_clock(Path(path))
        native = original(path)
        if not math.isclose(native.fps, float(clock.fps), abs_tol=1e-9, rel_tol=0):
            raise flora.FloraError(f"Native probe disagrees with decoded frame rate: {path}")
        verified[str(Path(path).resolve())] = core.VideoInfo(
            native.path, float(clock.fps), clock.frames, native.width, native.height)
        audit.append({"path": path, "native_packet_count": native.num_frames,
                      "decoded_frames": clock.frames, "fps": str(clock.fps),
                      "duration_s": clock.duration, "packet_count_corrected": native.num_frames != clock.frames})

    def probe(path):
        key = str(Path(path).resolve())
        if key not in verified:
            raise flora.FloraError(f"Native reader requested an unadmitted video: {path}")
        return verified[key]

    core.probe_video = probe
    try:
        yield audit
    finally:
        core.probe_video = original


def _native_media_preflight(jobs: list[dict], requests: list[dict], core, stage: str) -> None:
    """Read every frame the selected native stage will use, before engine creation."""
    expected = {request["id"]: request["duration_s"] for request in requests}
    for job in jobs:
        collage = core.Collage(job["videos"])
        try:
            # Upstream probes nonempty packets, which need not equal decoded frames.
            # Never let a different native clock relabel a byte-verified episode.
            if not math.isclose(collage.duration_s, expected[job["id"]], abs_tol=0.001, rel_tol=0):
                raise flora.FloraError("native packet-derived clock differs from decoded source clock")
            plans = []
            if stage in ("outcome", "both"):
                plans.append(core.terminal_frame_indices(collage.num_frames, collage.fps))
            if stage in ("dense", "both"):
                plans.extend(window.frame_indices for window in core.dense_windows(collage.num_frames, collage.fps))
            for indices in plans:
                collage.frames(indices)
        except (ValueError, OSError) as error:
            raise flora.FloraError(f"Native reward media rejected {job['id']}: {error}") from error
        finally:
            collage.release()


def read_results(path: Path, stage: str, requests: list[dict]) -> list[dict]:
    """Reject missing/duplicate/foreign episodes and invalid native substitutions."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = {request["id"]: request for request in requests}
    seen = set()
    for record in records:
        if not isinstance(record, dict) or record.get("id") not in expected or record["id"] in seen:
            raise flora.FloraError("reward batch includes duplicate or foreign episode results")
        seen.add(record["id"])
        flora.validate(stage, record, expected[record["id"]])
    if seen != expected.keys():
        raise flora.FloraError("reward batch omitted episodes")
    return records


def _publish(source: Path, target: Path) -> None:
    """Copy exclusively: Modal Volumes do not support hard links."""
    if target.exists():
        if source.read_bytes() != target.read_bytes():
            raise flora.FloraError("replayed reward output differs from retained native output")
        return
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())
    _sync_directory(target.parent)


def recover_outcome(requests: list[dict], output_dir: Path, journal_dir: Path) -> list[dict]:
    """Publish complete paid terminal results after a filesystem failure, CPU only.

    This reads the original request binding and generation receipts, not current
    model code or remote videos. It does not authorize any further model calls.
    Continue unfinished dense work under a new stage='dense' run identity.
    """
    output_dir, journal_dir = Path(output_dir), Path(journal_dir)
    binding = json.loads((output_dir / "batch.json").read_text())
    if binding.get("requests_sha256") != input_digest(requests) or not binding.get("stages", [])[0:1] == ["outcome"]:
        raise flora.FloraError("retained terminal batch does not bind these requests")
    size = binding.get("max_num_seqs")
    if type(size) is not int or not 1 <= size <= 64:
        raise flora.FloraError("retained batch has an invalid concurrency")
    texts = []
    generation_hashes = {}
    for index in range(math.ceil(len(requests) / size)):
        intent_path = journal_dir / f"{index:06d}.intent.json"
        result_path = journal_dir / f"{index:06d}.result.json"
        intent = json.loads(intent_path.read_text())
        retained = json.loads(result_path.read_text())
        if (intent.get("request_sha256") != binding.get("journal_sha256")
                or retained.get("binding") != intent):
            raise flora.FloraError("terminal generation receipt does not match the retained batch")
        completions = retained["value"]
        expected_count = min(size, len(requests) - index * size)
        if len(completions) != expected_count or any(len(row) != 1 for row in completions):
            raise flora.FloraError("terminal journal omitted or added completions")
        texts.extend(row[0]["text"] for row in completions)
        generation_hashes[result_path.name] = flora.digest(result_path)
    candidates = list(output_dir.glob("outcome.attempt-*.jsonl"))
    if (output_dir / "outcome.jsonl").exists():
        candidates.append(output_dir / "outcome.jsonl")
    matched = []
    for path in candidates:
        try:
            rows = read_results(path, "outcome", requests)
        except (ValueError, OSError):
            continue  # Keep partial native attempts, but never adopt them.
        if ([row["id"] for row in rows] != [request["id"] for request in requests]
                or [row.get("output") for row in rows] != texts):
            raise flora.FloraError("native terminal output differs from durable generation responses")
        matched.append((path, rows))
    if not matched or len({flora.digest(path) for path, _ in matched}) != 1:
        raise flora.FloraError("no unique complete native terminal output can be recovered")
    source, rows = matched[0]
    _publish(source, output_dir / "outcome.jsonl")
    recovery = {"schema": "argus.flora.terminal-recovery/v1", "binding": binding,
                "raw_sha256": flora.digest(source), "generation_sha256": generation_hashes,
                "recovery_adapter_sha256": flora.digest(Path(__file__)), "model_calls": 0}
    receipt = output_dir / "outcome.recovery.json"
    if receipt.exists():
        if json.loads(receipt.read_text()) != recovery:
            raise flora.FloraError("terminal recovery receipt changed")
    else:
        _write_json(receipt, recovery)
        _sync_directory(output_dir)
    return rows


def _dense_batch(module, argv) -> int:
    """Feed windows across episodes to vLLM, delegating model semantics upstream."""
    parser = module.base_parser("Argus pooled Flora dense reward")
    parser.add_argument("--max-model-len", type=int, default=16384)
    args = parser.parse_args(argv)
    jobs = module.read_jobs(args)
    model_path = module.resolve_model(args.model, args.revision)
    processor = module.load_processor(model_path)
    core = module.core
    llm = module.load_llm(model_path, args, max_model_len=args.max_model_len,
                         mm_processor_kwargs={"do_sample_frames": False,
                                              "size": module.COLLAGE_VIDEO_SIZE})
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    params = SamplingParams(temperature=0.0, max_tokens=core.DENSE_MAX_TOKENS,
                            structured_outputs=StructuredOutputsParams(regex=core.dense_regex()))
    episodes = []
    try:
        for job in jobs:
            collage = core.Collage(job["videos"])
            episodes.append((job, collage, core.dense_windows(collage.num_frames, collage.fps)))
        # Requests from successive episodes share a scheduler batch rather than
        # leaving slots idle at every short episode's boundary.
        windows = [(episode, window) for episode, (_, _, items) in enumerate(episodes) for window in items]
        texts = [[] for _ in episodes]
        decoded = set()
        for start in range(0, len(windows), args.max_num_seqs):
            batch = windows[start:start + args.max_num_seqs]
            inputs = []
            for episode, window in batch:
                job, collage, native_windows = episodes[episode]
                if episode not in decoded:
                    # One native decoding pass per episode, even across batches.
                    collage.frames(sorted({index for item in native_windows for index in item.frame_indices}))
                    decoded.add(episode)
                inputs.append({
                    "prompt": module.render(processor, core.dense_messages(job["task"], window), thinking=False),
                    "multi_modal_data": {"video": (
                        collage.frames(window.frame_indices),
                        core.video_metadata(window.frame_indices, collage.fps, collage.num_frames),
                    )},
                    "mm_processor_kwargs": {"do_sample_frames": False},
                })
            outputs = llm.generate(inputs, params, use_tqdm=False)
            if len(outputs) != len(batch):
                raise flora.FloraError("dense engine omitted window responses")
            for (episode, _), output in zip(batch, outputs, strict=True):
                texts[episode].append(output.outputs[0].text)
                if len(texts[episode]) == len(episodes[episode][2]):
                    episodes[episode][1].release()
        with module.open_output(args) as stream:
            for (job, collage, native_windows), answers in zip(episodes, texts, strict=True):
                record = module.summarize(job, collage, native_windows,
                                          [core.parse_dense(text) for text in answers], answers,
                                          core.DENSE_TAIL_TRIM_S, False)
                module.write_record(stream, record)
    finally:
        for _, collage, _ in episodes:
            collage.release()
    return 0


def run(requests: list[dict], output_dir: Path, journal: GenerationJournal,
        max_num_seqs: int = 8, *, stage: str = "both",
        with_rationale: bool = True) -> dict[str, list[dict]]:
    """Execute native --jobs, retaining JSONL and journaling every paid generation.

    A single fresh journal spans terminal then dense calls. Restart with a new
    GenerationJournal pointing to the same journal directory: completed calls
    replay, while an unsettled intent refuses another payment. Partial native
    JSONL remains in attempt files. A completed batch returns without model loads.
    """
    if type(max_num_seqs) is not int or not 1 <= max_num_seqs <= 64:
        raise flora.FloraError("max_num_seqs must be an integer in 1..64")
    if stage not in ("both", "outcome", "dense"):
        raise flora.FloraError("unknown reward stage")
    if type(with_rationale) is not bool:
        raise flora.FloraError("with_rationale must be a boolean")
    if os.environ.get("FLORA_REWARDER_REPO"):
        raise flora.FloraError("Unset FLORA_REWARDER_REPO; requests pin the repository")
    jobs = preflight(requests)
    stages = ("outcome", "dense") if stage == "both" else (stage,)
    binding = {"requests_sha256": input_digest(requests), "max_num_seqs": max_num_seqs,
               "stages": list(stages), "journal_sha256": journal.request_sha256,
               "with_rationale": with_rationale,
               "adapter_sha256": flora.digest(Path(__file__))}
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    guard = output_dir / ".running"
    try:
        guard.mkdir()
    except FileExistsError as error:
        raise flora.FloraError("a worker may still own this reward batch") from error
    native_scope = ExitStack()
    try:
        saved = output_dir / "batch.json"
        if saved.exists():
            if json.loads(saved.read_text()) != binding:
                raise flora.FloraError("reward batch inputs or implementation changed")
        else:
            if any(path not in (guard, journal.root.resolve()) for path in output_dir.iterdir()):
                raise flora.FloraError("reward batch output has unbound existing files")
            _write_json(saved, binding)
        receipt_path = output_dir / "receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("binding") != binding or any(
                receipt.get("raw_sha256", {}).get(name) != flora.digest(output_dir / f"{name}.jsonl")
                for name in stages
            ):
                raise flora.FloraError("retained reward batch evidence changed")
            return {name: read_results(output_dir / f"{name}.jsonl", name, requests) for name in stages}
        if journal.index != 0:
            raise flora.FloraError("reward batch requires a fresh journal cursor for deterministic replay")
        _, spec = _snapshot(requests[0], "rewarder")
        core = importlib.import_module("batch_utils.core")
        media_audit = native_scope.enter_context(_decoded_video_probe(core, jobs))
        _native_media_preflight(jobs, requests, core, stage)
        media_path = output_dir / "media-preflight.json"
        if media_path.exists():
            if json.loads(media_path.read_text()) != media_audit:
                raise flora.FloraError("Retained native media admission changed")
        else:
            _write_json(media_path, media_audit)
        jobs_path = output_dir / "jobs.jsonl"
        encoded_jobs = "".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs)
        if jobs_path.exists():
            if jobs_path.read_text() != encoded_jobs:
                raise flora.FloraError("retained native jobs changed")
        else:
            with jobs_path.open("x") as stream:
                stream.write(encoded_jobs)
                stream.flush()
                os.fsync(stream.fileno())
        engine = importlib.import_module("batch_utils.engine")
        loaded = None
        engine_binding = None
        active_stage = None

        def shared_engine(path, args, **kwargs):
            nonlocal loaded, engine_binding
            # Terminal needs score-token logprobs; dense can share that capacity.
            kwargs["max_logprobs"] = core.TERMINAL_TOP_LOGPROBS
            config = {"path": path, "kwargs": kwargs, "tensor_parallel_size": args.tensor_parallel_size,
                      "gpu_memory_utilization": args.gpu_memory_utilization,
                      "max_num_seqs": args.max_num_seqs, "seed": args.seed}
            if loaded is None:
                engine_binding = config
                native_engine = engine.load_llm(path, args, **kwargs)
                timing = {}

                class TimedEngine:
                    def __getattr__(self, name):
                        return getattr(native_engine, name)

                    def generate(self, *args, **kwargs):
                        timing["started_at_unix"] = time.time()
                        started = time.perf_counter()
                        outputs = native_engine.generate(*args, **kwargs)
                        timing["seconds"] = time.perf_counter() - started
                        timing["ended_at_unix"] = time.time()
                        return outputs

                journaled = _reward_generate(TimedEngine(), journal)

                class MeasuredEngine:
                    def __getattr__(self, name):
                        return getattr(journaled, name)

                    def generate(self, inputs, params, **generate_kwargs):
                        generation = journal.index
                        replayed = (journal.root / f"{generation:06d}.result.json").exists()
                        started = time.perf_counter()
                        outputs = journaled.generate(inputs, params, **generate_kwargs)
                        elapsed = time.perf_counter() - started
                        tokens = sum(len(completion.token_ids) for output in outputs for completion in output.outputs)
                        inference_seconds = None if replayed else timing["seconds"]
                        metric = {"stage": active_stage, "generation": generation, "replayed": replayed,
                                  "requests": len(inputs), "output_tokens": tokens,
                                  "seconds": inference_seconds, "journaled_seconds": elapsed,
                                  "started_at_unix": None if replayed else timing["started_at_unix"],
                                  "ended_at_unix": None if replayed else timing["ended_at_unix"],
                                  "output_tokens_per_second": None if replayed else tokens / inference_seconds}
                        # Paid responses are already journaled before optional metrics IO.
                        with (output_dir / "inference_metrics.jsonl").open("a") as stream:
                            stream.write(json.dumps(metric) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                        return outputs

                loaded = MeasuredEngine()
            elif config != engine_binding:
                raise flora.FloraError("native reward stages requested incompatible engine configurations")
            return loaded

        result = {}
        for name in stages:
            active_stage = name
            module = importlib.import_module("batch_utils.terminal_reward" if name == "outcome"
                                             else "batch_utils.dense_reward")
            original_load = module.load_llm
            module.load_llm = shared_engine
            attempt = output_dir / f"{name}.attempt-{uuid.uuid4().hex}.jsonl"
            args = ["--model", spec["repo"], "--revision", spec["weights_revision"],
                    "--jobs", str(jobs_path), "--output", str(attempt),
                    "--max-num-seqs", str(max_num_seqs), "--max-model-len", "16384"]
            if name == "outcome" and with_rationale:
                args.append("--with-rationale")
            try:
                status = module.main(args) if name == "outcome" else _dense_batch(module, args)
            finally:
                module.load_llm = original_load
            if status != 0:
                raise RuntimeError(f"Released Flora {name} exited with status {status}")
            # Retain even invalid responses; they never become accepted evidence.
            result[name] = read_results(attempt, name, requests)
            with attempt.open("rb") as stream:
                os.fsync(stream.fileno())
            _publish(attempt, output_dir / f"{name}.jsonl")
        _write_json(receipt_path, {"binding": binding,
                                  "raw_sha256": {name: flora.digest(output_dir / f"{name}.jsonl")
                                                 for name in stages}})
        return result
    finally:
        native_scope.close()
        guard.rmdir()
