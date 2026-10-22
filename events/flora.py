"""Research-only Flora captions and rewards over existing Argus event intervals.

Planning is offline. Execution uses separate, explicitly supplied GPU Python environments;
raw replies are durable before parsing and an unsettled stage is never retried automatically.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
import subprocess

from label.atomic import write_atomic

MODELS = {
    "captioner": {"repo": "nvidia/Flora-Captioner",
                  "revision": "f736ca36925ae55d2f7915993ed1bc9f87288bb3",
                  "weights_revision": "fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6"},
    "rewarder": {"repo": "nvidia/Flora-Rewarder",
                 "revision": "7de68d22c4a9decec5ddb7f86ddcaee8f82d3367",
                 "weights_revision": "7de68d22c4a9decec5ddb7f86ddcaee8f82d3367"},
}
STAGES = ("caption", "outcome", "dense")
WORKER = Path(__file__).with_name("flora_worker.py")


class FloraError(ValueError):
    """Missing or invalid research evidence; no usable annotation is published."""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FloraError("expected a finite number")
    return float(value)


def score(value: object, high: int) -> int:
    if type(value) is not int or not 1 <= value <= high:
        raise FloraError(f"expected an integer score in 1..{high}")
    return value


@dataclass(frozen=True)
class VideoClock:
    duration: float
    fps: Fraction
    frames: int


def video_clock(path: Path) -> VideoClock:
    """Refuse clocks the released rewarder cannot represent, before any GPU work."""
    import av

    if path.stat().st_size > 256 * 1024 * 1024:
        raise FloraError("Flora supports at most 256 MiB per video")
    with av.open(str(path), options={"err_detect": "explode"}) as container:
        if len(container.streams.video) != 1:
            raise FloraError("each file must contain one video stream")
        stream = container.streams.video[0]
        rate = stream.average_rate
        if rate is None or rate < 4 or max(stream.width, stream.height) > 4096:
            raise FloraError("Flora requires at least 4 FPS and at most 4096 pixels per side")
        first = None
        count = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                raise FloraError("every frame needs a source presentation timestamp")
            first = frame.pts if first is None else first
            elapsed = Fraction(frame.pts - first) * stream.time_base
            if abs(elapsed - Fraction(count, rate)) > Fraction(1, 1000000):
                raise FloraError("combined Flora rewards require constant-rate native video; do not resample")
            count += 1
            if count > 7200 or elapsed >= 120:
                raise FloraError("Flora supports at most 120 seconds and 7200 source frames")
        duration = float(Fraction(count, rate))
        if count < 2 or duration > 120:
            raise FloraError("unsupported video duration or frame count")
        return VideoClock(duration, rate, count)


def prepare(videos: list[Path], events_path: Path, task: str, profile: str, out: Path) -> dict:
    if not task.strip() or len(task.strip()) > 2000:
        raise FloraError("reward scoring requires an explicit episode task of at most 2000 characters")
    if profile not in {"human", "yam", "dex", "tianji"}:
        raise FloraError("unknown Flora profile")
    if len(videos) not in {1, 3} or (profile == "human" and len(videos) != 1):
        raise FloraError("use one camera or main, left wrist, right wrist; human requires one camera")
    videos = [path.resolve(strict=True) for path in videos]
    clocks = [video_clock(path) for path in videos]
    if any(clock != clocks[0] for clock in clocks):
        raise FloraError("camera videos must have identical native clocks and start together")
    source = json.loads(events_path.read_text())
    rows = source.get("events") if isinstance(source, dict) else None
    if not isinstance(rows, list) or not rows:
        raise FloraError("input must be an Argus result with a nonempty events list")
    events = []
    previous = -1.0
    # Flora's native source clock is rounded to microseconds. Preserve its
    # terminal boundary (e.g. 395/30 -> 13.166667), without extending to a frame
    # tolerance or rewriting any interval supplied by the segmenter.
    final_boundary = max(clocks[0].duration, round(clocks[0].duration, 6))
    for row in rows:
        if not isinstance(row, dict):
            raise FloraError("invalid event")
        start, end = number(row.get("start_s")), number(row.get("end_s"))
        if not 0 <= start < end <= final_boundary or start < previous:
            raise FloraError("events must be ordered, nonempty and within the native video clock")
        events.append({"start_s": start, "end_s": end})
        previous = start
    request = {"schema": "argus.flora.request/v1", "id": "episode", "use": "noncommercial_research",
               "videos": [str(path) for path in videos], "video_sha256": [digest(path) for path in videos],
               "events_sha256": digest(events_path), "events": events, "task": task.strip(), "profile": profile,
               "duration_s": clocks[0].duration, "models": MODELS,
               "adapter_sha256": digest(Path(__file__)), "worker_sha256": digest(WORKER),
               "journal_sha256": digest(Path(__file__).with_name("flora_journal.py"))}
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
    request["request_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    saved = out / "request.json"
    if saved.exists():
        if json.loads(saved.read_text()) != request:
            raise FloraError("run inputs or implementation changed; use a new output directory")
    else:
        if any(out.iterdir()):
            raise FloraError("output directory has files but no request; inspect them before continuing")
        try:
            with saved.open("x") as stream:
                json.dump(request, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as error:
            raise FloraError("another caller prepared this run; rerun to verify its identity") from error
    return request


def read_result(path: Path) -> dict:
    # Reward scripts emit one JSONL row; the caption worker emits one JSON object.
    try:
        value = json.loads(path.read_text())
    except (ValueError, OSError) as error:
        raise FloraError(f"unreadable retained result {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise FloraError(f"{path.name} is not a result object")
    return value


def validate(stage: str, raw: dict, request: dict) -> None:
    duration = request["duration_s"]
    if stage == "caption":
        rows = raw.get("segments")
        if not isinstance(rows, list) or len(rows) != len(request["events"]):
            raise FloraError("captioner omitted or added event intervals")
        for event, row in zip(request["events"], rows, strict=True):
            if not isinstance(row, dict) or row.get("status") != "complete":
                raise FloraError("captioner returned an incomplete event")
            if any(number(row.get(key)) != event[key] for key in ("start_s", "end_s")):
                raise FloraError("captioner changed an Argus event boundary")
            if not isinstance(row.get("caption"), str) or not row["caption"].strip():
                raise FloraError("captioner returned an empty caption")
        return
    if raw.get("id") != request["id"] or raw.get("task") != request["task"]:
        raise FloraError("reward belongs to another episode or task")
    if abs(number(raw.get("duration_s")) - duration) > 0.001:
        raise FloraError("reward clock differs from the native video")
    if stage == "outcome":
        score(raw.get("score"), 5)
        expected = number(raw.get("expected_score"))
        probabilities = raw.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != {str(i) for i in range(1, 6)}:
            raise FloraError("outcome lacks the five score probabilities")
        values = [number(probabilities[str(i)]) for i in range(1, 6)]
        if any(not 0 <= value <= 1 for value in values) or abs(sum(values) - 1) > 0.001:
            raise FloraError("invalid outcome probability distribution")
        mean = sum(i * value for i, value in enumerate(values, 1))
        if not 1 <= expected <= 5 or abs(expected - mean) > 0.002:
            raise FloraError("outcome expected score disagrees with its probabilities")
        if abs(number(raw.get("progress")) - (expected - 1) / 4) > 0.001:
            raise FloraError("outcome progress disagrees with its expected score")
    elif stage == "dense":
        clips = raw.get("clips")
        if type(raw.get("invalid_clips")) is not int or raw["invalid_clips"] != 0:
            raise FloraError("dense reward includes unparsed clips; normal substitutes are not evidence")
        if not isinstance(clips, list) or not clips or any(
            not isinstance(clip, dict) or clip.get("valid") is not True for clip in clips
        ):
            raise FloraError("dense reward lacks valid clip evidence")
        for clip in clips:
            quality = clip.get("quality")
            if not isinstance(quality, str) or len(quality) != 32 or set(quality) - set("123"):
                raise FloraError("dense clip lacks all 32 action-quality scores")
        segments = raw.get("segments")
        if not isinstance(segments, list) or not segments:
            raise FloraError("dense reward has no segments")
        end = 0.0
        for segment in segments:
            if not isinstance(segment, dict):
                raise FloraError("invalid dense segment")
            start, stop = number(segment.get("start_s")), number(segment.get("end_s"))
            if abs(start - end) > 0.001 or not start < stop <= duration + 0.001:
                raise FloraError("dense segments have gaps, overlaps or invalid bounds")
            score(segment.get("score"), 3)
            end = stop
        if abs(end - duration) > 0.001:
            raise FloraError("dense reward does not cover the episode")
        score(raw.get("worst_score"), 3)
    else:
        raise FloraError(f"unknown stage: {stage}")


def execute(request: dict, out: Path, captioner_python: str, rewarder_python: str, *,
            resume_interrupted: bool = False) -> Path:
    out = out.resolve()
    # One owner prevents concurrent callers from duplicating the same compute.
    guard = out / ".running"
    try:
        guard.mkdir()
    except FileExistsError as error:
        raise FloraError("a worker may still own this run; inspect .running before resuming") from error
    try:
        raw_outputs = {}
        for stage in STAGES:
            raw_path = out / f"{stage}.json"
            pending = out / f"{stage}.pending.json"
            receipt_path = out / f"{stage}.receipt.json"
            binding = {"request_sha256": request["request_sha256"], "stage": stage}
            if pending.exists() and read_result(pending) != binding:
                raise FloraError(f"{stage} pending request binding changed")
            if resume_interrupted and pending.exists() and raw_path.exists() and not receipt_path.exists():
                try:
                    read_result(raw_path)
                except FloraError:
                    # Native scripts open output before generating. Keep their incomplete bytes.
                    raw_path.rename(out / f"{stage}.interrupted-{uuid.uuid4().hex}.json")
            if receipt_path.exists():
                receipt = read_result(receipt_path)
                if (receipt.get("request_sha256") != request["request_sha256"]
                        or not raw_path.exists() or receipt.get("raw_sha256") != digest(raw_path)):
                    raise FloraError(f"{stage} retained evidence changed or belongs to another request")
            elif raw_path.exists() and (not pending.exists() or read_result(pending) != binding):
                raise FloraError(f"{stage} raw output has no matching request binding")
            if not raw_path.exists():
                if pending.exists() and not resume_interrupted:
                    raise FloraError(f"{stage} was interrupted; reconcile its retained logs before any new compute")
                write_atomic(pending, binding)
                python = captioner_python if stage == "caption" else rewarder_python
                with (out / f"{stage}.log").open("ab") as log:
                    subprocess.run([python, str(WORKER), "--stage", stage, "--request", str(out / "request.json"),
                                    "--output", str(raw_path)], stdout=log, stderr=subprocess.STDOUT, check=True)
            raw = read_result(raw_path)
            validate(stage, raw, request)
            raw_outputs[stage] = raw
            write_atomic(receipt_path, {**binding, "raw_sha256": digest(raw_path)})
            # Removing only our local marker acknowledges a retained, parsed result.
            pending.unlink(missing_ok=True)
        result = {"schema": "argus.flora.research/v1", "request": request,
                  "events": [{**event, "label": row["caption"].strip()} for event, row in
                             zip(request["events"], raw_outputs["caption"]["segments"], strict=True)],
                  "outcome_reward": raw_outputs["outcome"], "dense_reward": raw_outputs["dense"],
                  "caption_evidence": raw_outputs["caption"],
                  "raw_sha256": {stage: digest(out / f"{stage}.json") for stage in STAGES}}
        target = out / "result.json"
        write_atomic(target, result, allow_nan=False)
        return target
    finally:
        guard.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, nargs="+", required=True,
                        help="native clip-local video: main, then optional left wrist and right wrist")
    parser.add_argument("--events", type=Path, required=True, help="existing Argus result; intervals use clip-local seconds")
    parser.add_argument("--task", required=True)
    parser.add_argument("--profile", choices=("human", "yam", "dex", "tianji"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="run pinned models; otherwise prepare only, offline")
    parser.add_argument("--resume-interrupted", action="store_true",
                        help="replay journaled generations after reconciliation; never repeats an unsettled generation")
    parser.add_argument("--noncommercial-research", action="store_true", help="acknowledge the released model use restriction")
    parser.add_argument("--captioner-python", help="Python 3.12 GPU environment with Captioner requirements")
    parser.add_argument("--rewarder-python", help="Python 3.12 GPU environment with Rewarder requirements")
    args = parser.parse_args()
    try:
        if args.execute and not (args.noncommercial_research and args.captioner_python and args.rewarder_python):
            raise FloraError("execution requires --noncommercial-research and both explicit GPU Python environments")
        request = prepare(args.video, args.events, args.task, args.profile, args.out)
        print(execute(request, args.out, args.captioner_python, args.rewarder_python,
                      resume_interrupted=args.resume_interrupted)
              if args.execute else args.out / "request.json")
    except (FloraError, OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Flora: {error}\n")


if __name__ == "__main__":
    main()
