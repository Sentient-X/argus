"""Assemble completed hosted Flora evidence into Argus research results, CPU only.

Receipt foo.receipt.json reads sibling foo.artifacts/. Reward raw hashes identify
retained batch JSONL bytes; each result contains its episode's validated record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from .flora import FloraError, validate
from .flora_journal import input_digest
from .flora_worker import _write_json


def _load(receipt_path: Path, stage: str) -> tuple[dict, dict[str, bytes]]:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete" or not receipt_path.name.endswith(".receipt.json"):
        raise FloraError("Import requires a complete hosted .receipt.json")
    root = receipt_path.with_name(receipt_path.name.removesuffix(".receipt.json") + ".artifacts").resolve()
    files = receipt.get("files")
    if not isinstance(files, dict) or "manifest.json" not in files:
        raise FloraError("Receipt lacks a hashed manifest")
    retained = {}
    for name, expected in files.items():
        relative = Path(name)
        path = root / relative
        if (relative.is_absolute() or ".." in relative.parts
                or not path.resolve().is_relative_to(root) or not path.is_file()):
            raise FloraError("Artifact path escapes its evidence directory or is missing")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise FloraError(f"Artifact hash differs: {name}")
        retained[name] = data
    if {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()} != set(files):
        raise FloraError("Evidence directory contains unhashed files")
    manifest = json.loads(retained["manifest.json"])
    if (input_digest(manifest) != receipt.get("manifest_sha256")
            or manifest.get("stage") != stage or manifest.get("run_id") != receipt.get("run_id")):
        raise FloraError("Manifest identity or stage differs from its receipt")
    return manifest, retained


def assemble(caption_receipt: Path, reward_receipt: Path, out: Path) -> list[Path]:
    caption_manifest, captions = _load(caption_receipt, "caption")
    reward_manifest, rewards = _load(reward_receipt, "reward")
    requests = caption_manifest.get("requests")
    if (not isinstance(requests, list) or not 1 <= len(requests) <= 16
            or input_digest(requests) != input_digest(reward_manifest.get("requests"))):
        raise FloraError("Caption and reward request lists differ or are empty")
    ids = [request.get("id") for request in requests]
    if (any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value)
            for value in ids) or len(set(ids)) != len(ids)):
        raise FloraError("Requests require unique safe episode IDs")
    by_stage = {}
    for stage in ("outcome", "dense"):
        name = f"results/{stage}.jsonl"
        if name not in rewards:
            raise FloraError(f"Receipt lacks hashed {stage} evidence")
        rows = [json.loads(line) for line in rewards[name].splitlines() if line.strip()]
        if (len(rows) != len(ids) or any(not isinstance(row, dict) for row in rows)
                or {row.get("id") for row in rows} != set(ids)):
            raise FloraError(f"{stage} has duplicate, missing or foreign episodes")
        by_stage[stage] = {row["id"]: row for row in rows}
    root = out.resolve()
    outputs = []
    for request in requests:
        identifier = request["id"]
        name = f"results/{identifier}.caption.raw.json"
        if name not in captions:
            raise FloraError("Receipt lacks hashed caption evidence")
        caption = json.loads(captions[name])
        if not isinstance(caption, dict) or caption.get("id") != identifier:
            raise FloraError("Caption belongs to another episode")
        raw = {"caption": caption, **{stage: rows[identifier] for stage, rows in by_stage.items()}}
        for stage, record in raw.items():
            validate(stage, record, request)
        result = {"schema": "argus.flora.research/v1", "request": request,
                  "events": [{**event, "label": row["caption"].strip()} for event, row in
                             zip(request["events"], caption["segments"], strict=True)],
                  "outcome_reward": raw["outcome"], "dense_reward": raw["dense"],
                  "caption_evidence": caption,
                  "raw_sha256": {"caption": hashlib.sha256(captions[name]).hexdigest(), **{
                      stage: hashlib.sha256(rewards[f"results/{stage}.jsonl"]).hexdigest()
                      for stage in by_stage}}}
        path = root / identifier / "result.json"
        if not path.resolve().is_relative_to(root) or path.exists():
            raise FloraError("Output escapes its directory or already exists")
        json.dumps(result, allow_nan=False)  # Validate all serialized results before publishing any.
        outputs.append((path, result))
    for path, result in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, result)
    return [path for path, _ in outputs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caption-receipt", type=Path, required=True)
    parser.add_argument("--reward-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        for path in assemble(args.caption_receipt, args.reward_receipt, args.out):
            print(path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Flora import: {error}\n")


if __name__ == "__main__":
    main()
