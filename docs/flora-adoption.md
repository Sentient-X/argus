# Flora captioner and rewarder adoption

## Objective and boundary

Give an Argus researcher captions for existing event intervals and retrospective outcome and
process rewards for the same recorded episode. The vertical path is an Argus event result and
native video → pinned Flora inference → retained raw outputs → a research annotation document.
Existing production callers of `events.passes.annotate` require their structured skills, body
parts, scene and QC judgments; a caption or scalar reward does not replace those contracts.
No production default, QC publication, safety decision or training selection changes here.

## Source identity

| Source | Pin | License and evidence |
| --- | --- | --- |
| [Flora paper](https://research.nvidia.com/labs/gear/flora/) | Leng et al., 2026; inspected 2026-10-22 | Reported benchmarks, not reproduced |
| [Captioner](https://huggingface.co/nvidia/Flora-Captioner) | `f736ca36925ae55d2f7915993ed1bc9f87288bb3` | NVIDIA OneWay Noncommercial License; runtime artifacts downloaded separately |
| [Rewarder](https://huggingface.co/nvidia/Flora-Rewarder) | `7de68d22c4a9decec5ddb7f86ddcaee8f82d3367` | NVIDIA OneWay Noncommercial License; runtime artifacts downloaded separately |
| [Argus](https://github.com/Sentient-X/argus) | base `69f7567ad0d479e2f4cd8162678666edd7222f10` | Apache-2.0 |

Both model cards state non-commercial research and development only and exclude commercial
or production use. The adapter does not redistribute NVIDIA code or weights. Commercial adoption
requires separately established rights. A research acknowledgement does not itself establish them.

## Source model and concept map

| Source concept | Argus concept | Relation and decision |
| --- | --- | --- |
| Supplied half-open target interval | `Event.start_s/end_s` | Same; preserve times, caption each interval |
| Native before/target/after frame plans | Naming contact sheets | Replace only for Flora inference; use released preprocessing |
| One imperative caption | Event label | Refines; output a research label without inventing skill/body-part metadata |
| Outcome score 1–5 | Episode outcome judgment | Distinct; retain numeric reward, never translate it to QC success |
| Dense score 1–3 | Time-ranged quality evidence | Missing; retain retrospective spans and raw clips |
| TempSeg | Argus find/refine | Omit; existing Argus intervals are the input |
| Rubric reward | No requested consumer | Omit from this bounded integration |

Identity binds input video bytes, interval bytes, task, camera order, profile, model revisions and
adapter code. Intervals must be finite, ordered, nonempty and inside the native video duration.
Concurrent intervals remain distinct. Captions cannot change boundaries or omit an event. Scores
must be in their declared discrete ranges. A failed dense parse is unavailable evidence, never
normal quality. Every generation writes an intent and retains returned tokens or completion text/logprobs before native parsing. Runtime exceptions leave those records and a stage marker; no automatic paid retry.
Completed stages are reused only for the identical request. Separate processes avoid the two
upstream packages both named `batch_utils` and their different GPU dependencies.

## Benchmark contract

Required local evidence: input/bounds rejection; exact interval preservation; malformed/incomplete
caption and reward refusal; source identity and resume behavior; offline Argus event regressions.
Use real generated video for media validation. Scripted output fixtures test parsing only.

Required before claiming model quality: a fixed held-out episode set spanning the deployed rigs;
compare caption quality on identical Argus intervals, outcome MAE, dense failure recall, elapsed
inference time and measured GPU cost. Do not tune on the holdout or compare segmentation gains
when the segmenter is unchanged. NVIDIA reports WGO gold-window accuracy 64.7 and outcome MAE
0.40; these are upstream results, not Argus results.

Upstream requires Linux CUDA, Python 3.12 and an 80 GB GPU for each sequential model. This host
has no such GPU. Weight inference, quality parity and performance are not established locally.
No weights or paid model calls are needed for the local contract checks.

## Validation results

Local offline contract and existing event regression checks pass. Research support is not production adoption or model qualification. See the commands and final evidence below.

## Use from Argus

First produce event intervals with the existing Argus pipeline (for example a WGO run with
`--passes 2`), or use an existing Argus event JSON. Video and intervals must describe the same
whole clip in seconds relative to its first presentation timestamp. Packed videos, external MCAP
clocks, variable-rate video and recordings longer than 120 seconds are refused by this combined
caption/reward path; do not silently resample them. Single-camera and synchronized three-camera
inputs are supported. Three-camera arguments are **main, left wrist, right wrist**; the worker
reorders only the rewarder's camera collage to its native left/main/right order.

Prepare a request with CPU-only validation, no network or model calls:

```bash
.venv/bin/python -m events.flora \
  --video data/episode.mp4 --events data/events.json \
  --task 'Put the cup on the tray.' --profile human --out data/flora/episode
```

Create dedicated Python 3.12 CUDA environments on your managed GPU Job using the pinned
[captioner requirements](https://huggingface.co/nvidia/Flora-Captioner/blob/f736ca36925ae55d2f7915993ed1bc9f87288bb3/batch_utils/requirements.txt)
and [rewarder requirements](https://huggingface.co/nvidia/Flora-Rewarder/blob/7de68d22c4a9decec5ddb7f86ddcaee8f82d3367/batch_utils/requirements.txt).
Captioner also specifies Torch 2.11.0 / torchvision 0.26.0 with CUDA 12.8. The environments
may be separate because the released runtimes have different dependency constraints. Both
stages were validated upstream on an 80 GB GPU. The worker downloads pinned NVIDIA runtime
code and weights only when execution is requested. The adapter adds no GPU dependencies to
Argus's normal environment. Use `HF_TOKEN` through the Job's secret environment if required;
never put it in the request or command arguments.

For permitted noncommercial research, repeat the preparation command with:

```bash
  --execute --noncommercial-research \
  --captioner-python /opt/flora-captioner/bin/python \
  --rewarder-python /opt/flora-rewarder/bin/python
```

The captioner code revision binds weight revision
`fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6`, as declared by its released `models.json`.
The worker verifies that binding and retains its manifest/load audit. TempSeg and its base
model are not executed. Outcome and dense rewards use the pinned Rewarder weight revision.

`request.json` binds the inputs, task, profiles, model pins and adapter code. Each stage journals individual model generations before native parsing, then writes
its raw JSON and log before Argus validates the stage. A receipt binds completed output bytes to
the request. `result.json` contains captioned events, the complete outcome/dense reward records,
caption provenance and raw-output hashes. It deliberately has no invented skill/body-part
values or production QC summary. Reward `progress` ranks task completion; it is not a calibrated
success probability. Dense scores are retrospective and must not drive robot motion.

Rerunning the identical command reuses verified completed stages. A pending stage without
usable output stops: inspect its logs and reconcile the running Job before deciding whether
resuming is justified. Add `--resume-interrupted` to replay retained generations with exact input
checks. An intent without a saved response refuses to repeat that uncertain generation, even
with this flag. Incomplete stage-output bytes are kept under an `interrupted` filename. There
is no automatic retry. A stale `.running` directory likewise
requires checking whether its owner is still running before removing that local marker. Never
remove an input or anything under `raw/` to recover a run. Budget, admission and cost accounting
remain with the managed Job; this adapter does not provision GPUs or claim unmeasured cost.

## Local evidence and remaining parity

Commands executed in the Argus checkout:

```bash
uv sync --frozen --python /opt/homebrew/bin/python3.11
.venv/bin/python -m pytest -q tests/test_flora.py tests/test_events.py
.venv/bin/python -m py_compile events/flora.py events/flora_worker.py events/flora_journal.py
```

The installed interpreter was Python 3.11.14, within the project's 3.11 requirement; the local
uv version could not resolve the `.python-version` file's 3.11.15. No dependency lock changed.
The focused suite passed **43 tests**. A separate adversarial pass found the upstream runners'
in-memory generation buffering; the generation journal fixes that boundary, and the correction
pass found no remaining blocker. The reward hook replay test preserves raw token IDs and
nonfinite logprobs across a downstream parse failure without a second generation call.

A generated 32-frame, 8 FPS MP4 was decoded by both the Argus planning CLI and the pinned
Captioner `DataEnginePipeline.inspect_inputs`: both report 4 seconds and the same source hash,
and the NVIDIA validator reports zero model calls. Local inputs and requests are retained under
`data/flora-contract-evidence/` (ignored, synthetic media only).

| Dimension | Upstream reported | Reproduced here | Verdict |
| --- | --- | --- | --- |
| Native input clock | Source PTS retained | Same 32 frames / 8 FPS / 4 seconds and bytes | CPU input contract checked |
| Caption quality | WGO gold-window accuracy 64.7 | No weight inference | Unverified |
| Outcome quality | MAE 0.40 | No weight inference | Unverified |
| Dense failure detection | DenseRewardBench failure recall 53.3 | No weight inference | Unverified |
| Latency and cost | GPU-specific research measurements | No GPU run | Unverified |

Reduction decision: reuse both released inference implementations rather than copying their
prompts, frame planners, model loaders or stitching. Added one research CLI, its isolated worker
and one generation-journal invariant; no Argus dependencies, services or production settings
were added. Existing event and QC paths remain because they have real consumers and richer
contracts than these model outputs. No model-quality parity or production adoption is claimed.

The broader `.venv/bin/python -m pytest -q -ra` run completed with **2330 passed,
19 failed, 1 skipped** in 458 seconds. All 19 failures are the existing
`tests/test_transform_corpus.py::test_every_transform_keeps_the_picture_upright_whole_and_on_time`
parameterizations: the unchanged exo hand tracker invokes `ffmpeg -vsync`, which the local
ffmpeg rejects (`Unrecognized option 'vsync'`), leaving zero tracked frames. The optional
hand-pose readout test skipped because Torch is absent. The full run began before the final
journal correction; the final candidate's focused 43-test suite passed afterwards. This is
not a green repository-wide gate, and no unrelated hand-tracking code was changed.
