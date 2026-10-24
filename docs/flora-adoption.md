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
| TempSeg | Argus find/refine | Separate native research baseline; preserve unlabeled overlapping intervals for matched evaluation |
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

Upstream requires Linux CUDA, Python 3.12 and an 80 GB GPU. The optional Modal host uses
one H200 per job, Python 3.12, Torch 2.11.0, torchvision 0.26.0, Transformers 5.8.1
and vLLM 0.24.0. Local contract tests still need no weights or paid calls.

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

For optimized hosted inference, use the Modal commands below. For the native differential
reference, create dedicated Python 3.12 CUDA environments on your managed GPU Job using the pinned
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
The worker verifies that binding and retains its manifest/load audit. The optional
`segment-native` stage executes TempSeg and its pinned Qwen3.5-9B base independently.
Outcome and dense rewards use the pinned Rewarder weight revision.

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
remove an input or anything under `raw/` to recover a run. The local GPU-environment path leaves placement and cost admission to its caller. The Modal
qualification path below adds bounded allocation and retained list-price estimates.

## Qualification evidence

The initial full offline Argus suite passed **2,393 tests**, with one optional Torch-dependent
hand-pose test skipped. The live qualification uses three fixed WGO development episodes:
`robointer_droid_000013`, `galaxea_049`, and `homer_33`. Actual Argus find/refine produced
one, one, and six intervals respectively; gold windows were not passed to Flora. Source
videos are native CFR bytes, bound by SHA-256. This is a functional/performance cohort,
not a held-out model-quality evaluation. The WGO source data is CC BY-NC-SA 4.0.

Native Transformers caption inference completed all six human intervals on an H200.
A downstream metadata serialization failure was recovered from retained native output
and six generation receipts, without new inference. The unknown tail of the native load
audit remains explicitly unavailable; complete FP32 GDN checkpoint-value verification
was retained. The first reward run similarly retained all three outcome generations;
CPU recovery published them after fixing unsupported Modal Volume hard links. Both failed
run receipts remain failed, alongside separate recovery receipts. New output serialization
is checked before opening its destination, and publication uses exclusive fsynced copying.

GPU measurements and the final qualification verdict are recorded in the
[qualification report](flora-qualification-2026-10-22.md). Functional validity and throughput do not establish caption accuracy,
reward calibration, failure recall, production rights, or qualified robot behavior.

The follow-up [quality pilot](flora-quality-2026-10-22.md) adds matched Astra/Sol and native
TempSeg comparisons, resident caption serving and independent outcome-reward labels. The
current offline suite passes **2,460 tests**, with two optional Torch-dependent tests skipped;
the native TempSeg CPU tests were also exercised separately with Torch installed.

## Modal hosting and inference optimization

`events/flora_modal.py` defines the authenticated `argus-flora-research` Modal app. Its
model-cache and evidence volumes survive containers. CPU prefetch downloads pinned runtime
code and weights before GPU allocation. H200 jobs scale to zero, permit one container, have
no automatic retries, a 600-second startup timeout and a 1,800-second execution timeout.
The child worker has a 1,650-second deadline. CPU and memory have explicit hard limits.
CPU admission has a separate 1,200-second deadline: the 24-interval Sol pilot needed
545 seconds before GPU dispatch. These CPU verification costs are measured separately
from model generation and remain part of end-to-end throughput and economics.

The local entrypoint reserves $4 before each allocation in `modal-budget.jsonl` beside the
manifest, with a $24 campaign ceiling. Completed receipts release only the conservative unused
reservation, retaining the entire startup allowance. Unsettled dispatches keep their reservation;
identical run IDs cannot dispatch again. Keep all manifests for a qualification campaign in the
same directory. Model downloads and input annotation have separate bounded budgets.
GPU/CPU/memory estimates use Modal's published list prices; they are not a provider invoice.

```bash
# Separate CLI dependency; Argus's CPU environment stays unchanged.
uvx --from modal==1.5.2 modal run events/flora_modal.py --prefetch-only
uvx --from modal==1.5.2 modal run events/flora_modal.py --manifest data/flora/qualification/caption.json
uvx --from modal==1.5.2 modal deploy events/flora_modal.py
# Invoke the deployed functions, with the same budget and evidence checks:
uvx --from modal==1.5.2 modal run events/flora_modal.py --deployed --admit-only --manifest data/flora/qualification/caption.json
uvx --from modal==1.5.2 modal run events/flora_modal.py --deployed --submit-only --manifest data/flora/qualification/caption.json --prepared-cache data/flora/qualification/caption.admission.json
uvx --from modal==1.5.2 modal run events/flora_modal.py --collect-only --manifest data/flora/qualification/caption.json
```

Deployed submission uses Modal's durable `spawn` and saves the function-call ID before
returning. Collection retrieves that same call and resumes interrupted evidence downloads;
it does not start inference or reserve another allocation. Keep the dispatch file with its
manifest. A submission with an uncertain outcome remains reserved until its remote evidence
is reconciled. This matters when the local connection drops: synchronous driver calls can
be canceled even though paid generations have already been retained.

A manifest supplies `run_id`, `stage` (`caption`, `caption-session`, `caption-native`,
`segment-native`, `reward`, `reward-dense`, or `reward-outcome`),
`use: "noncommercial_research"`, `max_num_seqs` (1, 2, 4, 8, or 16), and `requests`, a list
of prepared request objects. Give each episode a unique `id`. The CLI verifies and uploads
native video bytes, records every Flora implementation hash, and maps paths into the mounted
evidence volume. The remote function verifies those hashes before model work. The native
caption reference accepts one episode per isolated process; the optimized paths accept a batch.
The authenticated function requires submitted implementation hashes; use the supplied entrypoint.
`reward-dense` completes only dense work when a prior terminal result was already retained.
`reward-outcome` accepts up to 64 episodes and runs only the terminal 1–5 rubric. Set
`reward_rationale: false` for the released score-only mode; the default remains rationale
mode. This choice is bound into the generation evidence and cannot change on replay.
The deployed app is `argus-flora-research` in Modal workspace `sentient-x`; deployment itself
is distinct from a successful inference receipt.

`caption-session` supplies 1..8 `batches`, each with 1..16 requests, and a top-level
`requests` list equal to their ordered concatenation. CPU admission prepares every batch
before GPU dispatch. One bounded worker retains its verified engine and processor across
the backlog; vLLM continuously schedules intervals within each batch. Identical caption
inputs across batches are refused. Each batch has its own durable journal and output folder.
`--prepared-cache` accepts the list of CPU descriptors returned by session admission.

`segment-native` requests include `tempseg_models` from `events.flora_segment.TEMPSEG_MODELS`.
Prefetch those artifacts with `--prefetch-only --segmenter-only`. This is the released
Transformers reference, capped at 64 generations and the same worker deadline. It keeps
one verified TempSeg model across episodes, preserves overlaps and stitching provenance,
and does not feed unlabeled intervals through Argus's label-dependent refinement.
An interrupted native run can continue under a new run ID with an explicit `segment_replay`
descriptor naming the prior manifest and hashes of each settled generation intent/result.
Replay requires identical source/model bindings and exact prepared generation inputs. It
retains the reused output in the new journal before decoding; missing windows alone perform
new inference. An unsettled generation remains a refusal, not permission to retry.
See the [quality-first comparison protocol](flora-quality-plan.md) for candidate selection
and the limits on SOTA, parity and cost claims.

CPU admission retains native frame/prompt inputs as hashed JSON and NumPy arrays (`allow_pickle=False`).
It runs the real vLLM processor and checks its expanded token IDs, pixels and grids before GPU
allocation. A later batch with identical requests and implementation may reuse that admitted
cache using `--prepared-cache`; changed arrays, model manifests, model-file sizes/timestamps or
requests are refused. CPU reuse checks the retained parity certificate and processor binding
without rereading every weight shard; the first GPU engine load still hashes all checkpoint
contents before loading. Subsequent batches reuse that verified engine. This avoids repeating native frame preparation on a rented GPU. Checkpoint content SHA-256
verification remains required before loading: file timestamps alone do not establish weight
identity. The GPU worker also verifies the exact saved FP32 GDN values.

The captioner backend runs Flora's released frame planning and processor, then verifies vLLM's
FP32 processor pixels, integer video grids and prompt tokens before generation. It separately
checks the exact BF16 cast that the native vision embedding applies before its convolution;
vLLM performs that cast in its input processor. A worker hook preserves
`dt_bias` in FP32 **before checkpoint loading**; it verifies both `dt_bias` and `A_log` against
the saved checkpoint. Recurrent state uses FP32. Greedy decoding, native output-token limits,
end-token checks and the original intervals remain intact. Native bare video markers are wrapped
in vLLM's required transport delimiters; the fully expanded prompt still must equal native tokens. It pools intervals across episodes
into a continuously scheduled vLLM batch. The upstream Transformers path is the differential
reference, not an alternate product default.

The rewarder reuses one vLLM engine for terminal and dense rewards. Terminal requests batch
across episodes; the adapter also pools dense windows across episodes, while retaining native
frame indices, prompts, constrained decoding, parsing and stitching. NVIDIA's released dense
CLI only batches windows within an episode. Generation receipts commit to the Modal volume
before parsing, and intent commits precede inference. Native reward frame selection is exercised
on CPU before GPU admission. The released probe counts encoded packets, which can exceed
decoded frames; the adapter substitutes the verified decoded count and duration while preserving
frame bytes, native indices and the sampling formula. Real timestamp gaps remain unsupported.
The packet and decoded counts are retained in `media-preflight.json`. Returned tokens and log probabilities
remain available when a downstream parser fails. An unsettled generation never retries itself.

Each batch retains model/runtime provenance, raw replies, parsed outputs, timing, token counts,
and one-second GPU utilization/memory/power samples. Compare identical inputs at concurrency 1
and 8; separate model load, preparation and inference. A small functional batch cannot establish
fleet throughput, model quality or deployment economics.

## Differences from NVIDIA's released usage

The official project page links the Hugging Face model collection; no official GitHub runtime
repository is linked there. The audited executable source is the pinned `batch_utils` code in
the two model repositories. NVIDIA's full caption pipeline first runs TempSeg and then captions
its intervals. Argus supplies find/refine intervals directly and skips TempSeg. Native captions
are imperative action strings; Argus's ordinary annotation schema carries richer action, skill,
body-part and QC fields. Those cannot be manufactured from caption text or reward scores.

The model card describes neighboring-event context, while the released planner uses its own
before/target/after blocks (including fixed two-second context in the three-view planner).
This integration follows the pinned executable planner. It does not send Argus contact sheets
to Flora. Input order is main/left/right; only reward collage ordering becomes left/main/right.
DROID and Galaxea evaluated through the YAM profile are outside its native embodiment and do
not establish YAM, Dex or Tianji qualification. Human captioning upstream supports VFR, but the
combined reward path requires CFR and rejects VFR rather than resampling it.

## Import hosted results

After both hosted batches complete, assemble research annotations on CPU:

```bash
.venv/bin/python -m events.flora_import \
  --caption-receipt data/flora/qualification/caption.receipt.json \
  --reward-receipt data/flora/qualification/reward.receipt.json \
  --out data/flora/research-results
```

The importer verifies every downloaded artifact against its receipt, manifest/request identity,
and all caption/reward contracts, then writes one `<episode>/result.json` with the existing research
schema. It refuses failed batches, missing/foreign episodes, modified intervals, invalid rewards
and overwritten outputs. It makes no model calls. Retained failures require explicit recovery;
changing a failed receipt to claim success is not a recovery mechanism.
