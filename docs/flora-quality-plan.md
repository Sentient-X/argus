# Argus and Flora: quality-first evaluation

The acceptance target is the best measured combination for segmentation, captioning and
retrospective reward modeling. The user clarified on 2026-10-22 that $0.70 per source-video
hour is a reference, not a release gate. Functional GPU qualification is not quality parity.
The optional research integration can be reviewed independently; a SOTA or default-replacement
claim requires the matched evidence below.

## Candidates and semantics

| Stage | Candidates | Comparison |
| --- | --- | --- |
| Event discovery and boundaries | Argus find/refine with Astra medium or Sol 6.1 high; native Flora TempSeg | Same native videos, instructions and reference version; raw and snapped temporal IoU F1 |
| Caption text | Argus naming with the same frontier model; Flora Captioner on each candidate's intervals | Fixed-window text accuracy first, then joint temporal/semantic F1 |
| Outcome reward | Flora Rewarder and frontier-model baseline under the same rubric | Ordinal MAE and trajectory ranking against independently supplied labels |
| Dense action quality | Flora Rewarder and frontier-model baseline under the same rubric | Per-class/macro F1, balanced accuracy and failure recall; invalid outputs remain missing |

TempSeg intervals are unlabeled and may overlap. Argus refinement prompts require labels,
and Argus cleaning removes overlaps. Do not feed blank or placeholder labels through that
path. A potential refinement experiment needs preliminary captions, an overlap-preserving
refiner, and fresh captions for changed windows; it is not implemented or proven beneficial.
Reward output is separate evidence, never a replacement for caption accuracy, QC acceptance
or a calibrated probability of success.

## Evidence boundaries

The current three clips are development smoke tests: 101.8 seconds spanning DROID, Galaxea
and HomER. Their revised VideoDB reference has only four segments. They cannot establish
SOTA, broad model parity, or robot-embodiment generalization. Do not tune against the fixed
69-episode WGO test split. The 31 development episodes may inform one candidate selection;
freeze the candidate and evaluator before running the held-out comparison.

Use one pinned reference and judge for every paired score. Report all of: pooled micro F1,
episode-macro F1 and equal-weight source F1, with raw and endpoint-snapped versions. NVIDIA's
published WGO number is source-equal, while the optimized Argus blog reports global micro.
Neither is directly interchangeable with the other. Original Macrodata (743 segments) and
revised VideoDB (534 segments) are distinct references; missing references remain unavailable.
A text-only reference judge does not establish visual factuality, which needs independent
video review. A three-episode pilot cannot provide a useful noninferiority confidence bound.

`python -m bench.flora_compare` prepares hash-bound comparison manifests and scores retained
outputs offline using the existing WGO matcher. Missing model outputs or judge verdicts
remain unmeasured. It exports pending text judgments for separately budgeted execution.
Paid results are journaled before parsing; returned responses are replayed without inference.
Any blocked dispatch must be explicitly reconciled before a continuation.

## Serving and cost

The first H200 caption qualification allocated 427.11 worker seconds for 101.8 source seconds:
144.52 seconds preparing/verifying inputs, 236.96 seconds loading the engine, 21.88 seconds
rechecking processor parity and 9.96 seconds generating eight captions. The GPU worker
estimate was $22.78/source-hour, plus $3.20/source-hour for cold CPU admission, before startup,
storage and Argus discovery. This tiny cold workload is not a steady-state price measurement.
The $0.53/source-hour generation-only component excludes these overheads and covers sparse
intervals, so it is not an achieved end-to-end cost.

A bounded caption session now accepts up to eight prepared batches and retains one vLLM
engine and processor. It verifies checkpoint content before first load, verifies CPU parity
certificates and tensor bytes per batch, and retains every raw response before parsing.
Identical requests in the backlog are rejected to avoid buying the same output twice. A
ready backlog and the existing wall-clock deadline bound idle rent. The measured benefit
must be taken from real session receipts; local tests do not prove speedup or saturation.

NVIDIA's Appendix D gives $0.7224/source-hour as a modeled estimate: eight GB300s, batch16,
$10/GPU-hour, segmentation plus captioning, excluding deployment overhead. Its assumed
production GPU-hour denominator is unverified. Treat this as a workload-specific estimate,
not a performance guarantee for a cold H200 job.

Sources: [Flora paper](https://research.nvidia.com/labs/gear/flora/assets/paper/flora.pdf?v=2026-10-19c),
[optimized Argus evaluation](https://www.midcentury.xyz/blog/optimizing-argus),
[Modal pricing](https://modal.com/pricing).

## Paired pilot and independent reward labels

The frontier pilot runs Argus's complete find/refine/name pipeline on the same three native
clips with Astra (medium) and Sol 6.1 (high). Flora then captions each frontier model's exact
intervals. Identical interval inputs shared by the two pipelines are evaluated once and reused
with provenance. Native TempSeg is a separate segmentation candidate, followed by Flora
captions without deleting overlapping intervals. A single pinned Gemini 3.5 Flash text judge
scores all candidates against both reference versions; these are development comparisons.

The outcome-reward pilot freezes 50 RoboReward test examples at revision
`469b9af76e8539e8d2ac553b081307738fd92ca5`, selected deterministically with ten examples per
human score. Gold labels are held outside inference inputs. Forty-seven pass the native
clock contract; three have a real timestamp gap before the last frame. They remain in the
coverage denominator, with no replacement or resampling. `python -m bench.flora_reward_eval`
checks source, mode, journal and receipt hashes before calculating ordinal MAE and accuracy.
This score-stratified subset is not an estimate of the full dataset's natural prevalence.
Outcome scores do not establish dense reward quality or trajectory-ranking performance.

The released reward probe counts nonempty encoded packets instead of decoded frames. A Linux
CPU audit found nine packet-count mismatches among the 47 valid examples. The adapter now uses
the strict native-clock decoder's frame count before upstream frame selection. All 47 pass
CPU media preflight; the 38 unaffected cases retain identical sampled indices and pixel hashes.
The failed GPU attempt generated no outcomes; its receipt is preserved and bound into the
changed attempt. The three genuine timestamp-gap examples remain unsupported.
