# Flora quality pilot and resident inference

This follow-up evaluates the combination of Argus and Flora, with quality taking precedence
over a fixed captioning price. It extends the [initial GPU qualification](flora-qualification-2026-10-22.md)
with matched Astra and Sol 6.1 pipelines, native TempSeg, a resident caption engine, and an
independently labeled outcome-reward cohort. It does not establish SOTA or model parity.

## What was compared

The same three WGO development clips contain 101.8 seconds of distinct source video: DROID
`robointer_droid_000013`, Galaxea `galaxea_049`, and HomER `homer_33`. Argus runs its complete
find/refine/name pipeline with Astra medium and Sol 6.1 high. Flora captions each pipeline's
unchanged intervals. Native TempSeg supplies a third set of intervals, preserving overlaps,
followed by Flora captions. The identical Galaxea input shared by Astra and Sol is generated
once and reused with explicit provenance.

Astra produced 16 intervals, Sol 25, and TempSeg 36. The original GLM-based qualification has
eight. These counts are predictions, not annotation quality. All systems use the same pinned
text-judge protocol and are scored separately against original Macrodata and revised VideoDB
references. The three selected clips have 13 original reference segments but only four revised
ones. This difference can reverse rankings. Raw and snapped temporal scores, pooled micro,
episode macro and source-equal aggregates remain in the [evidence record](flora-quality-2026-10-22.json).
Text matching against references does not independently establish visual factuality.

The [candidate protocol](flora-quality-plan.md) reserves the 69-episode WGO test split for a
frozen candidate. These three development clips cannot establish noninferiority to Astra or
Sol 6.1. Dense reward quality and a matched frontier-model reward baseline remain unmeasured.

## Matched quality results

At snapped temporal IoU 0.75, joint temporal/semantic F1 is shown below as
**pooled micro / source-equal**. Each source contributes one episode in this pilot;
these numbers cannot be compared directly with the full published WGO benchmark.

| Pipeline | Revised VideoDB | Original Macrodata |
| --- | ---: | ---: |
| Argus + GLM baseline | 0.1667 / 0.3333 | 0.3810 / 0.1905 |
| Same GLM intervals + Flora captions | 0.3333 / 0.6667 | 0.3810 / 0.1905 |
| Argus + Astra | 0.3000 / 0.7083 | 0.1379 / 0.0606 |
| Same Astra intervals + Flora captions | 0.2000 / 0.6667 | 0.0690 / 0.0303 |
| Argus + Sol 6.1 | 0.2069 / 0.6933 | 0.0526 / 0.0215 |
| Same Sol intervals + Flora captions | 0.1379 / 0.6667 | 0.1053 / 0.0430 |
| Native TempSeg + Flora captions | 0.0000 / 0.0000 | 0.2041 / 0.4231 |

Flora does not consistently improve the frontier pipelines. On Astra intervals, accepted
captions fell from 3/3 to 2/3 matched windows against the revised reference and from 2/2 to
1/2 against the original. On Sol intervals the change was 3/3 to 2/3 and 1/2 to 2/2,
respectively. These tiny, temporally selected denominators are not gold-window caption
accuracy or a noninferiority test.

Native TempSeg's original-reference segmentation F1 is 0.4082 pooled / 0.7521 source-equal,
but only five of ten matched captions were accepted. Its 36 narrow/overlapping intervals
have no matches at this threshold against the revised broad segments. The correct next
choice is a reference contract and a frozen development candidate, followed by held-out
qualification—not a blanket replacement of either frontier annotator.

A qualitative spot-check also exposes a judge limitation: it rejected Flora's “dice onion”
caption against “slice the purple onion,” while accepting a frontier description of parallel
cuts followed by cross-cuts. Six sampled frames in the 50–72 second interval confirm onion
cutting and the stated hand/knife assignment, but do not adjudicate every temporal detail.
The original verdicts remain unchanged; this disagreement needs independent adjudication
before reading the numeric difference as a visual quality difference.

All 51 distinct text-judge verdicts are retained. Fourteen were new for this comparison;
37 earlier verdicts were reused. There are no pending verdicts in the report.

## Resident caption performance

All 76 unique interval requests completed in one H200 worker. The request variants cover
292.23 seconds when counted per candidate, but only 101.8 seconds of distinct video. These
are three experimental pipelines, so treating their combined workload as unique production
video would overstate throughput.

| Batch | Captions | Engine reused | Preparation / verification | Engine load | Generation | Batch total | Generation GPU mean |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Astra intervals | 16 | No | 132.99 s | 238.72 s | 32.10 s | 408.19 s | 30.6% |
| Sol intervals, excluding shared Galaxea | 24 | Yes | 21.68 s | 0.024 s | 12.69 s | 39.19 s | 98.2% |
| Native TempSeg intervals | 36 | Yes | 25.37 s | 0.023 s | 18.69 s | 50.40 s | 100.0% |

Generation used 63.47 seconds total for 1,309 output tokens. The warm batches demonstrate
engine reuse and near-full GPU utilization during generation (13 and 19 one-second samples).
They are different inputs, so their relative rates are not a controlled concurrency speedup.
The first batch includes cold kernel work. Overall worker time was 511.39 seconds and average
GPU utilization was only 10.4%: loading and verification still dominate this bounded run.

At H200 plus 8 CPU / 64 GiB list prices ($5.428368/hour), measured worker time costs about
$0.771. Initial CPU preparation took 404.07, 544.88 and 595.50 seconds respectively, about
$0.381 in total before later admission, startup and storage. These are benchmark costs,
not a steady-state pipeline price or provider invoice. For the warm native-interval batch
alone, generation is approximately $1.00/source-hour and the full batch $2.69/source-hour,
excluding its initial CPU preparation, segmentation and cold startup. The $0.70 target has
not been reached. This batch has 36 intervals rather than the initial qualification's eight;
caption density and context affect cost materially.

The next throughput improvement should overlap verified CPU input delivery with GPU work
and feed a longer ready backlog to the retained engine. Hardware/concurrency selection needs
a matched workload sweep. Increasing the sequence limit alone cannot eliminate verification
and model-loading time. NVIDIA's $0.7224 estimate uses eight GB300s and batch16 for segmentation
plus captioning under its Appendix D assumptions; it is not a measured Modal H200 price.

## Runtime changes and recovered failures

The captioner uses vLLM 0.24.0 on Modal H200 with a sequence limit of 16. A bounded session
reuses one engine and processor across prepared batches. Native frame plans, prompt tokens,
FP32/BF16 pixel parity and exact FP32 GDN checkpoint values remain checked. CPU preparation
is durable; subsequent admission verifies the cache, arrays, processor certificate and model
metadata without rereading all weight contents. Full checkpoint hashes are still checked
before the first GPU engine load.

Modal calls use durable submission and a saved call ID. Collection resumes evidence transfer
without another inference allocation. An interrupted TempSeg run was continued from exact
Linux-prepared generation inputs and retained token receipts after verifying the old worker
had stopped; only missing windows incurred new inference. Genuine uncertain generations
remain blocked. Native TempSeg endpoint rounding is admitted to microsecond precision,
without extending the allowed boundary by a frame.

The released reward probe counted packets instead of decoded frames. Nine of the 47 valid
benchmark videos have extra packets. The first outcome attempt failed before generating any
score; its failed receipt remains retained. The corrected adapter uses the verified decoded
count and keeps the native sampling formula. Linux CPU checks admitted all 47 videos and
proved identical selected indices and pixel hashes for the 38 unaffected cases. Three genuine
timestamp gaps remain unsupported, with no replacement examples.

## Outcome reward results

The frozen RoboReward test pilot selects ten examples per human score at source revision
`469b9af76e8539e8d2ac553b081307738fd92ca5`. Gold labels remain outside inference inputs.
All 47 admissible videos returned valid score-only outputs; three true timestamp-gap failures
remain missing among the original 50, giving **94% coverage** without replacement.

- Ordinal MAE: **21/47 = 0.4468** on the 1–5 scale.
- Exact score accuracy: **33/47 = 70.2%**.
- Weakest group: gold score 4, with **3/10 correct** and MAE **1.20**.
- Full 50-example MAE bounds allowing any missing prediction: **0.42–0.56**. These are
  missing-result bounds, not statistical confidence intervals.

| Human score | Examples answered | Exact matches | MAE |
| --- | ---: | ---: | ---: |
| 1 | 10 | 9 | 0.10 |
| 2 | 9 | 6 | 0.5556 |
| 3 | 8 | 6 | 0.25 |
| 4 | 10 | 3 | 1.20 |
| 5 | 10 | 9 | 0.10 |

The full confusion matrix and all example-level outcomes are retained in the JSON evidence.
The 47 predictions used 38.66 seconds of generation within a 254.38-second worker run.
Generation GPU utilization averaged 38.1%; whole-worker utilization was 9.85%. This run
qualifies the packet-count fix and independent ordinal evaluation; it does not establish
optimal reward serving throughput. The previously qualified dense batching remains separate.

This score-stratified subset does not estimate the natural test-set prevalence. NVIDIA's
reported outcome MAE of 0.40 is not a matched comparison with these 47 examples. There is no
frontier reward baseline, dense human-label evaluation or trajectory-ranking result here.
The score-4 errors are a concrete reason to retain human review before using reward thresholds.

## Validation

The full offline suite passed **2,460 tests**, with two optional Torch-dependent tests skipped.
Native TempSeg's CPU suite separately passed 30 tests in a Torch environment. A separate
review exercised durable collection, continuation refusal and checkpoint tampering; no scoped
blocker remained. No new generation was needed to repair collected output or score results.
The immutable run receipts and compact metrics are linked in the accompanying JSON evidence.

## Decision boundary

The integration remains an optional noncommercial research path. NVIDIA's released terms do
not authorize commercial production deployment. Caption strings and scalar rewards do not
replace Argus's structured annotation and QC contracts. A production/default choice needs
independent visual review, a frozen held-out comparison, relevant dense reward labels, and
separately established rights. Functional GPU completion and a green PR are narrower evidence.
