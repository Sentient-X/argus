# Flora GPU qualification — 2026-10-22

Flora Rewarder completed terminal and dense inference on three real Argus episodes on
Modal H200. Increasing vLLM concurrency from one to eight preserved the final discrete
reward timelines on this cohort and increased dense decoding throughput. Native Flora
Captioner produced six captions on actual Argus intervals; its responses were recovered
without new inference after an output-serialization failure. The optimized vLLM captioner
completed all eight intervals at concurrency one and eight. All three episodes were imported
with their captions and terminal/dense rewards into validated Argus research results.

This is functional research qualification and a small performance measurement, not a
held-out accuracy benchmark, production QC rollout or commercial-use approval. Flora's
NVIDIA OneWay Noncommercial License remains applicable.

## Inputs and runtime

The fixed WGO development cohort uses original video bytes and event intervals produced
by actual Argus find/refine calls. Reference annotation files were retained as controls
but were not used as inference intervals. No test-set tuning was performed.

| Episode | Native duration | Supplied intervals | Argus model | Flora profile |
| --- | ---: | ---: | --- | --- |
| `robointer_droid_000013` | 16.400 s | 1 | `z-ai/glm-5.3-flash` | single-camera YAM |
| `galaxea_049` | 13.167 s | 1 | `z-ai/glm-5.3-flashx` | single-camera YAM |
| `homer_33` | 72.233 s | 6 | `z-ai/glm-5.3-flash` | Human |

DROID and Galaxea use the robot/YAM preprocessing profile outside its native embodiment;
this does not qualify either embodiment. The initial Galaxea GLM Flash finder returned no
events. That result was retained, and a changed-model FlashX run supplied its interval.
Another candidate, `homer_39`, failed constant-rate admission; its source bytes were kept
and it was not resampled. All admitted videos are single-camera, constant-rate inputs.
Three-camera synchronization remains unqualified by this cohort.

| Source video | SHA-256 |
| --- | --- |
| DROID | `99a81db00989bf4f7c50c37f0de262aa3557804fc93e971bdd2db951637fa0aa` |
| Galaxea | `771a004b42304b0a8ab3a7caba1fcbb12756bbe6517a122a8bfdd99611f0f89c` |
| HomER | `c27a72aa1f474e19ef4e738453bf75ef53604969e2dcaeea0af5d106810a2690` |

The hosted image uses Python 3.12, Torch 2.11.0, Transformers 5.8.1, vLLM 0.24.0,
PyAV 16.0.1 and Hugging Face Hub 1.33.0 on one H200. Rewarder code and weights are pinned
to `7de68d22c4a9decec5ddb7f86ddcaee8f82d3367`. Captioner code is pinned to
`f736ca36925ae55d2f7915993ed1bc9f87288bb3`, with weights
`fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6` and the native mixed-precision loading audit.

Reward runs share cohort identity
`9decc0e9137f414952d0067ba8a2b949682c0dcf3d585369f97c648b05a52ddf`
and request-list identity
`321a9d537d5306b4ad230684e96af13f67a8235d0813dc842d33001c95fe71ca`.
Run manifests separately retain exact implementation hashes, request bindings and environment
versions. The native caption baseline uses only HomER's six intervals, so its aggregate rate
must not be compared with the eight-interval, three-episode optimized-caption cohort as a
controlled speedup.

## Reward inference and batching

Terminal requests batch across episodes. Dense requests pool native windows across episode
boundaries, then use the released prompt, frame metadata, constrained decoder, parser and
stitcher. One loaded vLLM engine handles both stages in the concurrency-eight run. Every
paid generation has an intent and retained completion before parsing; Modal Volume commits
bracket the paid call. Native JSONL, per-generation metrics and 1 Hz GPU samples are retained.

| Run | Stage | vLLM maximum sequences | Submitted batches | Generate time | Output tokens | Tokens/s |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `reward-seq1-v1` | terminal | 1 | 1 + 1 + 1 | 39.870 s | 119 | 2.98 |
| `reward-dense-seq1-v1` | dense | 1 | twelve single requests | 37.902 s | 820 | 21.63 |
| `reward-seq8-v1` | terminal | 8 | 3 | 27.816 s | 119 | 4.28 |
| `reward-seq8-v1` | dense | 8 | 8 + 4 | 9.053 s | 822 | 90.80 |

These times bracket `LLM.generate` and exclude journal writes, frame preparation and model
loading. They include any first-call kernel or constrained-decoder initialization. The first
sequential terminal call took 37.031 s and the first sequential dense call took 20.220 s;
subsequent sequential dense calls took 1.467–1.865 s. Excluding that first sequential dense
call yields 751 tokens in 17.683 s, or 42.47 tokens/s. The batched dense run followed terminal
inference on an already active engine. Its roughly 2.14× rate relative to that warm subset
is indicative, not a repeated, equally warmed comparison on identical timed subsets. The
4.19× ratio of all dense generation times is strongly affected by cold initialization.

During the two batched dense calls, the nine available GPU samples averaged 77.8% utilization
and reached 100%. Whole-worker utilization was much lower because preparation, loading and
compilation dominate this small job. The complete concurrency-eight worker took 427.958 s;
its GPU samples during all generation calls averaged 24.3%. Some individual calls have fewer
than five samples, so short-window percentiles are weak evidence. The workload exercised
batching and improved measured decoding rate; it did not establish sustained GPU saturation
or the best possible concurrency for this hardware.

## Reward agreement across concurrency settings

| Episode | Terminal score, seq1 / seq8 | Expected score, seq1 / seq8 | Dense worst / mean, both runs | Invalid dense clips |
| --- | --- | --- | --- | ---: |
| DROID | 5 / 5 | 4.9999 / 4.9999 | 2 / 2.5976 | 0 |
| Galaxea | 5 / 5 | 4.9999 / 4.9999 | 2 / 2.7139 | 0 |
| HomER | 3 / 3 | 3.1244 / 3.1244 | 1 / 2.3862 | 0 |

All three terminal rationale strings agree exactly. DROID's rounded terminal probabilities
changed by 0.0001 between scores four and five; the rounded expected score did not change.
For every episode, the final dense `(start_s, end_s, score)` timeline, per-second scores,
score fractions, worst score and mean score agree exactly.

Raw dense output is not entirely invariant. Galaxea has a minor rationale wording change.
HomER's window beginning at 28 s changes one quality character at zero-based group 24 from
2 to 3; this lies in the overlapping tail discarded by native stitching, so the final
score timeline does not change. Another HomER rationale changes from the knife hitting a
round object to the left hand repositioning the onion. These are substantive differences
in explanatory text despite equal final scores. Neither rationale is treated as verified
truth, and zero temperature is not a guarantee of batch-invariant prose.

## Failures, retained work and recovery

- `reward-seq1-v1` completed its three terminal generations, then failed publishing a hard
  link: Modal Volumes returned `EPERM`. Publication now uses an exclusive, fsynced copy.
  CPU recovery verified the original batch binding, completion texts and validated native
  rows, then retained the three results with a recovery receipt. It made zero model calls.
  Dense work ran separately as `reward-dense-seq1-v1`; terminal inference was not repeated.
- `native-baseline-v4` completed all six native captions, then failed serializing a `set`
  in the native model-loading audit. Six durable generations and the complete caption
  portion survived. CPU recovery retained complete JSON members, verified video, interval,
  code and weight bindings, and made zero model calls. The missing `loading_info` audit
  fields are explicitly unknown; the original hosted run remains marked failed.
- The first optimized-caption attempt, `caption-seq1-v1`, failed before retaining a model
  generation. It is not counted as caption inference evidence.

The recovered native caption baseline measured 43.700 s and 142 output tokens over six
calls. Its timing scope includes native generation, journal work and decoding, unlike the
reward engine-only measurements above. The sampled GPU utilization during those calls
averaged 32.8%. This is a functional native baseline, not an optimized serving result.

## Existing Argus annotation comparison

The HomER comparison verifies the same video SHA-256, the exact original Argus annotation
file hash and all six unchanged intervals. These are existing GLM-5.3-Flash labels compared
with native Flora output, not reference labels or an accuracy judgment.

| Interval | Existing Argus label | Native Flora caption |
| --- | --- | --- |
| 0.75–3.25 s | pick up the knife | pick up knife with right hand |
| 4.25–12.50 s | cut off the root end of the onion and drop the piece | pick up onion with left hand, cut onion with knife in right hand, place onion on counter with right hand |
| 12.75–41.50 s | peel the outer skin off the onion | hold onion with left hand, peel onion with knife in right hand |
| 42.00–48.50 s | cut the onion in half | hold onion with left hand, cut onion with knife in right hand, separate onion with both hands |
| 48.50–62.50 s | slice one half of the onion | hold onion with left hand, place onion on cutting board with right hand, slice onion with knife in right hand |
| 63.00–72.10 s | slice the other half of the onion | slice the left half of the onion diagonally using the knife with the right hand, rotate the left half of the onion counterclockwise on the chopping board with the left hand, and slice the left half of the onion diagonally using the knife with the right hand |

Flora adds hand and motion detail. The root-end disposal versus counter placement and the
last interval's half identity and rotation require video review; this report does not choose
a winner. Argus's sampled-footage duration is 72.125 s, while native decoding gives
72.233333 s; every compared event ends at or before 72.1 s. This comparison therefore
assesses text on fixed intervals, not event discovery or complete-episode coverage.

Existing Astra QC records were also inspected. They concern other recordings and provide
QC findings and ratings, not a verified matching set of native videos and Argus caption
intervals. No matched Astra caption comparison is claimed. A tracking-quality rating is
not equivalent to Flora's visual task-success reward. No private episode identifiers,
findings or recordings are included here.

## Optimized captioner result

All eight intervals across all three episodes passed real CPU vLLM processor admission.
Expanded prompt tokens and integer grids matched native inputs exactly. FP32 normalized
pixels matched exactly; vLLM's BF16 inputs matched the exact cast performed by the native
Qwen3.5 vision patch embedding. No tolerance comparison was used. Preparation took
366.499 seconds, including full checkpoint verification and video/frame work, and its
reusable cache has SHA-256
`cac345d9e7a9c69548b680999929dc4186f0b637489204582bb75bc026890f67`.
GPU loading still rehashes every checkpoint file and checks saved FP32 GDN values.

Both final runs completed all eight intervals and preserved exact event boundaries. Every
returned caption ended normally and passed the Argus contract. The two concurrency settings
produced identical caption text for all eight intervals. Five of the six shared HomER captions
exactly matched the recovered native Transformers output. The last vLLM caption is longer
and describes a final vertical cut, where native output describes a diagonal cut. This is a
real output difference, not demonstrated semantic equivalence or an accuracy improvement.

| Run | Sequence cap | Generate time | Output tokens | Tokens/s | Generation GPU mean / samples | Worker elapsed |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `caption-seq1-v3` | 1 | 9.842 s | 226 | 22.96 | 88.7% / 9 | 444.337 s |
| `caption-seq8-v3` | 8 | 9.964 s | 226 | 22.68 | 65.6% / 9 | 427.109 s |

Both calls processed 52,295 prompt tokens. Concurrency eight did **not** improve caption
inference time on this small cohort. Different cache/JIT warmness and one trial per setting
prevent a general ranking. The sequential engine reused a compiled graph; the batched run
compiled one and logged first-call JIT warnings. Both use asynchronous vLLM scheduling and
chunked prefill. The native six-interval timing includes journal/decode work and is not a
controlled comparison with these eight-interval engine-only timings.

CPU-cache verification/read-back on the GPU took 159.016 s / 144.525 s; engine loading took
229.313 s / 236.961 s; processor parity rechecks took 32.052 s / 21.878 s (seq1 / seq8).
The batched worker's full-job GPU utilization averaged only 4.5%. Loading and verification
dominate this small cohort despite high utilization during parts of generation. Pooling more
real intervals per job can amortize that overhead; sustained service utilization, ideal GPU
choice and optimal concurrency remain unmeasured. No saturation or best-performance claim
is made for the complete job.

The authenticated deployed app, `argus-flora-research` in Modal workspace `sentient-x`, ran
both the successful batched caption and reward jobs. `events.flora_import` verified their
receipt hashes, identical request lists and output contracts, then wrote three
`argus.flora.research/v1` results. Read-back confirmed eight captions, terminal scores
5 / 5 / 3 and zero invalid dense clips. Raw evidence remains in the local qualification directory and hosted evidence volume;
imported results were read back from the local qualification directory. A compact, source-footage-free
[evidence record](flora-qualification-2026-10-22.json) contains the cohort, outputs, timings
and retained hashes used by this report.

## Offline validation

The final candidate passed **2,393 tests**, with one optional Torch-dependent hand-pose
readout test skipped, in 399.15 seconds. The focused Flora suite passed 73 tests. The
FFmpeg 9 compatibility failure was fixed by replacing removed `-vsync passthrough` with
`-fps_mode passthrough`; the video-transform corpus passes. A separate adversarial check
confirmed that changing checkpoint bytes while preserving size and mtime is rejected by
full content verification. These checks do not substitute for the live GPU evidence above.

## Retained artifact identities

Full raw evidence remains outside Git under `data/flora-qualification/`. The following
hashes identify the results used here without publishing source footage or private data.

| Artifact relative to that directory | SHA-256 |
| --- | --- |
| `native-baseline-recovered.json` | `cb4b797e065a7197f388138827da0a2280914f37f6cdecd2212c61b7cd1223c8` |
| `reward-seq1-recovered/outcome.jsonl` | `d4cf80c87015468b3033f4ed9a1010cdb7c96d18370ac279bb74d3fb1b56cacc` |
| `reward-dense-seq1.artifacts/results/dense.jsonl` | `35005b604d616571bb6bb53eeaa13ffc4eb6ea715e8acd726b0fbd582b5cf05a` |
| `reward-seq8.artifacts/results/outcome.jsonl` | `f6c1f52991931b70fd325e36944ee8b5c6d3ad2082c586a893a292b80a86982e` |
| `reward-seq8.artifacts/results/dense.jsonl` | `1220d9f875d3ab861869b756e4de76a87fcdeaa0cd03bb3eb0278a9df32f5057` |

`native-baseline-recovered.receipt.json` and `reward-seq1-recovered/outcome.recovery.json`
retain recovery provenance. Each hosted run retains its submitted manifest, environment,
worker log, generation intents/results, metrics and GPU samples. All conclusions above
are limited to this cohort and these recorded attempts.
