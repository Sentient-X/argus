# Flora bulk captioning qualification — 2026-10-23

The target is useful first-pass labels for finished recordings at about $0.70 per
source-video hour. This is a captioning serving experiment, not a change to the
production annotator or an assertion of parity with frontier models.

## Frozen behavior and acceptance

Use the released captioner code `f736ca36925ae55d2f7915993ed1bc9f87288bb3`, weights
`fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6`, vLLM 0.24.0, BF16 with verified FP32 GDN,
native before/target/after frames, temperature zero and the native 512-token ceiling.
Preserve every supplied interval. Do not obtain a speedup by dropping frames, captions,
cameras from an already-frozen input, or failed episodes from coverage reporting.

The finished-data cohort is chosen in immutable source-hash order from retained Catalog
recordings with existing Argus annotations. Its native head-camera stream is remuxed
without reencoding; decoded pixels and timestamp alignment are checked against the MCAP.
The view and the existing intervals are frozen before inference. This is single-view
captioning of human-operated grippers, not a synchronized three-camera robot qualification.
The selected cohort contains retained GLM 5.3 Flash descriptions. These are comparison candidates, not human ground truth. The prior Argus reader saw additional camera views, so that comparison does not isolate model quality.

The released human caption profile supports variable frame timestamps. Caption-only
requests use that native contract; combined caption/reward requests still require the
rewarder's constant-rate clock. Scope is part of request identity and is enforced before
model work. No frame resampling is introduced to make recordings pass admission.

## Runtime change

Hosted caption sessions keep one verified engine and prepare one successor batch on a CPU
thread while the current batch generates. Each cache is limited to 4 GiB of array files and
64 MiB of metadata; the stream holds at most the current and one successor batch. The first
checkpoint load completes before lookahead begins. The backlog is bounded to 32 batches, and the worker deadline remains unchanged. Caption sessions admit up to 32 active sequences; reward concurrency is unchanged. Paid generations and complete outputs are persisted before a malformed successor
can fail the session. There are no automatic uncertain retries.

CPU admission now uses four CPU cores and 16 GiB (the main cohort used 32 GiB), the existing durable journal, and records its execution time and cost
estimate. A thread-scoped PNG encoder removes compression from upstream temporary frame
files while preserving decoded pixels. Explicit encoder settings and other threads are untouched.
All native processor parity checks remain. Completed preparation caches survive runtime
changes that do not change their pinned caption preparation implementation. A saved call ID permits collection without buying the work again. GPU input-wait,
preparation-overlap, model-load and generation timings distinguish utilization from useful
throughput; merely using continuous batching does not establish GPU saturation.

## Interpreting the published target

Appendix D models $0.7224/video-hour using eight GB300s, batch 16 and an assumed
$10/GPU-hour. It explicitly excludes deployment/storage overhead and states that
the production GPU-hour denominator is unverified. Its workload assumption is
890 captions over 1.24543 source hours (714.6 captions/hour). This frozen finished-data
cohort has 1,315.2 captions/hour, about 1.84 times that caption density. Hardware,
segment workload and accounting therefore differ; the paper number remains a useful
target rather than a measured rate for this Modal pipeline. No intervals are dropped
to make the target easier.

## Cost accounting

Report distinct completed source seconds, caption density, complete/failed coverage, CPU
preparation, GPU worker time, startup, transfer and retained-storage assumptions separately.
Use current Modal list rates, not credits: H200 $0.001261/s, CPU $0.0000131/core/s,
memory $0.00000222/GiB/s, egress $0.04/GiB, volume $0.09/GiB-month. Local source extraction
is timed and reported separately; it is not silently priced at zero as rented compute.
Client submit-to-collection wall time includes queueing and delayed collection and is only
an upper bound on allocated time. A worker-only estimate must not be labelled all-in.

The existing $24 GPU campaign ceiling is unchanged. This pass can reserve one $4 GPU
allocation from its remaining balance. CPU admission has a separate $3.60 ceiling. Original calls reserve $0.45 each; the
smaller four-core/32-GiB worker reserves $0.225, covering its configured 600-second
startup plus 1,200-second execution ceiling at list rates. Up to three admissions can
prepare independent episodes concurrently; their reservations share that same ceiling. Completed charges are reconciled from retained evidence;
a missing result does not release a reservation.

## Evidence status

The 16-episode immutable cohort contains 1,339.639170 unique source seconds and 456
unchanged intervals (1,225.4 captions per video hour). All 40,190 decoded frames match the
original MCAP camera stream. The intervals cover 748.74 seconds; gaps are retained. This
measures captioning on supplied intervals, not the cost of finding new intervals.

The first CPU control admission took 540.68 seconds, including 162.83 seconds reading
109.45 GB for checkpoint integrity. The first finished batch retained 60 captions after
623.79 seconds of CPU preparation. The remaining slower admission was explicitly stopped
after that batch committed, and the completed cache was retained for reuse. Queued small
admissions were canceled and their conservative cost reservations reconciled separately.

On 27 fixed native audit frames, three rotated local trials measured default PNG save+SHA
at 1.504 seconds, compression zero at 0.332 seconds, and compression one at 0.542 seconds.
Compression zero used 2.87 times as many ephemeral file bytes. Every decoded pixel matched.
These local results are not Modal throughput measurements.

Before the final prefetch delta, the full offline suite passed 2,482 tests with two
optional skips. This covered overlap,
paid-output preservation, memory bounds, lossless serialization, native-timestamp admission
and the 32-batch backlog. The focused Flora suite passes 169 tests with one optional skip after the
prefetch implementation.
A separate money/data review found no remaining blocker in these changes.
The frozen extension acquired all 41 selected sources (5,954,659,926 bytes) and
admitted 38 episodes: 3,197.038861 unique native seconds and 1,168 intervals. Three
clock mismatches are retained as exclusions, without shifting or clipping events.
Original 16-episode proofs compare decoded RGB; the other 22 admitted episodes
compare compressed H.264 payloads, exact timestamps and frame counts.

A controlled CPU memory trial rebuilt the largest admitted batch (96 captions,
2.633 GB arrays) at 16 GiB. Its request digest, native prepared objects and all 288
array hashes exactly matched the 32-GiB baseline. The whole CPU execution took
486.669 seconds ($0.042788 at list rates), including 160.931 seconds verifying
109.447 GB of model files. This is a cold trial; the earlier 300.407-second warm
batch is not a paired speedup comparison. The 16-GiB resource rate is 28.8% lower;
the main cohort is still charged at its actual 32-GiB allocation.

## Collected GPU result

The bounded GPU run reached its unchanged 1,650-second deadline. It retained 323
completed captions for 11 episodes / 889.239707 source seconds (14.82 minutes), out
of 38 admitted episodes. Three of 41 selected sources were excluded for explicit
clock mismatches. Batch 006 has an intent without a result and remains uncertain;
batches 007–018 were never started. Neither is silently retried or counted as
completed. All 37 returned artifacts were downloaded and hash-verified. The GPU
container terminated; no GPU was left running.

| Measured boundary | Result |
| --- | ---: |
| Completed production generation only | $1.023 / source-video hour |
| Production span including initial input wait and inter-batch work | $4.066 / source-video hour |
| Whole GPU worker plus measured execution of CPU cache creators | $11.523 / completed source-video hour |
| Whole GPU worker, including cold start and failed tail | 1,653.107 s; $2.49269 |
| Mean GPU utilization during completed production span | 23.48% |
| Optimized CPU preparation, 12 nonfirst caches at actual 4 CPU / 32 GiB | $0.467 / source-video hour |

**The $0.70 target was not reached.** The whole-execution figure includes cold
setup, control and failed GPU work and whole CPU creator calls (some prepared
additional retained caches). It is a realized partial-pilot cost, not a mature
production rate. It still excludes container startup, local extraction, ingress,
storage and the already-supplied Argus segmentation. It is therefore a lower bound
on that accounting scope, not an all-in claim. Even generation alone exceeds the
target for this source mix; input prefetching alone cannot close the entire gap.

The engine load took 560.42 seconds, including 282.18 seconds of compilation;
the first control generation incurred additional kernel compilation. Later
engine reuse took about 0.03 seconds. Compiler caches are already persisted on
the model volume; changing concurrency created new compilation work in this run.
The complete campaign's measured CPU worker execution was $0.86525, including
controls, the memory trial and prepared caches not yet used by the GPU. Canceled
call upper bounds and provisional shared-app billing remain separate in the
local cost audit. The GPU campaign settled conservatively at $23.24140 against
its unchanged $24 ceiling, leaving $0.75860; there is no outstanding GPU call.

## Input starvation and the recorded intervention

Serial cold array reads from Modal Volume measured about 5.7 MB/s, while warm
memory-map copies and hashing were hundreds of MB/s. Batch 002 spent 243.31
seconds waiting for input and only 34.31 seconds generating. At 19:54:44 UTC a
bounded four-thread read-only SHA-verifying warmer was started in that same
container, limited to the current and successor caches. It stopped before the
worker deadline. The ordinary loader still checked all input hashes.

Cold caches 004–006 were read and verified at 26.1–29.2 MB/s. Batch 004 waited
30.94 seconds and batch 005 waited 4.47 seconds; batch 005 generated 58 captions
in 28.99 seconds. Batch 003 was partly warm before intervention and is not a
clean speedup measurement. These are different batches and the run mixes
configurations: no paired end-to-end speedup or sustained saturation claim is
made. The retained helper source, SHA, timestamps and partial-cache stop are
included in the evidence. The permanent bounded prefetch implementation uses four streaming readers with
256-KiB buffers, a four-item queue and only the current and successor cache. It
advances after the normal batch validation and durable commit. Future-cache
errors are telemetry; the ordinary loader still rejects invalid input before
its model call. Cancellation waits at most 0.5 seconds for readers, so blocked
remote storage cannot hold completed results hostage. This implementation
requires its own uninterrupted GPU qualification.

## Quality preservation and its limits

All 16 retained controls match the earlier 16-sequence baseline exactly in prompt
token IDs, output token IDs and caption text. No frame, interval, prompt, model
precision, sampling policy or token ceiling was reduced to obtain speed.
The 16-GiB CPU trial independently preserves the largest batch's prepared inputs.

Only three of nine predeclared visual audit events have completed Flora outputs
in this bounded run; the other six remain missing. The two ring-placement
captions agree with the sampled visual action. The white-box caption describes
a handover, whereas the saved Argus caption and pre-output observation describe
a lift; three stills do not resolve the complete motion. This sparse review is
not a quality score or evidence of Astra/Sol parity. The review retains both
candidates alongside native frames and the pre-output observations.

Aggregate timings, hashes and limits are retained in
[the machine-readable result](flora-throughput-2026-10-23.json). Private episode
inputs, raw candidates, journals, billing snapshots and the visual comparison
remain in the ignored `data/flora-throughput-run/` and
`data/flora-qualification/finished-throughput-gpu-01.artifacts/` evidence folders.

Sources: [released Flora paper, Appendix D](https://research.nvidia.com/labs/gear/flora/assets/paper/flora.pdf?v=2026-10-19c),
[Modal prices](https://modal.com/pricing), [prior measured pilot](flora-quality-2026-10-22.md).
