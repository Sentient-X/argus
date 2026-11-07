# Flora local staging qualification

The retained October 24 run did not meet the $0.80/video-hour target: completed
caption generation cost $1.264/video-hour, and GPU worker plus required CPU
creator execution cost $9.579/completed video-hour, excluding some other costs.
See [the measured result](flora-throughput-2026-10-24.md).

The remote warmer read and SHA-verified arrays but discarded their bytes. The
native loader subsequently hashed and loaded those arrays from the remote volume
again. Local staging replaces this implicit filesystem-cache assumption with an
explicit verified local copy. It does not change the model, frames, precision,
intervals or native preparation contract. It is an engineering candidate, not a
measured speedup or a claim that the cost target has been achieved.

The native preparation implementation remains byte-identical so paid CPU caches
remain valid. Hosted inference hands bounded pairs of staged caches to the existing
resident stream, preserving within-pair preparation overlap and model reuse. A
pair boundary can wait for local staging; cross-pair native preparation overlap is
not claimed. Transfer, hashing, local-copy preparation, generation, startup and
completed source duration must all remain visible in the next receipt.

## Prepared acceptance boundary

The next candidate is the 12 episodes in batches 006–011 of
`finished-throughput-gpu-prefetch-02`: 974.560103 source seconds and 408 intervals.
The retained run has no generation intents or outputs for those batches. Reuse
their paid CPU caches. Before dispatch, recheck the journals for any later intent;
never regenerate an uncertain or completed caption automatically.

Compare GPU input waits and realized cost against the retained pilot, while
reporting the changed cohort explicitly. This is not a paired speedup experiment.
Require all submitted intervals to complete, all output hashes to verify, the same
native input/model/processor bindings, and a visual audit against retained Argus
annotations. Report separately generation-only, entire GPU allocation, CPU
preparation, transfer/storage and segmentation/reward exclusions. Do not label a
partial accounting boundary all-in.

A proposed single Modal H200 call uses the existing 1,650-second worker deadline
and a $4 reservation. The recorded campaign settlement is $26.647773 against the
approved $28 ceiling. A new $4 reservation therefore requires explicit approval to
raise that ceiling; the previous one-call authorization has been consumed. The
existing no-new-GPU instruction remains in effect until explicitly replaced.
No dispatch, deployment or new model benchmark is part of this CPU-only change.

RunPod requires a separately qualified provider adapter and a live capacity/price
quote. Cheaper allocation alone cannot prove lower cost per completed video-hour;
interruptions, model reloads, input transfer and incomplete coverage belong in the
same denominator. No provider switch is implied by merging this staging change.

## Local verification

The affected Flora suite passed 87 tests with one optional SDK-dependent skip.
The staging checks verify exact copied bytes and NumPy values, corruption refusal,
current/successor bounds, committed-order advancement and deletion of owned scratch
only. Native resident and hosted checks remain green. `flora_vllm.py` is unchanged,
so its existing preparation/cache code identity is preserved. This is local CPU
verification, not a new hosted throughput measurement.

An independent review found that ignored scratch-deletion errors could grow the
disk horizon. Cleanup now records the error and refuses advancement; the new
regression test proves that a third cache is not staged after such a failure.
