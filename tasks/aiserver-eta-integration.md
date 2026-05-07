# reslock — accommodate aiserver per-request ETA + cross-consumer wait prediction

Filed 2026-05-07 from `~/workspace/straiqr/aiserver`. Cross-repo: see
`~/workspace/straiqr/aiserver/tasks/per-request-lease-eta.md` for the
aiserver-side context.

## Why

Aiserver shipped `/v1/estimate/{asr,dia,llm}` endpoints that return
`wait_sec` to the scriba CPU frontend. The wait estimate peeks
`pool.status()` to find in-flight work that would block our request.

Aiserver also propagates its own predicted runtime onto its lease via
`LeaseHandle.update(estimated_seconds=N)` so peer consumers (other
aiservers, scriba, kirk-rpcserver) can include aiserver in their own
queue calculations.

Three rough edges came out of that work — two are reslock-shaped, one
is a documentation gap.

## Issue 1 — `Lease.acquired_at` reflects acquisition, not "current request started"

Aiserver's lease is a persistent model-load lease (reclaimable, lives
across many requests so the weights stay in VRAM). On each incoming
request we update `estimated_seconds=N`, but `acquired_at` is whenever
the model was first loaded — could be 10 minutes ago.

Peer-side wait math today:
```python
elapsed = (now - lease.acquired_at).total_seconds()
remaining = lease.estimated_seconds - elapsed
```
For a 30s ETA on a 10-min-old lease: `30 - 600 = -570 → clamped to 0`.
The peer thinks "no wait" the moment we set the new ETA.

Workaround that needs no reslock change: **set `progress=0.0` together
with `estimated_seconds=N` at request entry**, and have the peer
prefer `estimated_seconds * (1 - progress)` when `progress` is set.
Reslock 0.7.1's `LeaseHandle.update()` already accepts both, so this
is purely a convention issue.

**Reslock-side ask** (optional): document this idiom in
`LeaseHandle.update()` and in the `Lease.acquired_at` /
`Lease.estimated_seconds` field docstrings. Something like:

> ``estimated_seconds`` is the consumer's prediction of remaining
> wall-clock time for *whatever the lease is currently doing*, not
> for "until release". On a lease that handles many sub-requests,
> reset ``progress=0.0`` whenever you update ``estimated_seconds`` so
> peer consumers can compute remaining time correctly via
> ``estimated_seconds * (1 - progress)`` instead of subtracting
> ``elapsed`` from acquisition time.

Optional second part: add a separate `Lease.eta_set_at: datetime`
that's auto-stamped by `update(estimated_seconds=...)`, so peers can
compute `(now - eta_set_at)` for the elapsed term without depending
on consumer discipline. Schema bump.

## Issue 2 — `QueueEntry.estimated_seconds` doesn't exist

`pool.status().queue` is a list of `QueueEntry` items. They carry
`pid`, `resources`, `priority`, `queued_at`, `label` — but no ETA.

Aiserver currently only sums remaining time over **in-flight leases**
on the GPUs we'd need. If three jobs are queued ahead of us, our
estimate is just "time until current lease releases" — completely
ignores the queue tail.

**Reslock-side ask**: extend `QueueEntry` to carry the
`estimated_seconds` value passed to `acquire()` /
`acquire_async()` / `_acquire_blocking()`. The acquire APIs already
take that parameter (line 267, 290 in `pool.py`); plumb it through
into the `QueueEntry` written in `_enqueue` so peers walking the
queue can sum ETAs.

Schema impact: `QueueEntry.estimated_seconds: int | None = None`
(additive Optional, no schema bump if existing readers handle the
new field with `extra="allow"` semantics — but `model_config = {"extra": "forbid"}` on `QueueEntry` today means coordinated bump
across consumers).

Wire-level: trivial. Conceptual: the queue ETA is "what the consumer
*would* run for once promoted", not "remaining time" — peers
computing wait_sec for their own request would do something like:

```python
total = max(active_leases_remaining_eta_on_my_gpus)
for q in queue_ahead_of_me:
    total += q.estimated_seconds  # roughly; could overlap if multi-GPU
```

(Real math is fiddlier with multi-GPU + priority but the data is
necessary.)

## Issue 3 — In-place reclaimability toggle

Aiserver's lease stays `reclaimable=True` so the model can be evicted
under VRAM pressure — but during active inference, eviction would
orphan the in-flight request. Today there's no clean way to mark a
lease "busy, do not reclaim until current work finishes" without
either (a) taking a second non-reclaimable lease per request or (b)
releasing-and-reacquiring the model lease as non-reclaimable.

Both work, both are ugly.

**Reslock-side ask**: either

- `LeaseHandle.update(reclaimable=...)` — letting the consumer flip
  the lease in place. Update path in `pool.py:_update` is trivial;
  `update()`'s signature gains one more keyword. No schema bump.
- Or a dedicated `Lease.busy: bool` field plus a corresponding
  `update(busy=...)` toggle, and the reclaim path treats `busy=True`
  as "skip even when reclaimable". This is more explicit (preserves
  the long-term reclaimable=True intent while saying "not right now")
  and arguably cleaner semantically. Schema bump.

I'd lean toward `Lease.busy` because it preserves the consumer's
declared "this lease is reclaimable in principle" while giving them
a temporary opt-out — and peer consumers see `busy=True` and know
"this won't yield right now" without misreading
`reclaimable=False` as a permanent decision.

## Acceptance

- A peer aiserver / scriba can compute `wait_sec` for "I want X VRAM
  on GPU 3" that correctly includes (a) any in-flight lease's
  remaining ETA on that GPU, (b) any queued requests' projected
  ETAs ahead of us, (c) without false zeros caused by stale
  `acquired_at` on a long-lived lease.
- An aiserver can flip its model-load lease into "busy, please don't
  reclaim me" for the duration of a request without releasing it.

## Non-goals

- A new persistence model. The JSON state file + transact() API
  stays. All asks are additive Pydantic field additions plus one
  parameter to `LeaseHandle.update()`.
- Tracking historical ETA accuracy (consumer-side learning lives in
  aiserver's `calibration_learner.py`).
