# `QueueEntry.complete()` should be crash-loud or persist-then-retry

Filed 2026-06-02 by claude session `d0c419e0-8982-4265-a4a7-2ff4b28a267c` working on `straiqr/aiserver`.

## Problem

When `QueueEntry.complete()` fails (e.g. transient state-file write error) it
returns control to the caller without raising. If the caller swallows or
logs-and-continues that failure — as scriba was doing at
`backend/scriba/simplemodel_tools.py::_lease_start_work` before
SCRIBA-384 — the leaked `QueueEntry` stays alive on the lease forever:

- `reclaim` skips the lease as long as any entry references it (the entry
  is what tells the reclaim path "this lease is mid-request, don't evict").
- Peer consumers see "busy ~N sec left" indefinitely.
- The owning process keeps running fine; nothing surfaces the bug until
  another consumer needs that GPU and waits behind the leaked entry.

This is what wedged `aiserver-prod` on kirk 2026-05-30 (broadcast urgent,
session 1e02039b): scriba's lease from 2026-05-29T09:41:21Z still held a
QueueEntry from a request that had long since returned. Workaround on the
scriba side landed as SCRIBA-384 / commit `6a82b7d3`: retry `complete()` up
to 3× with private-flag reset, warn on persistent failure. But that's
defensive; the root issue is at the reslock contract level — `complete()`
should either:

- **Raise loud and clear on failure** so callers can't accidentally swallow
  it. Optional retry inside the library still useful, but a final failure
  must surface (the scriba contract is at-least-once delivery).
- **Persist-then-retry in the background** — write the intent to state and
  let a watchdog reconcile on the next `transact()`. Crash-tolerant, but
  needs careful design so a process death doesn't leave the intent file
  half-written.

Either path beats the current "best-effort, silent on failure" semantics.

## Cross-refs

- aiserver `CLAUDE.md` section "Diagnosing a reslock wedge" — fingerprint
  for end-users.
- scriba `tasks/reslock-entry-complete-leak-2026-05-30.md` — original task
  (trashed once SCRIBA-384 shipped, but the workaround sits on the wrong
  side of the API).
- aiserver `tests/test_unit.py::TestAcquireReslockCleanup` — regression
  test pinning aiserver's own clean release behavior.
- broadcast log 2026-05-30T19:50:41 + 19:58:04 — incident timeline.

## Constraints

- Schema bump is acceptable if needed (we coordinated v3 migration; can
  coordinate again).
- Backward-compat on the wire: state-file format may change but client API
  surface should stay stable so consumers (scriba, aiserver,
  kirk-rpcserver) don't all need synchronized re-deploys.
- Watch for the multi-consumer eviction-during-complete race: if reclaim
  fires between the entry's intent-to-complete and the actual state write,
  the entry's lease may be evicted out from under it. That's worth a
  thought-through invariant before shipping.
