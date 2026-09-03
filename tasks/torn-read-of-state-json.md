# `transact()` releases the flock before the buffered write reaches the file — readers see a torn `state.json`

Filed 2026-09-03 from the aiserver repo (session d83deaab, task
`aiserver/tasks/reclaim-checker-dies-on-torn-state-read.md`), left uncommitted here as a
message to the next reslock session. The aiserver side (a reclaim loop that died on the first
such error and never reclaimed again, 30 hours) is fixed there; this file is the reslock half.

## What happened on kirk

aiserver-dev's journal, 2026-09-02 ~15:3x CEST:

```
Traceback (most recent call last):
  File "/app/src/aiserver/backends/llama_server.py", line 2350, in check_reclaim
    reclaim_wanted = (srv.reslock_handle is not None and srv.reslock_handle.reclaim_requested) or (
  File "/app/.venv/lib/python3.12/site-packages/reslock/pool.py", line 387, in reclaim_requested
pydantic_core._pydantic_core.ValidationError: 1 validation error for State
  Invalid JSON: EOF while parsing a string at line 137 column 18 [type=json_invalid, input_value='{\n  "version": 5,\n  "r...ue,\n      "reclaim_req', input_type=str]
```

`LeaseHandle.reclaim_requested` (`pool.py:387`) calls `read_state()`, which takes the portalocker
lock, reads, and validates. The file it read ended mid-string. reslock 0.12.1, portalocker 3.2.0,
Python 3.12, in the `aiserver-gpu:kirk-dev` container; the live `/var/lib/reslock/state.json` is
148 lines / 4374 bytes today.

## Mechanism (measured, not guessed)

`state.transact()` writes **in place** through the locked handle:

```python
with portalocker.Lock(str(path), "r+", timeout=5) as fh:
    data = fh.read()
    ...
    fh.seek(0)
    fh.truncate()
    fh.write(new_data)      # buffered TextIOWrapper — nothing reaches the file yet
```

and portalocker 3.2.0 `Lock.release()` (`utils.py:318`) is

```python
portalocker.unlock(self.fh)   # flock dropped here ...
self.fh.close()               # ... and the buffered content is flushed HERE
```

So between `unlock` and `close` the file is truncated and unlocked while the new content is still
in Python's write buffer. A reader that wins the flock in that window reads whatever has landed.

`truncate()` is a syscall and takes effect immediately; `write()` stays in the `TextIOWrapper`
pending buffer until it exceeds the 8 KiB chunk size, at which point it is written through.
Measured with a hook on `portalocker.unlock` that `stat()`s the path at the instant the lock is
released (`/private/tmp/.../scratchpad/unlock_size_probe.py` in the aiserver session; the script
is 25 lines, easy to recreate):

| state size (final) | on disk at unlock, macOS | on disk at unlock, kirk dev container (Linux) |
|---|---|---|
| 1220 B | **0 B** | **0 B** |
| 3520 B | **0 B** | **0 B** |
| 6970 B | **0 B** | **0 B** |
| 9270 B | 9270 B | 9270 B |
| 13870 B | 13870 B | 13870 B |

Kirk's file is 4.3 KB, i.e. squarely in the "0 bytes on disk at unlock" band. Above 8 KiB the
single `write()` is larger than the buffer and goes straight through, which is why a first repro
with a 9.3 KB file never tore.

Two-process race (writer loops `transact`, reader loops `read_state`, 4 s, `NRES=150` -> 3.5 KB):

| writer | reads ok | torn reads |
|---|---|---|
| `transact()` as shipped | 11 | 0 (window is microseconds; the reader's 0.25 s retry cadence almost never lands in it) |
| `transact()` with `unlock` widened by 2 ms (mechanism unchanged) | 25 | **722**, every one `EOF while parsing a value at line 1 column 0`, file size 0 B |
| `read_state` + `write_state` (tmp + `os.replace`) — the control | 35589 | 0 |

The kirk traceback cuts at line 137 column 18, which is byte ~4121 of the file — just past the
4096-byte page boundary. That is consistent with a reader that got the flock after `unlock` and
read while `close()`'s 4374-byte `write()` syscall had landed its first page and not yet the
second. Inferred from the offset, not observed directly; the 0-byte case is the one measured.

Every consumer polls this: aiserver's `check_reclaim` reads the file every 2 s per backend per
instance, scriba's rpcserver and the tenants transact on every model call, isidore reads it for
monitoring. Once in ~30 hours of dev traffic is the observed rate.

## Fix (reslock side)

1. **Flush before the lock is released.** In `transact()` (and `force_reset_state()`, which has
   the same in-place shape) add `fh.flush()` — and `os.fsync(fh.fileno())` if durability across a
   crash matters, it does not for the race — as the last statement inside the `with`. That alone
   closes the window: the content is on disk before `unlock`.

   Do **not** switch `transact()` to the `write_state` tmp + `os.replace` pattern while the
   flock lives on the state file itself: the lock is per inode, so after a replace a waiter that
   was blocked on the old inode acquires a lock on an unlinked file, reads stale content and
   writes into the void — lost update instead of torn read. Atomic replace needs a separate,
   never-replaced lock file (`state.json.lock`) first.

2. **Readers retry on a parse error.** `read_state()` should retry a `ValidationError` whose
   error type is `json_invalid` (and a bare `JSONDecodeError`) a few times with a short sleep
   before raising — belt and braces for any consumer still on an older reslock that keeps
   writing the old way. Keep `SchemaVersionMismatch` un-retried, that one is real.

3. **Regression test:** hook `portalocker.unlock` in a test, run `transact()` on a ~4 KB state,
   assert `path.stat().st_size == len(new_data)` at the instant of unlock. Deterministic; the
   two-process race is only a demo. Table above is the expected red/green pair.

4. Patch-level release (no schema change), then bump the pin in aiserver/scriba/isidore at
   leisure — this is additive, no coordinated window.

## The probe (so the table above is reproducible without the aiserver session)

```python
"""On-disk size of state.json at the instant transact() drops the flock."""
import os, sys, tempfile
import portalocker, portalocker.utils as pu
from pathlib import Path
from reslock import state as st

d = Path(tempfile.mkdtemp(prefix="reslock-probe-")); path = d / "state.json"
st.ensure_state_file(path)
seen = {}
real_unlock = portalocker.unlock
def probe(fh):
    seen["size_at_unlock"] = os.stat(path).st_size
    real_unlock(fh)
pu.portalocker.unlock = probe
for n in (int(a) for a in sys.argv[1:]):          # e.g. 50 150 300 400 600
    def fn(s, n=n):
        s.resources.clear()
        for i in range(n): s.resources[f"gpu:{i:04d}"] = 24135
    st.transact(path, fn)
    final = path.stat().st_size
    print(f"resources={n:4d} final={final:6d}B size_at_unlock={seen['size_at_unlock']:6d}B "
          f"{'TORN-WINDOW' if seen['size_at_unlock'] != final else 'complete'}")
```

Run with `uv run python probe.py 50 150 300 400 600`; on 0.12.1 the first three rows print
`size_at_unlock=0B TORN-WINDOW`. After the fix every row must print `complete`.

## Not this file's problem

aiserver now isolates each backend's `check_reclaim`, keeps its loop alive through this error,
logs it at WARNING once per minute per backend, and exposes `reclaim_checker.stale` on `/health`
(503), `/status` and `/metrics` (`aiserver_reclaim_checker_alive`) so a dead loop pages instead
of holding 53 GB in silence. Fixed and deployed to aiserver-dev on 2026-09-03/04.

## Dispatch (Moritz 2026-09-03 22:30: "dispatch to reslock first")

Coordinator session 390aee1f (scriba_bugfixer). Scope for the sibling working this file:

1. **Writer atomic:** in `state.transact()` flush + `os.fsync` BEFORE the lock is released, and
   prefer writing to a sibling temp file followed by `os.replace` under the lock so a reader can
   never see a truncated file at all (also covers a crash between truncate and write). Keep the
   portalocker lock semantics for writers.
2. **Reader tolerant:** `read_state()` retries a bounded number of times (e.g. 5 × 20 ms) on
   `ValidationError`/`json.JSONDecodeError`/empty file before raising; a 0-byte file is "retry",
   not "empty state". Never silently reset the state (0.12.0 rule stands).
3. **Regression test** built from the measured mechanism: a hook on `portalocker.unlock` that
   `stat()`s the path at release must see the full size for the 1 KiB / 3.5 KiB / 7 KiB states
   (the table above shows 0 B today); plus a concurrent reader/writer test that runs the old code
   red (torn reads > 0) and the new code green over a few hundred transactions.
4. **Version 0.12.2**, CHANGELOG/HISTORY per this repo's convention, `uv run pytest` green
   (note the known-flaky nvml test in AGENTS.md), commit-flow.py with explicit paths.
5. **Release:** per AGENTS.md "Publishing" — GitHub release with tag `v0.12.2` triggers trusted
   publishing; then VERIFY per the "Verifying a release" note (`pypi.org/pypi/reslock/json` must
   serve 0.12.2, `uvx` is not proof). No schema change → consumers can move rolling, no
   coordinated restart. Then post one bus line to the coordinator
   (`~/workspace/claude-setup/broadcast.py post --to-agent 390aee1f "reslock 0.12.2 on PyPI: …"`);
   the coordinator bumps the pins in scriba and aiserver. `/finish` afterwards.

Consumers (AGENTS.md: the whole list is aiserver and scriba; kirk-rpcserver is scriba):
aiserver `50bebef` already isolates its loop; scriba's `clear_model_cache_thread` is being
hardened in parallel (scriba_bugfixer `tasks/scriba-reslock-loop-hardening.md`).
