# Guard: warn when set_resources registers GPU capacity below the physical total

**Status: DONE (2026-07-20)** — implemented and released in v0.10.1.
`ResourcePool._warn_gpu_capacity_mismatch()` warns (below *and* above the NVML
total), `nvml_total_vram_mb()` caches physical totals per process, docstring
updated, tests in `tests/test_set_resources_guard.py` (5 tests, incl. the
optional above-total branch and a non-GPU-keys-skip-NVML test).

**Filed:** 2026-07-20 from the aiserver prod-cut session (straiqr/aiserver,
session 6c53ecaf). Companion to the scriba-side fix in
`scriba/tasks/reslock-capacity-clobber-refresh-totals.md`.

## Incident (2026-07-20, kirk)

scriba's `_refresh_reslock_gpu_totals()` persisted a free-based snapshot
(~17–19 GB/card instead of 24135 MB) into the shared pool's GPU **capacity**
via `pool.set_resources()`. The snapshot went stale after the VRAM freed, and
a 10x19000 MB acquire could then never promote on a physically idle host
(aiserver 503 "Resources unavailable" after the full 300 s timeout). Nothing
in reslock flagged that the registered capacity was below what NVML reports as
`memory.total` — the state looked plausible and the failure surfaced as an
opaque wait-timeout two consumers away.

The scriba helper is being deleted (it predates the v0.7.1 NVML pre-flight and
is redundant since then), but reslock should carry a tripwire so the *next*
consumer that abuses capacity this way is caught immediately.

## What to implement

In `ResourcePool.set_resources()` (pool.py:798): when a submitted key is a GPU
VRAM capacity (`parse_gpu_vram_key` matches) and the value is **below** the
NVML-detected `memory.total` for that UUID, log a **WARNING** naming the key,
submitted value, physical total, and the caller pid — then register the value
anyway.

- **Warn, don't clamp/refuse**: registering less than total can be legitimate
  (deliberate headroom reserves — cf. `detect_ram_mb(reserve_mb=...)`), and
  capacity above total is equally suspicious but not this incident — optional
  second warn branch, agent's call.
- NVML lookup must fail soft (no nvidia-smi / no NVML → no warning, no error)
  and must not meaningfully slow `set_resources` (it's called at consumer
  startup and, until scriba's fix lands, every ~2 s during scriba waits — cache
  the totals per process).
- Docstring: state explicitly that capacity is the *physical total*, that
  external/transient VRAM usage is handled by the NVML pre-flight at placement
  time, and that consumers must NOT write derived free-based values here.

## Tests

- `set_resources` with a GPU value below the (mocked/injected) physical total →
  warning emitted, value still registered.
- Value equal to total → no warning.
- NVML unavailable → no warning, no crash.

## Release

Patch release per AGENTS.md publishing flow (version bump `0.10.0` → `0.10.1`,
GitHub release tag `v0.10.1` → PyPI trusted publishing; verify via the
version-specific PyPI endpoint, not `info.version`). Mix-safe: no schema or
behavior change, log-only — kirk consumers (aiserver, scriba, kirk-rpcserver,
all on 0.10.0 with `>=`-pins) pick it up on their next deploys, no coordinated
restart.
