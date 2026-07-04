# CPU/RAM-Ressourcen: offene v2-Punkte (NUMA, Disk)

v1 ist umgesetzt (v0.9.0, 2026-07-04): first-class `cpu_cores`/`ram_mb` als
host-globale Zähler — Standard-Keys (`CPU_CORES_KEY`/`RAM_MB_KEY`), explizite
acquire-Kwargs, `detect_ram_mb(reserve_mb=...)` (MemTotal/cgroup/sysctl),
`reslock init` detektiert CPU+RAM, Queue/Priority/Reclaim getestet in
`tests/test_cpu_ram_resources.py`. Kein Schema-Bump (additiv, weiter v3).
Ursprüngliche Anforderung siehe git-History dieser Datei (Commit 99678ad).

## Offen für v2

1. **NUMA-Awareness** (kirk: 2 Nodes, Cross-NUMA drückt effektive Bandbreite
   auf ~30 von 280 GB/s): per-Node-Kapazitäten `cpu_cores@node<N>` /
   `ram_mb@node<N>` (Namensschema in v1 reserviert, Keys sind frei-form —
   additiv möglich), Scheduler vergibt Kern-Sets + `numactl --membind`
   konsistent. Für Modelle > Node-RAM bleibt Interleave der Default.
2. **Disk**: Entscheidung ausstehend, ob `disk_mb` lease-förmig sinnvoll ist —
   Modelle belegen Platz dauerhaft, nicht lease-förmig. Evtl. reicht
   Kapazitäts-Reporting/Warnschwelle. `detect_disk_mb()` existiert bereits
   für Kapazitäts-Registrierung.
3. **Konsumenten-Rollout**: aiserver CPU-LLM-Starts (Kimi K2.6 / GLM 5.2 auf
   kirk, siehe `aiserver/tasks/cpu-llm-models-kimi-glm-reslock.md`) auf
   `reslock>=0.9.0` mit `acquire(cpu_cores=..., ram_mb=...)` umstellen;
   Reserve-Wert für `detect_ram_mb()` auf kirk festlegen (Produktion:
   scriba rpcserver, aiserver, OCR).
