# First-class CPU / RAM (/ Disk?) Ressourcen-Felder

Angefragt von Moritz 2026-07-04. Kontext: CPU-LLM-Serving auf kirk (Kimi K2.6
544 GB + GLM 5.2 436 GB als llama.cpp-CPU-Instanzen, siehe
`aiserver/tasks/cpu-llm-models-kimi-glm-reslock.md`). Die Jobs müssen gegen-
einander UND gegen die Produktion (scriba rpcserver, aiserver, OCR) koordiniert
werden — heute geht das nur über ad-hoc `non_gpu`-Keys.

## Ist-Zustand

- `acquire(..., non_gpu={"ram_mb": 8000})` existiert bereits: frei benannte
  Zähler-Ressourcen. Aber: keine Standard-Namen (jeder Konsument könnte
  `ram_mb`/`mem_mb`/`host_ram` erfinden), keine auto-detektierten Kapazitäten
  (detect.py registriert nur GPU-VRAM), kein CLI-/Status-Support analog VRAM,
  keine NUMA-Semantik.

## Wunsch

1. **Standardisierte, erst-klassige Felder** analog `vram_mb_each`/`num_gpus`:
   - `cpu_cores` (Kapazität = physische Kerne, auto-detektiert; Anfrage = Anzahl)
   - `ram_mb` (Kapazität = MemTotal minus konfigurierbarer Reserve; Anfrage = MB)
   - optional `disk_mb` pro konfigurierbarem Mount (offene Frage: brauchen wir
     das wirklich als Lease, oder reicht Kapazitäts-Anzeige? Modelle belegen
     Platz dauerhaft, nicht lease-förmig — evtl. nur Reporting/Warnschwelle)
2. **Detection** in detect.py (nproc/MemTotal, optional statvfs pro Mount) +
   Anzeige in `reslock status` wie bei VRAM (belegt/frei/Queue).
3. **Reclaim/Priority/Queue-Semantik identisch zu VRAM** (funktioniert über
   die bestehenden Zähler vermutlich schon — testen + dokumentieren).
4. **Schema-Vorsicht**: v3 resettet den State beim ersten Read eines neuen
   Schemas; additive Felder bevorzugen, sonst koordiniertes Upgrade über alle
   kirk-Konsumenten (aiserver, scriba rpcserver, OCR) einplanen.

## Ja, mem-Architektur (NUMA) ist ein Thema — aber v2

Messung kirk (2x Xeon 6258R, 12x DDR4-2933, 2 NUMA-Nodes): Kimi-CPU-Decode
erreicht effektiv nur ~30 GB/s von theoretisch ~280 GB/s — Hauptgründe
Cross-NUMA-Traffic (544-GB-Modell > 512 GB pro Node, Interleave erzwungen)
plus MoE-Zugriffs-Muster. Konsequenzen für das Design:

- Ein `cpu_cores`-Lease ohne Node-Bindung ist für bandbreiten-hungrige Jobs
  fast wertlos — die eigentliche knappe Ressource ist **Speicherbandbreite
  pro NUMA-Node**, nicht Kern-Anzahl.
- v1 pragmatisch: `cpu_cores`/`ram_mb` als host-globale Zähler (reicht zum
  Serialisieren der dicken CPU-LLMs: 1 Slot = alle Kerne minus Reserve).
- v2-Feld-Design NUMA-offen halten: optionales `numa_node`-Attribut auf
  Kapazitäten/Anfragen (z.B. `cpu_cores@node0`), damit später ein Scheduler
  Kern-Sets + `numactl --membind` konsistent vergeben kann, ohne das Schema
  nochmal zu brechen. Für Modelle > Node-RAM bleibt Interleave der Default.

## Akzeptanz

- aiserver kann einen CPU-LLM-Start mit `{cpu_cores: 48, ram_mb: 650000}`
  leasen; zweiter CPU-LLM-Start queued; VRAM-Konsumenten unbeeinflusst.
- `reslock status` zeigt CPU/RAM-Belegung + Queue.
- Bestehende v3-Konsumenten laufen ohne State-Reset weiter (oder Upgrade-Plan).
