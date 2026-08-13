# kirk-rpcserver gegen den gelöschten `detect.detect_gpu_vram_mb()` prüfen

**Status:** offen, angelegt 2026-08-13 (Sitzung b2e998d7). Klein, aber unverifiziert.

## Worum es geht

In `84670ea` ist die tote, nicht exportierte Dublette `detect_gpu_vram_mb()` aus
`src/reslock/detect.py` gelöscht worden. Vor dem Löschen wurden die Verbraucher geprüft —
aber nur die, die auf diesem Mac ausgecheckt sind:

- **aiserver** — `src/aiserver/server_state.py:218` holt den Namen aus dem Paketwurzel
  (`from reslock import ResourcePool, detect_gpu_vram_mb`), also aus `resources.py`. Nicht
  betroffen.
- **scriba** — `backend/scriba/simplemodel_tools.py:166` genauso. Die beiden
  `from reslock.detect import`-Stellen (`cli_server.py:269`, `cli_rpc_server.py:99`) holen
  `get_host_pid`, ein anderes Symbol. Nicht betroffen.
- **kirk-rpcserver** — **nicht geprüft**, liegt nicht auf diesem Rechner.

## Was zu tun ist

Ein Grep im kirk-rpcserver-Checkout:

```bash
grep -rn "reslock\.detect\|from reslock import detect\|detect_gpu_vram_mb" --include='*.py' .
```

- Kein Treffer auf `reslock.detect.detect_gpu_vram_mb` → erledigt, Datei wegwerfen.
- Treffer → durch `resources.detect_gpu_vram_mb_nvidia_smi()` ersetzen, **nicht** durch die
  Kette: wer sich die Dublette geholt hat, wollte ausdrücklich nvidia-smi und den
  physischen Total, nicht den CUDA-sichtbaren Wert (24576 vs 24135 MB auf einer RTX 3090).

## Wie groß das Risiko wirklich ist

Gering. Die Funktion war nie exportiert (`__init__.py` reicht den Namen aus `resources.py`
durch), ein Verbraucher hätte also bewusst nach `reslock.detect` hineingreifen müssen.
Trotzdem ist es eine Löschung ohne vollständige Gegenprobe, und genau das gehört
aufgeschrieben statt gehofft.

## Nebenbei

`main` liegt seit dem Release einen Quellcode-Commit vor dem Tag `v0.12.0`, während
`pyproject.toml` weiter `version = "0.12.0"` sagt — „0.12.0" bezeichnet damit zwei
verschiedene Stände (PyPI und main). Kein Handlungsbedarf: der Veröffentlichungsablauf in
AGENTS.md hebt die Version als ersten Schritt sowieso. Nur nicht vergessen, dass das nächste
Release **0.12.1** heißt und nicht wieder 0.12.0.
