# Tote Dublette: zweites `detect_gpu_vram_mb()` in detect.py

**Status:** offen, gefunden am 2026-08-12 beim v0.11.1-Fix (Sitzung 58cc6108).
Bewusst nicht mitgefixt — außerhalb des damaligen Auftrags, Moritz hat entschieden,
es getrennt zu betrachten.

## Befund

Es gibt zwei Funktionen dieses Namens:

- `src/reslock/resources.py:23` — die **exportierte**: Kette CUDA-Treiber → torch →
  nvidia-smi. Das ist die, die `__init__.py` re-exportiert und die im README steht.
- `src/reslock/detect.py:81` — eine **tote Dublette**, nur nvidia-smi, praktisch
  byte-gleich zu `detect_gpu_vram_mb_nvidia_smi()` in resources.py. Nichts importiert
  sie: `__init__.py` holt aus `detect` nur die Key-Helfer (`gpu_vram_key`,
  `parse_gpu_vram_key`, …), nicht diese Funktion.

## Warum das mehr als Kosmetik ist

Die beiden liefern **verschiedene Zahlen**: die Dublette gibt den physischen Total
(24576 MB auf einer RTX 3090), die echte den CUDA-sichtbaren (24135). Wer eine
VRAM-Erkennungszahl debuggt und in `detect.py` landet — der Dateiname lädt dazu ein —
liest die falsche Funktion und bekommt eine plausible, aber falsche Antwort. Genau
diese Verwechslung hat beim 0.10.1-Stolperdraht Zeit gekostet.

## Vorschlag

Löschen. Vorher prüfen:

1. `grep -rn "from reslock.detect import\|detect\.detect_gpu_vram_mb"` über src/, tests/
   und die Verbraucher (aiserver, scriba, kirk-rpcserver) — sie ist nicht exportiert,
   aber ein Verbraucher könnte sie direkt importiert haben.
2. Falls doch jemand sie nutzt: durch `resources.detect_gpu_vram_mb_nvidia_smi()`
   ersetzen, nicht durch die Kette — der Aufrufer wollte dann explizit nvidia-smi.

Rein additiv/subtraktiv, kein Schema, kein koordiniertes Fenster.
