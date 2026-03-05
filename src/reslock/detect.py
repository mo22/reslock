from __future__ import annotations

import shutil
import subprocess


def detect_gpu_vram_mb() -> dict[str, int]:
    """Detect total GPU VRAM via nvidia-smi. Returns empty dict if not available."""
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
        total = 0
        for line in result.stdout.strip().splitlines():
            total += int(line.strip())
        if total > 0:
            return {"vram_mb": total}
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return {}
