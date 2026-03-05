#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["reslock", "fastapi", "uvicorn", "httpx"]
# ///
"""
Reclaimable llama-server proxy.

Manages a llama-server process behind a FastAPI proxy. Acquires a reslock VRAM
lease (reclaimable), starts llama-server, and proxies requests to it. When
another process needs the VRAM, the lease is reclaimed: llama-server is stopped
and the VRAM is freed. Once resources become available again, llama-server is
restarted automatically.

Usage:
    uvx --script examples/llama_server_proxy.py -- \
        --vram 8G \
        --cmd "llama-server -m model.gguf --port 8081" \
        --backend-port 8081 \
        --port 8080

Then point clients at http://localhost:8080 (OpenAI-compatible).
When the model is unloaded, requests return 503 Service Unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

from reslock import ResourcePool

logger = logging.getLogger("llama-proxy")


def parse_size(value: str) -> int:
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([gGmM]?)$", value)
    if not m:
        raise ValueError(f"Invalid size: {value}")
    num = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "G":
        return int(num * 1024)
    return int(num)


class LlamaManager:
    """Manages the llama-server lifecycle with reclaimable VRAM leases."""

    def __init__(
        self,
        cmd: list[str],
        resources: dict[str, int],
        backend_port: int,
        priority: int = 0,
        label: str | None = None,
    ) -> None:
        self.cmd = cmd
        self.resources = resources
        self.backend_port = backend_port
        self.priority = priority
        self.label = label or "llama-server"
        self.pool = ResourcePool()
        self.proc: subprocess.Popen[bytes] | None = None
        self.ready = False
        self._task: asyncio.Task[None] | None = None
        self._shutdown = False

    async def start(self) -> None:
        self._shutdown = False
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._shutdown = True
        self._kill_server()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        """Main loop: acquire VRAM, run server, watch for reclaim, repeat."""
        while not self._shutdown:
            logger.info("Waiting for resources: %s", self.resources)
            handle = await self.pool.acquire_async(
                reclaimable=True,
                priority=self.priority,
                label=self.label,
                **self.resources,
            )
            try:
                logger.info("Lease acquired (%s), starting llama-server", handle.id)
                self._start_server()
                await self._wait_for_healthy()
                self.ready = True
                logger.info("llama-server is ready")

                # Wait for either reclaim or server exit
                while not self._shutdown:
                    if handle.reclaim_requested:
                        logger.info("Reclaim requested, stopping llama-server")
                        self.ready = False
                        self._kill_server()
                        break
                    if self.proc and self.proc.poll() is not None:
                        logger.warning(
                            "llama-server exited unexpectedly (code %d)", self.proc.returncode
                        )
                        self.ready = False
                        break
                    await asyncio.sleep(0.5)
            finally:
                self.ready = False
                self._kill_server()
                handle.release()

            if not self._shutdown:
                logger.info("Will re-acquire resources after cooldown")
                await asyncio.sleep(1)

    def _start_server(self) -> None:
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )

    def _kill_server(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None

    async def _wait_for_healthy(self, timeout: float = 120) -> None:
        """Poll the backend health endpoint until it responds."""
        url = f"http://127.0.0.1:{self.backend_port}/health"
        deadline = asyncio.get_event_loop().time() + timeout
        async with httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(url, timeout=2)
                    if resp.status_code == 200:
                        return
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(1)
        raise TimeoutError("llama-server did not become healthy")


def create_app(manager: LlamaManager) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        await manager.start()
        yield
        await manager.stop()

    app = FastAPI(lifespan=lifespan)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy(request: Request, path: str) -> Response:
        if not manager.ready:
            return Response(
                content='{"error": "Model not loaded, try again later"}',
                status_code=503,
                media_type="application/json",
            )

        url = f"http://127.0.0.1:{manager.backend_port}/{path}"
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method=request.method,
                url=url,
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                content=await request.body(),
                params=dict(request.query_params),
                timeout=300,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Reclaimable llama-server proxy")
    parser.add_argument("--vram", required=True, help="VRAM to reserve (e.g., 8G, 4096M)")
    parser.add_argument("--cmd", required=True, help="llama-server command to run")
    parser.add_argument(
        "--backend-port", type=int, default=8081, help="Port llama-server listens on"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port for the proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--label", default="llama-server")
    args = parser.parse_args()

    resources = {"vram_mb": parse_size(args.vram)}
    manager = LlamaManager(
        cmd=args.cmd.split(),
        resources=resources,
        backend_port=args.backend_port,
        priority=args.priority,
        label=args.label,
    )

    app = create_app(manager)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
