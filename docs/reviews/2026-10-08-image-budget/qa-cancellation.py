"""Offline image-worker cancellation and shield diagnostic reproduction.

Author: Zeno Ren

Run against application modules baked into the candidate image. Only this script
is mounted; the synthetic worker never decodes user images or calls a provider.
"""
import asyncio
import hashlib
import json
import logging
import platform
import sys
import threading
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio

sys.path.insert(0, str(Path.cwd()))
import main
from admission import AdmissionController, AdmissionMiddleware, CURRENT_LEASE


EXPECTED_MAIN_SHA256 = "2dbd73d9db841043b52f9d6b308fbdd511e0488f0d2e829b369d3acf46032a57"
logging.disable(logging.CRITICAL)


async def exercise(mode):
    started, release = threading.Event(), threading.Event()
    holder = {}
    controller = AdmissionController(
        max_active=1, max_queued=0, wait_timeout=1,
        body_budget=100, tenant_limits={},
    )
    semaphore = asyncio.Semaphore(1)

    def failing_worker(payload):
        started.set()
        if not release.wait(10):
            raise RuntimeError("Synthetic worker was not released")
        raise ValueError("synthetic worker failure")

    async def app(scope, receive, send):
        CURRENT_LEASE.get().reserve_body(32)
        await main.compress_images_async({})

    middleware = AdmissionMiddleware(
        app, controller=controller, tenant_for_scope=lambda scope: "qa",
    )

    async def invoke():
        await middleware(
            {"type": "http", "method": "POST", "path": "/v1/responses", "headers": []},
            AsyncMock(), AsyncMock(),
        )

    async def owner():
        if mode == "anyio_level_cancel":
            with anyio.CancelScope() as scope:
                holder["scope"] = scope
                await invoke()
        else:
            await invoke()

    with patch.object(main, "_img_compress_sem", semaphore), patch.object(
        main, "compress_images_in_payload", failing_worker,
    ):
        task = asyncio.create_task(owner())
        try:
            async with asyncio.timeout(5):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            if mode == "anyio_level_cancel":
                holder["scope"].cancel()
            else:
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
            await asyncio.sleep(0.03)
            assert not task.done(), "Caller abandoned its running image worker"
            assert controller.active == 1 and controller.body_bytes == 32
            assert semaphore.locked(), "Compression slot was released early"
        finally:
            release.set()
        try:
            await task
        except asyncio.CancelledError:
            assert mode == "repeated_task_cancel"
        else:
            assert mode == "anyio_level_cancel"
        assert controller.active == 0 and controller.body_bytes == 0
        assert semaphore._value == 1, "Compression slot was leaked or double-released"
    return {"mode": mode, "functional_checks": "passed", "active": 0, "body_bytes": 0}


async def run():
    diagnostics = []
    asyncio.get_running_loop().set_exception_handler(
        lambda loop, context: diagnostics.append({
            "message": context.get("message"),
            "exception_type": type(context.get("exception")).__name__,
        })
    )
    cases = [await exercise(mode) for mode in ("repeated_task_cancel", "anyio_level_cancel")]
    await asyncio.sleep(0)
    report = {
        "author": "Zeno Ren",
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "application_path": main.__file__,
        "main_sha256": hashlib.sha256(Path(main.__file__).read_bytes()).hexdigest(),
        "cases": cases,
        "shield_diagnostics": diagnostics,
        "diagnostic_check": "passed" if not diagnostics else "observed",
    }
    print(json.dumps(report, indent=2))
    assert report["main_sha256"] == EXPECTED_MAIN_SHA256, "Unexpected candidate application"
    assert not diagnostics, "Cancellation generated shield exception diagnostics"


if __name__ == "__main__":
    asyncio.run(run())
