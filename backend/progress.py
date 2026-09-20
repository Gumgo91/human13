"""Per-request SSE status, including heartbeats during external waits."""
import asyncio
import logging
import time
import uuid
from contextvars import ContextVar

sink = ContextVar("progress_sink", default=None)
logger = logging.getLogger("uvicorn.error")
HEARTBEAT_SECONDS = 2.


async def report(message, phase):
    callback = sink.get()
    if callback:
        await callback("progress", {"message": message, "phase": phase})


async def stream_work(work):
    queue = asyncio.Queue()
    started = time.monotonic()
    phase_started = started
    request_id = uuid.uuid4().hex[:8]
    current = {"stage": 0, "fraction": 0., "message": "Starting the simulation.", "phase": "starting"}

    def status(heartbeat=False):
        return {**current, "elapsed_seconds": round(time.monotonic()-started, 1),
                "phase_elapsed_seconds": round(time.monotonic()-phase_started, 1), "heartbeat": heartbeat}

    async def emit(kind, data):
        nonlocal current, phase_started
        if kind == "progress":
            if data.get("message", current["message"]) != current["message"]:
                phase_started = time.monotonic()
                logger.info("Simulation %s: %s (%.1fs)", request_id, data["message"], phase_started-started)
            current = {**current, **data}
            data = status()
        await queue.put((kind, data))

    async def run():
        token = sink.set(emit)
        try:
            await work(emit)
        finally:
            sink.reset(token)
            await queue.put(None)

    task = asyncio.create_task(run())
    next_item = asyncio.create_task(queue.get())
    try:
        while True:
            ready, _ = await asyncio.wait({next_item}, timeout=HEARTBEAT_SECONDS)
            if not ready:
                yield "progress", status(heartbeat=True)
                continue
            item = next_item.result()
            if item is None:
                await task
                break
            yield item
            next_item = asyncio.create_task(queue.get())
    finally:
        if not next_item.done():
            next_item.cancel()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, next_item, return_exceptions=True)
