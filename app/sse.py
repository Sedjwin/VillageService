"""
Server-Sent Events broadcast manager.
"""
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class SSEManager:
    """
    Manages a pool of subscriber queues.
    Broadcast sends a formatted SSE message to all live subscribers.
    Dead (full) queues are pruned automatically.
    """

    def __init__(self):
        self._queues: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        """Register a new SSE subscriber and return its queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._queues.add(q)
        logger.debug("SSE subscriber added. Total: %d", len(self._queues))
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Remove a subscriber queue (called when the client disconnects)."""
        self._queues.discard(q)
        logger.debug("SSE subscriber removed. Total: %d", len(self._queues))

    async def broadcast(self, event_type: str, data: dict):
        """
        Push an SSE-formatted message to all subscribers.
        Queues that are full are considered dead and removed.
        """
        msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        dead: list[asyncio.Queue] = []
        for q in self._queues:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._queues.discard(q)
        if dead:
            logger.debug("Pruned %d dead SSE subscriber(s).", len(dead))

    @property
    def subscriber_count(self) -> int:
        return len(self._queues)


sse_manager = SSEManager()
