"""
Public API routes — no auth required.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Conversation, EventLog, Tile, VillageAgent, WorldState
from app.schemas import (
    AgentStateOut,
    ConversationOut,
    EventLogOut,
    TileOut,
    VillageStateOut,
    agent_to_out,
    event_to_out,
    tile_to_out,
    world_to_out,
)
from app.sse import sse_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/village", tags=["village"])


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

async def _build_village_state(db: AsyncSession) -> VillageStateOut:
    result = await db.execute(select(WorldState).where(WorldState.id == 1))
    world = result.scalar_one_or_none()
    if not world:
        raise HTTPException(status_code=503, detail="World not initialised yet.")

    agents_result = await db.execute(
        select(VillageAgent).where(VillageAgent.in_simulation == True)
    )
    agents = agents_result.scalars().all()

    events_result = await db.execute(
        select(EventLog).order_by(EventLog.created_at.desc()).limit(20)
    )
    events = events_result.scalars().all()

    tiles_result = await db.execute(select(Tile))
    tiles = tiles_result.scalars().all()

    return VillageStateOut(
        world_state=world_to_out(world),
        agents=[agent_to_out(a) for a in agents],
        recent_events=[event_to_out(e) for e in events],
        tiles=[tile_to_out(t) for t in tiles],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/state", response_model=VillageStateOut)
async def get_village_state(db: AsyncSession = Depends(get_db)):
    """Full snapshot — used for initial page load."""
    return await _build_village_state(db)


@router.get("/stream")
async def village_stream(db: AsyncSession = Depends(get_db)):
    """
    SSE stream. Sends tick_update events.
    On first connect, immediately sends current state.
    """
    queue = sse_manager.subscribe()

    async def event_generator():
        # Send current state immediately
        try:
            state = await _build_village_state(db)
            import json
            initial = f"event: tick_update\ndata: {state.model_dump_json()}\n\n"
            yield initial
        except Exception as exc:
            logger.warning("SSE initial state error: %s", exc)

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield msg
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            sse_manager.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/map", response_model=list[TileOut])
async def get_map(
    x1: int = Query(-10),
    y1: int = Query(-10),
    x2: int = Query(10),
    y2: int = Query(10),
    db: AsyncSession = Depends(get_db),
):
    """Return tile data for a bounding box."""
    # Clamp to reasonable size
    if (x2 - x1) * (y2 - y1) > 2500:
        raise HTTPException(status_code=400, detail="Requested area too large (max 50x50).")

    result = await db.execute(
        select(Tile).where(
            Tile.x >= x1, Tile.x <= x2,
            Tile.y >= y1, Tile.y <= y2,
        )
    )
    tiles = result.scalars().all()
    return [tile_to_out(t) for t in tiles]


@router.get("/agents", response_model=list[AgentStateOut])
async def list_agents(db: AsyncSession = Depends(get_db)):
    """All agents — in simulation or not."""
    result = await db.execute(select(VillageAgent))
    agents = result.scalars().all()
    return [agent_to_out(a) for a in agents]


@router.get("/agents/{agent_id}", response_model=AgentStateOut)
async def get_agent(agent_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(VillageAgent).where(VillageAgent.agent_id == agent_id)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found.")
    return agent_to_out(agent)


@router.get("/events", response_model=list[EventLogOut])
async def get_events(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    event_type: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    query = select(EventLog).order_by(EventLog.created_at.desc())
    if event_type:
        query = query.where(EventLog.event_type == event_type)
    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    events = result.scalars().all()
    return [event_to_out(e) for e in events]


@router.get("/conversations", response_model=list[ConversationOut])
async def get_conversations(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.completed == True)
        .order_by(Conversation.tick.desc())
        .limit(limit)
    )
    convos = result.scalars().all()
    return [
        ConversationOut(
            id=c.id,
            tick=c.tick,
            x=c.x,
            y=c.y,
            participants=c.participants,
            messages=c.messages,
            completed=c.completed,
        )
        for c in convos
    ]
