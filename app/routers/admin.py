"""
Admin API routes — require Dashboard JWT with role == "admin".
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import Creature, EventLog, ModelConfig, Tile, VillageAgent, WorldState
from app.schemas import SpawnConfirmToken, agent_to_out, world_to_out
from app.sse import sse_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/village", tags=["admin"])

# In-memory pending token store: token -> {agent_id, action, expires_at}
_pending_tokens: dict[str, dict] = {}
_TOKEN_TTL = 10  # seconds


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def require_admin(authorization: str = Header(...)):
    """Validate JWT against UserManager /auth/validate. Require admin role."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{settings.usermanager_url}/auth/validate",
                headers={"Authorization": authorization},
            )
    except httpx.RequestError as exc:
        logger.error("Auth check failed (network): %s", exc)
        raise HTTPException(status_code=503, detail="Auth service unreachable.")

    if resp.status_code != 200:
        raise HTTPException(status_code=503, detail="Auth service error.")

    data = resp.json()
    if not data.get("valid"):
        raise HTTPException(status_code=401, detail="Not authenticated.")
    if data.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required.")
    return data


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class EngineConfigUpdate(BaseModel):
    tick_rate_seconds: int


class HintRequest(BaseModel):
    text: str


class TeleportRequest(BaseModel):
    x: int
    y: int


class GiveRequest(BaseModel):
    item: str
    qty: int


class AdminEventRequest(BaseModel):
    type: str
    x: int | None = None
    y: int | None = None
    params: dict = {}


class DropItemRequest(BaseModel):
    x: int
    y: int
    item: str
    qty: int


class WeatherUpdate(BaseModel):
    weather: str


class SeasonUpdate(BaseModel):
    season: str


class ModelConfigUpdate(BaseModel):
    task: str
    model_id: str
    temperature: float = 0.9
    max_tokens: int = 512


class RelationshipUpdate(BaseModel):
    agent_a: str
    agent_b: str
    score: int


class GatewayConfigUpdate(BaseModel):
    url: str | None = None
    token: str | None = None


class TilePatchRequest(BaseModel):
    terrain: str | None = None
    add_feature: str | None = None
    remove_feature: str | None = None
    add_item: str | None = None
    add_item_qty: int = 1
    remove_item: str | None = None


class GoalOverrideRequest(BaseModel):
    goal: str


# ---------------------------------------------------------------------------
# Engine control
# ---------------------------------------------------------------------------

@router.post("/engine/start")
async def engine_start(_: dict = Depends(require_admin)):
    from app.engine import engine
    await engine.start()
    return {"status": "started"}


@router.post("/engine/pause")
async def engine_pause(_: dict = Depends(require_admin)):
    from app.engine import engine
    await engine.pause()
    return {"status": "paused"}


@router.post("/engine/stop")
async def engine_stop(_: dict = Depends(require_admin)):
    from app.engine import engine
    await engine.stop()
    return {"status": "stopped"}


@router.post("/engine/step")
async def engine_step(_: dict = Depends(require_admin)):
    from app.engine import engine
    await engine.step_once()
    return {"status": "stepped"}


@router.put("/engine/config")
async def engine_config(
    body: EngineConfigUpdate,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(select(WorldState).where(WorldState.id == 1))
    world = result.scalar_one_or_none()
    if not world:
        raise HTTPException(status_code=503, detail="World not initialised.")
    world.tick_rate_seconds = max(1, body.tick_rate_seconds)
    await db.commit()
    return {"tick_rate_seconds": world.tick_rate_seconds}


@router.get("/engine/status")
async def engine_status(
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(select(WorldState).where(WorldState.id == 1))
    world = result.scalar_one_or_none()
    if not world:
        raise HTTPException(status_code=503, detail="World not initialised.")
    return {
        "state": world.engine_state,
        "tick": world.tick,
        "game_day": world.game_day,
        "game_hour": world.game_hour,
        "tick_rate_seconds": world.tick_rate_seconds,
    }


@router.get("/engine/sim-config")
async def get_sim_config(
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(select(WorldState).where(WorldState.id == 1))
    world = result.scalar_one_or_none()
    if not world:
        raise HTTPException(status_code=503, detail="World not initialised.")
    return world.sim_config


@router.put("/engine/sim-config")
async def set_sim_config(
    body: dict,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(select(WorldState).where(WorldState.id == 1))
    world = result.scalar_one_or_none()
    if not world:
        raise HTTPException(status_code=503, detail="World not initialised.")
    cfg = world.sim_config
    # Validate and clamp values
    float_keys = {"hunger_rate", "rest_rate", "warmth_rate", "social_rate",
                  "cooked_food_restore", "raw_food_restore", "event_chance"}
    for k in float_keys:
        if k in body:
            cfg[k] = round(max(0.0, min(10.0, float(body[k]))), 3)
    if "hours_per_tick" in body:
        # Supported values: 0.25, 0.5, 1, 2, 3, 4
        allowed = {0.25, 0.5, 1.0, 2.0, 3.0, 4.0}
        val = float(body["hours_per_tick"])
        # Snap to nearest allowed value
        val = min(allowed, key=lambda x: abs(x - val))
        cfg["hours_per_tick"] = val
    world.sim_config = cfg
    await db.commit()
    return cfg


# ---------------------------------------------------------------------------
# Agent spawn / despawn
# ---------------------------------------------------------------------------

def _issue_token(agent_id: str, action: str) -> str:
    token = secrets.token_urlsafe(16)
    _pending_tokens[token] = {
        "agent_id": agent_id,
        "action": action,
        "expires_at": time.time() + _TOKEN_TTL,
    }
    return token


def _redeem_token(token: str, expected_action: str, agent_id: str) -> bool:
    entry = _pending_tokens.pop(token, None)
    if not entry:
        return False
    if time.time() > entry["expires_at"]:
        return False
    if entry["agent_id"] != agent_id or entry["action"] != expected_action:
        return False
    return True


@router.post("/agents/{agent_id}/spawn", response_model=SpawnConfirmToken)
async def spawn_agent(
    agent_id: str,
    _: dict = Depends(require_admin),
):
    token = _issue_token(agent_id, "spawn")
    return SpawnConfirmToken(
        confirm_token=token,
        expires_in=_TOKEN_TTL,
        agent_id=agent_id,
        action="spawn",
    )


@router.post("/agents/{agent_id}/spawn/confirm", status_code=201)
async def spawn_agent_confirm(
    agent_id: str,
    confirm_token: str,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    if not _redeem_token(confirm_token, "spawn", agent_id):
        raise HTTPException(status_code=400, detail="Invalid or expired confirm token.")

    # Check for existing agent record
    result = await db.execute(
        select(VillageAgent).where(VillageAgent.agent_id == agent_id)
    )
    existing = result.scalar_one_or_none()
    if existing and existing.in_simulation:
        raise HTTPException(status_code=409, detail="Agent already in simulation.")

    # Fetch agent info from AgentManager (via Dashboard proxy or direct)
    agent_data = await _fetch_agent_info(agent_id)
    av = _derive_avatar_config(agent_data)

    if existing:
        existing.in_simulation = True
        # Refresh avatar from profile on every re-spawn
        existing.avatar_primary_color   = av["primary_color"]
        existing.avatar_secondary_color = av["secondary_color"]
        existing.avatar_body_shape      = av["body_shape"]
        existing.avatar_eye_style       = av["eye_style"]
        agent = existing
    else:
        # Determine spawn position (random small offset from origin)
        import random
        ox = random.randint(-3, 3)
        oy = random.randint(-3, 3)

        agent = VillageAgent(
            agent_id=agent_id,
            name=agent_data.get("name", agent_id),
            personality_summary=agent_data.get("personality_summary", agent_data.get("description", "")),
            in_simulation=True,
            x=ox,
            y=oy,
            state="idle",
            avatar_primary_color=av["primary_color"],
            avatar_secondary_color=av["secondary_color"],
            avatar_body_shape=av["body_shape"],
            avatar_eye_style=av["eye_style"],
        )
        agent.inventory = {}
        agent.skills = {}
        agent.needs = {"hunger": 30, "rest": 20, "warmth": 20, "social": 40}
        agent.travel_path = []
        agent.village_memory = []
        agent.relationship_scores = {}
        db.add(agent)

    # Generate starter crate
    result_world = await db.execute(select(WorldState).where(WorldState.id == 1))
    world = result_world.scalar_one_or_none()

    from app.engine import engine
    crate_model = settings.world_agent_model
    bible = world.world_bible if world else ""
    crate = await engine.world_agent.generate_starter_crate(
        agent.name, agent.personality_summary, bible, crate_model
    )

    # Place crate items on spawn tile
    result_tile = await db.execute(
        select(Tile).where(Tile.x == agent.x, Tile.y == agent.y)
    )
    spawn_tile = result_tile.scalar_one_or_none()
    if spawn_tile:
        items = spawn_tile.items
        for item_type, qty in crate["items"].items():
            found = False
            for existing_item in items:
                if existing_item.get("type") == item_type:
                    existing_item["qty"] = existing_item.get("qty", 0) + qty
                    found = True
                    break
            if not found:
                items.append({"type": item_type, "qty": qty})
        spawn_tile.items = items

    # Log + broadcast
    if world:
        log = EventLog(
            id=str(uuid.uuid4()),
            tick=world.tick,
            game_day=world.game_day,
            game_hour=world.game_hour,
            event_type="spawn",
            description=f"{agent.name} arrived at the village. {crate.get('narrative', '')}",
            x=agent.x,
            y=agent.y,
        )
        log.agents_involved = [agent_id]
        db.add(log)
        agent.joined_tick = world.tick

    await db.commit()

    await sse_manager.broadcast("agent_spawned", {
        "agent_id": agent_id,
        "name": agent.name,
        "x": agent.x,
        "y": agent.y,
        "crate_narrative": crate.get("narrative", ""),
    })

    return {"agent_id": agent_id, "name": agent.name, "x": agent.x, "y": agent.y}


@router.delete("/agents/{agent_id}", response_model=SpawnConfirmToken)
async def despawn_agent(
    agent_id: str,
    _: dict = Depends(require_admin),
):
    token = _issue_token(agent_id, "despawn")
    return SpawnConfirmToken(
        confirm_token=token,
        expires_in=_TOKEN_TTL,
        agent_id=agent_id,
        action="despawn",
    )


@router.delete("/agents/{agent_id}/confirm")
async def despawn_agent_confirm(
    agent_id: str,
    confirm_token: str,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    if not _redeem_token(confirm_token, "despawn", agent_id):
        raise HTTPException(status_code=400, detail="Invalid or expired confirm token.")

    result = await db.execute(
        select(VillageAgent).where(VillageAgent.agent_id == agent_id)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found.")

    agent.in_simulation = False
    agent.state = "idle"
    await db.commit()

    await sse_manager.broadcast("agent_despawned", {"agent_id": agent_id})
    return {"status": "despawned", "agent_id": agent_id}


# ---------------------------------------------------------------------------
# God mode
# ---------------------------------------------------------------------------

@router.post("/agents/{agent_id}/hint")
async def set_hint(
    agent_id: str,
    body: HintRequest,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(
        select(VillageAgent).where(VillageAgent.agent_id == agent_id)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found.")
    agent.god_hint = body.text
    await db.commit()
    return {"status": "hint_set"}


@router.post("/agents/{agent_id}/teleport")
async def teleport_agent(
    agent_id: str,
    body: TeleportRequest,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(
        select(VillageAgent).where(VillageAgent.agent_id == agent_id)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found.")
    agent.x = body.x
    agent.y = body.y
    agent.travel_path = []
    agent.state = "idle"
    agent.add_memory(f"Suddenly found myself at ({body.x},{body.y}).")
    await db.commit()
    return {"status": "teleported", "x": body.x, "y": body.y}


@router.post("/agents/{agent_id}/give")
async def give_item(
    agent_id: str,
    body: GiveRequest,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(
        select(VillageAgent).where(VillageAgent.agent_id == agent_id)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found.")
    inv = agent.inventory
    inv[body.item] = inv.get(body.item, 0) + body.qty
    agent.inventory = inv
    agent.add_memory(f"Received {body.qty}x {body.item} from somewhere.")
    await db.commit()
    return {"status": "given", "item": body.item, "qty": body.qty}


@router.put("/agents/{agent_id}/needs")
async def set_agent_needs(
    agent_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(
        select(VillageAgent).where(VillageAgent.agent_id == agent_id)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found.")
    needs = agent.needs
    for key in ("hunger", "rest", "warmth", "social"):
        if key in body:
            needs[key] = max(0.0, min(100.0, float(body[key])))
    agent.needs = needs
    from app.physics import compute_mood
    agent.mood = compute_mood(needs)
    await db.commit()
    return {"status": "ok", "needs": needs}


# ---------------------------------------------------------------------------
# World management
# ---------------------------------------------------------------------------

@router.post("/admin/event")
async def trigger_event(
    body: AdminEventRequest,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    from app.engine import engine
    from app.models import WorldState, VillageAgent

    result_world = await db.execute(select(WorldState).where(WorldState.id == 1))
    world = result_world.scalar_one_or_none()
    agents_result = await db.execute(
        select(VillageAgent).where(VillageAgent.in_simulation == True)
    )
    agents = list(agents_result.scalars().all())

    description = await apply_event(
        body.type, world, agents, db, engine.world_agent, settings.world_agent_model
    )
    if world and description:
        log = EventLog(
            id=str(uuid.uuid4()),
            tick=world.tick,
            game_day=world.game_day,
            game_hour=world.game_hour,
            event_type=body.type,
            description=description,
            x=body.x,
            y=body.y,
        )
        log.agents_involved = []
        db.add(log)
    await db.commit()
    return {"status": "event_applied", "description": description}


@router.post("/admin/drop-item")
async def drop_item(
    body: DropItemRequest,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(select(Tile).where(Tile.x == body.x, Tile.y == body.y))
    tile = result.scalar_one_or_none()
    if not tile:
        raise HTTPException(status_code=404, detail="Tile not found. Generate it first.")
    items = tile.items
    for existing in items:
        if existing.get("type") == body.item:
            existing["qty"] = existing.get("qty", 0) + body.qty
            break
    else:
        items.append({"type": body.item, "qty": body.qty})
    tile.items = items
    await db.commit()
    return {"status": "dropped", "x": body.x, "y": body.y, "item": body.item}


@router.put("/admin/weather")
async def set_weather(
    body: WeatherUpdate,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    valid = {"clear", "cloudy", "rainy", "stormy", "snowy"}
    if body.weather not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid weather. Valid: {valid}")
    result = await db.execute(select(WorldState).where(WorldState.id == 1))
    world = result.scalar_one_or_none()
    if not world:
        raise HTTPException(status_code=503, detail="World not initialised.")
    world.weather = body.weather
    await db.commit()
    return {"weather": body.weather}


@router.put("/admin/season")
async def set_season(
    body: SeasonUpdate,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    valid = {"spring", "summer", "autumn", "winter"}
    if body.season not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid season. Valid: {valid}")
    result = await db.execute(select(WorldState).where(WorldState.id == 1))
    world = result.scalar_one_or_none()
    if not world:
        raise HTTPException(status_code=503, detail="World not initialised.")
    world.season = body.season
    await db.commit()
    return {"season": body.season}


@router.get("/admin/gateway-config")
async def get_gateway_config(
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(select(WorldState).where(WorldState.id == 1))
    world = result.scalar_one_or_none()
    if not world:
        raise HTTPException(status_code=503, detail="World not initialised.")
    cfg = world.gateway_config
    token = cfg.get("token", "")
    return {
        "token_set": bool(token),
        "token_hint": token[-4:] if token else "",
    }


@router.put("/admin/gateway-config")
async def set_gateway_config(
    body: GatewayConfigUpdate,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(select(WorldState).where(WorldState.id == 1))
    world = result.scalar_one_or_none()
    if not world:
        raise HTTPException(status_code=503, detail="World not initialised.")
    cfg = world.gateway_config
    if body.token:
        cfg["token"] = body.token.strip()
    world.gateway_config = cfg
    await db.commit()
    return {"status": "ok"}


@router.get("/admin/available-models")
async def get_available_models(
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    """Fetch model list from AIGateway using the effective token."""
    result = await db.execute(select(WorldState).where(WorldState.id == 1))
    world = result.scalar_one_or_none()
    gw = world.gateway_config if world else {}
    effective_token = gw.get("token") or settings.aigateway_token

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{settings.aigateway_url.rstrip('/')}/v1/models",
                headers={"Authorization": f"Bearer {effective_token}"},
            )
            resp.raise_for_status()
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            return {"models": sorted(models)}
    except Exception as exc:
        logger.warning("Could not fetch models from gateway: %s", exc)
        return {"models": []}


@router.put("/admin/models")
async def update_model_config(
    body: ModelConfigUpdate,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    valid_tasks = {"world_agent", "agent_brain", "conversation", "narration"}
    if body.task not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task. Valid: {valid_tasks}")

    result = await db.execute(select(ModelConfig).where(ModelConfig.task == body.task))
    config = result.scalar_one_or_none()
    if config:
        config.model_id = body.model_id
        config.temperature = body.temperature
        config.max_tokens = body.max_tokens
    else:
        config = ModelConfig(
            task=body.task,
            model_id=body.model_id,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
        db.add(config)
    await db.commit()
    return {"task": body.task, "model_id": body.model_id}


@router.get("/admin/models")
async def get_model_configs(
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(select(ModelConfig))
    configs = result.scalars().all()
    return [
        {"task": c.task, "model_id": c.model_id, "temperature": c.temperature, "max_tokens": c.max_tokens}
        for c in configs
    ]


@router.put("/admin/relationship")
async def update_relationship(
    body: RelationshipUpdate,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    score = max(-10, min(10, body.score))
    for agent_id, other_id in [(body.agent_a, body.agent_b), (body.agent_b, body.agent_a)]:
        result = await db.execute(
            select(VillageAgent).where(VillageAgent.agent_id == agent_id)
        )
        agent = result.scalar_one_or_none()
        if agent:
            scores = agent.relationship_scores
            scores[other_id] = score
            agent.relationship_scores = scores
    await db.commit()
    return {"agent_a": body.agent_a, "agent_b": body.agent_b, "score": score}


@router.get("/admin/log")
async def get_admin_log(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    """Return recent event log entries (used to monitor LLM activity)."""
    result = await db.execute(
        select(EventLog).order_by(EventLog.created_at.desc()).limit(limit)
    )
    events = result.scalars().all()
    return [
        {
            "id": e.id,
            "tick": e.tick,
            "game_day": e.game_day,
            "game_hour": e.game_hour,
            "event_type": e.event_type,
            "description": e.description,
        }
        for e in events
    ]


# ---------------------------------------------------------------------------
# AgentManager fetch helper
# ---------------------------------------------------------------------------

async def _fetch_agent_info(agent_id: str) -> dict:
    """
    Fetch agent profile from AgentManager.
    Tries the configured AIGATEWAY_URL base, then falls back to localhost:8003.
    """
    urls_to_try = [
        f"http://localhost:8003/agents/{agent_id}",
        f"{settings.aigateway_url.rstrip('/')}/agents/{agent_id}",
    ]
    for url in urls_to_try:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {settings.aigateway_token}"},
                )
                if resp.status_code == 200:
                    return resp.json()
        except Exception as exc:
            logger.debug("AgentManager fetch failed at %s: %s", url, exc)

    logger.warning("Could not fetch agent info for %s — using defaults.", agent_id)
    return {"name": agent_id, "personality_summary": "", "description": ""}


# ---------------------------------------------------------------------------
# Tile admin
# ---------------------------------------------------------------------------

@router.patch("/tiles/{x}/{y}")
async def patch_tile(
    x: int,
    y: int,
    body: TilePatchRequest,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(select(Tile).where(Tile.x == x, Tile.y == y))
    tile = result.scalar_one_or_none()
    if not tile:
        raise HTTPException(status_code=404, detail="Tile not found.")

    if body.terrain:
        from app.crafting import TERRAIN_MOVEMENT_COST
        if body.terrain not in TERRAIN_MOVEMENT_COST:
            raise HTTPException(status_code=400, detail=f"Unknown terrain: {body.terrain}")
        tile.terrain = body.terrain

    if body.add_feature:
        feats = tile.features
        if body.add_feature not in feats:
            feats.append(body.add_feature)
        tile.features = feats

    if body.remove_feature:
        feats = tile.features
        tile.features = [f for f in feats if f != body.remove_feature]

    if body.add_item:
        items = tile.items
        for it in items:
            if it.get("type") == body.add_item:
                it["qty"] = it.get("qty", 0) + body.add_item_qty
                break
        else:
            items.append({"type": body.add_item, "qty": body.add_item_qty})
        tile.items = items

    if body.remove_item:
        items = tile.items
        tile.items = [it for it in items if it.get("type") != body.remove_item]

    await db.commit()
    from app.schemas import tile_to_out
    return tile_to_out(tile).model_dump()


# ---------------------------------------------------------------------------
# Agent goal override
# ---------------------------------------------------------------------------

@router.patch("/agents/{agent_id}/goal")
async def override_agent_goal(
    agent_id: str,
    body: GoalOverrideRequest,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(select(VillageAgent).where(VillageAgent.agent_id == agent_id))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found.")
    agent.current_goal = body.goal
    from app.models import WorldState
    ws_result = await db.execute(select(WorldState).where(WorldState.id == 1))
    world = ws_result.scalar_one_or_none()
    if world:
        agent.goal_set_tick = world.tick
    agent.goal_resource_brief = None  # clear old brief; will regenerate on next tick
    agent.add_memory(f"[Admin] Goal set to: {body.goal}")
    await db.commit()
    return {"agent_id": agent_id, "goal": agent.current_goal}


# ---------------------------------------------------------------------------
# Creature admin
# ---------------------------------------------------------------------------

@router.get("/admin/creatures")
async def list_creatures(
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(select(Creature).order_by(Creature.creature_type))
    creatures = result.scalars().all()
    return [
        {
            "id": c.id,
            "creature_type": c.creature_type,
            "x": c.x,
            "y": c.y,
            "state": c.state,
            "spawned_tick": c.spawned_tick,
        }
        for c in creatures
    ]


@router.delete("/creatures/{creature_id}", status_code=204)
async def delete_creature(
    creature_id: str,
    db: AsyncSession = Depends(get_db),
    _: dict = Depends(require_admin),
):
    result = await db.execute(select(Creature).where(Creature.id == creature_id))
    creature = result.scalar_one_or_none()
    if not creature:
        raise HTTPException(status_code=404, detail="Creature not found.")
    await db.delete(creature)
    await db.commit()


def _derive_avatar_config(agent_data: dict) -> dict:
    """
    Derive village avatar config from AgentManager profile.appearance fields.
    face_roundness  → body_shape:  <0.25=square, >0.70=circle, else tall_rectangle
    eye_shape_roundness → eye_style: >=0.5=round, else square
    """
    import json as _json

    profile_raw = agent_data.get("profile")
    if isinstance(profile_raw, str):
        try:
            profile_raw = _json.loads(profile_raw)
        except Exception:
            profile_raw = None

    appearance = (profile_raw or {}).get("appearance", {}) if isinstance(profile_raw, dict) else {}

    primary   = appearance.get("primary_color",   "#4a9eff")
    secondary = appearance.get("secondary_color", "#1a3a6a")

    roundness = appearance.get("face_roundness", 0.5)
    if roundness < 0.25:
        body_shape = "square"
    elif roundness > 0.70:
        body_shape = "circle"
    else:
        body_shape = "tall_rectangle"

    eye_round = appearance.get("eye_shape_roundness", 0.5)
    eye_style = "round" if eye_round >= 0.5 else "square"

    return {
        "primary_color":   primary,
        "secondary_color": secondary,
        "body_shape":      body_shape,
        "eye_style":       eye_style,
    }
