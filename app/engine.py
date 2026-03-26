"""
VillageEngine — the heart of the simulation.
Runs the tick loop, processes agent decisions, manages world state.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
import time as _time
from datetime import datetime

import json

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal
from app.events import check_random_event, apply_event, _SEASONS, _SEASON_WEATHER
from app.los import calculate_los
from app.models import Creature, EventLog, ModelConfig, Tile, TickSnapshot, VillageAgent, WorldState, Conversation
from app.physics import (
    apply_skill_gain,
    can_gather,
    compute_mood,
    decay_needs,
    do_craft,
    do_gather,
    find_path,
    get_terrain_movement_cost,
    satisfy_need,
    validate_craft,
)
from app.social import run_conversation
from app.sse import sse_manager
from app.world_agent import WorldAgent
from app.agent_brain import AgentBrain
from app.crafting import RECIPES

logger = logging.getLogger(__name__)

_DIRECTION_DELTAS = {
    "n": (0, 1), "s": (0, -1), "e": (1, 0), "w": (-1, 0),
    "ne": (1, 1), "nw": (-1, 1), "se": (1, -1), "sw": (-1, -1),
}

# How many game-days per season
_DAYS_PER_SEASON = 30

# Max snapshots to keep (rolling window)
_MAX_SNAPSHOTS = 2000


# ---------------------------------------------------------------------------
# Tick snapshot
# ---------------------------------------------------------------------------

async def _save_tick_snapshot(db: AsyncSession, world: WorldState, agents: list[VillageAgent]):
    """Save a compact snapshot of this tick for timeline replay."""
    snap = TickSnapshot(
        tick=world.tick,
        world_json=json.dumps({
            "tick": world.tick,
            "game_day": world.game_day,
            "game_hour": world.game_hour,
            "season": world.season,
            "weather": world.weather,
            "engine_state": world.engine_state,
            "tick_rate_seconds": world.tick_rate_seconds,
        }),
        agents_json=json.dumps([{
            "agent_id": a.agent_id,
            "name": a.name,
            "in_simulation": a.in_simulation,
            "x": a.x,
            "y": a.y,
            "state": a.state,
            "mood": a.mood,
            "needs": a.needs,
            "current_goal": a.current_goal,
            "inventory": a.inventory,
            "avatar_config": {
                "primary_color": a.avatar_primary_color,
                "secondary_color": a.avatar_secondary_color,
                "body_shape": a.avatar_body_shape,
                "eye_style": a.avatar_eye_style,
            },
        } for a in agents]),
    )
    db.add(snap)

    # Prune old snapshots every 50 ticks
    if world.tick % 50 == 0:
        cutoff = world.tick - _MAX_SNAPSHOTS
        await db.execute(delete(TickSnapshot).where(TickSnapshot.tick < cutoff))


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _get_world_state(db: AsyncSession) -> WorldState | None:
    result = await db.execute(select(WorldState).where(WorldState.id == 1))
    return result.scalar_one_or_none()


async def _get_active_agents(db: AsyncSession) -> list[VillageAgent]:
    result = await db.execute(
        select(VillageAgent).where(VillageAgent.in_simulation == True)
    )
    return list(result.scalars().all())


async def _get_tile(db: AsyncSession, x: int, y: int) -> Tile | None:
    result = await db.execute(select(Tile).where(Tile.x == x, Tile.y == y))
    return result.scalar_one_or_none()


async def _get_all_tiles_map(db: AsyncSession) -> dict[tuple[int, int], Tile]:
    result = await db.execute(select(Tile))
    tiles = result.scalars().all()
    return {(t.x, t.y): t for t in tiles}


async def _log_event(
    db: AsyncSession,
    world: WorldState,
    event_type: str,
    description: str,
    agent_ids: list[str],
    x: int | None = None,
    y: int | None = None,
):
    log = EventLog(
        id=str(uuid.uuid4()),
        tick=world.tick,
        game_day=world.game_day,
        game_hour=world.game_hour,
        event_type=event_type,
        description=description,
        x=x,
        y=y,
        created_at=datetime.utcnow(),
    )
    log.agents_involved = agent_ids
    db.add(log)


# ---------------------------------------------------------------------------
# Tile generation
# ---------------------------------------------------------------------------

async def _ensure_tile_exists(
    x: int,
    y: int,
    db: AsyncSession,
    world: WorldState,
    world_agent: WorldAgent,
    model: str = "",
) -> Tile:
    """Return existing tile or generate a new one via WorldAgent."""
    tile = await _get_tile(db, x, y)
    if tile:
        return tile

    # Gather adjacent tiles for context
    adj_tiles = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        adj = await _get_tile(db, x + dx, y + dy)
        if adj:
            adj_tiles.append({
                "x": adj.x, "y": adj.y,
                "terrain": adj.terrain,
                "elevation": adj.elevation,
                "narrative": adj.narrative,
            })

    effective_model = model or settings.world_agent_model
    data = await world_agent.generate_tile(x, y, adj_tiles, world.world_bible, effective_model)

    tile = Tile(
        x=x, y=y,
        terrain=data["terrain"],
        elevation=data["elevation"],
        narrative=data.get("narrative", ""),
    )
    tile.features = data.get("features", [])
    tile.resource_nodes = data.get("resources", [])
    tile.items = []
    tile.buildings = []
    tile.explored_by = []
    db.add(tile)

    # Chance to spawn a creature on new tiles
    from app.creatures import maybe_spawn_creature
    await maybe_spawn_creature(x, y, tile.terrain, db, world)

    return tile


# ---------------------------------------------------------------------------
# Agent movement
# ---------------------------------------------------------------------------

async def _step_agent_movement(
    agent: VillageAgent,
    db: AsyncSession,
    world: WorldState,
    world_agent: WorldAgent,
):
    """Advance agent one step along their travel_path."""
    path = agent.travel_path
    if not path:
        agent.state = "idle"
        agent.travel_path = []
        return

    next_step = path[0]
    nx, ny = next_step[0], next_step[1]

    # Ensure destination tile exists
    tile = await _ensure_tile_exists(nx, ny, db, world, world_agent)

    # Check passability
    cost = get_terrain_movement_cost(tile.terrain)
    if cost is None:
        # Blocked — clear path and let agent re-plan
        agent.travel_path = []
        agent.state = "idle"
        agent.add_memory(f"Couldn't pass through {tile.terrain} at ({nx},{ny}).")
        return

    # Move
    agent.x = nx
    agent.y = ny
    remaining_path = path[1:]
    agent.travel_path = remaining_path

    if not remaining_path:
        agent.state = "idle"

    # Mark explored
    explored = tile.explored_by
    if agent.agent_id not in explored:
        explored.append(agent.agent_id)
        tile.explored_by = explored

    # Skill gain for navigation
    agent.skills = apply_skill_gain(agent.skills, "navigation")


# ---------------------------------------------------------------------------
# Needs processing
# ---------------------------------------------------------------------------

async def _tick_needs(
    agent: VillageAgent,
    db: AsyncSession,
    world: WorldState,
):
    """Decay needs and update mood. Handle critical need consequences."""
    # Check for nearby shelter / fire
    tile = await _get_tile(db, agent.x, agent.y)
    tile_buildings = tile.buildings if tile else []
    tile_features = tile.features if tile else []
    has_shelter = any(
        b.get("type") in ("basic_shelter", "house") for b in tile_buildings
    )
    has_fire = "campfire" in tile_features or any(
        b.get("type") == "campfire" for b in tile_buildings
    )

    cfg = world.sim_config
    needs = decay_needs(agent.needs, world.season, has_shelter, has_fire, cfg)

    cooked_restore = 45 * cfg.get("cooked_food_restore", 1.0)
    raw_restore    = 25 * cfg.get("raw_food_restore",    1.0)

    # Auto-eat only when genuinely hungry
    if needs.get("hunger", 0) >= 82 and agent.inventory.get("cooked_food", 0) > 0:
        inv = agent.inventory
        inv["cooked_food"] -= 1
        if inv["cooked_food"] == 0:
            del inv["cooked_food"]
        agent.inventory = inv
        needs = satisfy_need(needs, "hunger", cooked_restore)
        agent.add_memory("Ate some cooked food — felt better immediately.")
    elif needs.get("hunger", 0) >= 91 and agent.inventory.get("raw_food", 0) > 0:
        inv = agent.inventory
        inv["raw_food"] -= 1
        if inv["raw_food"] == 0:
            del inv["raw_food"]
        agent.inventory = inv
        needs = satisfy_need(needs, "hunger", raw_restore)
        agent.add_memory("Had to eat something raw. Better than nothing.")

    # Rest: if exhausted and standing still, auto-rest
    if needs.get("rest", 0) >= 90 and agent.state == "idle":
        needs = satisfy_need(needs, "rest", 30)
        agent.state = "idle"
        agent.add_memory("Couldn't keep going — had to rest.")

    # Soft death / collapse: both hunger and rest critical → agent collapses
    if (
        needs.get("hunger", 0) >= 95
        and needs.get("rest", 0) >= 95
        and agent.state != "collapsed"
    ):
        agent.state = "collapsed"
        agent.travel_path = []
        needs["hunger"] = 100
        needs["rest"] = 100
        agent.add_memory(
            "My body gave out — I collapsed. Everything went dark. I need help."
        )
        await _log_event(
            db, world, "collapse",
            f"{agent.name} collapsed from exhaustion and hunger at ({agent.x},{agent.y}).",
            [agent.agent_id], agent.x, agent.y,
        )

    agent.needs = needs
    agent.mood = compute_mood(needs)


# ---------------------------------------------------------------------------
# Spontaneous conversations
# ---------------------------------------------------------------------------

async def _check_conversations(
    agents: list[VillageAgent],
    db: AsyncSession,
    world: WorldState,
    engine: "VillageEngine",
):
    """Check for agents sharing a tile who might want to talk."""
    import random

    # Group agents by location
    location_map: dict[tuple[int, int], list[VillageAgent]] = {}
    for agent in agents:
        key = (agent.x, agent.y)
        location_map.setdefault(key, []).append(agent)

    for (x, y), group in location_map.items():
        if len(group) < 2:
            continue
        # 25% chance of conversation per tick per group
        if random.random() > 0.25:
            continue
        # Pick two random agents
        a, b = random.sample(group, 2)
        if a.state in ("socializing", "sleeping") or b.state in ("socializing", "sleeping"):
            continue
        if a.state == "traveling" or b.state == "traveling":
            continue

        a.state = "socializing"
        b.state = "socializing"

        trigger = "seek_help" if (a.needs.get("hunger", 0) > 70 or b.needs.get("hunger", 0) > 70) else "encounter"

        try:
            convo = await run_conversation(
                agent_a=a,
                agent_b=b,
                trigger=trigger,
                conversation_model=engine._eff_conv_model,
                aigateway_url=engine._eff_url,
                token=engine._eff_token,
                db=db,
                world_state=world,
            )
            await _log_event(
                db, world, "speak",
                f"{a.name} and {b.name} spoke at ({x},{y}).",
                [a.agent_id, b.agent_id], x, y,
            )
        except Exception as exc:
            logger.error("Conversation failed between %s and %s: %s", a.name, b.name, exc)
            a.state = "idle"
            b.state = "idle"


# ---------------------------------------------------------------------------
# Individual agent turn processing
# ---------------------------------------------------------------------------

async def _process_agent_turn(
    agent: VillageAgent,
    all_agents: list[VillageAgent],
    db: AsyncSession,
    world: WorldState,
    sem: asyncio.Semaphore,
    engine: "VillageEngine",
):
    """Process a single agent's decision turn. Never raises."""
    try:
        async with sem:
            await _do_agent_turn(agent, all_agents, db, world, engine)
    except Exception as exc:
        logger.error("Agent turn failed for %s: %s", agent.name, exc, exc_info=True)
        agent.state = "idle"


async def _do_agent_turn(
    agent: VillageAgent,
    all_agents: list[VillageAgent],
    db: AsyncSession,
    world: WorldState,
    engine: "VillageEngine",
):
    """Core agent decision and action execution."""
    tiles_map = await _get_all_tiles_map(db)

    # Compute LOS
    visible_coords = calculate_los(
        agent.x, agent.y, agent.skills, tiles_map,
        world.game_hour, world.weather,
    )

    # Build visible tiles list
    visible_tiles = []
    for (vx, vy) in visible_coords:
        tile = tiles_map.get((vx, vy))
        if tile is None:
            # Generate tile if not yet known
            tile = await _ensure_tile_exists(vx, vy, db, world, engine.world_agent, engine._eff_world_model)
            tiles_map[(vx, vy)] = tile
        visible_tiles.append({
            "x": tile.x, "y": tile.y,
            "terrain": tile.terrain,
            "elevation": tile.elevation,
            "features": tile.features,
            "resource_nodes": tile.resource_nodes,
            "buildings": tile.buildings,
            "items": tile.items,
            "narrative": tile.narrative,
        })
        # Mark explored
        explored = tile.explored_by
        if agent.agent_id not in explored:
            explored.append(agent.agent_id)
            tile.explored_by = explored

    # Build nearby agents list
    nearby_agents = [
        {"agent_id": a.agent_id, "name": a.name, "x": a.x, "y": a.y, "state": a.state}
        for a in all_agents
        if a.agent_id != agent.agent_id
        and abs(a.x - agent.x) <= 6
        and abs(a.y - agent.y) <= 6
    ]

    # Determine model
    model = agent.brain_model or engine._eff_brain_model

    # Ask brain for action
    action = await engine.agent_brain.decide_action(
        agent=agent,
        visible_tiles=visible_tiles,
        world_state=world,
        nearby_agents=nearby_agents,
        model=model,
        aigateway_url=engine._eff_url,
        token=engine._eff_token,
    )

    # Clear god_hint after use
    if agent.god_hint:
        agent.god_hint = None

    # Execute action
    await _execute_action(agent, action, db, world, engine, tiles_map, all_agents)

    agent.last_tick = world.tick


async def _execute_action(
    agent: VillageAgent,
    action: dict,
    db: AsyncSession,
    world: WorldState,
    engine: "VillageEngine",
    tiles_map: dict,
    all_agents: list[VillageAgent],
):
    """Dispatch and execute an agent action."""
    act = action.get("action", "wait")
    thought = action.get("thought", "")

    if act == "move":
        await _action_move(agent, action, db, world, engine, tiles_map)

    elif act == "gather":
        await _action_gather(agent, action, db, world, tiles_map)

    elif act == "craft":
        await _action_craft(agent, action, db, world, tiles_map)

    elif act == "build":
        await _action_build(agent, action, db, world, tiles_map)

    elif act == "speak":
        await _action_speak(agent, action, db, world, all_agents)

    elif act == "eat":
        await _action_eat(agent, action, db, world)

    elif act == "rest":
        await _action_rest(agent, action, db, world)

    elif act == "examine":
        await _action_examine(agent, action, db, world, tiles_map)

    elif act == "loot":
        await _action_loot(agent, action, db, world, engine, tiles_map)

    elif act == "write":
        await _action_write(agent, action, db, world, tiles_map)

    elif act == "set_goal":
        goal_text = action.get("goal", "")
        agent.current_goal = goal_text
        agent.goal_set_tick = world.tick
        agent.add_memory(f"Set new goal: {agent.current_goal}")
        # Fire-and-forget: generate resource brief in background
        if goal_text and hasattr(engine, 'world_agent') and engine.world_agent:
            import asyncio as _asyncio
            _asyncio.create_task(
                _update_goal_resource_brief(
                    agent.agent_id, agent.name, goal_text, agent.inventory, engine
                )
            )

    elif act == "wait":
        agent.state = "idle"
        # Small social need relief when consciously choosing to rest/wait
        needs = agent.needs
        needs["rest"] = max(0, needs.get("rest", 0) - 2)
        agent.needs = needs


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------

async def _update_goal_resource_brief(agent_id: str, agent_name: str, goal: str, inventory: dict, engine_ref) -> None:
    """Background task: generate a resource brief for the agent's new goal and persist it."""
    from app.crafting import RECIPES
    try:
        brief = await engine_ref.world_agent.get_resource_brief(
            agent_name=agent_name,
            agent_goal=goal,
            agent_inventory=inventory,
            item_catalog=RECIPES,
            model=engine_ref._eff_world_model,
        )
        if not brief:
            return
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(VillageAgent).where(VillageAgent.agent_id == agent_id))
            agent = result.scalar_one_or_none()
            if agent:
                agent.goal_resource_brief = brief.strip()
                await db.commit()
    except Exception as exc:
        logger.error("_update_goal_resource_brief failed: %s", exc)


async def _action_move(
    agent: VillageAgent,
    action: dict,
    db: AsyncSession,
    world: WorldState,
    engine: "VillageEngine",
    tiles_map: dict,
):
    direction = action.get("direction", "n").lower()
    delta = _DIRECTION_DELTAS.get(direction, (0, 1))
    tx, ty = agent.x + delta[0], agent.y + delta[1]

    dest_tile = await _ensure_tile_exists(tx, ty, db, world, engine.world_agent, engine._eff_world_model)
    cost = get_terrain_movement_cost(dest_tile.terrain)

    if cost is None:
        agent.add_memory(f"Tried to move {direction} but {dest_tile.terrain} blocked the way.")
        return

    agent.x = tx
    agent.y = ty
    agent.state = "idle"

    # Mark explored
    explored = dest_tile.explored_by
    if agent.agent_id not in explored:
        explored.append(agent.agent_id)
        dest_tile.explored_by = explored

    agent.skills = apply_skill_gain(agent.skills, "navigation")
    agent.add_memory(f"Moved {direction} to ({tx},{ty}) — {dest_tile.terrain}.")

    await _log_event(
        db, world, "movement",
        f"{agent.name} moved {direction} to ({tx},{ty}).",
        [agent.agent_id], tx, ty,
    )


async def _action_gather(
    agent: VillageAgent,
    action: dict,
    db: AsyncSession,
    world: WorldState,
    tiles_map: dict,
):
    resource = action.get("resource", "raw_food")
    tile = tiles_map.get((agent.x, agent.y))
    if tile is None:
        agent.add_memory(f"Couldn't find {resource} — strange ground.")
        return

    if not can_gather(tile, resource):
        # Check if it's a found/gathereable type via recipe
        recipe = RECIPES.get(resource, {})
        valid_terrains = recipe.get("found_in", [])
        if tile.terrain not in valid_terrains:
            agent.add_memory(f"No {resource} to gather here.")
            return
        # Resource not seeded on tile — try to auto-seed if terrain is right
        nodes = tile.resource_nodes
        nodes.append({"type": resource, "qty": 2, "max_qty": 4, "regen_at_tick": 0})
        tile.resource_nodes = nodes

    item_type, qty, updated_nodes = do_gather(tile, resource, agent.skills)
    if qty == 0:
        agent.add_memory(f"Tried to gather {resource} but found nothing left.")
        return

    tile.resource_nodes = updated_nodes

    inv = agent.inventory
    inv[item_type] = inv.get(item_type, 0) + qty
    agent.inventory = inv

    skill = "gathering" if resource in ("raw_food", "tinder", "paper") else "foraging"
    agent.skills = apply_skill_gain(agent.skills, skill)

    agent.state = "working"
    agent.add_memory(f"Gathered {qty}x {item_type} at ({agent.x},{agent.y}).")

    await _log_event(
        db, world, "gather",
        f"{agent.name} gathered {qty}x {item_type}.",
        [agent.agent_id], agent.x, agent.y,
    )


async def _action_craft(
    agent: VillageAgent,
    action: dict,
    db: AsyncSession,
    world: WorldState,
    tiles_map: dict,
):
    recipe_name = action.get("recipe", "")
    tile = tiles_map.get((agent.x, agent.y))
    tile_features = tile.features if tile else []

    can, reason = validate_craft(recipe_name, agent.inventory, tile_features, agent.skills)
    if not can:
        agent.add_memory(f"Couldn't craft {recipe_name}: {reason}")
        return

    agent.inventory = do_craft(recipe_name, agent.inventory)
    agent.skills = apply_skill_gain(agent.skills, "crafting")
    agent.state = "working"
    agent.add_memory(f"Crafted {recipe_name}.")

    await _log_event(
        db, world, "craft",
        f"{agent.name} crafted {recipe_name}.",
        [agent.agent_id], agent.x, agent.y,
    )


async def _action_build(
    agent: VillageAgent,
    action: dict,
    db: AsyncSession,
    world: WorldState,
    tiles_map: dict,
):
    structure = action.get("structure", "")
    recipe = RECIPES.get(structure)
    if not recipe or not recipe.get("is_building", False):
        agent.add_memory(f"Don't know how to build {structure}.")
        return

    tile = tiles_map.get((agent.x, agent.y))
    tile_features = tile.features if tile else []

    can, reason = validate_craft(structure, agent.inventory, tile_features, agent.skills)
    if not can:
        agent.add_memory(f"Couldn't build {structure}: {reason}")
        return

    agent.inventory = do_craft(structure, agent.inventory)
    agent.skills = apply_skill_gain(agent.skills, "building")
    agent.state = "working"

    if tile:
        building_id = str(uuid.uuid4())[:8]
        buildings = tile.buildings
        buildings.append({
            "id": building_id,
            "type": structure,
            "builder_id": agent.agent_id,
            "health": 100,
        })
        tile.buildings = buildings

        # If campfire, add feature too for near_req checks
        if structure == "campfire":
            features = tile.features
            if "campfire" not in features:
                features.append("campfire")
            tile.features = features

        # Road tile: change terrain so movement cost (0.5) kicks in automatically
        if structure == "road_tile":
            tile.terrain = "road"

    agent.add_memory(f"Built {structure} at ({agent.x},{agent.y}).")

    await _log_event(
        db, world, "build",
        f"{agent.name} built a {structure} at ({agent.x},{agent.y}).",
        [agent.agent_id], agent.x, agent.y,
    )


async def _action_speak(
    agent: VillageAgent,
    action: dict,
    db: AsyncSession,
    world: WorldState,
    all_agents: list[VillageAgent],
):
    target = action.get("target", "broadcast")
    message = action.get("message", "")

    agent.add_memory(f"Said to {target}: \"{message[:80]}\"")
    agent.needs = satisfy_need(agent.needs, "social", 10)

    # Find target agent if specific
    target_agent = None
    if target != "broadcast":
        target_agent = next(
            (a for a in all_agents if a.name.lower() == target.lower()), None
        )
        if target_agent:
            target_agent.add_memory(f"{agent.name} said: \"{message[:80]}\"")
            target_agent.needs = satisfy_need(target_agent.needs, "social", 5)

    description = f"{agent.name} said: \"{message[:120]}\""
    await _log_event(
        db, world, "speak",
        description,
        [agent.agent_id] + ([target_agent.agent_id] if target_agent else []),
        agent.x, agent.y,
    )


async def _action_eat(
    agent: VillageAgent,
    action: dict,
    db: AsyncSession,
    world: WorldState,
):
    food = action.get("food", "")
    inv = agent.inventory
    cfg = world.sim_config
    cooked_restore = 40 * cfg.get("cooked_food_restore", 1.0)
    raw_restore    = 22 * cfg.get("raw_food_restore",    1.0)

    if food == "cooked_food" and inv.get("cooked_food", 0) > 0:
        inv["cooked_food"] -= 1
        if inv["cooked_food"] == 0:
            del inv["cooked_food"]
        agent.inventory = inv
        agent.needs = satisfy_need(agent.needs, "hunger", cooked_restore)
        agent.add_memory("Ate cooked food — properly satisfied.")
        await _log_event(db, world, "need", f"{agent.name} ate cooked food.", [agent.agent_id], agent.x, agent.y)
    elif food in ("raw_food", "") and inv.get("raw_food", 0) > 0:
        inv["raw_food"] -= 1
        if inv["raw_food"] == 0:
            del inv["raw_food"]
        agent.inventory = inv
        agent.needs = satisfy_need(agent.needs, "hunger", raw_restore)
        agent.add_memory("Ate raw food. Not ideal, but filling enough.")
        await _log_event(db, world, "need", f"{agent.name} ate raw food.", [agent.agent_id], agent.x, agent.y)
    else:
        agent.add_memory("Wanted to eat but had nothing to eat.")


async def _action_rest(
    agent: VillageAgent,
    action: dict,
    db: AsyncSession,
    world: WorldState,
):
    hours = min(int(action.get("hours", 1)), 4)
    rest_relief = hours * 20.0
    agent.needs = satisfy_need(agent.needs, "rest", rest_relief)
    agent.state = "sleeping"
    agent.add_memory(f"Rested for {hours} hour(s).")
    await _log_event(
        db, world, "need",
        f"{agent.name} rested.",
        [agent.agent_id], agent.x, agent.y,
    )


async def _action_examine(
    agent: VillageAgent,
    action: dict,
    db: AsyncSession,
    world: WorldState,
    tiles_map: dict,
):
    target = action.get("target", "surroundings")
    tile = tiles_map.get((agent.x, agent.y))
    tile_info = ""
    if tile:
        tile_info = (
            f"{tile.terrain} terrain. "
            f"Features: {', '.join(tile.features) or 'none'}. "
            f"Resources available: {', '.join(r['type'] for r in tile.resource_nodes if r.get('qty',0)>0) or 'none'}."
        )
    agent.add_memory(f"Examined {target}: {tile_info[:100]}")
    await _log_event(
        db, world, "discovery",
        f"{agent.name} examined {target}.",
        [agent.agent_id], agent.x, agent.y,
    )


async def _action_loot(
    agent: VillageAgent,
    action: dict,
    db: AsyncSession,
    world: WorldState,
    engine: "VillageEngine",
    tiles_map: dict,
):
    tile = tiles_map.get((agent.x, agent.y))
    if tile is None:
        agent.add_memory("Tried to search for loot but the ground felt hollow.")
        return

    features = tile.features
    if "ruins" not in features:
        agent.add_memory("Nothing to loot here — no ruins.")
        return

    if "ruins_looted" in features:
        agent.add_memory("Already picked through these ruins. Nothing left.")
        return

    # Generate loot via WorldAgent
    loot = await engine.world_agent.generate_ruins_loot(
        agent_name=agent.name,
        personality=agent.personality_summary,
        tile_narrative=tile.narrative,
        world_bible="",
        model=engine._eff_world_model,
    )

    # Give items to agent
    inv = agent.inventory
    items_found = []
    for item_type, qty in loot["items"].items():
        inv[item_type] = inv.get(item_type, 0) + qty
        items_found.append(f"{qty}x {item_type}")
    agent.inventory = inv

    # Mark ruins as looted
    features.append("ruins_looted")
    tile.features = features

    agent.state = "working"
    narrative = loot["narrative"]
    agent.add_memory(f"Looted ruins at ({agent.x},{agent.y}): {narrative[:80]}")

    await _log_event(
        db, world, "discovery",
        f"{agent.name} searched ruins at ({agent.x},{agent.y}) — found {', '.join(items_found)}.",
        [agent.agent_id], agent.x, agent.y,
    )


async def _action_write(
    agent: VillageAgent,
    action: dict,
    db: AsyncSession,
    world: WorldState,
    tiles_map: dict,
):
    content = action.get("content", "")
    medium = action.get("medium", "note")

    # Check for required materials
    inv = agent.inventory
    if medium == "note":
        if inv.get("paper", 0) < 1 or inv.get("charcoal", 0) < 1:
            agent.add_memory("Tried to write a note but didn't have paper and charcoal.")
            return
        inv["paper"] = inv.get("paper", 1) - 1
        inv["charcoal"] = inv.get("charcoal", 1) - 1
        if inv["paper"] == 0:
            del inv["paper"]
        if inv["charcoal"] == 0:
            del inv["charcoal"]
        # Add note to tile items
        tile = tiles_map.get((agent.x, agent.y))
        if tile:
            items = tile.items
            items.append({"type": "note", "qty": 1, "content": content[:200], "author": agent.name})
            tile.items = items
        agent.inventory = inv
        agent.add_memory(f"Wrote a note: \"{content[:60]}\"")

    await _log_event(
        db, world, "speak",
        f"{agent.name} wrote: \"{content[:80]}\"",
        [agent.agent_id], agent.x, agent.y,
    )


# ---------------------------------------------------------------------------
# Resource regeneration
# ---------------------------------------------------------------------------

async def _tick_resource_regen(db: AsyncSession, world: WorldState):
    """Regenerate depleted resource nodes that have hit their regen_at_tick."""
    result = await db.execute(select(Tile))
    tiles = result.scalars().all()
    for tile in tiles:
        nodes = tile.resource_nodes
        changed = False
        updated = []
        for node in nodes:
            regen_at = node.get("regen_at_tick", 0)
            if node.get("qty", 0) < node.get("max_qty", 3):
                if regen_at == 0 or world.tick >= regen_at:
                    # Slowly restore 1 unit
                    node = dict(node)
                    node["qty"] = min(node.get("max_qty", 3), node.get("qty", 0) + 1)
                    node["regen_at_tick"] = world.tick + 5  # regen again in 5 ticks
                    changed = True
            updated.append(node)
        if changed:
            tile.resource_nodes = updated


# ---------------------------------------------------------------------------
# SSE broadcast
# ---------------------------------------------------------------------------

async def _broadcast_state(
    world: WorldState,
    agents: list[VillageAgent],
    db: AsyncSession,
):
    from app.schemas import world_to_out, agent_to_out, event_to_out, tile_to_out, CreatureOut

    # Get recent events
    events_result = await db.execute(
        select(EventLog)
        .order_by(EventLog.created_at.desc())
        .limit(10)
    )
    recent_events = list(events_result.scalars().all())

    # Get all tiles (viewers see everything)
    tiles_result = await db.execute(select(Tile))
    all_tiles = tiles_result.scalars().all()

    # Get creatures
    creatures_result = await db.execute(select(Creature))
    all_creatures = creatures_result.scalars().all()

    payload = {
        "world_state": world_to_out(world).model_dump(),
        "agents": [agent_to_out(a).model_dump() for a in agents],
        "recent_events": [event_to_out(e).model_dump() for e in recent_events],
        "tiles": [tile_to_out(t).model_dump() for t in all_tiles],
        "creatures": [
            CreatureOut(id=c.id, creature_type=c.creature_type, x=c.x, y=c.y, state=c.state).model_dump()
            for c in all_creatures
        ],
    }
    await sse_manager.broadcast("tick_update", payload)


# ---------------------------------------------------------------------------
# VillageEngine
# ---------------------------------------------------------------------------

class VillageEngine:
    def __init__(self):
        self.world_agent = WorldAgent(settings.aigateway_url, settings.aigateway_token)
        self.agent_brain = AgentBrain()
        self._task: asyncio.Task | None = None
        self._stepping = False
        # Effective runtime config — refreshed each tick from DB overrides
        self._eff_url: str = settings.aigateway_url
        self._eff_token: str = settings.aigateway_token
        self._eff_brain_model: str = settings.agent_brain_model
        self._eff_conv_model: str = settings.conversation_model
        self._eff_world_model: str = settings.world_agent_model

    async def start(self):
        async with AsyncSessionLocal() as db:
            world = await _get_world_state(db)
            if not world:
                logger.error("No WorldState found — cannot start engine.")
                return
            world.engine_state = "running"
            await db.commit()

        if self._task and not self._task.done():
            logger.info("Engine already running.")
            return

        self._task = asyncio.create_task(self._loop())
        logger.info("VillageEngine started.")

    async def pause(self):
        async with AsyncSessionLocal() as db:
            world = await _get_world_state(db)
            if world:
                world.engine_state = "paused"
                await db.commit()
        logger.info("VillageEngine paused.")

    async def stop(self, persist: bool = True):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if persist:
            async with AsyncSessionLocal() as db:
                world = await _get_world_state(db)
                if world:
                    world.engine_state = "stopped"
                    await db.commit()
        logger.info("VillageEngine stopped.")

    async def step_once(self):
        """Execute exactly one tick, regardless of engine state."""
        if self._stepping:
            return
        self._stepping = True
        try:
            await self._process_tick()
        finally:
            self._stepping = False

    async def _loop(self):
        """Main tick loop."""
        while True:
            try:
                async with AsyncSessionLocal() as db:
                    world = await _get_world_state(db)
                    if not world:
                        await asyncio.sleep(5)
                        continue

                    if world.engine_state == "stopped":
                        break
                    if world.engine_state == "paused":
                        await asyncio.sleep(2)
                        continue

                    tick_rate = world.tick_rate_seconds

                await self._process_tick()
                await asyncio.sleep(tick_rate)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Engine loop error: %s", exc, exc_info=True)
                await asyncio.sleep(10)

    async def _process_tick(self):
        """Main tick handler — runs once per game hour."""
        async with AsyncSessionLocal() as db:
            world = await _get_world_state(db)
            if not world:
                return

            # 0. Refresh effective gateway + model config from DB overrides
            gw = world.gateway_config
            self._eff_url = (gw.get("url") or settings.aigateway_url).rstrip("/")
            self._eff_token = gw.get("token") or settings.aigateway_token
            self.world_agent.aigateway_url = self._eff_url
            self.world_agent.token = self._eff_token
            mc_result = await db.execute(select(ModelConfig))
            for _mc in mc_result.scalars().all():
                if _mc.task == "world_agent" and _mc.model_id:
                    self._eff_world_model = _mc.model_id
                elif _mc.task == "agent_brain" and _mc.model_id:
                    self._eff_brain_model = _mc.model_id
                elif _mc.task == "conversation" and _mc.model_id:
                    self._eff_conv_model = _mc.model_id

            # 1. Advance game time
            world.tick += 1
            hours_per_tick = float(world.sim_config.get("hours_per_tick", 1.0))
            accum = (world.game_hour_accum or 0.0) + hours_per_tick
            hours_to_advance = int(accum)
            world.game_hour_accum = accum - hours_to_advance  # keep fractional remainder

            prev_hour = world.game_hour
            new_hour = prev_hour + hours_to_advance
            world.game_hour = new_hour % 24
            if new_hour >= 24:
                world.game_day += 1
                # Seasonal change every N days
                if world.game_day % _DAYS_PER_SEASON == 0:
                    current_idx = _SEASONS.index(world.season) if world.season in _SEASONS else 0
                    world.season = _SEASONS[(current_idx + 1) % 4]
                    import random
                    import app.events as ev_mod
                    world.weather = random.choice(ev_mod._SEASON_WEATHER[world.season])
                    await _log_event(
                        db, world, "world_event",
                        f"Season changed to {world.season}.",
                        [],
                    )

            # 2. Load active agents
            agents = await _get_active_agents(db)

            # 3. Process movement
            for agent in agents:
                if agent.travel_path:
                    try:
                        await _step_agent_movement(agent, db, world, self.world_agent)
                    except Exception as exc:
                        logger.error("Movement error for %s: %s", agent.name, exc)

            # 4. Decay needs
            for agent in agents:
                try:
                    await _tick_needs(agent, db, world)
                except Exception as exc:
                    logger.error("Needs decay error for %s: %s", agent.name, exc)

            # 5. Spontaneous conversations
            try:
                await _check_conversations(agents, db, world, self)
            except Exception as exc:
                logger.error("Conversation check error: %s", exc)

            # 6. Agent decisions (idle agents only)
            # Auto-wake: sleeping agents get 1 tick of sleep state (rest already applied
            # in full when the action ran), then return to idle.  Conversations get 2
            # ticks of leeway before we treat them as stuck and force-idle.
            for _a in agents:
                ticks_stale = world.tick - (_a.last_tick or 0)
                if _a.state == "sleeping" and ticks_stale >= 1:
                    _a.state = "idle"
                elif _a.state == "socializing" and ticks_stale >= 2:
                    _a.state = "idle"
                elif _a.state == "collapsed" and ticks_stale >= 4:
                    # Slow recovery after collapse
                    _a.state = "idle"
                    needs = _a.needs
                    needs["hunger"] = min(needs.get("hunger", 100), 70)
                    needs["rest"] = min(needs.get("rest", 100), 60)
                    _a.needs = needs
                    _a.mood = compute_mood(needs)
                    _a.add_memory("Slowly came back to consciousness. Still weak, but upright.")
                    await _log_event(
                        db, world, "collapse",
                        f"{_a.name} recovered from collapse at ({_a.x},{_a.y}).",
                        [_a.agent_id], _a.x, _a.y,
                    )

            idle_agents = [
                a for a in agents
                if not a.travel_path and a.state not in ("socializing", "sleeping", "collapsed")
            ]
            sem = asyncio.Semaphore(3)
            tasks = [
                _process_agent_turn(agent, agents, db, world, sem, self)
                for agent in idle_agents
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.error("Agent task exception: %s", r)

            # 7. Resource regeneration
            try:
                await _tick_resource_regen(db, world)
            except Exception as exc:
                logger.error("Resource regen error: %s", exc)

            # 8. Random world event
            try:
                model = self._eff_world_model
                await check_random_event(world, agents, db, self.world_agent, model)
            except Exception as exc:
                logger.error("World event check error: %s", exc)

            # 8.5. Creature movement
            try:
                from app.creatures import process_creatures
                creatures_result = await db.execute(select(Creature))
                creatures = list(creatures_result.scalars().all())
                if creatures:
                    tiles_result = await db.execute(select(Tile))
                    tiles_map_dict = {
                        f"{t.x},{t.y}": {"terrain": t.terrain}
                        for t in tiles_result.scalars().all()
                    }
                    await process_creatures(creatures, agents, tiles_map_dict, world, db)
            except Exception as exc:
                logger.error("Creature processing error: %s", exc)

            # 9. Save + snapshot
            world.last_tick_at = _time.time()
            await _save_tick_snapshot(db, world, agents)
            await db.commit()

            # 10. Broadcast SSE
            try:
                await _broadcast_state(world, agents, db)
            except Exception as exc:
                logger.error("SSE broadcast error: %s", exc)

        logger.debug("Tick %d complete. Day %d, Hour %02d:00", world.tick, world.game_day, world.game_hour)


# Singleton
engine = VillageEngine()
