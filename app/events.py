"""
World event system — random events that give the world texture and consequence.
"""
from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import EventLog, WorldState, VillageAgent, Tile

logger = logging.getLogger(__name__)


POSSIBLE_EVENTS = [
    {"type": "weather_change", "weight": 10},
    {"type": "resource_discovery", "weight": 5},
    {"type": "animal_sighting", "weight": 8},
    {"type": "storm", "weight": 3},
    {"type": "mysterious_object", "weight": 2},
    {"type": "windfall", "weight": 4},       # branch falls, scatters materials
    {"type": "cold_snap", "weight": 3},       # warmth need spikes
    {"type": "fog_bank", "weight": 4},        # vision penalty for a few ticks
    {"type": "meteor", "weight": 1},
    {"type": "seasonal_change", "weight": 1},
]

_EVENT_BASE_CHANCE = 0.08  # 8% chance per tick that SOMETHING happens

# Season cycle: spring -> summer -> autumn -> winter -> spring
_SEASONS = ["spring", "summer", "autumn", "winter"]

# Weather valid for each season
_SEASON_WEATHER = {
    "spring": ["clear", "cloudy", "rainy"],
    "summer": ["clear", "clear", "cloudy"],
    "autumn": ["clear", "cloudy", "rainy", "stormy"],
    "winter": ["clear", "cloudy", "snowy", "stormy"],
}


async def check_random_event(
    world_state: WorldState,
    active_agents: list[VillageAgent],
    db: AsyncSession,
    world_agent,
    model: str,
) -> None:
    """
    Called each tick. Small chance to trigger a world event.
    """
    event_chance = _EVENT_BASE_CHANCE * world_state.sim_config.get("event_chance", 1.0)
    if random.random() > event_chance:
        return

    # Weighted random selection
    total = sum(e["weight"] for e in POSSIBLE_EVENTS)
    r = random.uniform(0, total)
    cumulative = 0.0
    chosen_type = "weather_change"
    for event in POSSIBLE_EVENTS:
        cumulative += event["weight"]
        if r <= cumulative:
            chosen_type = event["type"]
            break

    try:
        description = await apply_event(
            chosen_type, world_state, active_agents, db, world_agent, model
        )
        if description:
            log = EventLog(
                id=str(uuid.uuid4()),
                tick=world_state.tick,
                game_day=world_state.game_day,
                game_hour=world_state.game_hour,
                event_type=chosen_type,
                description=description,
                created_at=datetime.utcnow(),
            )
            log.agents_involved = [a.agent_id for a in active_agents]
            db.add(log)
            logger.info("World event triggered: %s — %s", chosen_type, description[:80])
    except Exception as exc:
        logger.error("check_random_event failed for %s: %s", chosen_type, exc)


async def apply_event(
    event_type: str,
    world_state: WorldState,
    active_agents: list[VillageAgent],
    db: AsyncSession,
    world_agent,
    model: str,
) -> str:
    """
    Apply event effects and return a narrative description.
    """
    if event_type == "weather_change":
        return _apply_weather_change(world_state)

    elif event_type == "seasonal_change":
        return _apply_seasonal_change(world_state)

    elif event_type == "storm":
        return _apply_storm(world_state)

    elif event_type == "cold_snap":
        return _apply_cold_snap(active_agents)

    elif event_type == "windfall":
        return await _apply_windfall(db, world_state)

    elif event_type == "resource_discovery":
        return await _apply_resource_discovery(db, world_state, world_agent, model, active_agents)

    elif event_type == "animal_sighting":
        return _apply_animal_sighting(active_agents, world_state)

    elif event_type == "mysterious_object":
        return await _apply_mysterious_object(db, world_state, world_agent, model)

    elif event_type == "fog_bank":
        return _apply_fog_bank(world_state)

    elif event_type == "meteor":
        return await _apply_meteor(db, world_state)

    else:
        return ""


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def _apply_weather_change(world_state: WorldState) -> str:
    valid = _SEASON_WEATHER.get(world_state.season, ["clear", "cloudy"])
    old_weather = world_state.weather
    new_weather = random.choice(valid)
    world_state.weather = new_weather
    return f"The weather shifted from {old_weather} to {new_weather}."


def _apply_seasonal_change(world_state: WorldState) -> str:
    current_idx = _SEASONS.index(world_state.season) if world_state.season in _SEASONS else 0
    new_season = _SEASONS[(current_idx + 1) % 4]
    old_season = world_state.season
    world_state.season = new_season
    world_state.weather = random.choice(_SEASON_WEATHER[new_season])
    return f"{old_season.capitalize()} gave way to {new_season}. The world shifted."


def _apply_storm(world_state: WorldState) -> str:
    world_state.weather = "stormy"
    return "A storm rolled in fast — dark clouds, rain in sheets, wind that makes movement slow."


def _apply_cold_snap(active_agents: list[VillageAgent]) -> str:
    for agent in active_agents:
        needs = agent.needs
        needs["warmth"] = min(100, needs.get("warmth", 0) + 20)
        agent.needs = needs
    return "A cold snap swept through. Everyone felt it in their bones."


async def _apply_windfall(db: AsyncSession, world_state: WorldState) -> str:
    from sqlalchemy import select
    # Add materials to a random explored tile
    result = await db.execute(select(Tile).limit(20))
    tiles = result.scalars().all()
    if not tiles:
        return ""

    tile = random.choice(tiles)
    nodes = tile.resource_nodes
    # Add long_sticks — branches fell
    found = False
    for node in nodes:
        if node.get("type") == "long_stick":
            node["qty"] = min(node.get("max_qty", 4), node["qty"] + 3)
            found = True
            break
    if not found:
        nodes.append({"type": "long_stick", "qty": 3, "max_qty": 5, "regen_at_tick": 0})
    tile.resource_nodes = nodes

    return f"A tree came down somewhere near ({tile.x},{tile.y}). Branches everywhere."


async def _apply_resource_discovery(
    db: AsyncSession,
    world_state: WorldState,
    world_agent,
    model: str,
    active_agents: list[VillageAgent],
) -> str:
    from sqlalchemy import select
    result = await db.execute(select(Tile).limit(30))
    tiles = result.scalars().all()
    if not tiles:
        return ""

    tile = random.choice(tiles)
    nodes = tile.resource_nodes
    resource = random.choice(["sharp_rock", "vine", "raw_food", "long_stick"])
    for node in nodes:
        if node.get("type") == resource:
            node["qty"] = node.get("max_qty", 4)
            tile.resource_nodes = nodes
            return f"A rich deposit of {resource} appeared near ({tile.x},{tile.y})."

    nodes.append({"type": resource, "qty": 4, "max_qty": 6, "regen_at_tick": 0})
    tile.resource_nodes = nodes
    return f"A new vein of {resource} was uncovered near ({tile.x},{tile.y})."


def _apply_animal_sighting(
    active_agents: list[VillageAgent],
    world_state: WorldState,
) -> str:
    animals = [
        "a red deer pausing at the forest edge",
        "a hawk circling overhead for a long moment",
        "a fox watching from between the trees",
        "a heron standing motionless in the shallows",
        "a badger rooting through fallen leaves nearby",
    ]
    sighting = random.choice(animals)
    for agent in active_agents:
        agent.add_memory(f"Saw {sighting}.")
    return f"Wildlife spotted: {sighting.capitalize()}."


async def _apply_mysterious_object(
    db: AsyncSession,
    world_state: WorldState,
    world_agent,
    model: str,
) -> str:
    from sqlalchemy import select
    result = await db.execute(select(Tile).limit(20))
    tiles = result.scalars().all()
    if not tiles:
        return "Something strange was seen in the distance."

    tile = random.choice(tiles)
    features = tile.features
    mysterious = random.choice([
        "carved_marker",
        "old_campfire_ring",
        "buried_chest",
        "stone_arrangement",
        "rusted_mechanism",
        "faded_rope_bridge",
    ])
    if mysterious not in features:
        features.append(mysterious)
        tile.features = features
    return f"Something unexpected was found near ({tile.x},{tile.y}): a {mysterious.replace('_', ' ')}."


def _apply_fog_bank(world_state: WorldState) -> str:
    # Fog is represented as cloudy weather with a note
    world_state.weather = "cloudy"
    return "A dense fog rolled in from the lowlands. Vision was cut to almost nothing."


async def _apply_meteor(
    db: AsyncSession,
    world_state: WorldState,
) -> str:
    # Create an impact crater tile
    import random as r
    from app.models import Tile

    ix = r.randint(-20, 20)
    iy = r.randint(-20, 20)
    crater = Tile(
        x=ix, y=iy,
        terrain="hills",
        elevation=1,
        narrative="A fresh impact crater, still warm at the edges. Glassy rock fragments everywhere.",
    )
    crater.features = ["impact_crater", "glass_fragments"]
    crater.resource_nodes = [{"type": "sharp_rock", "qty": 8, "max_qty": 8, "regen_at_tick": 0}]
    crater.items = []
    crater.explored_by = []
    crater.buildings = []
    db.add(crater)

    return f"A meteor struck near ({ix},{iy}). The sky lit up. A crater smokes where it hit."
