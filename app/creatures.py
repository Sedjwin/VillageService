"""
Creature / wildlife system.
Animals wander, flee agents, and add organic life to the world.
"""
from __future__ import annotations

import random
import uuid
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Creature, WorldState

logger = logging.getLogger(__name__)

# Which terrain types each creature is at home in
_CREATURE_HABITAT: dict[str, set[str]] = {
    "rabbit": {"grass", "light_forest", "beach"},
    "deer":   {"grass", "light_forest", "hills"},
    "wolf":   {"dense_forest", "light_forest", "hills"},
    "bird":   {"grass", "light_forest", "beach", "hills", "mountain"},
}

# Terrain that creatures cannot enter
_IMPASSABLE = {"water", "mountain", "cave"}

# Chebyshev distance at which each type reacts to agents
_FLEE_RADIUS: dict[str, int] = {
    "rabbit": 3,
    "deer":   4,
    "wolf":   8,  # wolf is drawn toward agents rather than away
    "bird":   3,
}

_DIRECTIONS = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
_MAX_CREATURES = 28


def _dist(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(abs(x1 - x2), abs(y1 - y2))


async def maybe_spawn_creature(
    x: int,
    y: int,
    terrain: str,
    db: AsyncSession,
    world: WorldState,
) -> Creature | None:
    """Called when a new tile is generated. 20 % chance to spawn wildlife."""
    result = await db.execute(select(Creature))
    if len(result.scalars().all()) >= _MAX_CREATURES:
        return None

    candidates = [ct for ct, habitats in _CREATURE_HABITAT.items() if terrain in habitats]
    if not candidates or random.random() > 0.20:
        return None

    c = Creature(
        id=str(uuid.uuid4()),
        creature_type=random.choice(candidates),
        x=x,
        y=y,
        state="idle",
        last_tick=world.tick,
        spawned_tick=world.tick,
    )
    db.add(c)
    logger.debug("Spawned %s at (%d,%d)", c.creature_type, x, y)
    return c


async def process_creatures(
    creatures: list[Creature],
    agents: list,
    tiles_map: dict[str, dict],
    world: WorldState,
    db: AsyncSession,
) -> None:
    """Move all creatures. Called every tick from the engine."""
    for creature in creatures:
        # Rabbits and birds move every other tick; deer every 2 ticks; wolves every tick
        move_freq = {"rabbit": 2, "deer": 2, "wolf": 1, "bird": 2}
        if world.tick % move_freq.get(creature.creature_type, 2) != 0:
            continue
        # 65 % chance to actually move this tick
        if random.random() > 0.65:
            creature.state = "idle"
            continue

        # Find nearest agent
        nearest_agent = None
        nearest_dist = 9999
        for agent in agents:
            d = _dist(creature.x, creature.y, agent.x, agent.y)
            if d < nearest_dist:
                nearest_dist = d
                nearest_agent = agent

        flee_r = _FLEE_RADIUS.get(creature.creature_type, 3)
        is_wolf = creature.creature_type == "wolf"

        def passable(nx: int, ny: int) -> bool:
            t = tiles_map.get(f"{nx},{ny}")
            return bool(t and t.get("terrain", "grass") not in _IMPASSABLE)

        if nearest_agent is not None and nearest_dist < flee_r and not is_wolf:
            # Flee — move directly away from the nearest agent
            creature.state = "fleeing"
            dx = creature.x - nearest_agent.x
            dy = creature.y - nearest_agent.y
            dirs = sorted(_DIRECTIONS, key=lambda d: -(d[0] * dx + d[1] * dy))
            for dd in dirs[:4]:
                nx, ny = creature.x + dd[0], creature.y + dd[1]
                if passable(nx, ny):
                    creature.x, creature.y = nx, ny
                    break

        elif is_wolf and nearest_agent is not None and nearest_dist < flee_r:
            # Wolf: stalk toward agent
            creature.state = "stalking"
            dx = nearest_agent.x - creature.x
            dy = nearest_agent.y - creature.y
            dirs = sorted(_DIRECTIONS, key=lambda d: -(d[0] * dx + d[1] * dy))
            for dd in dirs[:3]:
                nx, ny = creature.x + dd[0], creature.y + dd[1]
                if passable(nx, ny):
                    creature.x, creature.y = nx, ny
                    break

        else:
            # Random wander — prefer habitat terrain
            creature.state = "wandering"
            habitat = _CREATURE_HABITAT.get(creature.creature_type, set())
            shuffled = list(_DIRECTIONS)
            random.shuffle(shuffled)
            # Try to stay in preferred terrain first, then accept any
            for dd in shuffled:
                nx, ny = creature.x + dd[0], creature.y + dd[1]
                t = tiles_map.get(f"{nx},{ny}")
                if t and t.get("terrain") in habitat:
                    creature.x, creature.y = nx, ny
                    break
            else:
                for dd in shuffled:
                    nx, ny = creature.x + dd[0], creature.y + dd[1]
                    if passable(nx, ny):
                        creature.x, creature.y = nx, ny
                        break

        creature.last_tick = world.tick
