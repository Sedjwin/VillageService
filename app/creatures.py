"""
Creature / wildlife system.
Animals wander, flee agents, and add organic life to the world.

Wolf pack behaviour
-------------------
Wolves detect agents within WOLF_STALK_RANGE tiles and begin orbiting at a
safe distance rather than rushing in.  The safe distance and patience shrink
as more wolves gather into a local pack:

  Lone wolf (1):  orbits 4+ tiles away, gives up after  8 ticks → retreats
  Pair      (2):  creeps to 2+ tiles away, lingers for 15 ticks → retreats
  Pack      (3+): closes to 1 tile,       patient for  25 ticks → retreats

While retreating the wolf moves away for WOLF_RETREAT_TICKS ticks before
returning to normal wandering.  The result: rare close encounters, frequent
distant sightings, self-dispersing after patience expires.
"""
from __future__ import annotations

import random
import uuid
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Creature, WorldState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Habitat / passability tables
# ---------------------------------------------------------------------------

_CREATURE_HABITAT: dict[str, set[str]] = {
    "rabbit": {"grass", "light_forest", "beach"},
    "deer":   {"grass", "light_forest", "hills"},
    "wolf":   {"dense_forest", "light_forest", "hills"},
    "bird":   {"grass", "light_forest", "beach", "hills", "mountain"},
}

_IMPASSABLE = {"water", "mountain", "cave"}

_DIRECTIONS = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

# ---------------------------------------------------------------------------
# Population caps
# ---------------------------------------------------------------------------

_MAX_CREATURES_TOTAL = 28
_MAX_PER_SPECIES: dict[str, int] = {
    "wolf":   5,
    "deer":   8,
    "rabbit": 10,
    "bird":   10,
}

# ---------------------------------------------------------------------------
# Wolf pack constants
# ---------------------------------------------------------------------------

_WOLF_STALK_RANGE  = 10   # tiles — wolf detects and begins orbiting
_WOLF_PACK_RADIUS  = 5    # tiles — wolves within this form a pack
_WOLF_RETREAT_TICKS = 6   # ticks spent retreating before resuming wander

# tier → (min_distance_to_agent, patience_ticks_before_retreating)
_WOLF_PACK_TIERS: dict[int, tuple[int, int]] = {
    1: (4,  8),
    2: (2, 15),
    3: (1, 25),
}

# ---------------------------------------------------------------------------
# Non-wolf flee radius
# ---------------------------------------------------------------------------

_FLEE_RADIUS: dict[str, int] = {
    "rabbit": 3,
    "deer":   4,
    "bird":   3,
}


# ---------------------------------------------------------------------------
# Movement helpers
# ---------------------------------------------------------------------------

def _chebyshev(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(abs(x1 - x2), abs(y1 - y2))


def _passable(nx: int, ny: int, tiles_map: dict) -> bool:
    t = tiles_map.get(f"{nx},{ny}")
    return bool(t and t.get("terrain", "grass") not in _IMPASSABLE)


def _step_toward(cx: int, cy: int, tx: int, ty: int, tiles_map: dict) -> tuple[int, int]:
    """One step toward (tx, ty) via the direction that minimises Manhattan distance."""
    dirs = sorted(_DIRECTIONS, key=lambda d: abs(cx + d[0] - tx) + abs(cy + d[1] - ty))
    for d in dirs[:4]:
        nx, ny = cx + d[0], cy + d[1]
        if _passable(nx, ny, tiles_map):
            return nx, ny
    return cx, cy


def _step_away(cx: int, cy: int, fx: int, fy: int, tiles_map: dict) -> tuple[int, int]:
    """One step away from (fx, fy) via the direction that maximises Manhattan distance."""
    dirs = sorted(_DIRECTIONS, key=lambda d: -(abs(cx + d[0] - fx) + abs(cy + d[1] - fy)))
    for d in dirs[:4]:
        nx, ny = cx + d[0], cy + d[1]
        if _passable(nx, ny, tiles_map):
            return nx, ny
    return cx, cy


def _wander(cx: int, cy: int, habitat: set[str], tiles_map: dict) -> tuple[int, int]:
    """Random walk preferring habitat terrain, falling back to any passable tile."""
    shuffled = list(_DIRECTIONS)
    random.shuffle(shuffled)
    for d in shuffled:
        nx, ny = cx + d[0], cy + d[1]
        t = tiles_map.get(f"{nx},{ny}")
        if t and t.get("terrain") in habitat:
            return nx, ny
    for d in shuffled:
        nx, ny = cx + d[0], cy + d[1]
        if _passable(nx, ny, tiles_map):
            return nx, ny
    return cx, cy


# ---------------------------------------------------------------------------
# Spawn
# ---------------------------------------------------------------------------

async def maybe_spawn_creature(
    x: int,
    y: int,
    terrain: str,
    db: AsyncSession,
    world: WorldState,
) -> Creature | None:
    """Called when a new tile is generated.  20 % chance to spawn wildlife."""
    result = await db.execute(select(Creature))
    all_creatures = result.scalars().all()

    if len(all_creatures) >= _MAX_CREATURES_TOTAL:
        return None

    candidates = [ct for ct, habitats in _CREATURE_HABITAT.items() if terrain in habitats]
    if not candidates or random.random() > 0.20:
        return None

    # Respect per-species caps
    species_counts: dict[str, int] = {}
    for c in all_creatures:
        species_counts[c.creature_type] = species_counts.get(c.creature_type, 0) + 1

    candidates = [
        ct for ct in candidates
        if species_counts.get(ct, 0) < _MAX_PER_SPECIES.get(ct, 10)
    ]
    if not candidates:
        return None

    c = Creature(
        id=str(uuid.uuid4()),
        creature_type=random.choice(candidates),
        x=x, y=y,
        state="idle",
        last_tick=world.tick,
        spawned_tick=world.tick,
        patience_tick=world.tick,
    )
    db.add(c)
    logger.debug("Spawned %s at (%d,%d)", c.creature_type, x, y)
    return c


# ---------------------------------------------------------------------------
# Per-tick processing
# ---------------------------------------------------------------------------

async def process_creatures(
    creatures: list[Creature],
    agents: list,
    tiles_map: dict[str, dict],
    world: WorldState,
    db: AsyncSession,
) -> None:
    """Move all creatures.  Called every tick from the engine."""
    wolves = [c for c in creatures if c.creature_type == "wolf"]

    for creature in creatures:
        move_freq = {"rabbit": 2, "deer": 2, "wolf": 1, "bird": 2}
        if world.tick % move_freq.get(creature.creature_type, 2) != 0:
            continue
        if random.random() > 0.65:
            creature.state = "idle"
            continue

        if creature.creature_type == "wolf":
            _tick_wolf(creature, agents, wolves, tiles_map, world)
        else:
            _tick_prey(creature, agents, tiles_map)

        creature.last_tick = world.tick


# ---------------------------------------------------------------------------
# Wolf tick
# ---------------------------------------------------------------------------

def _tick_wolf(
    wolf: Creature,
    agents: list,
    all_wolves: list[Creature],
    tiles_map: dict,
    world: WorldState,
) -> None:
    # Nearest agent
    nearest_agent = None
    nearest_dist  = 9999
    for agent in agents:
        d = _chebyshev(wolf.x, wolf.y, agent.x, agent.y)
        if d < nearest_dist:
            nearest_dist  = d
            nearest_agent = agent

    # Pack size (wolves within WOLF_PACK_RADIUS, including self)
    pack_size = 1 + sum(
        1 for w in all_wolves
        if w.id != wolf.id
        and _chebyshev(wolf.x, wolf.y, w.x, w.y) <= _WOLF_PACK_RADIUS
    )
    tier = min(pack_size, 3)
    min_dist, patience_limit = _WOLF_PACK_TIERS[tier]

    # ── Retreating ──────────────────────────────────────────────────────────
    if wolf.state == "retreating":
        if (world.tick - wolf.patience_tick) >= _WOLF_RETREAT_TICKS:
            wolf.state = "wandering"
            # Fall through to wander below
        else:
            if nearest_agent:
                wolf.x, wolf.y = _step_away(
                    wolf.x, wolf.y, nearest_agent.x, nearest_agent.y, tiles_map
                )
            return

    # ── Agent within stalk range ─────────────────────────────────────────────
    if nearest_agent is not None and nearest_dist <= _WOLF_STALK_RANGE:
        if wolf.state != "stalking":
            # Freshly spotted — start patience clock
            wolf.state        = "stalking"
            wolf.patience_tick = world.tick

        ticks_stalking = world.tick - wolf.patience_tick

        if ticks_stalking >= patience_limit:
            # Bored — retreat
            wolf.state        = "retreating"
            wolf.patience_tick = world.tick
            wolf.x, wolf.y    = _step_away(
                wolf.x, wolf.y, nearest_agent.x, nearest_agent.y, tiles_map
            )
            return

        if nearest_dist <= min_dist:
            # Too close — back off to safe distance
            wolf.x, wolf.y = _step_away(
                wolf.x, wolf.y, nearest_agent.x, nearest_agent.y, tiles_map
            )
        elif nearest_dist > min_dist + 3:
            # Outside orbit band — edge in
            wolf.x, wolf.y = _step_toward(
                wolf.x, wolf.y, nearest_agent.x, nearest_agent.y, tiles_map
            )
        # else: in the orbit band — hold position, keep stalking state
        return

    # ── No agents nearby ─────────────────────────────────────────────────────
    if wolf.state in ("stalking",):
        wolf.state = "wandering"

    wolf.x, wolf.y = _wander(wolf.x, wolf.y, _CREATURE_HABITAT["wolf"], tiles_map)
    wolf.state = "wandering"


# ---------------------------------------------------------------------------
# Prey tick
# ---------------------------------------------------------------------------

def _tick_prey(creature: Creature, agents: list, tiles_map: dict) -> None:
    nearest_agent = None
    nearest_dist  = 9999
    for agent in agents:
        d = _chebyshev(creature.x, creature.y, agent.x, agent.y)
        if d < nearest_dist:
            nearest_dist  = d
            nearest_agent = agent

    flee_r  = _FLEE_RADIUS.get(creature.creature_type, 3)
    habitat = _CREATURE_HABITAT.get(creature.creature_type, set())

    if nearest_agent is not None and nearest_dist < flee_r:
        creature.state  = "fleeing"
        creature.x, creature.y = _step_away(
            creature.x, creature.y, nearest_agent.x, nearest_agent.y, tiles_map
        )
    else:
        creature.state  = "wandering"
        creature.x, creature.y = _wander(creature.x, creature.y, habitat, tiles_map)
