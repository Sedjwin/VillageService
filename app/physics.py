"""
Game physics: movement, needs, gathering, crafting, pathfinding.
"""
from __future__ import annotations

import heapq
import logging
import random
from typing import TYPE_CHECKING

from app.crafting import RECIPES, TERRAIN_MOVEMENT_COST

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def get_terrain_movement_cost(terrain: str) -> float | None:
    """Returns movement cost (ticks) for terrain, or None if impassable."""
    return TERRAIN_MOVEMENT_COST.get(terrain, 1.0)


# ---------------------------------------------------------------------------
# Mood
# ---------------------------------------------------------------------------

def compute_mood(needs: dict) -> float:
    """
    Derive mood from needs.
    Each need contributes negatively when high (needs are "pressure" values).
    Returns a 0.0–1.0 float.
    """
    hunger = needs.get("hunger", 50)
    rest = needs.get("rest", 50)
    warmth = needs.get("warmth", 50)
    social = needs.get("social", 50)

    # Weight: hunger > rest > warmth > social
    pressure = (hunger * 0.35) + (rest * 0.30) + (warmth * 0.20) + (social * 0.15)
    mood = max(0.0, min(1.0, 1.0 - (pressure / 100.0)))
    return round(mood, 3)


# ---------------------------------------------------------------------------
# Needs decay
# ---------------------------------------------------------------------------

_SEASON_WARMTH_MULTIPLIER = {
    "spring": 1.0,
    "summer": 0.4,
    "autumn": 1.2,
    "winter": 2.5,
}


def decay_needs(
    needs: dict,
    season: str,
    has_shelter: bool,
    has_fire_nearby: bool,
    sim_config: dict | None = None,
) -> dict:
    """
    Apply per-tick needs decay.
    Needs are pressure values (0 = fine, 100 = critical).
    Returns updated needs dict.
    """
    cfg = sim_config or {}
    n = dict(needs)

    n["hunger"] = min(100, n.get("hunger", 0) + cfg.get("hunger_rate", 0.15))
    n["rest"]   = min(100, n.get("rest",   0) + cfg.get("rest_rate",   0.10))

    warmth_mult = _SEASON_WARMTH_MULTIPLIER.get(season, 1.0)
    warmth_rate = cfg.get("warmth_rate", 0.09) * warmth_mult
    if has_shelter:
        warmth_rate *= 0.3
    if has_fire_nearby:
        warmth_rate *= 0.2
    n["warmth"] = min(100, n.get("warmth", 0) + warmth_rate)

    n["social"] = min(100, n.get("social", 0) + cfg.get("social_rate", 0.06))

    return {k: round(v, 2) for k, v in n.items()}


def satisfy_need(needs: dict, need_name: str, amount: float) -> dict:
    """Reduce a need value (clamped at 0)."""
    n = dict(needs)
    n[need_name] = max(0.0, n.get(need_name, 0) - amount)
    return n


# ---------------------------------------------------------------------------
# Gathering
# ---------------------------------------------------------------------------

def can_gather(tile, resource_type: str) -> bool:
    """Check whether the resource_type is available on this tile."""
    for node in tile.resource_nodes:
        if node.get("type") == resource_type and node.get("qty", 0) > 0:
            return True
    return False


def do_gather(
    tile,
    resource_type: str,
    skills: dict,
) -> tuple[str, int, list]:
    """
    Gather resource_type from tile.
    Returns (item_type, qty_gathered, updated_resource_nodes).
    Skill level increases yield slightly.
    """
    nodes = tile.resource_nodes
    updated = []
    gathered = 0
    item_type = resource_type

    # Skill bonus: gathering or foraging adds up to +2 at skill 100
    relevant_skill = max(skills.get("gathering", 0), skills.get("foraging", 0))
    skill_bonus = int(relevant_skill / 50)  # 0, 1, or 2

    for node in nodes:
        if node.get("type") == resource_type and gathered == 0:
            available = node.get("qty", 0)
            if available > 0:
                take = min(available, 1 + skill_bonus)
                gathered = take
                new_qty = available - take
                updated_node = dict(node)
                updated_node["qty"] = new_qty
                updated.append(updated_node)
            else:
                updated.append(node)
        else:
            updated.append(node)

    return item_type, gathered, updated


# ---------------------------------------------------------------------------
# Skill gain
# ---------------------------------------------------------------------------

_SKILL_GAIN_CHANCE = 0.2  # 20% chance per use


def apply_skill_gain(skills: dict, skill_name: str) -> dict:
    """
    Small probabilistic skill increment on each use.
    Skills cap at 100.
    """
    s = dict(skills)
    current = s.get(skill_name, 0)
    if current >= 100:
        return s
    if random.random() < _SKILL_GAIN_CHANCE:
        s[skill_name] = min(100, current + 1)
    return s


# ---------------------------------------------------------------------------
# Crafting validation & execution
# ---------------------------------------------------------------------------

def validate_craft(
    recipe_name: str,
    inventory: dict,
    tile_features: list,
    skills: dict,
) -> tuple[bool, str]:
    """
    Returns (can_craft, reason_if_not).
    """
    recipe = RECIPES.get(recipe_name)
    if recipe is None:
        return False, f"Unknown recipe: {recipe_name}"

    # Check input materials
    for item, qty in recipe.get("requires", {}).items():
        if qty == 0:
            continue  # tool_req handled separately
        if inventory.get(item, 0) < qty:
            return False, f"Need {qty}x {item}, have {inventory.get(item, 0)}"

    # Check tool requirement
    tool = recipe.get("tool_req")
    if tool and inventory.get(tool, 0) < 1:
        return False, f"Requires {tool} in inventory"

    # Check proximity requirement
    near = recipe.get("near_req")
    if near and near not in tile_features:
        return False, f"Must be near a {near}"

    # Check skill requirement
    for skill, level in recipe.get("skill_req", {}).items():
        if skills.get(skill, 0) < level:
            return False, f"Requires {skill} >= {level} (have {skills.get(skill, 0)})"

    return True, ""


def do_craft(recipe_name: str, inventory: dict) -> dict:
    """
    Consume ingredients and add output item to inventory.
    Returns updated inventory.
    """
    recipe = RECIPES.get(recipe_name)
    if recipe is None:
        return inventory

    inv = dict(inventory)

    # Consume inputs
    for item, qty in recipe.get("requires", {}).items():
        if qty == 0:
            continue
        inv[item] = max(0, inv.get(item, 0) - qty)
        if inv[item] == 0:
            del inv[item]

    # Add output
    if not recipe.get("is_building", False):
        qty_out = recipe.get("qty_out", 1)
        inv[recipe_name] = inv.get(recipe_name, 0) + qty_out

    return inv


# ---------------------------------------------------------------------------
# Pathfinding (A*)
# ---------------------------------------------------------------------------

def find_path(
    from_x: int,
    from_y: int,
    to_x: int,
    to_y: int,
    tiles_map: dict[tuple[int, int], object],
    max_steps: int = 200,
) -> list[tuple[int, int]]:
    """
    A* pathfinding over the tile grid.
    Returns list of (x, y) steps from start (exclusive) to destination (inclusive).
    Returns [] if no path exists or destination is impassable.
    """
    def heuristic(ax: int, ay: int) -> float:
        return abs(ax - to_x) + abs(ay - to_y)

    def terrain_cost(x: int, y: int) -> float | None:
        tile = tiles_map.get((x, y))
        if tile is None:
            return 1.5  # unknown tile — passable but costly
        terrain = tile.terrain if hasattr(tile, "terrain") else tile.get("terrain", "grass")
        return TERRAIN_MOVEMENT_COST.get(terrain, 1.0)

    start = (from_x, from_y)
    goal = (to_x, to_y)

    # Check destination passability
    dest_cost = terrain_cost(to_x, to_y)
    if dest_cost is None:
        return []

    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    DIRECTIONS = [
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            # Reconstruct path (exclude start, include goal)
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = current
            while node is not None and node != start:
                path.append(node)
                node = came_from.get(node)
            path.reverse()
            return path

        if len(came_from) > max_steps:
            break

        cx, cy = current
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            cost = terrain_cost(nx, ny)
            if cost is None:
                continue  # impassable

            # Diagonal movement costs slightly more
            step_cost = cost * (1.414 if dx != 0 and dy != 0 else 1.0)
            tentative_g = g_score[current] + step_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(nx, ny)
                heapq.heappush(open_heap, (f, neighbor))
                came_from[neighbor] = current

    return []  # No path found
