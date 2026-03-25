"""
Line-of-sight calculator using Bresenham's ray algorithm.

Blocking rules:
- dense_forest blocks LOS at distance > 2 from the blocking tile
- mountain always blocks tiles behind it
- hills block tiles that are further away (shadow effect)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


def calculate_los(
    agent_x: int,
    agent_y: int,
    agent_skills: dict,
    all_tiles_map: dict[tuple[int, int], object],
    time_of_day: int,
    weather: str,
) -> set[tuple[int, int]]:
    """
    Returns the set of (x, y) tiles visible to the agent.

    Parameters
    ----------
    agent_skills   : {"navigation": int, ...}
    all_tiles_map  : {(x, y): Tile ORM object or dict with .terrain attribute}
    time_of_day    : game_hour (0-23)
    weather        : current weather string
    """
    base_radius = 6
    nav_bonus = agent_skills.get("navigation", 0) // 20

    # Time of day penalty
    if 22 <= time_of_day or time_of_day < 6:
        time_penalty = 3
    elif time_of_day >= 20:
        time_penalty = 1
    else:
        time_penalty = 0

    weather_penalty = {"stormy": 4, "rainy": 2, "cloudy": 1}.get(weather, 0)

    # Standing on hills gives a bonus
    agent_tile = all_tiles_map.get((agent_x, agent_y))
    hill_bonus = 3 if _get_terrain(agent_tile) == "hills" else 0

    effective_radius = max(1, base_radius + nav_bonus + hill_bonus - time_penalty - weather_penalty)

    visible: set[tuple[int, int]] = set()

    # Always see your own tile
    visible.add((agent_x, agent_y))

    for dx in range(-effective_radius, effective_radius + 1):
        for dy in range(-effective_radius, effective_radius + 1):
            if dx * dx + dy * dy > effective_radius * effective_radius:
                continue
            tx, ty = agent_x + dx, agent_y + dy
            if tx == agent_x and ty == agent_y:
                continue
            if _has_los(agent_x, agent_y, tx, ty, all_tiles_map):
                visible.add((tx, ty))

    return visible


def _get_terrain(tile) -> str:
    if tile is None:
        return "unknown"
    if isinstance(tile, dict):
        return tile.get("terrain", "unknown")
    return getattr(tile, "terrain", "unknown")


def _has_los(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    tiles_map: dict[tuple[int, int], object],
) -> bool:
    """
    Bresenham's line check from (x0,y0) to (x1,y1).
    Returns False if any intermediate tile blocks LOS to (x1,y1).
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    cx, cy = x0, y0
    dist = 0

    while True:
        # Compute approximate Chebyshev distance from origin
        dist = max(abs(cx - x0), abs(cy - y0))

        if cx == x1 and cy == y1:
            return True  # Reached the target — visible

        terrain = _get_terrain(tiles_map.get((cx, cy)))

        # Blocking checks (skip the origin tile itself)
        if not (cx == x0 and cy == y0):
            if terrain == "mountain":
                # Mountain blocks all tiles behind it
                return False
            if terrain == "dense_forest" and dist > 2:
                # Dense forest blocks beyond distance 2 from observer
                return False
            if terrain == "hills":
                # Hills create shadow: block tiles more than 3 further in same direction
                remaining = max(abs(x1 - cx), abs(y1 - cy))
                if remaining > 3:
                    return False

        # Advance Bresenham
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            cx += sx
        if e2 < dx:
            err += dx
            cy += sy

        # Safety: if somehow we cycle (shouldn't happen with correct Bresenham)
        if abs(cx - x0) > abs(x1 - x0) + 1 or abs(cy - y0) > abs(y1 - y0) + 1:
            break

    return False
