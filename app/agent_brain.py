"""
Per-agent LLM decision loop.
Each agent is given its full sensory context and asked what it wants to do.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import httpx

logger = logging.getLogger(__name__)


_FALLBACK_ACTION = {"action": "wait", "thought": "Couldn't decide — taking a breath."}

_INVENTORY_EMPTY = "Nothing."
_MEMORY_EMPTY = "Nothing yet."


def _mood_description(mood: float) -> str:
    if mood >= 0.85:
        return "content and energised"
    if mood >= 0.65:
        return "doing alright"
    if mood >= 0.45:
        return "a bit worn"
    if mood >= 0.25:
        return "struggling"
    return "desperate"


def _needs_summary(needs: dict) -> str:
    """Needs are pressure values: 0 = fully satisfied, 100 = critical/dangerous."""
    THRESHOLDS = [
        (30,  "fine"),
        (55,  "mild"),
        (72,  "noticeable"),
        (85,  "pressing"),
        (95,  "URGENT"),
        (101, "CRITICAL"),
    ]
    CRITICAL_LABEL = {
        "hunger": "STARVING",
        "rest":   "EXHAUSTED",
        "warmth": "FREEZING",
        "social": "ISOLATED",
    }
    def _status(key, v):
        for threshold, label in THRESHOLDS:
            if v < threshold:
                return label
        return CRITICAL_LABEL.get(key, "CRITICAL")

    parts = []
    for key, label in [("hunger", "Hunger"), ("rest", "Fatigue"), ("warmth", "Cold"), ("social", "Lonely")]:
        val = needs.get(key, 0)
        parts.append(f"{label}: {int(val)}/100 ({_status(key, val)})")
    header = "Needs [0=satisfied · 100=critical]: "
    return header + "  ".join(parts)


def _needs_advice(needs: dict, inventory: dict) -> str:
    """
    Compute practical inventory-aware advice for the agent's current needs.
    This short-circuits common failure modes (starving while holding food, etc.).
    """
    lines = []
    hunger  = needs.get("hunger",  0)
    fatigue = needs.get("rest",    0)
    cold    = needs.get("warmth",  0)
    lonely  = needs.get("social",  0)

    has_cooked = inventory.get("cooked_food", 0) > 0
    has_raw    = inventory.get("raw_food",    0) > 0

    if hunger >= 72:
        if has_cooked:
            lines.append(f"⚠ Hunger {int(hunger)}/100 — you have cooked_food in your inventory. Eat it now with the eat action.")
        elif has_raw:
            lines.append(f"⚠ Hunger {int(hunger)}/100 — you have raw_food in your inventory. Eat it now (raw is fine when nothing else is available).")
        else:
            lines.append(f"⚠ Hunger {int(hunger)}/100 — no food in inventory. Find or gather food.")
    if fatigue >= 72:
        lines.append(f"⚠ Fatigue {int(fatigue)}/100 — you need rest. Use the rest action.")
    if cold >= 72:
        lines.append(f"⚠ Cold {int(cold)}/100 — find warmth (a campfire, shelter, or build one).")

    all_comfortable = hunger < 50 and fatigue < 50 and cold < 50 and lonely < 50
    if all_comfortable:
        lines.append("All needs comfortable — this is your time. Pursue your goal, explore, build, or talk.")

    return "\n".join(lines) if lines else ""


def _format_inventory(inventory: dict) -> str:
    if not inventory:
        return _INVENTORY_EMPTY
    from app.crafting import RECIPES
    parts = []
    for item, qty in inventory.items():
        recipe = RECIPES.get(item, {})
        if item == "note":
            desc = "blank writing medium — use 'write' action to inscribe content and leave on tile"
        else:
            desc = recipe.get("description", "")
        tag = f" — {desc}" if desc else ""
        label = item.replace("_", " ")
        parts.append(f"{qty}x {label}{tag}")
    return "\n  ".join(parts)


def _format_visible_tiles(visible_tiles: list[dict]) -> str:
    if not visible_tiles:
        return "  (nothing visible beyond your feet)"
    lines = []
    for t in visible_tiles[:12]:  # limit context size
        x, y = t.get("x", 0), t.get("y", 0)
        terrain = t.get("terrain", "?")
        features = t.get("features", [])
        resources = [r.get("type", "") for r in t.get("resource_nodes", []) if r.get("qty", 0) > 0]
        buildings = [b.get("type", "") for b in t.get("buildings", [])]
        detail_parts = []
        if features:
            feature_str = ", ".join(features)
            # Highlight lootable ruins distinctly
            if "ruins" in features and "ruins_looted" not in features:
                feature_str += " [LOOTABLE]"
            detail_parts.append("features: " + feature_str)
        if resources:
            detail_parts.append("resources: " + ", ".join(resources))
        if buildings:
            detail_parts.append("buildings: " + ", ".join(buildings))
        detail = " | ".join(detail_parts) if detail_parts else terrain
        lines.append(f"  ({x},{y}) {terrain} — {detail}")
    return "\n".join(lines)


_COORD_RE = re.compile(r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)')


def _extract_known_destinations(goal: str | None, memory: list[str], agent_x: int, agent_y: int) -> str:
    """
    Scan the agent's goal and recent memory for explicit coordinates and return
    them as a navigable destinations list.  Deduplicates and excludes the
    agent's current tile.
    """
    seen: dict[tuple[int, int], str] = {}

    def _scan(text: str, source: str):
        for m in _COORD_RE.finditer(text):
            x, y = int(m.group(1)), int(m.group(2))
            if (x, y) == (agent_x, agent_y):
                continue
            if (x, y) not in seen:
                seen[(x, y)] = source

    if goal:
        _scan(goal, "current goal")
    for entry in reversed(memory[-10:]):
        _scan(entry, "memory")

    if not seen:
        return ""
    lines = []
    for (x, y), src in seen.items():
        dist = abs(x - agent_x) + abs(y - agent_y)
        lines.append(f"  ({x},{y}) — {dist} tiles away [from {src}]")
    return "\n".join(lines)


def _format_landmarks(visible_tiles: list[dict], agent_x: int, agent_y: int, nearby_agents: list[dict] | None = None) -> str:
    """Extract notable navigable landmarks from visible tiles, with occupancy counts."""
    NOTABLE_FEATURES = {"campfire", "well", "notice_board", "ruins", "workshop", "forge", "dock"}
    NOTABLE_BUILDINGS = {"campfire", "well", "notice_board", "house", "basic_shelter", "workshop", "forge", "road_tile"}

    # Count agents per tile
    occupancy: dict[tuple, int] = {}
    for a in (nearby_agents or []):
        k = (a.get("x", 0), a.get("y", 0))
        occupancy[k] = occupancy.get(k, 0) + 1

    landmarks = []
    for t in visible_tiles:
        x, y = t.get("x", 0), t.get("y", 0)
        if x == agent_x and y == agent_y:
            continue
        features = t.get("features", [])
        buildings = [b.get("type", "") for b in t.get("buildings", [])]
        notable = [f for f in features if f in NOTABLE_FEATURES] + [b for b in buildings if b in NOTABLE_BUILDINGS]
        if notable:
            dist = abs(x - agent_x) + abs(y - agent_y)
            occ = occupancy.get((x, y), 0)
            occ_str = f" — {occ} agent{'s' if occ != 1 else ''} already here" if occ else ""
            landmarks.append(f"  ({x},{y}) — {', '.join(notable)} [{dist} tiles]{occ_str}")
    if not landmarks:
        return "  None visible from here."
    return "\n".join(landmarks[:8])


def _format_nearby_agents(nearby: list[dict]) -> str:
    if not nearby:
        return "  No one nearby."
    lines = []
    for a in nearby[:6]:
        name = a.get("name", "?")
        state = a.get("state", "idle")
        state_note = " [COLLAPSED — needs help!]" if state == "collapsed" else ""
        lines.append(f"  {name} at ({a.get('x',0)},{a.get('y',0)}) — {state}{state_note}")
    return "\n".join(lines)


def _format_memory(memory: list[str]) -> str:
    if not memory:
        return _MEMORY_EMPTY
    return "\n".join(f"  - {m}" for m in memory[-10:])


class AgentBrain:

    async def decide_action(
        self,
        agent,
        visible_tiles: list[dict],
        world_state,
        nearby_agents: list[dict],
        model: str,
        aigateway_url: str,
        token: str,
    ) -> dict:
        """
        Ask the agent's LLM what it wants to do.
        Returns action dict: {action, thought, ...params}
        """
        current_tile = next(
            (t for t in visible_tiles if t.get("x") == agent.x and t.get("y") == agent.y),
            None,
        )
        landmarks_str = _format_landmarks(visible_tiles, agent.x, agent.y, nearby_agents)
        known_destinations_str = _extract_known_destinations(
            agent.current_goal, agent.village_memory, agent.x, agent.y
        )
        tile_narrative = current_tile.get("narrative", "unfamiliar ground") if current_tile else "unfamiliar ground"

        god_hint_block = ""
        if agent.god_hint:
            god_hint_block = f"\n[IMPORTANT MESSAGE FROM ABOVE]: {agent.god_hint}\n"

        # Summarise available recipes for context
        from app.crafting import RECIPES
        tile_features = current_tile.get("features", []) if current_tile else []
        tile_buildings = [b.get("type", "") for b in (current_tile.get("buildings", []) if current_tile else [])]
        all_nearby_features = set(tile_features + tile_buildings)

        craftable_now = []
        craftable_nearby = []  # have materials, but need facility
        for name, recipe in RECIPES.items():
            reqs = recipe.get("requires", {})
            if not reqs:
                continue
            has_materials = all(agent.inventory.get(k, 0) >= v for k, v in reqs.items() if v > 0)
            tool = recipe.get("tool_req")
            has_tool = not tool or agent.inventory.get(tool, 0) >= 1
            near_req = recipe.get("near_req")
            has_near = not near_req or near_req in all_nearby_features

            if has_materials and has_tool:
                if has_near:
                    craftable_now.append(name)
                else:
                    craftable_nearby.append(f"{name}(needs {near_req})")

        craftable_str = ", ".join(craftable_now[:8]) if craftable_now else "none right now"
        if craftable_nearby:
            craftable_str += f" | with facility: {', '.join(craftable_nearby[:4])}"

        system_prompt = f"""\
You are {agent.name}. {agent.personality_summary}

You live in a growing settlement. Each tick you must choose ONE concrete action.
Follow this priority order every single turn:

PRIORITY 1 — IMMEDIATE SURVIVAL (needs > 72)
  • Hungry? Check inventory first. Carrying food? → eat it right now.
  • No food in inventory? → gather or find food before anything else.
  • Exhausted? → rest.
  • Freezing? → move near a campfire or build one.
  Needs are 0–100 pressure values. 0 = fine. 100 = critical. Ignore anything below 50.

PRIORITY 2 — ADVANCE YOUR PLAN
  • Look at your current plan/goal. What is the very next concrete step?
  • If the next step needs an item you already carry → do it now (craft, build, eat, write).
  • If the next step needs an item you don't have → go gather or forage it.
  • If the next step needs a facility (campfire, workshop) → navigate there.
  • If your plan is complete or stale → update it with set_goal.

PRIORITY 3 — CONTRIBUTE TO THE SETTLEMENT
  • If you have no plan: look at what the settlement most needs (food, shelter, tools, paths).
  • Talk to nearby people, share resources, coordinate.
  • Explore new tiles to expand the map.

Always check your inventory before travelling anywhere. You may already have what you need.
Never wait or stand idle when your needs are fine and you have a plan.
Your personality shapes HOW you pursue these priorities — not WHETHER you pursue them.
"""

        user_prompt = f"""\
CURRENT STATE:
  Location: ({agent.x}, {agent.y}) — {tile_narrative}
  Time: Day {world_state.game_day}, {world_state.game_hour:02d}:00 | {world_state.season} | {world_state.weather}
  Mood: {_mood_description(agent.mood)}
  {_needs_summary(agent.needs)}

WHAT YOU CAN SEE:
{_format_visible_tiles(visible_tiles)}

NEARBY PEOPLE:
{_format_nearby_agents(nearby_agents)}

YOUR INVENTORY:
  {_format_inventory(agent.inventory)}
{f"{chr(10)}{_needs_advice(agent.needs, agent.inventory)}{chr(10)}" if _needs_advice(agent.needs, agent.inventory) else ""}
CRAFTABLE RIGHT NOW: {craftable_str}

NEARBY LANDMARKS (visible):
{landmarks_str if landmarks_str else "  None visible from here."}
{f"KNOWN DESTINATIONS (from goal & memory — use navigate to reach these):{chr(10)}{known_destinations_str}" if known_destinations_str else ""}
CURRENT GOAL: {agent.current_goal or "None — decide what matters to you"}{
  f" (set {world_state.tick - agent.goal_set_tick} ticks ago)" if agent.current_goal and agent.goal_set_tick else ""
}{
  "\n⚠ Your needs are all comfortable and this goal is more than 20 ticks old. Is it still what you want? Consider using set_goal to redefine what matters to you right now."
  if agent.current_goal
     and (world_state.tick - agent.goal_set_tick) > 20
     and all(v < 50 for v in agent.needs.values())
  else ""
}{
  f"\nGOAL RESOURCES:\n  {agent.goal_resource_brief.strip()}"
  if agent.current_goal and getattr(agent, 'goal_resource_brief', None)
  else ""
}

RECENT MEMORY:
{_format_memory(agent.village_memory)}{god_hint_block}

Choose your next action. Be yourself. Think practically about your needs.

Valid actions (respond with ONLY valid JSON, one action):
{{"action":"navigate","x":0,"y":0,"thought":"..."}}                          ← pathfind to coordinates (multi-tick journey, handles terrain automatically)
{{"action":"navigate","destination":"campfire","thought":"..."}}              ← pathfind to a named landmark (campfire/well/notice_board/camp/home)
{{"action":"move","direction":"n/s/e/w/ne/nw/se/sw","thought":"..."}}       ← single step in a direction (exploring adjacent tiles)
{{"action":"gather","resource":"type","thought":"..."}}
{{"action":"eat","food":"raw_food or cooked_food","thought":"..."}}
{{"action":"craft","recipe":"name","thought":"..."}}
{{"action":"build","structure":"type","thought":"..."}}
{{"action":"speak","target":"agent_name_or_broadcast","message":"...","thought":"..."}}
{{"action":"rest","hours":1,"thought":"..."}}
{{"action":"examine","target":"description","thought":"..."}}
{{"action":"loot","thought":"..."}}  ← only valid if current tile has "ruins" feature
{{"action":"write","content":"...","medium":"note","thought":"..."}}
{{"action":"set_goal","goal":"...","thought":"..."}}
{{"action":"wait","thought":"..."}}

Use 'navigate' to travel anywhere — visible or not. If your goal names a coordinate, navigate there directly. Use 'move' only for casual single-step exploration of adjacent tiles.
If a landmark shows multiple agents already there, consider navigating to a different one or building your own nearby rather than crowding the same spot.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            url = f"{aigateway_url.rstrip('/')}/v1/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "max_tokens": 400,
                "temperature": 0.9,
            }
            async with httpx.AsyncClient(timeout=25.0) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                )
                resp.raise_for_status()
                data = resp.json()
                content = (data["choices"][0]["message"].get("content") or "").strip()

            return self._parse_action(content)

        except httpx.HTTPStatusError as exc:
            logger.warning("AgentBrain LLM call failed for %s: HTTP %d", agent.name, exc.response.status_code)
        except httpx.RequestError as exc:
            logger.warning("AgentBrain LLM unreachable for %s: %s", agent.name, exc)
        except Exception as exc:
            logger.error("AgentBrain unexpected error for %s: %s", agent.name, exc)

        return _FALLBACK_ACTION

    def _parse_action(self, text: str) -> dict:
        """Extract and validate JSON action from LLM response."""
        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())

        # Find first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            logger.warning("AgentBrain: no JSON found in response: %r", text[:100])
            return dict(_FALLBACK_ACTION)

        try:
            action = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("AgentBrain: JSON parse failed: %r", text[:100])
            return dict(_FALLBACK_ACTION)

        # Validate action field
        valid_actions = {
            "move", "gather", "eat", "craft", "build", "speak",
            "rest", "examine", "loot", "write", "set_goal", "wait", "navigate",
        }
        if action.get("action") not in valid_actions:
            logger.warning("AgentBrain: unknown action %r, defaulting to wait", action.get("action"))
            return {"action": "wait", "thought": action.get("thought", "Reconsidering.")}

        return action
