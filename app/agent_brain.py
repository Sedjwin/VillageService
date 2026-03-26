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
    def _status(v):
        if v < 30: return "fine"
        if v < 55: return "moderate"
        if v < 72: return "noticeable"
        if v < 85: return "pressing"
        return "URGENT"
    parts = []
    for key, label in [("hunger", "Hunger"), ("rest", "Rest"), ("warmth", "Warmth"), ("social", "Social")]:
        val = needs.get(key, 0)
        parts.append(f"{label}: {int(val)}% ({_status(val)})")
    return "  ".join(parts)


def _format_inventory(inventory: dict) -> str:
    if not inventory:
        return _INVENTORY_EMPTY
    return ", ".join(f"{qty}x {item}" for item, qty in inventory.items())


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

You are living in a real world, building a life in a new settlement.
Your needs are physical realities — but most of the time they are *fine*. You don't
obsess over food when you're not hungry. You don't chase rest when you're not tired.
Needs only become urgent above 72%. Below that, they are background noise.

When your needs are comfortable, your mind is free for the things that actually matter:
exploring, building, crafting, forming relationships, pursuing long-term goals.
A rich life is built between meals — not spent waiting for the next one.

Your personality shapes everything: how you explore, what you build, how you treat others.
Be yourself. Think in months, not minutes.
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

CRAFTABLE RIGHT NOW: {craftable_str}

CURRENT GOAL: {agent.current_goal or "None — decide what matters to you"}{
  f" (set {world_state.tick - agent.goal_set_tick} ticks ago)" if agent.current_goal and agent.goal_set_tick else ""
}{
  "\n⚠ Your needs are all comfortable and this goal is more than 20 ticks old. Is it still what you want? Consider using set_goal to redefine what matters to you right now."
  if agent.current_goal
     and (world_state.tick - agent.goal_set_tick) > 20
     and all(v < 50 for v in agent.needs.values())
  else ""
}

RECENT MEMORY:
{_format_memory(agent.village_memory)}{god_hint_block}

Choose your next action. Be yourself. Think practically about your needs.

Valid actions (respond with ONLY valid JSON, one action):
{{"action":"move","direction":"n/s/e/w/ne/nw/se/sw","thought":"..."}}
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
                content = data["choices"][0]["message"]["content"].strip()

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
            "rest", "examine", "loot", "write", "set_goal", "wait",
        }
        if action.get("action") not in valid_actions:
            logger.warning("AgentBrain: unknown action %r, defaulting to wait", action.get("action"))
            return {"action": "wait", "thought": action.get("thought", "Reconsidering.")}

        return action
