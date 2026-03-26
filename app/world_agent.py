"""
World Agent — the unseen intelligence that gives form to uncharted land.
Handles tile generation, starter crates, world events, and action narration.
"""
from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_WORLD_WEAVER_SYSTEM = """\
You are the World Weaver — the unseen intelligence that gives form to uncharted land.

You describe what exists at the edge of the known with specificity and imagination.
Your world is grounded, interesting, quietly strange. It has history without explanation.
Things have been here before the agents arrived. The land has texture and character.

You remember what you've placed and stay consistent.
When you describe a forest, make it feel like a real forest — wet bark, birdsong, gaps in canopy.
When you fill a crate, fill it with things that feel like they belonged to someone specific.
When you write narrative, write for someone who will read it on a small map tile.

Constraints:
- Brief but evocative. 1-3 sentences of narrative.
- Stay grounded — no magic, no monsters, no fantasy tropes unless asked.
- Respond ONLY in valid JSON. No markdown fences. No explanation.
"""

_NARRATION_SYSTEM = """\
You write brief, vivid log entries for a living village simulation.
Each entry is 1-2 sentences. Past tense. Specific. No purple prose.
Capture what happened and why it matters to the agent.
Respond with only the log entry text — no JSON, no prefix.
"""


class WorldAgent:
    def __init__(self, aigateway_url: str, token: str):
        self.aigateway_url = aigateway_url.rstrip("/")
        self.token = token

    async def _call_llm(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 400,
        temperature: float = 0.85,
    ) -> str:
        """Make a chat completion call and return the content string."""
        url = f"{self.aigateway_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.token}"},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _extract_json(self, text: str) -> dict | list:
        """Strip markdown fences and parse JSON from LLM response."""
        text = text.strip()
        # Remove ```json ... ``` or ``` ... ```
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    async def generate_tile(
        self,
        x: int,
        y: int,
        adjacent_tiles: list[dict],
        world_bible: str,
        model: str,
    ) -> dict:
        """
        Generate data for a newly discovered tile at (x, y).

        Returns dict with keys: terrain, elevation, features, resources, narrative
        """
        adj_summary = ""
        if adjacent_tiles:
            parts = []
            for t in adjacent_tiles[:4]:
                parts.append(
                    f"  ({t['x']},{t['y']}): {t['terrain']}, elev={t['elevation']} — {t['narrative'][:80]}"
                )
            adj_summary = "Adjacent known tiles:\n" + "\n".join(parts)
        else:
            adj_summary = "This is a frontier tile — no adjacent known tiles yet."

        bible_ctx = f"\nWorld context:\n{world_bible[:400]}" if world_bible else ""

        prompt = f"""\
Generate the tile at grid position ({x}, {y}).
{adj_summary}{bible_ctx}

Consider what would naturally exist here given the surroundings.
Be consistent with adjacent terrain. Allow gradual transitions.
Place resources that make sense for the terrain.
Occasionally (roughly 1 in 12 tiles) include "ruins" in features — crumbled structures,
old foundations, collapsed walls. They hint at prior habitation and can be looted.

Respond in this exact JSON format:
{{
  "terrain": "<grass|light_forest|dense_forest|hills|mountain|water|beach|cave|road>",
  "elevation": <0-10>,
  "features": ["<feature1>", "<feature2>"],
  "resources": [
    {{"type": "<resource>", "qty": <int>, "max_qty": <int>}}
  ],
  "narrative": "<1-2 evocative sentences describing this tile>"
}}"""

        messages = [
            {"role": "system", "content": _WORLD_WEAVER_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = await self._call_llm(messages, model, max_tokens=350)
            data = self._extract_json(raw)
            # Validate and normalise
            return {
                "terrain": data.get("terrain", "grass"),
                "elevation": int(data.get("elevation", 2)),
                "features": data.get("features", []),
                "resources": [
                    {
                        "type": r.get("type", "raw_food"),
                        "qty": int(r.get("qty", 1)),
                        "max_qty": int(r.get("max_qty", 3)),
                        "regen_at_tick": 0,
                    }
                    for r in data.get("resources", [])
                ],
                "narrative": data.get("narrative", ""),
            }
        except Exception as exc:
            logger.error("WorldAgent.generate_tile failed at (%d,%d): %s", x, y, exc)
            # Fallback: procedural tile
            return _procedural_tile(x, y)

    async def generate_starter_crate(
        self,
        agent_name: str,
        personality: str,
        world_bible: str,
        model: str,
    ) -> dict:
        """
        Decide what goes into this agent's starter crate.
        Returns {"items": {item_type: qty, ...}, "narrative": str}
        """
        bible_ctx = world_bible[:300] if world_bible else "A wild land at the edge of settlement."

        prompt = f"""\
An agent named {agent_name} is about to arrive in the village.
Their character: {personality[:200]}

World context: {bible_ctx}

Design a starter crate tailored to this specific person — not generic survival gear,
but something that feels like it was packed for them. Include 4-6 items.
Available item types: tinder, rope, knife, axe, raw_food, cooked_food, long_stick,
sharp_rock, vine, torch, fishing_rod, paper, charcoal, note, flute.

Respond in this exact JSON format:
{{
  "items": {{"item_type": qty, ...}},
  "narrative": "<1-2 sentences describing the crate and its contents>"
}}"""

        messages = [
            {"role": "system", "content": _WORLD_WEAVER_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = await self._call_llm(messages, model, max_tokens=200)
            data = self._extract_json(raw)
            items = {str(k): int(v) for k, v in data.get("items", {}).items()}
            return {
                "items": items,
                "narrative": data.get(
                    "narrative",
                    "A weathered crate, contents still dry. Someone packed this with care.",
                ),
            }
        except Exception as exc:
            logger.error("WorldAgent.generate_starter_crate failed for %s: %s", agent_name, exc)
            return _default_crate()

    async def generate_world_event(
        self,
        world_state,
        active_agents: list,
        model: str,
    ) -> dict | None:
        """
        Occasionally invent a world event. Returns None if no event is warranted.
        Only called when the dice say to — so assume an event IS happening.
        Returns: {type, description, x, y, effects: dict}
        """
        agent_names = [a.name for a in active_agents[:6]]
        agents_str = ", ".join(agent_names) if agent_names else "no one yet"

        prompt = f"""\
Something unusual is happening in the village world.
Current state: Day {world_state.game_day}, hour {world_state.game_hour}:00,
{world_state.season}, {world_state.weather}.
Active agents: {agents_str}.

Invent a world event that would be interesting for these agents to encounter or discover.
It should be grounded — no magic. Could be: unusual weather, animal behaviour,
a discovery, structural collapse, fire, flood edge, strange sound, abandoned object.
Keep it short and consequential.

Respond in JSON:
{{
  "type": "<event_type_slug>",
  "description": "<1-2 sentence narrative description>",
  "x": <int or null>,
  "y": <int or null>,
  "effects": {{}}
}}"""

        messages = [
            {"role": "system", "content": _WORLD_WEAVER_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = await self._call_llm(messages, model, max_tokens=200)
            data = self._extract_json(raw)
            return data
        except Exception as exc:
            logger.error("WorldAgent.generate_world_event failed: %s", exc)
            return None

    async def generate_ruins_loot(
        self,
        agent_name: str,
        personality: str,
        tile_narrative: str,
        world_bible: str,
        model: str,
    ) -> dict:
        """
        Generate loot for an agent searching ruins on a tile.
        Returns {"items": {item_type: qty, ...}, "narrative": str}
        """
        bible_ctx = world_bible[:200] if world_bible else "A land with a forgotten past."
        tile_ctx = tile_narrative[:150] if tile_narrative else "Crumbled stonework, half-buried."

        prompt = f"""\
{agent_name} is searching through ruins.
The ruins: {tile_ctx}
World context: {bible_ctx}
Agent character: {personality[:150]}

What do they find? Something that was left behind — tools, scraps, written fragments,
partial useful objects. Keep it grounded. 2-4 items. Modest quantities.

Available item types: sharp_rock, long_stick, rope, vine, tinder, raw_food,
paper, charcoal, note, knife, torch, axe, fishing_rod.

Respond in this exact JSON format:
{{
  "items": {{"item_type": qty}},
  "narrative": "<1-2 sentences describing what they found and the mood of the discovery>"
}}"""

        messages = [
            {"role": "system", "content": _WORLD_WEAVER_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = await self._call_llm(messages, model, max_tokens=200)
            data = self._extract_json(raw)
            items = {str(k): max(1, int(v)) for k, v in data.get("items", {}).items()}
            return {
                "items": items,
                "narrative": data.get(
                    "narrative",
                    "Digging through the rubble turned up a few useful things.",
                ),
            }
        except Exception as exc:
            logger.error("WorldAgent.generate_ruins_loot failed: %s", exc)
            # Procedural fallback
            return {
                "items": {
                    random.choice(["sharp_rock", "long_stick", "rope", "tinder"]): random.randint(1, 2),
                    random.choice(["vine", "paper", "charcoal"]): 1,
                },
                "narrative": "Old stone and splinters — not much, but enough.",
            }

    async def narrate_action(
        self,
        agent_name: str,
        action: str,
        result: str,
        context: str,
        model: str,
    ) -> str:
        """
        Generate a brief, vivid log entry for an agent's action.
        Returns a plain text string (1-2 sentences).
        """
        prompt = f"""\
Agent: {agent_name}
Action: {action}
Result: {result}
Context: {context[:200]}

Write a 1-2 sentence log entry in past tense. Be specific and vivid. No prefix."""

        messages = [
            {"role": "system", "content": _NARRATION_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            return await self._call_llm(messages, model, max_tokens=80, temperature=0.7)
        except Exception as exc:
            logger.error("WorldAgent.narrate_action failed: %s", exc)
            return f"{agent_name} {action}. {result}."

    async def grow_world_bible(
        self,
        current_bible: str,
        recent_events: list[str],
        model: str,
    ) -> str:
        """
        Periodically update the world bible with significant events.
        Returns updated bible text (max 800 chars).
        """
        if not recent_events:
            return current_bible

        events_str = "\n".join(f"- {e}" for e in recent_events[-10:])
        prompt = f"""\
Current world chronicle (may be empty for a new world):
{current_bible[:500] if current_bible else "(none yet — this world is brand new)"}

Recent significant events:
{events_str}

Update the world chronicle to include what matters from these events.
Be concise — keep the total under 600 characters.
Focus on facts, discoveries, and changes that future agents should know about.
Respond with only the updated chronicle text."""

        messages = [
            {"role": "system", "content": _WORLD_WEAVER_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            result = await self._call_llm(messages, model, max_tokens=200, temperature=0.7)
            return result.strip()[:800]
        except Exception as exc:
            logger.error("WorldAgent.grow_world_bible failed: %s", exc)
            return current_bible


# ---------------------------------------------------------------------------
# Fallback procedural generation (no LLM)
# ---------------------------------------------------------------------------

def _procedural_tile(x: int, y: int) -> dict:
    """Generate a tile procedurally when the LLM is unavailable."""
    # Simple noise-like selection based on position
    r = ((x * 73856093) ^ (y * 19349663)) % 100

    if r < 30:
        terrain = "grass"
        resources = [{"type": "raw_food", "qty": 2, "max_qty": 4, "regen_at_tick": 0}]
        narrative = "Open grassland, windswept."
    elif r < 50:
        terrain = "light_forest"
        resources = [
            {"type": "long_stick", "qty": 2, "max_qty": 4, "regen_at_tick": 0},
            {"type": "vine", "qty": 2, "max_qty": 3, "regen_at_tick": 0},
        ]
        narrative = "Scattered trees, light filtering through."
    elif r < 65:
        terrain = "dense_forest"
        resources = [{"type": "vine", "qty": 3, "max_qty": 5, "regen_at_tick": 0}]
        narrative = "Thick canopy. The light barely reaches the floor."
    elif r < 75:
        terrain = "hills"
        resources = [{"type": "sharp_rock", "qty": 3, "max_qty": 5, "regen_at_tick": 0}]
        narrative = "Rolling hills, exposed rock at the crests."
    elif r < 82:
        terrain = "beach"
        resources = [{"type": "sharp_rock", "qty": 1, "max_qty": 2, "regen_at_tick": 0}]
        narrative = "Pale sand and smooth stones."
    elif r < 88:
        terrain = "water"
        resources = [{"type": "raw_food", "qty": 2, "max_qty": 4, "regen_at_tick": 0}]
        narrative = "Clear water, shallow near the edges."
    elif r < 93:
        terrain = "mountain"
        resources = [{"type": "sharp_rock", "qty": 4, "max_qty": 6, "regen_at_tick": 0}]
        narrative = "Sheer rock face."
    else:
        terrain = "cave"
        resources = [{"type": "sharp_rock", "qty": 2, "max_qty": 3, "regen_at_tick": 0}]
        narrative = "A low cave mouth. Cool air drifts out."

    elev_r = ((x * 1234567) ^ (y * 7654321)) % 11
    return {
        "terrain": terrain,
        "elevation": elev_r,
        "features": [],
        "resources": resources,
        "narrative": narrative,
    }


def _default_crate() -> dict:
    return {
        "items": {"tinder": 3, "rope": 1, "knife": 1, "raw_food": 2},
        "narrative": "A weathered crate, contents still dry. Someone packed this with care.",
    }
