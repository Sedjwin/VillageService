"""
Conversation and social interaction engine.
Two agents talk — each is fully themselves.
"""
from __future__ import annotations

import json
import logging
import re
import uuid

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, VillageAgent, WorldState

logger = logging.getLogger(__name__)


_CONVERSATION_SYSTEM = """\
You are playing yourself in a conversation. Respond as your character would —
naturally, briefly, in-voice. 1-3 sentences. No stage directions. No asterisks.
This is a real exchange between two people who exist in the same world.
"""


async def _get_agent_response(
    speaking_agent: VillageAgent,
    other_agent: VillageAgent,
    conversation_history: list[dict],
    trigger: str,
    world_state: WorldState,
    model: str,
    aigateway_url: str,
    token: str,
) -> str:
    """Ask one agent to respond in the conversation."""
    # Build history string
    history_lines = []
    for msg in conversation_history:
        history_lines.append(f"{msg['name']}: {msg['text']}")
    history_str = "\n".join(history_lines) if history_lines else "(conversation just started)"

    trigger_context = {
        "encounter": f"You've just run into {other_agent.name} at ({speaking_agent.x}, {speaking_agent.y}).",
        "trade_offer": f"{other_agent.name} is interested in trading with you.",
        "seek_help": f"{other_agent.name} seems to need something.",
    }.get(trigger, f"You're talking with {other_agent.name}.")

    system_prompt = f"""\
You are {speaking_agent.name}. {speaking_agent.personality_summary}

{_CONVERSATION_SYSTEM}"""

    user_prompt = f"""\
{trigger_context}
Day {world_state.game_day}, {world_state.game_hour:02d}:00. {world_state.season}, {world_state.weather}.

{other_agent.name} has the following inventory (roughly): {_summarise_inventory(other_agent.inventory)}
Your inventory: {_summarise_inventory(speaking_agent.inventory)}
Your mood: {_mood_word(speaking_agent.mood)}

Conversation so far:
{history_str}

What do you say next? (1-3 sentences, in character, no prefix)"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    url = f"{aigateway_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": 120,
        "temperature": 0.95,
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


def _summarise_inventory(inventory: dict) -> str:
    if not inventory:
        return "nothing"
    return ", ".join(f"{qty}x {item}" for item, qty in list(inventory.items())[:6])


def _mood_word(mood: float) -> str:
    if mood >= 0.75:
        return "good"
    if mood >= 0.5:
        return "okay"
    if mood >= 0.3:
        return "tired"
    return "distressed"


def _update_relationship(agent: VillageAgent, other_id: str, delta: int):
    """Adjust relationship score between two agents (clamped -10 to 10)."""
    scores = agent.relationship_scores
    current = scores.get(other_id, 0)
    scores[other_id] = max(-10, min(10, current + delta))
    agent.relationship_scores = scores


async def run_conversation(
    agent_a: VillageAgent,
    agent_b: VillageAgent,
    trigger: str,
    conversation_model: str,
    aigateway_url: str,
    token: str,
    db: AsyncSession,
    world_state: WorldState,
) -> Conversation:
    """
    Run a 2-4 exchange conversation between agent_a and agent_b.
    Agents alternate speaking. Updates relationship scores.
    Saves the conversation to DB.
    """
    convo = Conversation(
        id=str(uuid.uuid4()),
        tick=world_state.tick,
        x=agent_a.x,
        y=agent_a.y,
        completed=False,
    )
    convo.participants = [agent_a.agent_id, agent_b.agent_id]
    convo.messages = []

    # Decide number of exchanges (2-4)
    import random
    num_exchanges = random.randint(2, 4)

    # Alternate who speaks first based on trigger
    if trigger == "seek_help":
        order = [agent_b, agent_a]  # the needing agent speaks first
    else:
        order = [agent_a, agent_b]

    messages: list[dict] = []
    relationship_delta = 0

    for i in range(num_exchanges * 2):  # each exchange = 2 turns
        speaker = order[i % 2]
        listener = order[(i + 1) % 2]

        try:
            text = await _get_agent_response(
                speaking_agent=speaker,
                other_agent=listener,
                conversation_history=messages,
                trigger=trigger,
                world_state=world_state,
                model=conversation_model,
                aigateway_url=aigateway_url,
                token=token,
            )
        except Exception as exc:
            logger.error("Conversation LLM error (turn %d): %s", i, exc)
            text = "..."  # silent turn rather than crashing

        messages.append({
            "agent_id": speaker.agent_id,
            "name": speaker.name,
            "text": text,
        })

        # Talking boosts social need satisfaction
        speaker.needs = _reduce_social(speaker.needs)

        # Assume most conversations are slightly positive
        relationship_delta += 1

    convo.messages = messages
    convo.completed = True

    # Update memories
    summary_a = f"Spoke with {agent_b.name}: \"{messages[0]['text'][:60]}...\""
    summary_b = f"Spoke with {agent_a.name}: \"{messages[0]['text'][:60]}...\""
    agent_a.add_memory(summary_a)
    agent_b.add_memory(summary_b)

    # Update relationships
    _update_relationship(agent_a, agent_b.agent_id, relationship_delta)
    _update_relationship(agent_b, agent_a.agent_id, relationship_delta)

    # Set states back to idle after conversation
    agent_a.state = "idle"
    agent_b.state = "idle"

    db.add(convo)
    return convo


def _reduce_social(needs: dict) -> dict:
    n = dict(needs)
    n["social"] = max(0.0, n.get("social", 50) - 15.0)
    return n
