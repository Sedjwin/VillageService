from typing import Any
from pydantic import BaseModel


class TileOut(BaseModel):
    x: int
    y: int
    terrain: str
    elevation: int
    features: list[str]
    resource_nodes: list[dict]
    buildings: list[dict]
    items: list[dict]
    explored_by: list[str]
    narrative: str


class AgentStateOut(BaseModel):
    agent_id: str
    name: str
    in_simulation: bool
    x: int
    y: int
    state: str
    inventory: dict
    skills: dict
    needs: dict
    mood: float
    current_goal: str | None
    relationship_scores: dict
    avatar_config: dict
    village_memory: list[str]


class WorldStateOut(BaseModel):
    tick: int
    game_day: int
    game_hour: int
    season: str
    weather: str
    engine_state: str
    tick_rate_seconds: int


class EventLogOut(BaseModel):
    id: str
    tick: int
    game_day: int
    game_hour: int
    event_type: str
    description: str
    agents_involved: list[str]
    x: int | None
    y: int | None


class VillageStateOut(BaseModel):
    world_state: WorldStateOut
    agents: list[AgentStateOut]
    recent_events: list[EventLogOut]


class SpawnConfirmToken(BaseModel):
    confirm_token: str
    expires_in: int
    agent_id: str
    action: str


class ConversationOut(BaseModel):
    id: str
    tick: int
    x: int
    y: int
    participants: list[str]
    messages: list[dict]
    completed: bool


# --- Helper converters ---

def agent_to_out(agent) -> AgentStateOut:
    return AgentStateOut(
        agent_id=agent.agent_id,
        name=agent.name,
        in_simulation=agent.in_simulation,
        x=agent.x,
        y=agent.y,
        state=agent.state,
        inventory=agent.inventory,
        skills=agent.skills,
        needs=agent.needs,
        mood=agent.mood,
        current_goal=agent.current_goal,
        relationship_scores=agent.relationship_scores,
        avatar_config={
            "primary_color": agent.avatar_primary_color,
            "secondary_color": agent.avatar_secondary_color,
            "body_shape": agent.avatar_body_shape,
            "eye_style": agent.avatar_eye_style,
        },
        village_memory=agent.village_memory,
    )


def world_to_out(world) -> WorldStateOut:
    return WorldStateOut(
        tick=world.tick,
        game_day=world.game_day,
        game_hour=world.game_hour,
        season=world.season,
        weather=world.weather,
        engine_state=world.engine_state,
        tick_rate_seconds=world.tick_rate_seconds,
    )


def event_to_out(event) -> EventLogOut:
    return EventLogOut(
        id=event.id,
        tick=event.tick,
        game_day=event.game_day,
        game_hour=event.game_hour,
        event_type=event.event_type,
        description=event.description,
        agents_involved=event.agents_involved,
        x=event.x,
        y=event.y,
    )


def tile_to_out(tile) -> TileOut:
    return TileOut(
        x=tile.x,
        y=tile.y,
        terrain=tile.terrain,
        elevation=tile.elevation,
        features=tile.features,
        resource_nodes=tile.resource_nodes,
        buildings=tile.buildings,
        items=tile.items,
        explored_by=tile.explored_by,
        narrative=tile.narrative,
    )
