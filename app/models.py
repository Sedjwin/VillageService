import json
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _load_json(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


# ---------------------------------------------------------------------------
# WorldState
# ---------------------------------------------------------------------------

class WorldState(Base):
    __tablename__ = "world_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    tick: Mapped[int] = mapped_column(Integer, default=0)
    game_day: Mapped[int] = mapped_column(Integer, default=1)
    game_hour: Mapped[int] = mapped_column(Integer, default=8)
    season: Mapped[str] = mapped_column(String(16), default="spring")
    weather: Mapped[str] = mapped_column(String(16), default="clear")
    engine_state: Mapped[str] = mapped_column(String(16), default="stopped")
    tick_rate_seconds: Mapped[int] = mapped_column(Integer, default=60)
    world_bible: Mapped[str] = mapped_column(Text, default="")


# ---------------------------------------------------------------------------
# Tile
# ---------------------------------------------------------------------------

class Tile(Base):
    __tablename__ = "tiles"

    x: Mapped[int] = mapped_column(Integer, primary_key=True)
    y: Mapped[int] = mapped_column(Integer, primary_key=True)
    terrain: Mapped[str] = mapped_column(String(32), default="grass")
    elevation: Mapped[int] = mapped_column(Integer, default=0)
    _features: Mapped[str] = mapped_column(Text, default="[]")
    _resource_nodes: Mapped[str] = mapped_column(Text, default="[]")
    _buildings: Mapped[str] = mapped_column(Text, default="[]")
    _items: Mapped[str] = mapped_column(Text, default="[]")
    _explored_by: Mapped[str] = mapped_column(Text, default="[]")
    narrative: Mapped[str] = mapped_column(Text, default="")
    last_updated: Mapped[datetime | None] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    @property
    def features(self) -> list:
        return _load_json(self._features, [])

    @features.setter
    def features(self, value: list):
        self._features = json.dumps(value)

    @property
    def resource_nodes(self) -> list:
        return _load_json(self._resource_nodes, [])

    @resource_nodes.setter
    def resource_nodes(self, value: list):
        self._resource_nodes = json.dumps(value)

    @property
    def buildings(self) -> list:
        return _load_json(self._buildings, [])

    @buildings.setter
    def buildings(self, value: list):
        self._buildings = json.dumps(value)

    @property
    def items(self) -> list:
        return _load_json(self._items, [])

    @items.setter
    def items(self, value: list):
        self._items = json.dumps(value)

    @property
    def explored_by(self) -> list:
        return _load_json(self._explored_by, [])

    @explored_by.setter
    def explored_by(self, value: list):
        self._explored_by = json.dumps(value)


# ---------------------------------------------------------------------------
# VillageAgent
# ---------------------------------------------------------------------------

class VillageAgent(Base):
    __tablename__ = "village_agents"

    agent_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), default="Unknown")
    personality_summary: Mapped[str] = mapped_column(Text, default="")
    in_simulation: Mapped[bool] = mapped_column(Boolean, default=False)
    x: Mapped[int] = mapped_column(Integer, default=0)
    y: Mapped[int] = mapped_column(Integer, default=0)
    state: Mapped[str] = mapped_column(String(32), default="idle")
    _inventory: Mapped[str] = mapped_column(Text, default="{}")
    _skills: Mapped[str] = mapped_column(Text, default="{}")
    _needs: Mapped[str] = mapped_column(
        Text,
        default='{"hunger":30,"rest":20,"warmth":20,"social":40}',
    )
    mood: Mapped[float] = mapped_column(Float, default=0.7)
    current_goal: Mapped[str | None] = mapped_column(Text, nullable=True)
    _travel_path: Mapped[str] = mapped_column(Text, default="[]")
    _village_memory: Mapped[str] = mapped_column(Text, default="[]")
    god_hint: Mapped[str | None] = mapped_column(Text, nullable=True)
    _relationship_scores: Mapped[str] = mapped_column(Text, default="{}")
    brain_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    avatar_primary_color: Mapped[str] = mapped_column(String(16), default="#4a9eff")
    avatar_secondary_color: Mapped[str] = mapped_column(String(16), default="#1a3a6a")
    avatar_body_shape: Mapped[str] = mapped_column(String(16), default="circle")
    avatar_eye_style: Mapped[str] = mapped_column(String(16), default="round")
    joined_tick: Mapped[int] = mapped_column(Integer, default=0)
    last_tick: Mapped[int] = mapped_column(Integer, default=0)
    _starter_crate_opened: Mapped[str] = mapped_column(Text, default="false")

    # --- JSON property helpers ---

    @property
    def inventory(self) -> dict:
        return _load_json(self._inventory, {})

    @inventory.setter
    def inventory(self, value: dict):
        self._inventory = json.dumps(value)

    @property
    def skills(self) -> dict:
        return _load_json(self._skills, {})

    @skills.setter
    def skills(self, value: dict):
        self._skills = json.dumps(value)

    @property
    def needs(self) -> dict:
        return _load_json(
            self._needs,
            {"hunger": 30, "rest": 20, "warmth": 20, "social": 40},
        )

    @needs.setter
    def needs(self, value: dict):
        self._needs = json.dumps(value)

    @property
    def travel_path(self) -> list:
        return _load_json(self._travel_path, [])

    @travel_path.setter
    def travel_path(self, value: list):
        self._travel_path = json.dumps(value)

    @property
    def village_memory(self) -> list:
        return _load_json(self._village_memory, [])

    @village_memory.setter
    def village_memory(self, value: list):
        self._village_memory = json.dumps(value[-20:])  # keep last 20

    @property
    def relationship_scores(self) -> dict:
        return _load_json(self._relationship_scores, {})

    @relationship_scores.setter
    def relationship_scores(self, value: dict):
        self._relationship_scores = json.dumps(value)

    @property
    def starter_crate_opened(self) -> bool:
        return _load_json(self._starter_crate_opened, False)

    @starter_crate_opened.setter
    def starter_crate_opened(self, value: bool):
        self._starter_crate_opened = json.dumps(value)

    def add_memory(self, entry: str):
        mem = self.village_memory
        mem.append(entry)
        self.village_memory = mem


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    tick: Mapped[int] = mapped_column(Integer, default=0)
    x: Mapped[int] = mapped_column(Integer, default=0)
    y: Mapped[int] = mapped_column(Integer, default=0)
    _participants: Mapped[str] = mapped_column(Text, default="[]")
    _messages: Mapped[str] = mapped_column(Text, default="[]")
    completed: Mapped[bool] = mapped_column(Boolean, default=False)

    @property
    def participants(self) -> list:
        return _load_json(self._participants, [])

    @participants.setter
    def participants(self, value: list):
        self._participants = json.dumps(value)

    @property
    def messages(self) -> list:
        return _load_json(self._messages, [])

    @messages.setter
    def messages(self, value: list):
        self._messages = json.dumps(value)


# ---------------------------------------------------------------------------
# EventLog
# ---------------------------------------------------------------------------

class EventLog(Base):
    __tablename__ = "event_log"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    tick: Mapped[int] = mapped_column(Integer, default=0)
    game_day: Mapped[int] = mapped_column(Integer, default=1)
    game_hour: Mapped[int] = mapped_column(Integer, default=0)
    event_type: Mapped[str] = mapped_column(String(32), default="world_event")
    description: Mapped[str] = mapped_column(Text, default="")
    _agents_involved: Mapped[str] = mapped_column(Text, default="[]")
    x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    @property
    def agents_involved(self) -> list:
        return _load_json(self._agents_involved, [])

    @agents_involved.setter
    def agents_involved(self, value: list):
        self._agents_involved = json.dumps(value)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class ModelConfig(Base):
    __tablename__ = "model_configs"

    task: Mapped[str] = mapped_column(String(32), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(128), default="x-ai/grok-3-mini")
    temperature: Mapped[float] = mapped_column(Float, default=0.9)
    max_tokens: Mapped[int] = mapped_column(Integer, default=512)
