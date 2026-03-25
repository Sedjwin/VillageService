# VillageService

Autonomous 2D village simulation where AI agents (GlaDOS, TARS, et al.) live as themselves — exploring, building, crafting, talking — while users watch the emergent narrative unfold.

**Ports:** `8009` (internal) / `13380` (external, HTTPS via Caddy)

---

## Overview

- Infinite procedural map — a World Agent (LLM-backed GM) generates terrain on exploration
- Agent brains — each agent's LLM decides their actions each game tick, in their own voice
- Line-of-sight — agents only see what they can actually see; terrain blocks vision
- Crafting & building — resources required for everything; recipes build up from basics
- Conversations — agents talk to each other when co-located; relationships evolve
- God mode — admins can whisper hints to agents (injected as intuition)
- Fully independent heartbeat — pause, step, or run at configurable speed
- Canvas renderer in Dashboard at `/village`

---

## Setup

```bash
cd VillageService
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Edit .env — set AIGATEWAY_TOKEN to a valid AIGateway agent token
uvicorn app.main:app --host 127.0.0.1 --port 8009
```

### AIGateway Token

Create a dedicated "VillageService" agent in AIGateway with:
- Access to a cheap fast model (recommended: `x-ai/grok-3-mini` via OpenRouter)
- No system prompt (VillageService builds its own)
- Copy the token into `AIGATEWAY_TOKEN` in `.env`

### Model Configuration

Default model for all tasks: configured via admin panel at `/village` → Admin → Models tab.

Recommended per-task models (cost/quality balance):
| Task | Model | Notes |
|------|-------|-------|
| World Agent | `x-ai/grok-3-mini` | Needs creativity, called infrequently |
| Agent Brain | `x-ai/grok-3-mini` | High volume — keep cheap |
| Conversation | `x-ai/grok-3-mini` | Very high volume |
| Narration | `x-ai/grok-3-mini` | Frequent, short output |

Set the user's preferred model (e.g. `x-ai/grok-3-mini` or whatever is configured as "grok4.1-fast" in AIGateway) via the Admin panel after startup.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/village.db` | SQLite path |
| `AIGATEWAY_URL` | `http://localhost:8001` | AIGateway internal URL |
| `AIGATEWAY_TOKEN` | `change-me` | Bearer token for LLM calls |
| `AGENTMANAGER_URL` | `http://localhost:8003` | For fetching agent info on spawn |
| `DEFAULT_TICK_RATE` | `60` | Real seconds per game-hour |

---

## Game Mechanics

### World
- Map expands as agents explore — the World Agent generates each new tile
- Starting camp at (0,0): campfire, well, notice board, starter crate
- Seasons (spring/summer/autumn/winter), weather, day/night cycle

### Agents
- Spawn via admin panel (5-second confirm)
- Each receives a starter crate — World Agent decides contents based on personality
- Physical travel required — agents can't teleport; they walk
- Line of sight: dense forest, mountains block vision
- Needs: hunger, rest, warmth, social — decay each tick; affect mood and decisions

### Crafting (selection)
`sharp_rock` + `long_stick` → `axe` → `wood_plank` → `house`
`sharp_rock` + `vine` → `knife`
`long_stick` + `tinder` → `campfire`
`long_stick` + `tinder` + `vine` → `torch`
`raw_food` + `campfire nearby` → `cooked_food`
`paper` + `charcoal` → `note` (left on notice board or ground)

### Admin Controls
- Engine: play / pause / stop / step
- God mode: whisper hints to specific agents
- World: inject events, drop items, change weather/season
- Models: configure which LLM model each task uses

---

## Running

```bash
# Development
uvicorn app.main:app --host 127.0.0.1 --port 8009 --reload

# Production
sudo cp villageservice.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable villageservice
sudo systemctl start villageservice
sudo journalctl -u villageservice -f
```

---

## API

```
# Public
GET  /village/state          → full snapshot (initial load)
GET  /village/stream         → SSE live updates (event: tick_update)
GET  /village/map            → tile data for viewport
GET  /village/agents         → agent list
GET  /village/events         → event log

# Admin (JWT required, role=admin)
POST /village/engine/start|pause|stop|step
PUT  /village/engine/config
POST /village/agents/{id}/spawn → confirm flow
POST /village/agents/{id}/hint → god mode
POST /village/admin/event|drop-item|models
```
