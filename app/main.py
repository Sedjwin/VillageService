"""
VillageService — FastAPI application entry point.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from app.config import settings
from app.database import AsyncSessionLocal, init_db
from app.models import Tile, WorldState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger("village")


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

async def _init_world_state():
    """Create WorldState row and starting camp tiles on first run."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(WorldState).where(WorldState.id == 1))
        if result.scalar_one_or_none():
            logger.info("World state already exists — skipping init.")
            return

        logger.info("First run detected — creating world state and starting camp.")
        world = WorldState(
            id=1,
            tick=0,
            game_day=1,
            game_hour=8,
            season="spring",
            weather="clear",
            engine_state="stopped",
            tick_rate_seconds=settings.default_tick_rate,
            world_bible=(
                "A clearing has been carved out of the wilderness. "
                "A notice board stands at the centre, a campfire smoulders nearby. "
                "The world beyond is uncharted. Spring has just begun."
            ),
        )
        db.add(world)
        await _generate_starting_camp(db)
        await db.commit()
        logger.info("World initialised. Starting camp created.")


async def _generate_starting_camp(db):
    """
    Create the hand-crafted starting camp — a clearing with character.
    Tiles are laid out around the origin (0,0).
    """
    camp_layout = [
        # (x, y, terrain, elevation, features, resources)
        (0,  0,  "grass",        2, ["notice_board", "well"],
            [{"type": "raw_food", "qty": 3, "max_qty": 5}]),
        (-1, 0,  "grass",        2, ["campfire"],
            [{"type": "tinder", "qty": 3, "max_qty": 4}]),
        (1,  0,  "grass",        2, [],
            [{"type": "long_stick", "qty": 3, "max_qty": 4}]),
        (0,  1,  "grass",        2, ["starter_crate"],
            []),
        (0,  -1, "grass",        2, [],
            [{"type": "raw_food", "qty": 2, "max_qty": 4}]),
        (-1, -1, "light_forest", 3, [],
            [{"type": "vine", "qty": 3, "max_qty": 5}, {"type": "raw_food", "qty": 2, "max_qty": 4}]),
        (1,  -1, "light_forest", 3, [],
            [{"type": "long_stick", "qty": 3, "max_qty": 5}]),
        (-1, 1,  "light_forest", 3, [],
            [{"type": "raw_food", "qty": 2, "max_qty": 3}]),
        (1,  1,  "hills",        4, [],
            [{"type": "sharp_rock", "qty": 4, "max_qty": 6}]),
        (2,  0,  "light_forest", 3, [],
            [{"type": "vine", "qty": 2, "max_qty": 4}]),
        (-2, 0,  "light_forest", 3, [],
            [{"type": "long_stick", "qty": 2, "max_qty": 4}]),
        (2,  1,  "hills",        4, [],
            [{"type": "sharp_rock", "qty": 2, "max_qty": 4}]),
        (-2, 1,  "dense_forest", 4, [],
            [{"type": "vine", "qty": 4, "max_qty": 6}]),
        (0,  2,  "grass",        2, [],
            [{"type": "tinder", "qty": 2, "max_qty": 3}]),
        (0,  -2, "beach",        1, [],
            [{"type": "sharp_rock", "qty": 2, "max_qty": 3}]),
        (2,  -1, "beach",        1, [],
            [{"type": "raw_food", "qty": 2, "max_qty": 3}]),
        (-2, -1, "light_forest", 3, [],
            [{"type": "long_stick", "qty": 3, "max_qty": 4}]),
    ]

    narratives = {
        (0,  0):  "The heart of the camp. A notice board stands upright, its surface clean and waiting. A hand-dug well sits beside it.",
        (-1, 0):  "A campfire ring of flat stones. The fire has been lit recently — the coals are still warm.",
        (1,  0):  "Fallen branches, mostly straight. Someone stacked them here but never came back for them.",
        (0,  1):  "A weathered crate sits here, lid half-open. It looks like it was meant to be found.",
        (0,  -1): "Open grass, good for sitting. Berries grow in a low tangle at the edge.",
        (-1, -1): "Young trees and thick undergrowth. Vines loop between the trunks.",
        (1,  -1): "A stand of tall straight trees. Good timber, not yet touched.",
        (-1, 1):  "Dappled shade. Something rustles in the canopy but doesn't show itself.",
        (1,  1):  "A rocky outcrop, flint-grey. Flakes litter the base — something worked stone here before.",
        (2,  0):  "Forest edge, tangled with vine. The light is softer here.",
        (-2, 0):  "Solid hardwood trees. Branches litter the ground after last season's storms.",
        (2,  1):  "Steep ground, loose rock. You could find good flint if you look.",
        (-2, 1):  "Dense canopy, almost dark at midday. The undergrowth is thick with vine.",
        (0,  2):  "A grassy slope leading north. The wind comes from this direction most mornings.",
        (0,  -2): "Sandy ground gives way to a rocky beach. The sound of water is close.",
        (2,  -1): "Beach sand and flat pebbles. Something was dragged ashore here not long ago.",
        (-2, -1): "Tall trees, good shade. A branch has fallen recently — the wood is still pale at the break.",
    }

    for (x, y, terrain, elev, features, resources) in camp_layout:
        tile = Tile(
            x=x, y=y,
            terrain=terrain,
            elevation=elev,
            narrative=narratives.get((x, y), ""),
        )
        tile.features = features
        tile.resource_nodes = [
            {
                "type": r["type"],
                "qty": r["qty"],
                "max_qty": r["max_qty"],
                "regen_at_tick": 0,
            }
            for r in resources
        ]
        tile.items = []
        tile.buildings = []
        tile.explored_by = []
        db.add(tile)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

async def _migrate_db():
    """Lightweight migrations for columns added after initial deploy."""
    from app.database import engine as _engine
    from sqlalchemy import text
    async with _engine.begin() as conn:
        for stmt in [
            "ALTER TABLE world_state ADD COLUMN last_tick_at REAL",
            "ALTER TABLE world_state ADD COLUMN _sim_config TEXT DEFAULT '{}'",
            "ALTER TABLE world_state ADD COLUMN _gateway_config TEXT DEFAULT '{}'",
            "ALTER TABLE village_agents ADD COLUMN goal_set_tick INTEGER DEFAULT 0",
            "ALTER TABLE tick_snapshots ADD COLUMN tiles_json TEXT",
        ]:
            try:
                await conn.execute(text(stmt))
            except Exception:
                pass  # column already exists


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    await _migrate_db()
    await _init_world_state()

    # Import here to avoid circular on startup
    from app.engine import engine
    from app.models import WorldState

    # Auto-resume if engine was running before restart
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(WorldState).where(WorldState.id == 1))
        world = result.scalar_one_or_none()
        if world and world.engine_state == "running":
            logger.info("Resuming engine — was running before restart.")
            await engine.start()
        else:
            logger.info("VillageService ready. Engine is stopped — admin must press play.")

    yield

    # Shutdown — don't persist state so engine auto-resumes on next start
    from app.engine import engine as eng
    await eng.stop(persist=False)
    logger.info("VillageService shutdown complete.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VillageService",
    description="AI village simulation — agents, world, and everything in between.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
from app.routers.public import router as public_router
from app.routers.admin import router as admin_router

app.include_router(public_router)
app.include_router(admin_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "VillageService"}


@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=13380, reload=False)
