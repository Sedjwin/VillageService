"""
Crafting recipe registry, skill definitions, and terrain constants.
"""

RECIPES: dict = {
    # --- Gathered (not crafted) ---
    "sharp_rock": {
        "requires": {},
        "found_in": ["hills", "mountain", "beach"],
        "description": "A flint-edged stone that holds a decent edge.",
    },
    "long_stick": {
        "requires": {},
        "found_in": ["light_forest", "dense_forest", "grass"],
        "description": "A fallen branch, straight and sturdy.",
    },
    "tinder": {
        "requires": {},
        "found_in": ["grass", "light_forest"],
        "description": "Dry grass and crumbled bark.",
    },
    "vine": {
        "requires": {},
        "found_in": ["light_forest", "dense_forest"],
        "description": "A long woody vine, still supple.",
    },
    "raw_food": {
        "requires": {},
        "found_in": ["light_forest", "dense_forest", "grass", "beach", "water"],
        "description": "Berries, roots, caught fish — whatever the land offers.",
    },
    "paper": {
        "requires": {},
        "found_in": ["water"],  # reeds at water's edge
        "description": "Dried reed paper, slightly waxy to the touch.",
    },

    # --- Simple crafts ---
    "rope": {
        "requires": {"vine": 3},
        "skill_req": {"crafting": 0},
        "description": "Three vines twisted tight. Holds a surprising amount.",
    },
    "knife": {
        "requires": {"sharp_rock": 1, "vine": 1},
        "skill_req": {"crafting": 0},
        "description": "Stone bound to vine — not elegant, but reliably sharp.",
    },
    "hollow_stick": {
        "requires": {"long_stick": 1, "sharp_rock": 1},
        "skill_req": {"crafting": 0},
        "description": "A carefully bored branch. Smells faintly of sawdust.",
    },

    # --- Tools ---
    "axe": {
        "requires": {"sharp_rock": 2, "long_stick": 1, "vine": 1},
        "skill_req": {"crafting": 5},
        "description": "A proper woodcutter's axe. The weight feels right.",
    },
    "fishing_rod": {
        "requires": {"long_stick": 1, "rope": 1, "sharp_rock": 1},
        "skill_req": {"crafting": 5},
        "description": "Balanced for the wrist. Good for water tiles.",
    },
    "torch": {
        "requires": {"long_stick": 1, "tinder": 2, "vine": 1},
        "skill_req": {"crafting": 0},
        "description": "Burns for 3 hours. Pushes back the dark.",
    },

    # --- Instruments ---
    "flute": {
        "requires": {"hollow_stick": 1, "sharp_rock": 1},
        "skill_req": {"crafting": 5},
        "description": "Simple but true-sounding. Raises social mood when played.",
    },

    # --- Processed resources ---
    "wood_plank": {
        "requires": {"long_stick": 2},
        "tool_req": "axe",
        "qty_out": 3,
        "skill_req": {"woodcutting": 5},
        "description": "Split and smoothed. The grain is clean.",
    },
    "charcoal": {
        "requires": {"long_stick": 1},
        "near_req": "campfire",
        "skill_req": {"crafting": 0},
        "description": "From the fire's edge — dense black sticks.",
    },
    "cooked_food": {
        "requires": {"raw_food": 1},
        "near_req": "campfire",
        "skill_req": {"crafting": 0},
        "description": "Hot, filling, and surprisingly good.",
    },

    # --- Written items ---
    "note": {
        "requires": {"paper": 1, "charcoal": 1},
        "skill_req": {"crafting": 0},
        "description": "A written note. What it says depends entirely on who wrote it.",
    },

    # --- Structures (is_building = True) ---
    "campfire": {
        "requires": {"tinder": 2, "long_stick": 3},
        "is_building": True,
        "skill_req": {"crafting": 0},
        "description": "A small fire. Warmth, cooked food, and something to gather around.",
    },
    "basic_shelter": {
        "requires": {"long_stick": 8, "rope": 3, "tinder": 4},
        "is_building": True,
        "skill_req": {"building": 0},
        "description": "A lean-to. Blocks wind, keeps rain off. Not glamorous.",
    },
    "house": {
        "requires": {"wood_plank": 20, "rope": 6, "sharp_rock": 5},
        "is_building": True,
        "skill_req": {"building": 10},
        "description": "A proper dwelling with four walls and a latchable door.",
    },
    "workshop": {
        "requires": {"wood_plank": 15, "rope": 4, "sharp_rock": 3},
        "is_building": True,
        "skill_req": {"building": 20},
        "description": "Speeds up crafting. Unlocks advanced recipes.",
    },
    "road_tile": {
        "requires": {"sharp_rock": 3},
        "is_building": True,
        "skill_req": {"building": 0},
        "description": "Cleared and surfaced. Movement cost halved.",
    },
    "notice_board": {
        "requires": {"wood_plank": 5, "rope": 2},
        "is_building": True,
        "skill_req": {"building": 5},
        "description": "A community message board. Notes can be pinned here.",
    },
    "well": {
        "requires": {"sharp_rock": 8, "rope": 4, "long_stick": 3},
        "is_building": True,
        "skill_req": {"building": 15},
        "description": "Dug deep enough to find clean water. Reduces thirst decay.",
    },
    "watchtower": {
        "requires": {"wood_plank": 10, "rope": 5, "sharp_rock": 4},
        "is_building": True,
        "skill_req": {"building": 20},
        "description": "Extends line-of-sight by 3 tiles for any agent standing here.",
    },
    "garden_plot": {
        "requires": {"long_stick": 4, "rope": 2},
        "is_building": True,
        "skill_req": {"farming": 5},
        "description": "Tilled soil. Slowly regenerates raw_food without a resource node.",
    },
}


SKILLS = [
    "woodcutting",
    "gathering",
    "building",
    "crafting",
    "navigation",
    "social",
    "farming",
    "foraging",
]


TERRAIN_MOVEMENT_COST: dict[str, float | None] = {
    "grass": 1.0,
    "light_forest": 1.5,
    "dense_forest": 2.0,
    "hills": 2.0,
    "mountain": 3.0,
    "water": None,   # impassable
    "beach": 1.0,
    "cave": 1.5,
    "road": 0.5,
    "unknown": 1.0,
}

TERRAIN_VISION_MODIFIER: dict[str, int] = {
    "grass": 0,
    "light_forest": -2,
    "dense_forest": -3,
    "hills": -1,   # +3 if standing ON hills
    "mountain": 0,  # blocks from behind
    "water": 0,
    "beach": 0,
    "cave": -3,
    "road": 0,
}

# Resources that naturally appear on each terrain type (used by WorldAgent)
TERRAIN_NATURAL_RESOURCES: dict[str, list[str]] = {
    "grass": ["raw_food", "tinder", "long_stick"],
    "light_forest": ["long_stick", "vine", "raw_food", "tinder"],
    "dense_forest": ["long_stick", "vine", "raw_food"],
    "hills": ["sharp_rock", "raw_food"],
    "mountain": ["sharp_rock"],
    "water": ["raw_food", "paper"],
    "beach": ["sharp_rock", "raw_food"],
    "cave": ["sharp_rock"],
    "road": [],
}
