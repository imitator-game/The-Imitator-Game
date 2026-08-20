from typing import List
from mani_skill import ASSET_DIR
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.io_utils import load_json

YCB_DATASET = dict()

# All item IDs (complete list)
all_items = [
    "002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can",
    "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box",
    "010_potted_meat_can", "011_banana", "012_strawberry", "013_apple",
    "014_lemon", "015_peach", "016_pear", "017_orange", "018_plum",
    "019_pitcher_base", "021_bleach_cleanser", "022_windex_bottle",
    "024_bowl", "025_mug", "026_sponge", "028_skillet_lid", "029_plate",
    "030_fork", "031_spoon", "032_knife", "033_spatula",
    "035_power_drill", "036_wood_block", "037_scissors", "038_padlock",
    "040_large_marker", "042_adjustable_wrench", "043_phillips_screwdriver",
    "044_flat_screwdriver", "048_hammer", "050_medium_clamp", "051_large_clamp",
    "052_extra_large_clamp", "053_mini_soccer_ball", "054_softball", "055_baseball",
    "056_tennis_ball", "057_racquetball", "058_golf_ball", "059_chain",
    "061_foam_brick", "062_dice", "063-a_marbles", "063-b_marbles",
    "065-a_cups", "065-b_cups", "065-c_cups", "065-d_cups", "065-e_cups",
    "065-f_cups", "065-g_cups", "065-h_cups", "065-i_cups", "065-j_cups",
    "070-a_colored_wood_blocks", "070-b_colored_wood_blocks", "071_nine_hole_peg_test",
    "072-a_toy_airplane", "072-b_toy_airplane", "072-c_toy_airplane", "072-d_toy_airplane",
    "072-e_toy_airplane", "073-a_lego_duplo", "073-b_lego_duplo", "073-c_lego_duplo",
    "073-d_lego_duplo", "073-e_lego_duplo", "073-f_lego_duplo", "073-g_lego_duplo",
    "077_rubiks_cube"
]

# Grouped by category
# Food and beverage related
food_and_beverage = {
    "cans": [
        "002_master_chef_can",
        "005_tomato_soup_can",
        "007_tuna_fish_can",
        "010_potted_meat_can"
    ],
    "boxes": [
        "003_cracker_box",
        "004_sugar_box",
        "008_pudding_box",
        "009_gelatin_box"
    ],
    "condiments": [
        "006_mustard_bottle"
    ],
    "fruits": [
        "011_banana",
        "012_strawberry",
        "013_apple",
        "014_lemon",
        "015_peach",
        "016_pear",
        "017_orange",
        "018_plum"
    ]
}

# Cleaning supplies
cleaning_supplies = [
    "019_pitcher_base",
    "021_bleach_cleanser",
    "022_windex_bottle"
]

# Dishes and kitchen items
kitchen_items = {
    "dishes": [
        "024_bowl",
        "025_mug",
        "029_plate"
    ],
    "utensils": [
        "030_fork",
        "031_spoon",
        "032_knife"
    ],
    "kitchen_tools": [
        "026_sponge",
        "028_skillet_lid",
        "033_spatula"
    ]
}

# Tools
tools = {
    "power_tools": [
        "035_power_drill"
    ],
    "hand_tools": [
        "037_scissors",
        "042_adjustable_wrench",
        "043_phillips_screwdriver",
        "044_flat_screwdriver",
        "048_hammer"
    ],
    "clamps": [
        "050_medium_clamp",
        "051_large_clamp",
        "052_extra_large_clamp"
    ],
    "other_tools": [
        "036_wood_block",
        "038_padlock",
        "040_large_marker"
    ]
}

# Sports items
sports_items = [
    "053_mini_soccer_ball",
    "054_softball",
    "055_baseball",
    "056_tennis_ball",
    "057_racquetball",
    "058_golf_ball"
]

# Toys and games
toys_and_games = {
    "basic_toys": [
        "059_chain",
        "061_foam_brick",
        "062_dice",
        "071_nine_hole_peg_test",
        "077_rubiks_cube"
    ],
    "marbles": [
        "063-a_marbles",
        "063-b_marbles"
    ],
    "cups": [
        "065-a_cups",
        "065-b_cups",
        "065-c_cups",
        "065-d_cups",
        "065-e_cups",
        "065-f_cups",
        "065-g_cups",
        "065-h_cups",
        "065-i_cups",
        "065-j_cups"
    ],
    "colored_blocks": [
        "070-a_colored_wood_blocks",
        "070-b_colored_wood_blocks"
    ],
    "toy_airplanes": [
        "072-a_toy_airplane",
        "072-b_toy_airplane",
        "072-c_toy_airplane",
        "072-d_toy_airplane",
        "072-e_toy_airplane"
    ],
    "lego_duplo": [
        "073-a_lego_duplo",
        "073-b_lego_duplo",
        "073-c_lego_duplo",
        "073-d_lego_duplo",
        "073-e_lego_duplo",
        "073-f_lego_duplo",
        "073-g_lego_duplo"
    ]
}

# Flattened category list (if you need a simple flat categorization)
categories_flat = {
    "food_cans": food_and_beverage["cans"],
    "food_boxes": food_and_beverage["boxes"],
    "condiments": food_and_beverage["condiments"],
    "fruits": food_and_beverage["fruits"],
    "cleaning": cleaning_supplies,
    "dishes": kitchen_items["dishes"],
    "utensils": kitchen_items["utensils"],
    "kitchen_tools": kitchen_items["kitchen_tools"],
    "power_tools": tools["power_tools"],
    "hand_tools": tools["hand_tools"],
    "clamps": tools["clamps"],
    "other_tools": tools["other_tools"],
    "sports": sports_items,
    "basic_toys": toys_and_games["basic_toys"],
    "marbles": toys_and_games["marbles"],
    "cups": toys_and_games["cups"],
    "colored_blocks": toys_and_games["colored_blocks"],
    "toy_airplanes": toys_and_games["toy_airplanes"],
    "lego": toys_and_games["lego_duplo"]
}

# Grouped by ID prefix (for quickly finding items in the same series)
items_by_prefix = {
    "063": ["063-a_marbles", "063-b_marbles"],
    "065": ["065-a_cups", "065-b_cups", "065-c_cups", "065-d_cups", "065-e_cups",
            "065-f_cups", "065-g_cups", "065-h_cups", "065-i_cups", "065-j_cups"],
    "070": ["070-a_colored_wood_blocks", "070-b_colored_wood_blocks"],
    "072": ["072-a_toy_airplane", "072-b_toy_airplane", "072-c_toy_airplane",
            "072-d_toy_airplane", "072-e_toy_airplane"],
    "073": ["073-a_lego_duplo", "073-b_lego_duplo", "073-c_lego_duplo",
            "073-d_lego_duplo", "073-e_lego_duplo", "073-f_lego_duplo", "073-g_lego_duplo"]
}

def _load_ycb_dataset():
    global YCB_DATASET
    YCB_DATASET = {
        "model_data": load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"),
    }


def get_ycb_builder(
    scene: ManiSkillScene,
    id: str,
    add_collision: bool = True,
    add_visual: bool = True,
    scales: List[float] = None,
):
    if "YCB" not in YCB_DATASET:
        _load_ycb_dataset()
    model_db = YCB_DATASET["model_data"]

    builder = scene.create_actor_builder()

    metadata = model_db[id]
    density = metadata.get("density", 1000)
    if scales is not None:
        # Backward compatible:
        # - [s] or scalar-like first element -> uniform scaling
        # - [sx, sy, sz] -> anisotropic xyz scaling
        # - [[sx, sy, sz], ...] -> use first xyz scaling
        if (
            len(scales) == 3
            and all(isinstance(v, (int, float)) for v in scales)
        ):
            scale_vec = [float(scales[0]), float(scales[1]), float(scales[2])]
        elif (
            len(scales) > 0
            and isinstance(scales[0], (list, tuple))
            and len(scales[0]) == 3
        ):
            scale_vec = [float(scales[0][0]), float(scales[0][1]), float(scales[0][2])]
        else:
            s = float(scales[0])
            scale_vec = [s, s, s]
    else:
        model_scales = metadata.get("scales", [0.8])
        s = float(model_scales[0])
        scale_vec = [s, s, s]

    physical_material = None
    model_dir = ASSET_DIR / "assets/mani_skill2_ycb/models" / id
    if add_collision:
        collision_file = str(model_dir / "collision.ply")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=scale_vec,
            material=physical_material,
            density=density,
        )
    if add_visual:
        visual_file = str(model_dir / "textured.obj")
        builder.add_visual_from_file(filename=visual_file, scale=scale_vec)

    return builder
