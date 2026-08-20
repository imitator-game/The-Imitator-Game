import random

from mani_skill.envs.sim_assets_catigory import ROBOTWIN_MODEL_IDS


# Helper function: gets the valid model_id for an item
def get_model_id(item_name: str, model_id: int = None, random_id: bool = False):
    """
    Get the model_id for a specified item.

    Args:
        item_name: name of the item (e.g. '054_baguette')
        model_id: preferred model_id; if not specified, the default value is used

    Returns:
        The valid model_id

    Raises:
        ValueError: if the item does not exist or the model_id is invalid
    """
    if item_name not in ROBOTWIN_MODEL_IDS:
        return None

    config = ROBOTWIN_MODEL_IDS[item_name]
    available_ids = config["model_ids"]

    if not available_ids:
        return None

    if model_id is not None:
        if model_id in available_ids:
            return model_id
        else:
            print(f"Warning: item {item_name} does not support model_id={model_id}, using default value {config['default']}")
            return config['default']

    if random_id:
        return random.choice(config["model_ids"])

    return config['default']


# Usage examples:
# model_id = get_model_id('054_baguette')  # use the default model_id
# model_id = get_model_id('054_baguette', 3)  # try using model_id=3


# Robotwin items (IDs 001-120)
robotwin_items = [
    "001_bottle", "002_bowl", "003_plate", "004_fluted-block", "005_french-fries",
    "006_hamburg", "007_shoe-box", "008_tray", "010_pen",
    "011_dustbin", "012_plant-pot", "013_dumbbell-rack", "014_bookcase", "015_laptop",
    "016_oven", "017_calculator", "018_microphone", "019_coaster", "020_hammer",
    "021_cup", "022_cup-with-liquid", "023_tissue-box", "024_scanner", "025_chips-tub",
    "026_pet-collar", "027_table-tennis", "028_roll-paper", "029_olive-oil", "030_drill",
    "031_jam-jar", "032_screwdriver", "033_fork", "034_knife", "035_apple",
    "036_cabinet", "037_box", "038_milk-box", "039_mug", "040_rack",
    "041_shoe", "042_wooden_box", "043_book", "044_microwave", "045_sand-clock",
    "046_alarm-clock", "047_mouse", "048_stapler", "049_shampoo", "050_bell",
    "051_candlestick", "052_dumbbell", "053_teanet", "054_baguette", "055_small-speaker",
    "056_switch", "057_toycar", "058_markpen", "059_pencup", "060_kitchenpot",
    "061_battery", "062_plasticbox", "063_tabletrashbin", "064_msg", "065_soy-sauce",
    "066_vinegar", "067_steamer", "068_boxdrink", "069_vagetable", "070_paymentsign",
    "071_can", "072_electronicscale", "073_rubikscube", "074_displaystand", "075_bread",
    "076_breadbasket", "077_phone", "078_phonestand", "079_remotecontrol", "080_pillbottle",
    "081_playingcards", "082_smallshovel", "083_brush", "084_woodenmallet", "085_gong",
    "086_woodenblock", "087_waterer",
    # "088_wineglass", "009_kettle",
    "089_globe", "090_trophy",
    "091_kettle", "092_notebook", "093_brush-pen", "094_rest", "095_glue",
    "096_cleaner", "097_screen", "098_speaker", "099_fan", "100_seal",
    "101_milk-tea", "102_roller", "103_fruit", "104_board", "105_sauce-can",
    "106_skillet", "107_soap", "108_block", "109_hydrating-oil", "110_basket",
    "111_callbell", "112_tea-box", "113_coffee-box", "114_bottle", "115_perfume",
    "116_keyboard", "117_whiteboard-eraser", "118_tooth-paste", "119_mini-chalkboard", "120_plant"
]

# Objaverse items and their available IDs
objaverse_items = {
    "bottle": ["006", "007", "013", "017", "019", "020", "025"],
    "bowl": ["001", "002", "003", "004", "005", "007", "010", "011", "012", "016", "018", "020", "021", "022", "024"],
    "brush": ["005", "006", "007"],
    "can": ["001", "005", "008", "009", "011", "013", "015", "017", "018"],
    "chip_can": ["003", "007", "008"],
    "clock": ["005", "006"],
    "drinkbox": ["003"],
    "hammer": ["001", "002", "003", "004", "005"],
    "marker": ["002", "004"],
    "notebook": ["001", "002", "003", "004", "008"],
    "pencil": ["001"],
    "plate": ["003", "004", "007", "008", "009"],
    "pot": ["002", "004", "005", "006", "007", "008", "009", "010"],
    "ramen_box": ["001", "002", "005", "012", "013", "015", "017", "018", "019", "022", "023", "024"],
    "remote": ["002", "004", "005", "006"],
    "slipper": ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012",
                "015", "020", "021", "022", "023", "024", "026", "027", "028", "029", "031", "033", "035"],
    "snack_box": ["018", "020", "022", "025", "031", "034", "047"],
    "snack_package": ["003", "004"],
    "sneaker": ["001", "002", "003", "005", "006", "007"],
    "spoon": ["011", "020"],
    "steel_tape": ["003"],
    "tape": ["001", "002", "003", "004"],
    "thermos": ["002", "003", "005"],
    "tissue": ["004", "005"],
    "toothbrush": ["009", "010", "019", "030"],
    "toy_car": ["001", "002", "003", "005", "006", "007", "009", "011", "013"],
    "wallet": ["003", "007", "010", "011", "012", "017"]
}

# RoboTwin objects (from robotwin.py) - Filter out items with no model_ids
ROBOTWIN_CATEGORIES = {
    "food_and_beverage": {
        "containers": ["001_bottle", "114_bottle", "021_cup", "022_cup-with-liquid", "039_mug",
                       # "088_wineglass",
                       "091_kettle"],  # Note: 009_kettle has no model_ids
        "food_items": ["005_french-fries", "006_hamburg", "035_apple", "054_baguette", "069_vagetable", "075_bread",
                       "103_fruit"],
        "food_packages": ["025_chips-tub", "031_jam-jar", "038_milk-box", "064_msg", "068_boxdrink", "071_can",
                          "101_milk-tea", "105_sauce-can", "112_tea-box", "113_coffee-box"],
        "condiments": ["029_olive-oil", "065_soy-sauce", "066_vinegar"]
    },
    "kitchen_items": {
        "dishes": ["002_bowl", "003_plate", "008_tray", "076_breadbasket"],
        "utensils": ["033_fork", "034_knife"],
        "cookware": ["067_steamer", "106_skillet", "053_teanet"],  # Note: 060_kitchenpot has no model_ids
        "kitchen_tools": ["087_waterer"]
    },
    "furniture_storage": {
        "furniture": ["014_bookcase", "040_rack"],  # Note: 036_cabinet has no model_ids
        "containers": ["007_shoe-box", "042_wooden_box", "062_plasticbox", "110_basket", "019_coaster"],
        # Note: 037_box has no model_ids
        "trash_bins": ["011_dustbin", "063_tabletrashbin"]
    },
    "electronics": {
        "computers": ["047_mouse", "116_keyboard"],  # Note: 015_laptop has no model_ids
        "devices": ["017_calculator", "018_microphone", "024_scanner", "077_phone", "079_remotecontrol", "097_screen"],
        "audio": ["055_small-speaker", "098_speaker"],
        "appliances": ["072_electronicscale", "099_fan"]  # Note: 016_oven and 044_microwave have no model_ids
    },
    "tools": {
        "hand_tools": ["020_hammer", "032_screwdriver", "082_smallshovel", "084_woodenmallet"],
        "power_tools": ["030_drill"],
        "other_tools": ["102_roller"]  # Note: 056_switch has no model_ids
    },
    "stationery": {
        "writing": ["058_markpen", "093_brush-pen", "083_brush"],  # Note: 010_pen has no model_ids
        "office_supplies": ["048_stapler", "092_notebook", "095_glue", "117_whiteboard-eraser", "119_mini-chalkboard"],
        "containers": ["059_pencup"],
        "other": ["043_book", "104_board"]
    },
    "personal_care": {
        "hygiene": ["049_shampoo", "107_soap", "118_tooth-paste", "096_cleaner", "109_hydrating-oil"],
        "accessories": ["115_perfume"],
        "health": ["080_pillbottle"]
    },
    "sports_toys": {
        "sports": ["027_table-tennis"],
        "toys": ["057_toycar", "073_rubikscube", "081_playingcards"],
        "fitness": ["052_dumbbell", "013_dumbbell-rack"]
    },
    "home_accessories": {
        "decorative": ["012_plant-pot", "120_plant", "051_candlestick", "089_globe", "090_trophy", "085_gong"],
        "functional": ["045_sand-clock", "046_alarm-clock", "050_bell", "111_callbell", "074_displaystand",
                       "078_phonestand", "094_rest"],
        "other": ["023_tissue-box", "028_roll-paper", "026_pet-collar", "070_paymentsign", "100_seal"]
    },
    "miscellaneous": {
        "blocks": ["004_fluted-block", "086_woodenblock", "108_block"],
        "footwear": ["041_shoe"],
        "electronics_accessories": ["061_battery"]
    }
}
