import os
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

def scan_model_ids(objects_dir: str) -> Dict[str, List[int]]:
    """
    Scan all items' model_ids in the specified directory

    Args:
        objects_dir: path to the robotwin objects directory

    Returns:
        A dict mapping each item name to its list of available model_ids
    """
    model_ids = {}
    
    # Iterate over all subdirectories in the objects directory
    for item_dir in os.listdir(objects_dir):
        item_path = os.path.join(objects_dir, item_dir)
        
        # Make sure it is a directory and matches the item naming convention (e.g., 001_bottle)
        if os.path.isdir(item_path) and re.match(r'^\d{3}_', item_dir):
            available_ids = []
            
            # Find all model_data{id}.json files
            for file in os.listdir(item_path):
                match = re.match(r'^model_data(\d+)\.json$', file)
                if match:
                    model_id = int(match.group(1))
                    available_ids.append(model_id)
            
            # If model_ids were found, sort and store them
            if available_ids:
                available_ids.sort()
                model_ids[item_dir] = available_ids
    
    return model_ids

def update_robotwin_items_with_model_ids(objects_dir: str):
    """
    Update the robotwin_items list by adding model_id info to each item
    """
    # Scan all model_ids
    model_ids_dict = scan_model_ids(objects_dir)
    
    # The original robotwin_items list
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
        "089_globe", "090_trophy", "091_kettle",
        "092_notebook", "093_brush-pen", "094_rest", "095_glue",
        "096_cleaner", "097_screen", "098_speaker", "099_fan", "100_seal",
        "101_milk-tea", "102_roller", "103_fruit", "104_board", "105_sauce-can",
        "106_skillet", "107_soap", "108_block", "109_hydrating-oil", "110_basket",
        "111_callbell", "112_tea-box", "113_coffee-box", "114_bottle", "115_perfume",
        "116_keyboard", "117_whiteboard-eraser", "118_tooth-paste", "119_mini-chalkboard", "120_plant"
    ]
    
    # Create a new data structure that includes model_id info
    robotwin_items_with_model_ids = {}
    
    for item in robotwin_items:
        if item in model_ids_dict:
            robotwin_items_with_model_ids[item] = {
                "name": item,
                "model_ids": model_ids_dict[item],
                "default_model_id": model_ids_dict[item][0]  # Use the first one by default
            }
        else:
            print(f"Warning: no model_data file found for {item}")
            robotwin_items_with_model_ids[item] = {
                "name": item,
                "model_ids": [],
                "default_model_id": None
            }
    
    return robotwin_items_with_model_ids

def generate_updated_dataset_code(objects_dir: str):
    """
    Generate the updated dataset code, including model_id info
    """
    # Get item info with model_ids
    items_with_model_ids = update_robotwin_items_with_model_ids(objects_dir)
    
    # Generate new Python code
    code = """# Auto-generated Robotwin dataset config (includes model_id info)
ROBOTWIN_OBJAVERSE_DATASET = dict()

# Robotwin item config (includes available model_ids)
ROBOTWIN_ITEMS_CONFIG = {
"""
    
    # Add config for each item
    for item_name, item_info in items_with_model_ids.items():
        model_ids_str = str(item_info['model_ids'])
        default_id = item_info['default_model_id']
        code += f'    "{item_name}": {{"model_ids": {model_ids_str}, "default": {default_id}}},\n'
    
    code += "}\n\n"
    
    # Add a helper function
    code += """# Helper function: get a valid model_id for an item
def get_model_id(item_name: str, preferred_id: int = None) -> int:
    \"\"\"
    Get the model_id for the given item
    
    Args:
        item_name: the item name (e.g. '054_baguette')
        preferred_id: the preferred model_id; if not given, the default is used
        
    Returns:
        A valid model_id
        
    Raises:
        ValueError: if the item does not exist or the model_id is invalid
    \"\"\"
    if item_name not in ROBOTWIN_ITEMS_CONFIG:
        raise ValueError(f"Item {item_name} does not exist")
    
    config = ROBOTWIN_ITEMS_CONFIG[item_name]
    available_ids = config["model_ids"]
    
    if not available_ids:
        raise ValueError(f"Item {item_name} has no available model_id")
    
    if preferred_id is not None:
        if preferred_id in available_ids:
            return preferred_id
        else:
            print(f"Warning: item {item_name} does not support model_id={preferred_id}; using the default value {config['default']}")
            return config['default']
    
    return config['default']

# Usage examples:
# model_id = get_model_id('054_baguette')  # Use the default model_id
# model_id = get_model_id('054_baguette', 3)  # Try to use model_id=3
"""
    
    return code

def main(output_path=None):
    """
    Main function: scan the directory and generate the updated config

    Args:
        output_path: the output file path; if not given, the default path is used
    """
    # Set the robotwin objects directory path
    objects_dir = os.path.expanduser("~/.maniskill/data/robotwin/objects")
    
    # Check whether the directory exists
    if not os.path.exists(objects_dir):
        print(f"Error: directory {objects_dir} does not exist")
        print("Please make sure the Robotwin dataset is installed correctly")
        return
    
    # Scan and display results
    print("Scanning the Robotwin objects directory...")
    model_ids = scan_model_ids(objects_dir)
    
    print(f"\nFound model_id info for {len(model_ids)} items:")
    for item, ids in sorted(model_ids.items())[:10]:  # Show only the first 10 as an example
        print(f"  {item}: model_ids = {ids}")
    print("  ...")
    
    # Generate the updated config code
    print("\nGenerating the updated config code...")
    updated_code = generate_updated_dataset_code(objects_dir)
    
    # Set the output path
    if output_path is None:
        output_file = "/mani_skill/utils/building/actors/robotwin_model_ids.py"
    else:
        output_file = output_path
    
    # If the path starts with /, it may need to be converted to a relative path or a path in the current project
    if output_file.startswith("/mani_skill"):
        # Try relative to the current directory
        current_dir = os.getcwd()
        if "mani_skill" in current_dir:
            # Find the mani_skill root directory
            root_idx = current_dir.find("mani_skill")
            mani_skill_root = current_dir[:root_idx + len("mani_skill")]
            output_file = os.path.join(mani_skill_root, output_file.replace("/mani_skill/", ""))
        else:
            # Use a relative path
            output_file = "." + output_file
    
    # Ensure the target directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Creating directory: {output_dir}")
        except PermissionError:
            print(f"Error: no permission to create directory {output_dir}")
            print("Try running with sudo, or specify a different output path")
            return
    
    # Write the file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(updated_code)
        print(f"\nConfig saved to {output_file}")
    except PermissionError:
        print(f"Error: no permission to write the file {output_file}")
        print("Try running with sudo, or specify a different output path")
        return
    
    # Show usage examples
    print("\nUsage examples:")
    print("```python")
    print("from mani_skill.utils.building.actors.robotwin_model_ids import ROBOTWIN_ITEMS_CONFIG, get_model_id")
    print("")
    print("# Get the appropriate model_id when creating an object")
    print("robotwin_model_id = '054_baguette'")
    print("model_id = get_model_id(robotwin_model_id)  # Get the default model_id")
    print("")
    print("robotwin_obj = create_actor(")
    print("    scene=self.scene,")
    print("    pose=initial_pose,")
    print("    modelname=robotwin_model_id,")
    print("    convex=True,")
    print("    model_id=model_id,")
    print(")")
    print("```")

if __name__ == "__main__":
    import sys
    
    # Support specifying the output path via a command-line argument
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
        print(f"Using the specified output path: {output_path}")
        main(output_path)
    else:
        main()