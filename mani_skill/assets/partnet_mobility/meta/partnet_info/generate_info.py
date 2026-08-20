import os
import json
import numpy as np
from pathlib import Path

def calculate_scale(bbox_path, target_size=1.0):
    """
    Compute the scale factor from a bounding box so the model is normalized to the target size.
    """
    try:
        with open(bbox_path, 'r') as f:
            bbox = json.load(f)
        
        min_coords = np.array(bbox['min'])
        max_coords = np.array(bbox['max'])
        
        # Compute the dimensions of the bounding box
        dimensions = max_coords - min_coords
        
        # Get the maximum dimension
        max_dimension = np.max(dimensions)
        
        # Compute the scale factor so the maximum dimension matches the target size
        scale = target_size / max_dimension if max_dimension > 0 else 1.0
        
        # Keep 3 decimal places
        return round(scale, 3)
    except:
        # Return the default value if the file cannot be read or the scale cannot be computed
        return 0.3

def count_movable_links(mobility_v2_path):
    """
    Count the number of movable links from mobility_v2.json.
    """
    try:
        with open(mobility_v2_path, 'r') as f:
            mobility_data = json.load(f)
        
        # Count the number of non-free joints
        movable_count = 0
        for part in mobility_data:
            joint_type = part.get('joint', '')
            # Count movable joints such as slider, hinge, continuous, etc.
            if joint_type and joint_type != 'free' and joint_type != 'fixed':
                movable_count += 1
        
        # If there are movable parts, return the movable part count + 1 (including the base)
        # If there are no movable parts, return 1
        return movable_count + 1 if movable_count > 0 else 1
    except:
        return 1

def process_category(category_path, category_name):
    """
    Process a single category and generate info_xxx.json.
    """
    info_dict = {}
    
    # Get all asset folders under this category
    asset_folders = []
    for item in os.listdir(category_path):
        item_path = os.path.join(category_path, item)
        # Only process folders, excluding .zip files and other files
        if os.path.isdir(item_path) and item.isdigit():
            asset_folders.append(item)
    
    # Sort by ID
    asset_folders.sort(key=int)
    
    print(f"Processing {category_name}: {len(asset_folders)} assets")
    
    for asset_id in asset_folders:
        asset_path = os.path.join(category_path, asset_id)
        
        # Get the necessary file paths
        bbox_path = os.path.join(asset_path, 'bounding_box.json')
        mobility_v2_path = os.path.join(asset_path, 'mobility_v2.json')
        
        # Compute the required data
        scale = calculate_scale(bbox_path)
        num_links = count_movable_links(mobility_v2_path)
        
        # Add to the dictionary
        info_dict[asset_id] = {
            "num_target_links": num_links,
            "partnet_mobility_id": int(asset_id),
            "scale": scale
        }
    
    # Save to a json file
    output_filename = f"info_{category_name.lower()}.json"
    output_path = os.path.join(category_path, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(info_dict, f, indent=4)
    
    print(f"Saved {output_filename} with {len(info_dict)} entries")
    return len(info_dict)

def main():
    """
    Main function: iterate over all categories in the dataset directory.
    """
    dataset_path = "dataset"
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} directory not found!")
        return
    
    # Get all category folders
    categories = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            categories.append(item)
    
    categories.sort()
    print(f"Found {len(categories)} categories: {', '.join(categories)}")
    print("-" * 50)
    
    total_assets = 0
    
    # Process each category
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        asset_count = process_category(category_path, category)
        total_assets += asset_count
        print("-" * 50)
    
    print(f"\nTotal assets processed: {total_assets}")

if __name__ == "__main__":
    main()