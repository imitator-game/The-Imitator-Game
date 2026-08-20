import os
import trimesh
from pathlib import Path

mesh_dir = Path("./inspire_urdf/meshes")

# Process all STL files in the directory
for stl_file in mesh_dir.glob("*.STL"):
    try:
        # Load the mesh
        mesh = trimesh.load(stl_file)
        
        # Generate convex hull
        convex_hull = mesh.convex_hull
        
        # Save as .convex.stl
        output_path = stl_file.with_suffix('.STL.convex.stl')
        convex_hull.export(output_path)
        print(f"Generated convex hull: {output_path}")
    except Exception as e:
        print(f"Error processing {stl_file}: {e}")
