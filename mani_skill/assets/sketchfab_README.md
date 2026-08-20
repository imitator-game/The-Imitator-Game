# Sketchfab / Hunyuan Asset Pipeline

This document describes the current GLB-based asset pipeline used for external 3D objects stored under the `sketchfab` asset namespace in ManiSkill.

The same pipeline is currently used for:

- assets downloaded from Sketchfab
- assets generated locally, such as Hunyuan-generated models

Even if an object is not from Sketchfab, it can still be registered and loaded through this pipeline as long as it follows the same directory structure and registry format.

## Asset Download Source

The currently used external GLB assets are also mirrored in the following Google Drive folder:

<https://drive.google.com/drive/folders/1W2Zmi3m_0pXwNnj8iwSR_5oBEc1Y-hwm?usp=sharing>

After downloading, each asset must be saved to the canonical local directory:

```text
~/.maniskill/data/sketchfab/objects/{object_key}/model.glb
```

Do not place the downloaded files directly under `Downloads` or under the project repository. The loader expects them under the `~/.maniskill/data/sketchfab/objects/` tree.

## Overview

The pipeline is centered around three components:

1. A local asset directory that stores GLB files.
2. A JSON registry that maps a stable `object_key` to metadata such as scale and source.
3. A loader in `mani_skill/utils/building/actors/sketchfab.py` that resolves the asset and builds an actor.

At runtime, tasks create these objects with `create_sketchfab_actor(...)`.

## Canonical Directory Layout

The recommended on-disk layout is:

```text
~/.maniskill/data/sketchfab/objects/{object_key}/model.glb
```

Example:

```text
~/.maniskill/data/sketchfab/objects/balance_scale/model.glb
```

Here:

- `{object_key}` is the registry key
- `model.glb` is the default filename expected by the loader

This layout keeps each object self-contained and avoids filename collisions across collaborators.

Example workflow:

1. Download an object directory or GLB from the Google Drive folder.
2. Create the local directory:

```text
~/.maniskill/data/sketchfab/objects/{object_key}/
```

3. Rename or place the file as:

```text
~/.maniskill/data/sketchfab/objects/{object_key}/model.glb
```

4. Add or update the registry entry in `mani_skill/assets/sketchfab_registry.json`.

## Registry File

Registry path:

`mani_skill/assets/sketchfab_registry.json`

Example entry:

```json
{
  "balance_scale": {
    "source_url": "",
    "source": "hunyuan",
    "name_on_sketchfab": "",
    "glb_filename": "model.glb",
    "scale": [0.35, 0.35, 0.35],
    "collision_mode": "nonconvex",
    "license": "CC Attribution",
    "notes": "Canonical path: ~/.maniskill/data/sketchfab/objects/balance_scale/model.glb"
  }
}
```

### Registry fields

- `object_key`
  - Not stored explicitly as a field.
  - It is the JSON key itself, for example `balance_scale`.

- `source_url`
  - Optional URL used by the loader if the local GLB is missing.
  - In practice, this only works if the URL points directly to a downloadable GLB file.

- `source`
  - Optional source tag such as `sketchfab` or `hunyuan`.
  - This is metadata only. The loader does not branch on it.

- `name_on_sketchfab`
  - Optional human-readable source name.
  - Can be left empty for non-Sketchfab assets.

- `glb_filename`
  - Filename inside the object directory.
  - Usually `model.glb`.

- `scale`
  - Default scale applied when the object is loaded.
  - Can be a single number or a 3-element list.

- `collision_mode`
  - Optional.
  - Supported values in the current implementation:
    - `convex`
    - `nonconvex`
  - Default is `convex`.

- `license`
  - Optional metadata field for attribution and reuse tracking.

- `notes`
  - Optional free-form notes.

## Resolution Logic

Implementation:

`mani_skill/utils/building/actors/sketchfab.py`

When `create_sketchfab_actor(...)` is called, the loader resolves the asset in this order:

1. Canonical path:

```text
~/.maniskill/data/sketchfab/objects/{object_key}/{glb_filename}
```

2. Fallback:
   any `*.glb` directly under:

```text
~/.maniskill/data/sketchfab/objects/{object_key}/
```

3. If no local GLB is found:
   try downloading from `source_url`

The file is then validated as a binary GLB by checking the file header (`glTF` magic).

## Creating an Actor in a Task

Typical usage:

```python
from mani_skill.utils.building.actors.sketchfab import create_sketchfab_actor

self.scale = create_sketchfab_actor(
    scene=self.scene,
    object_key="balance_scale",
    pose=self.scale_pose,
    name="balance_scale",
    is_static=True,
)
```

If you need to override the registry scale at runtime, use `get_sketchfab_builder(...)` with `scales=[sx, sy, sz]`.

## Collision Behavior

This is the most important practical detail.

In the current implementation, visual and collision geometry are both built from the same GLB file unless you manually simplify the mesh offline.

### `collision_mode = "convex"`

The loader uses:

```python
builder.add_multiple_convex_collisions_from_file(...)
```

This is the default mode.

Pros:

- usually faster than nonconvex collision
- safer for dynamic rigid bodies

Cons:

- often inaccurate for complex meshes
- especially poor for thin structures, open containers, and irregular AI-generated meshes

### `collision_mode = "nonconvex"`

The loader uses:

```python
builder.add_nonconvex_collision_from_file(...)
```

Pros:

- much closer to the original render mesh
- useful for static assets whose shape matters

Cons:

- much slower for dense meshes
- can become very unstable or very expensive for messy AI-generated topology
- generally not suitable for dynamic objects

### Recommendation

For Hunyuan-generated or other AI-generated meshes:

- use `nonconvex` only for static objects, and only as a short-term fix
- do not rely on the render mesh as collision if the topology is noisy or extremely dense

If a model causes severe slowdown or collision mismatch, the long-term solution is to provide a simplified collision representation instead of using the render mesh directly.

## Sketchfab Page URLs vs Direct Download URLs

The current downloader uses:

```python
urlretrieve(source_url, ...)
```

This means it expects a direct file URL.

A normal Sketchfab model page usually does not work, because:

- the page returns HTML, not a GLB file
- many downloads require clicking a format button in the browser
- some downloads require authentication or a redirect chain

If the source page is a normal Sketchfab landing page, automatic download will usually fail.

### Recommended workflow

If the model comes from Sketchfab:

1. Download the GLB manually.
2. Put it at:

```text
~/.maniskill/data/sketchfab/objects/{object_key}/model.glb
```

3. Add or update the registry entry.

If you happen to have a true direct GLB URL, you can store it in `source_url`.

If the model comes from the shared Google Drive instead:

1. Download the GLB or object folder from the Google Drive link above.
2. Save it under:

```text
~/.maniskill/data/sketchfab/objects/{object_key}/model.glb
```

3. Make sure `{object_key}` matches the registry entry.

## Team Workflow Recommendation

For multi-person collaboration, use `object_key` as the stable primary identifier.

Recommended rule:

- one object = one `object_key`
- one object directory = one folder under `~/.maniskill/data/sketchfab/objects/`
- one registry entry = one metadata record

This is preferable to indexing by remote asset ID only, because:

- it remains stable even if the source platform changes
- it lets you rename and document objects by task usage
- it works equally well for Sketchfab assets and Hunyuan-generated assets

## Current Limitations

The current pipeline does not yet support:

- separate visual and collision GLB files
- primitive collision definitions in the registry
- automatic generation of clean collision meshes
- robust download from interactive Sketchfab web pages

These limitations matter most for AI-generated meshes, where render geometry is often unsuitable for collision.

## Practical Guidance for AI-Generated Assets

If a Hunyuan-generated object looks good visually but behaves badly in simulation:

1. First try `collision_mode: "convex"` if speed matters more than geometric fidelity.
2. Try `collision_mode: "nonconvex"` only if the object is static and the mesh is not too large.
3. If both are unsatisfactory, prepare a separate simplified collision asset or approximate the object with primitive shapes.

For complex objects such as balance scales, racks, or open containers, a simplified collision model is usually the correct long-term solution.

## Troubleshooting

### `Unknown sketchfab object_key`

The key is missing from `mani_skill/assets/sketchfab_registry.json`.

### `Found file but it is not a valid .glb`

The file is not a real binary GLB, or an HTML page was downloaded instead of the model.

### Collision is badly mismatched

The collision mesh is currently being generated from the same GLB as the visual mesh. This is common with AI-generated topology. Try changing `collision_mode`, but expect to eventually need a simplified collision asset.

### Simulation becomes very slow after switching to `nonconvex`

This is expected for dense or messy meshes. Triangle-mesh collision is expensive, especially during repeated collision checks in motion planning.

### Object scale is wrong

Update the `scale` field in `mani_skill/assets/sketchfab_registry.json` first, then adjust task-specific pose offsets if needed.
