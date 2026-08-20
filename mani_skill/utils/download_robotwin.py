from huggingface_hub import snapshot_download
from pathlib import Path
from mani_skill import ASSET_DIR

local_dir = ASSET_DIR/"robotwin"
snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    allow_patterns=["objects.zip", "background_texture.zip"],
    local_dir=str(local_dir),
    repo_type="dataset",
    resume_download=True,
)