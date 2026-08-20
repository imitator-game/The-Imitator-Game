from typing import List
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actor_builder import ActorBuilder

from .common import *
from .sketchfab import create_sketchfab_actor, get_sketchfab_builder


def get_actor_builder(
    scene: ManiSkillScene,
    id: str,
    add_collision: bool = True,
    add_visual: bool = True,
    scales: List[float] = None,
) -> ActorBuilder:
    """Builds an actor or returns an actor builder given an ID specifying which dataset/source and then the actor ID

    Currently these IDs are hardcoded for a few datasets. The new Shapedex platform for hosting and managing all assets will be
    integrated in the future

    Args:
        scene: The ManiSkillScene. If building a custom task this is generally just self.scene
        id (str): The unique ID identifying the dataset and the ID of the actor in that dataset to build. The format should be
            "<dataset_id>:<actor_id_in_dataset>"
        add_collision (bool): Whether to include the collision shapes/meshes
        add_visual (bool): Whether to include visual shapes/meshes
    """
    splits = id.split(":")
    dataset_source = splits[0]
    actor_id = ":".join(splits[1:])

    if dataset_source == "ycb":
        from mani_skill.utils.building.actors.ycb import get_ycb_builder

        builder = get_ycb_builder(
            scene=scene, id=actor_id, add_collision=add_collision, add_visual=add_visual, scales=scales
        )
    elif dataset_source == "sketchfab":
        from mani_skill.utils.building.actors.sketchfab import get_sketchfab_builder

        builder = get_sketchfab_builder(
            scene=scene,
            object_key=actor_id,
            add_collision=add_collision,
            add_visual=add_visual,
            scales=scales,
        )
    else:
        raise RuntimeError(f"No dataset with id {dataset_source} was found")

    return builder
