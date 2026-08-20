import inspect
from typing import Iterable

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

DUAL_TASK_CAMERA_NAMES = ("cam1", "cam2", "cam3", "zed2i")


def make_dual_task_sensor_configs(
    cam_width: int = 640,
    cam_height: int = 480,
    zed_width: int = 1280,
    zed_height: int = 720,
    camera_names: Iterable[str] | None = None,
):
    """Return the shared 4-view camera layout used by tabletop dual tasks."""
    import numpy as np

    selected = set(iter_dual_task_cameras(camera_names))
    pose1 = sapien_utils.look_at(eye=[0.3, 0.3, 0.6], target=[-0.1, 0, -0.6])
    pose2 = sapien_utils.look_at(eye=[0.1, 0.0, 0.6], target=[0.0, 0, 0.0])
    pose3 = sapien_utils.look_at(eye=[0.3, -0.3, 0.6], target=[-0.1, 0, -0.6])
    pose4 = sapien_utils.look_at(eye=[-0.4, 0, 0.6], target=[0.0, 0, 0.2])
    configs = [
        CameraConfig("cam1", pose1, cam_width, cam_height, np.pi / 2, 0.01, 100),
        CameraConfig("cam2", pose2, cam_width, cam_height, np.pi / 2, 0.01, 100),
        CameraConfig("cam3", pose3, cam_width, cam_height, np.pi / 2, 0.01, 100),
        CameraConfig("zed2i", pose4, zed_width, zed_height, np.pi * 1.1 / 2, 0.01, 100),
    ]
    return [config for config in configs if config.uid in selected]


def _is_dual_task_env_class(obj) -> bool:
    if not inspect.isclass(obj) or not issubclass(obj, BaseEnv):
        return False
    module_name = getattr(obj, "__module__", "")
    return module_name.startswith(
        "mani_skill.envs.tasks.tabletop.dual_tasks._"
    ) or module_name.startswith("mani_skill.envs.tasks.tabletop.dual_tasks_l3._")


def patch_dual_task_camera_defaults(
    cam_width: int = 640,
    cam_height: int = 480,
    zed_width: int = 1280,
    zed_height: int = 720,
    camera_names: Iterable[str] | None = None,
) -> list[str]:
    """
    Monkey patch all 50 dual-task envs and all 50 standalone L3 envs in-process.

    This avoids hand-editing 100 task files just to enable the shared 4-view layout.
    """
    import mani_skill.envs.tasks.tabletop as tabletop_tasks

    def _default_sensor_configs(self):
        return make_dual_task_sensor_configs(
            cam_width=cam_width,
            cam_height=cam_height,
            zed_width=zed_width,
            zed_height=zed_height,
            camera_names=camera_names,
        )

    patched = []
    seen_ids = set()
    for _, obj in inspect.getmembers(tabletop_tasks, inspect.isclass):
        if not _is_dual_task_env_class(obj):
            continue
        if id(obj) in seen_ids:
            continue
        obj._default_sensor_configs = property(_default_sensor_configs)
        seen_ids.add(id(obj))
        patched.append(f"{obj.__module__}.{obj.__name__}")
    patched.sort()
    return patched


def configure_dual_task_level(level: str, mirror_robot_pose: bool = True) -> None:
    """
    Reproduce the same global L-level switches used by two_robot_run.py.

    L3 standalone envs should also call this with level="L3"; it clears legacy
    base-env L3 flags while keeping the runtime state deterministic.
    """
    normalized = level.upper()
    if normalized not in {"L0", "L1", "L2", "L3"}:
        raise ValueError(f"Unsupported level: {level}")

    L0_L3_utils.set_l1_enabled(False)
    L0_L3_utils.set_l2_enabled(False)
    L0_L3_utils.set_l3_enabled(False)
    L0_L3_utils.set_lr_mirror_enabled(None)
    L0_L3_utils.set_lr_mirror_robot_pose_enabled(mirror_robot_pose)

    if normalized == "L1":
        L0_L3_utils.set_l1_enabled(True)
    elif normalized == "L2":
        L0_L3_utils.set_l2_enabled(True)
    elif normalized == "L3":
        # New standalone L3 envs are separate registered classes.
        L0_L3_utils.set_l3_enabled(False)


def iter_dual_task_cameras(cameras: Iterable[str] | None = None) -> tuple[str, ...]:
    if cameras is None:
        return DUAL_TASK_CAMERA_NAMES
    selected = tuple(cameras)
    invalid = sorted(set(selected) - set(DUAL_TASK_CAMERA_NAMES))
    if invalid:
        raise ValueError(f"Unsupported camera names: {invalid}")
    return selected
