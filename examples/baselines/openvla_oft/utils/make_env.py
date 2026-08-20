import os
from pathlib import Path
from typing import Optional
import copy
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

try:
    from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
except Exception:
    L0_L3_utils = None



class SelectCameraObservationWrapper(gym.ObservationWrapper):
    """Keep only specified camera(s) in sensor_data to avoid mixed resolutions."""

    def __init__(self, env, camera_names: Optional[list[str]] = None):
        super().__init__(env)
        self.camera_names = camera_names or []
        if self.camera_names:
            obs = self.observation(copy.deepcopy(self.base_env._init_raw_obs))
            self.base_env.update_obs_space(obs)

    @property
    def base_env(self):
        return self.env.unwrapped

    def observation(self, observation):
        if not self.camera_names:
            return observation
        sensor_data = observation.get("sensor_data", {})
        sensor_param = observation.get("sensor_param", {})

        kept = {}
        kept_param = {}
        for name in self.camera_names:
            if name in sensor_data:
                kept[name] = sensor_data[name]
                if isinstance(sensor_param, dict) and name in sensor_param:
                    kept_param[name] = sensor_param[name]

        if not kept and sensor_data:
            first_name = next(iter(sensor_data))
            kept = {first_name: sensor_data[first_name]}
            if isinstance(sensor_param, dict) and first_name in sensor_param:
                kept_param = {first_name: sensor_param[first_name]}

        observation["sensor_data"] = kept
        if isinstance(sensor_param, dict):
            observation["sensor_param"] = kept_param
        return observation


_L_ENV_VAR_MAP = {
    "L1": "MANI_SKILL_L1",
    "L2": "MANI_SKILL_L2",
    "L3": "MANI_SKILL_L3",
}


def extract_base_env_name(env_id: str) -> str:
    if env_id.startswith("L") and "_" in env_id:
        parts = env_id.split("_", 1)
        if len(parts) == 2 and parts[0] in ["L0", "L1", "L2", "L3"]:
            level, base = parts[0], parts[1]
            if level == "L3":
                # New L3 tasks are registered as separate env ids, e.g.
                # L3_TwoRobotStirSpoon-v1 -> TwoRobotStirSpoonL3-v1
                if "-v" in base:
                    name_part, version_part = base.rsplit("-v", 1)
                    return f"{name_part}L3-v{version_part}"
                return f"{base}L3"
            return base
    return env_id


def extract_level(env_id: str) -> str:
    if env_id.startswith("L") and "_" in env_id:
        parts = env_id.split("_", 1)
        if len(parts) == 2 and parts[0] in ["L0", "L1", "L2", "L3"]:
            return parts[0]
    return "L0"


def parse_optional_bool_flag(raw: Optional[str]) -> Optional[bool]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"auto", "default", "none"}:
        return None
    if value in {"1", "true", "on", "yes", "enable", "enabled"}:
        return True
    if value in {"0", "false", "off", "no", "disable", "disabled"}:
        return False
    raise ValueError(f"Invalid boolean flag value: {raw}")


def configure_tabletop_task_flags(
    level: Optional[str] = None,
    lr_mirror_enabled: Optional[bool] = None,
    lr_mirror_robot_pose_enabled: Optional[bool] = None,
) -> None:
    if L0_L3_utils is None:
        return

    if level is not None:
        for env_var in _L_ENV_VAR_MAP.values():
            os.environ.pop(env_var, None)
        if level in {"L1", "L2"}:
            os.environ[_L_ENV_VAR_MAP[level]] = "1"

        L0_L3_utils.set_l1_enabled(level == "L1")
        L0_L3_utils.set_l2_enabled(level == "L2")
        # L3 is now handled by separate registered env classes under
        # mani_skill.envs.tasks.tabletop.dual_tasks_l3, not by the legacy
        # MANI_SKILL_L3 / set_l3_enabled switch on base dual_tasks envs.
        L0_L3_utils.set_l3_enabled(False)

    L0_L3_utils.set_lr_mirror_enabled(lr_mirror_enabled)
    L0_L3_utils.set_lr_mirror_robot_pose_enabled(lr_mirror_robot_pose_enabled)


def clear_tabletop_task_flags() -> None:
    if L0_L3_utils is None:
        return
    for env_var in _L_ENV_VAR_MAP.values():
        os.environ.pop(env_var, None)
    L0_L3_utils.set_l1_enabled(False)
    L0_L3_utils.set_l2_enabled(False)
    L0_L3_utils.set_l3_enabled(False)
    L0_L3_utils.set_lr_mirror_enabled(None)
    L0_L3_utils.set_lr_mirror_robot_pose_enabled(None)


def _set_eval_flags_in_subprocess(
    level: Optional[str] = None,
    lr_mirror_enabled: Optional[bool] = None,
    lr_mirror_robot_pose_enabled: Optional[bool] = None,
):
    """Set tabletop task flags inside a subprocess thunk.

    Root cause of the bug: forkserver creates a daemon process once during the
    first AsyncVectorEnv call. That daemon inherits env vars from that moment.
    Later env var changes in the main process do NOT propagate to workers forked
    from this daemon. So we must set env vars inside each worker thunk.
    """
    configure_tabletop_task_flags(
        level=level,
        lr_mirror_enabled=lr_mirror_enabled,
        lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
    )


def make_eval_envs(
    env_id,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    other_kwargs: dict,
    video_dir: Optional[str] = None,
    wrappers: list[gym.Wrapper] = [],
    l_level: Optional[str] = None,
    lr_mirror_enabled: Optional[bool] = None,
    lr_mirror_robot_pose_enabled: Optional[bool] = None,
):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments, each parallel environment records to its own
    subdirectory under ``video_dir`` so the total saved videos can match the total
    number of evaluated episodes. Callers may flatten these per-env directories
    afterwards if they want a single folder of mp4 files.
    For GPU vectorized environments all parallel environments are recorded through
    the vectorized wrapper itself.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
        l_level: the L-level string (e.g. "L0", "L1", "L2", "L3") to set
                 inside subprocess workers. Fixes forkserver env-var inheritance bug.
        lr_mirror_enabled: optional override for left-right scene mirror behavior.
        lr_mirror_robot_pose_enabled: optional override for swapping robot poses under mirror.
    """
    if sim_backend == "physx_cpu":

        def cpu_make_env(
            env_id, seed, video_dir=None, env_kwargs=dict(), other_kwargs=dict(),
            l_level=None, lr_mirror_enabled=None, lr_mirror_robot_pose_enabled=None,
        ):
            def thunk():
                if (
                    l_level is not None
                    or lr_mirror_enabled is not None
                    or lr_mirror_robot_pose_enabled is not None
                ):
                    _set_eval_flags_in_subprocess(
                        level=l_level,
                        lr_mirror_enabled=lr_mirror_enabled,
                        lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
                    )

                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        save_video=True,
                        info_on_video=True,
                        avoid_overwriting_video=True,
                        source_type="openvla_oft",
                        source_desc="openvla_oft evaluation rollout",
                    )
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk

        vector_cls = (
            gym.vector.SyncVectorEnv
            if num_envs == 1
            else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver")
        )
        env = vector_cls(
            [
                cpu_make_env(
                    env_id,
                    seed,
                    (
                        str(Path(video_dir) / f"env{seed}")
                        if video_dir and num_envs > 1
                        else video_dir
                    ),
                    env_kwargs,
                    other_kwargs,
                    l_level=l_level,
                    lr_mirror_enabled=lr_mirror_enabled,
                    lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
                )
                for seed in range(num_envs)
            ]
        )
    else:
        if (
            l_level is not None
            or lr_mirror_enabled is not None
            or lr_mirror_robot_pose_enabled is not None
        ):
            _set_eval_flags_in_subprocess(
                level=l_level,
                lr_mirror_enabled=lr_mirror_enabled,
                lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
            )

        env = gym.make(
            env_id,
            num_envs=num_envs,
            sim_backend=sim_backend,
            reconfiguration_freq=1,
            **env_kwargs
        )
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
        if video_dir:
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_trajectory=False,
                save_video=True,
                source_type="openvla_oft",
                source_desc="openvla_oft evaluation rollout",
                max_steps_per_video=max_episode_steps,
            )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env
