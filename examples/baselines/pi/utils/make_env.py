import os
from typing import Optional
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


_L_ENV_VAR_MAP = {
    "L1": "MANI_SKILL_L1",
    "L2": "MANI_SKILL_L2",
    "L3": "MANI_SKILL_L3",
}


def _set_tabletop_flags_in_subprocess(
    level: Optional[str] = None,
    lr_mirror_enabled: Optional[bool] = None,
    lr_mirror_robot_pose_enabled: Optional[bool] = None,
):
    """Set tabletop env vars and module globals inside a subprocess thunk.

    Root cause of the bug: forkserver creates a daemon process once during the
    first AsyncVectorEnv call. That daemon inherits env vars from that moment.
    Later env var changes in the main process do NOT propagate to workers forked
    from this daemon. So we must set env vars/module globals inside each worker
    thunk.
    """
    # Clear all L-level env vars first
    for env_var in _L_ENV_VAR_MAP.values():
        os.environ.pop(env_var, None)
    # Set the active level
    if level is not None and level in _L_ENV_VAR_MAP:
        os.environ[_L_ENV_VAR_MAP[level]] = "1"
    # Also set module globals in this subprocess so both paths agree
    from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
    L0_L3_utils.set_l1_enabled(level == "L1")
    L0_L3_utils.set_l2_enabled(level == "L2")
    L0_L3_utils.set_l3_enabled(level == "L3")
    L0_L3_utils.set_lr_mirror_enabled(lr_mirror_enabled)
    L0_L3_utils.set_lr_mirror_robot_pose_enabled(lr_mirror_robot_pose_enabled)


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
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
        l_level: the L-level string (e.g. "L0", "L1", "L2", "L3") to set
                 inside subprocess workers. Fixes forkserver env-var inheritance bug.
        lr_mirror_enabled: optional override for tabletop left-right mirror.
        lr_mirror_robot_pose_enabled: optional override for mirroring robot poses.
    """
    if sim_backend == "physx_cpu":

        def cpu_make_env(
            env_id, seed, video_dir=None, env_kwargs=dict(), other_kwargs=dict(),
            l_level=None, lr_mirror_enabled=None, lr_mirror_robot_pose_enabled=None,
        ):
            def thunk():
                _set_tabletop_flags_in_subprocess(
                    l_level,
                    lr_mirror_enabled,
                    lr_mirror_robot_pose_enabled,
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
                        info_on_video=True,
                        source_type="pi_05",
                        source_desc="pi_05 evaluation rollout",
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
                    video_dir if seed == 0 else None,
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
        # For GPU backend, also set flags before gym.make.
        _set_tabletop_flags_in_subprocess(
            l_level,
            lr_mirror_enabled,
            lr_mirror_robot_pose_enabled,
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
                source_type="pi_05",
                source_desc="pi_05 evaluation rollout",
                max_steps_per_video=max_episode_steps,
            )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env
