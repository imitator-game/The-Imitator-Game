import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode
from h5py import File, Group, Dataset

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PushCube-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

def main(args: Args):
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs,
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=gym_utils.find_max_episode_steps_value(env))

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    count = 0

    def load_content_from_h5_file(file):
        if isinstance(file, (File, Group)):
            return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
        elif isinstance(file, Dataset):
            return file[()]
        else:
            raise NotImplementedError(f"Unspported h5 file type: {type(file)}")

    file = File("/home/output_3000.h5", "r")
    keys = list(file.keys())
    # keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
    ret = {key: load_content_from_h5_file(file[key]) for key in keys}
    file.close()

    batch_id = 1

    io_arm_qpos_idx = [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]

    while count < len(ret['io_teleop_joint_states'][f'batch_{batch_id}']['data']['position']['data']) - 1:
        # action = env.action_space.sample() if env.action_space is not None else None
        image = ret['_zed2i_zed_node_rgb_image_rect_color_compressed'][f'batch_{batch_id}']['images']['images']

        arm_action = ret['io_teleop_joint_states'][f'batch_{batch_id}']['data']['position']['data'][count+1][io_arm_qpos_idx] -\
                    ret['io_teleop_joint_states'][f'batch_{batch_id}']['data']['position']['data'][count][io_arm_qpos_idx]
        action = env.action_space.sample() * 0.0
        # action[0] = arm_action[0]
        # action[0:6] = arm_action[0:6]
        # action[18:24] = arm_action[6:12]
        obs, reward, terminated, truncated, info = env.step(action)

        """ For absolute pose control """
        base_qpos = ret['io_teleop_target_base_move'][f'batch_{batch_id}']['data']['data']['data'][count]
        arm_qpos = ret['io_teleop_joint_states'][f'batch_{batch_id}']['data']['position']['data'][count][io_arm_qpos_idx]
        finger_qpos = ret['io_teleop_joint_states'][f'batch_{batch_id}']['data']['position']['data'][count][14:28]
        io2sapien_arm_qpos_idx = [0,6,1,7,2,8,3,9,4,10,5,11]
        io2sapien_finger_qpos_idx = []
        arm_qpos = arm_qpos[io2sapien_arm_qpos_idx]
        target_qpos = np.zeros(len(env.agent.robot.active_joints))
        target_qpos[5:17] = arm_qpos
        env.agent.robot.set_qpos(target_qpos)
        env.agent.robot.set_qvel(np.zeros(len(env.agent.robot.active_joints)))
        """ For delta pose control """
        if args.render_mode is not None:
            env.render()
        count += 1
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
