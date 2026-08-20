import multiprocessing as mp
import os
from copy import deepcopy
import time
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories

from mani_skill.examples.motionplanning.realman.solutions.rm_microwave import solveRealmanMicrowave

MP_SOLUTIONS = {
    "RealmanMicrowave-v1": solveRealmanMicrowave,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--env-id",
        type=str,
        default="RealmanMicrowave-v1",
        help="Environment to run motion planning solver on"
    )
    parser.add_argument(
        "-o", "--obs-mode",
        type=str,
        default="none",
        help="Observation mode to use"
    )
    parser.add_argument(
        "-n", "--num-traj",
        type=int,
        default=10,
        help="Number of trajectories to generate"
    )
    parser.add_argument(
        "--only-count-success",
        action="store_true",
        help="If true, generates trajectories until num_traj of them are successful"
    )
    parser.add_argument(
        "-b", "--sim-backend",
        type=str,
        default="auto",
        help="Which simulation backend to use"
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="rgb_array",
        help="Render mode"
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Whether to visualize the solution live"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Whether to save videos"
    )
    parser.add_argument(
        "--traj-name",
        type=str,
        help="Name of the trajectory file"
    )
    parser.add_argument(
        "--shader",
        default="default",
        type=str,
        help="Shader for rendering"
    )
    parser.add_argument(
        "--cam-width",
        type=int,
        default=None,
        help="Override camera sensor width (pixels)"
    )
    parser.add_argument(
        "--cam-height",
        type=int,
        default=None,
        help="Override camera sensor height (pixels)"
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="demos",
        help="Directory to save trajectories"
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=1,
        help="Number of parallel processes"
    )
    return parser.parse_args()


def _main(args, proc_id: int = 0, start_seed: int = 0) -> str:
    env_id = args.env_id

    # Create the environment
    sensor_cfg = dict(shader_pack=args.shader)
    if args.cam_width is not None:
        sensor_cfg["width"] = int(args.cam_width)
    if args.cam_height is not None:
        sensor_cfg["height"] = int(args.cam_height)

    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        sensor_configs=sensor_cfg,
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend
    )

    if env_id not in MP_SOLUTIONS:
        raise RuntimeError(f"No motion planning solution for {env_id}")

    if not args.traj_name:
        new_traj_name = time.strftime("%Y%m%d_%H%M%S") + "_microwave"
    else:
        new_traj_name = args.traj_name

    if args.num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)

    env = RecordEpisode(
        env,
        output_dir=osp.join(args.record_dir, env_id, "motionplanning"),
        trajectory_name=new_traj_name,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="Realman dual-arm microwave motion planning solution",
        video_fps=30,
        record_reward=False,
        save_on_reset=False
    )

    output_h5_path = env._h5_file.filename
    solve = MP_SOLUTIONS[env_id]

    print(f"Motion Planning Running on {env_id} with Realman dual-arm robot")
    print("Task: Open microwave and place object inside")

    pbar = tqdm(range(args.num_traj), desc=f"proc_id: {proc_id}")
    seed = start_seed
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    passed = 0

    while True:
        # try:
        res = solve(env, seed=seed, debug=True, vis=args.vis)
        # except Exception as e:
        #     print(f"Motion planning failed with error: {e}")
        #     res = -1

        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()
            solution_episode_lengths.append(elapsed_steps)

        successes.append(success)

        if args.only_count_success and not success:
            seed += 1
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            continue
        else:
            env.flush_trajectory()
            if args.save_video:
                env.flush_video()
            pbar.update(1)
            pbar.set_postfix(
                dict(
                    success_rate=np.mean(successes),
                    failed_motion_plan_rate=failed_motion_plans / (seed + 1),
                    avg_episode_length=np.mean(solution_episode_lengths) if solution_episode_lengths else 0,
                    max_episode_length=np.max(solution_episode_lengths) if solution_episode_lengths else 0,
                )
            )
            seed += 1
            passed += 1
            if passed == args.num_traj:
                break

    env.close()
    return output_h5_path


def main(args):
    if args.num_procs > 1:
        # Parallel processing logic...
        pass
    else:
        _main(args)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main(parse_args())
