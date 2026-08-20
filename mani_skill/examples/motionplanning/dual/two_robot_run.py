import multiprocessing as mp
import os
from copy import deepcopy
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
from mani_skill.examples.motionplanning.dual.solutions import *
from mani_skill.examples.motionplanning.dual.solutions_l3 import *
from mani_skill.examples.motionplanning.dual.solutions import solveTwoRobotPickCubeYCB
from mani_skill.examples.motionplanning.dual.solutions import solveTwoRobotPickWash
from mani_skill.examples.motionplanning.dual.solutions import solveTwoRobotCleanDesk
from mani_skill.examples.motionplanning.dual.solutions import solveTwoRobotTransFood
from mani_skill.examples.motionplanning.dual.solutions import solveTwoRobotPutBox
MP_SOLUTIONS = {
    "TwoRobotPickCubeYCB-v1": solveTwoRobotPickCubeYCB,
    "TwoRobotPickRemoteControl-v1": solveTwoRobotPickRemoteControl,
    "TwoRobotPickTennisBallGolfBall-v1": solveTwoRobotPickTennisBallGolfBall,
    "TwoRobotStirSpoon-v1": solveTwoRobotStirSpoon,
    "TwoRobotPourKettle-v1": solveTwoRobotPourKettle,
    "TwoRobotPickAppleBasket-v1": solveTwoRobotPickAppleBasket,
    "TwoRobotPlaceBookBookcase-v1": solveTwoRobotPlaceBookBookcase,
    "TwoRobotScanMilkBox-v1": solveTwoRobotScanMilkBox,
    "TwoRobotScanPillBottle-v1": solveTwoRobotScanPillBottle,
    "TwoRobotPlaceChipsRack-v1": solveTwoRobotPlaceChipsRack,
    "TwoRobotPlaceCommodityRack-v1": solveTwoRobotPlaceCommodityRack,
    "TwoRobotPlaceClothBasket-v1": solveTwoRobotPlaceClothBasket,
    "TwoRobotPickAppleBananaToBaskets-v1": solveTwoRobotPickAppleBananaToBaskets,
    "TwoRobotPickAppleToScale-v1": solveTwoRobotPickAppleToScale,
    "TwoRobotPickPillToRegions-v1": solveTwoRobotPickPillToRegions,
    "TwoRobotCutFruit-v1": solveTwoRobotCutFruit,
    "TwoRobotWipePot-v1": solveTwoRobotWipePot,
    "TwoRobotPickFood-v1": solveTwoRobotPickFood,
    "TwoRobotPutCubeOnScale-v1": solveTwoRobotPutCubeOnScale,
    "TwoRobotPourCup-v1": solveTwoRobotPourCup,
    "TwoRobotKnifeBowlFork-v1": solveTwoRobotKnifeBowlFork,
    "TwoRobotPickFruitsToPlate-v1": solveTwoRobotPickFruitsToPlate,
    "TwoRobotPlaceFoodScale-v1": solveTwoRobotPlaceFoodScale,
    "TwoRobotPlaceMagazineFolder-v1": solveTwoRobotPlaceMagazineFolder,
    "TwoRobotPlaceFileFolder-v1": solveTwoRobotPlaceFileFolder,
    "TwoRobotPressStapler-v1": solveTwoRobotPressStapler,
    "TwoRobotPlaceFruitBox-v1": solveTwoRobotPlaceFruitBox,
    "TwoRobotPourKetchupFries-v1": solveTwoRobotPourKetchupFries,
    "TwoRobotPlaceBrushRest-v1": solveTwoRobotPlaceBrushRest,
    "TwoRobotPourLiquidCup-v1": solveTwoRobotPourLiquidCup,
    "TwoRobotPlacePlateRack-v1": solveTwoRobotPlacePlateRack,
    "TwoRobotFoldBox-v1": solveTwoRobotFoldBox,
    "TwoRobotGrindFood-v1": solveTwoRobotGrindFood,
    "TwoRobotPlaceMugRack-v1": solveTwoRobotPlaceMugRack,
    "TwoRobotPlaceBurgerTray-v1": solveTwoRobotPlaceBurgerTray,
    "TwoRobotOpenBox-v1": solveTwoRobotOpenBox,
    "TwoRobotPourLiquidMug-v1": solveTwoRobotPourLiquidMug,
    "TwoRobotPourLiquidFilter-v1": solveTwoRobotPourLiquidFilter,
    "TwoRobotLiftLidFromSkillet-v1": solveTwoRobotLiftLidFromSkillet,
    "TwoRobotFoldTowel-v1": solveTwoRobotFoldTowel,
    "TwoRobotPlacePillBox-v1": solveTwoRobotPlacePillBox,
    "TwoRobotPlaceShoeBox-v1": solveTwoRobotPlaceShoeBox,
    "TwoRobotPlaceCupPlate-v1": solveTwoRobotPlaceCupPlate,
    "TwoRobotPlaceScrewdriver-v1": solveTwoRobotPlaceScrewdriver,
    "TwoRobotCleanCup-v1": solveTwoRobotCleanCup,
    "TwoRobotOpenLiquidCap-v1": solveTwoRobotOpenLiquidCap,
    "TwoRobotPressJuicer-v1": solveTwoRobotPressJuicer,
    "TwoRobotPickWash-v1": solveTwoRobotPickWash,
    "TwoRobotCleanDesk-v1": solveTwoRobotCleanDesk,
    "TwoRobotTransFood-v1": solveTwoRobotTransFood,
    "TwoRobotPutBox-v1": solveTwoRobotPutBox,

    "TwoRobotPickRemoteControlL3-v1": solveTwoRobotPickRemoteControlL3,
    "TwoRobotPickTennisBallGolfBallL3-v1": solveTwoRobotPickTennisBallGolfBallL3,
    "TwoRobotStirSpoonL3-v1": solveTwoRobotStirSpoonL3,
    "TwoRobotPourKettleL3-v1": solveTwoRobotPourKettleL3,
    "TwoRobotPickAppleBasketL3-v1": solveTwoRobotPickAppleBasketL3,
    "TwoRobotPlaceBookBookcaseL3-v1": solveTwoRobotPlaceBookBookcaseL3,
    "TwoRobotScanMilkBoxL3-v1": solveTwoRobotScanMilkBoxL3,
    "TwoRobotScanPillBottleL3-v1": solveTwoRobotScanPillBottleL3,
    "TwoRobotPlaceChipsRackL3-v1": solveTwoRobotPlaceChipsRackL3,
    "TwoRobotPlaceCommodityRackL3-v1": solveTwoRobotPlaceCommodityRackL3,
    "TwoRobotPlaceClothBasketL3-v1": solveTwoRobotPlaceClothBasketL3,
    "TwoRobotPickAppleBananaToBasketsL3-v1": solveTwoRobotPickAppleBananaToBasketsL3,
    "TwoRobotPickAppleToScaleL3-v1": solveTwoRobotPickAppleToScaleL3,
    "TwoRobotPickPillToRegionsL3-v1": solveTwoRobotPickPillToRegionsL3,
    "TwoRobotCutFruitL3-v1": solveTwoRobotCutFruitL3,
    "TwoRobotWipePotL3-v1": solveTwoRobotWipePotL3,
    "TwoRobotPickFoodL3-v1": solveTwoRobotPickFoodL3,
    "TwoRobotPutCubeOnScaleL3-v1": solveTwoRobotPutCubeOnScaleL3,
    "TwoRobotPourCupL3-v1": solveTwoRobotPourCupL3,
    "TwoRobotKnifeBowlForkL3-v1": solveTwoRobotKnifeBowlForkL3,
    "TwoRobotPickFruitsToPlateL3-v1": solveTwoRobotPickFruitsToPlateL3,
    "TwoRobotPlaceFoodScaleL3-v1": solveTwoRobotPlaceFoodScaleL3,
    "TwoRobotPlaceMagazineFolderL3-v1": solveTwoRobotPlaceMagazineFolderL3,
    "TwoRobotPlaceFileFolderL3-v1": solveTwoRobotPlaceFileFolderL3,
    "TwoRobotPressStaplerL3-v1": solveTwoRobotPressStaplerL3,
    "TwoRobotPlaceFruitBoxL3-v1": solveTwoRobotPlaceFruitBoxL3,
    "TwoRobotPourKetchupFriesL3-v1": solveTwoRobotPourKetchupFriesL3,
    "TwoRobotPlaceBrushRestL3-v1": solveTwoRobotPlaceBrushRestL3,
    "TwoRobotPourLiquidCupL3-v1": solveTwoRobotPourLiquidCupL3,
    "TwoRobotPlacePlateRackL3-v1": solveTwoRobotPlacePlateRackL3,
    "TwoRobotFoldBoxL3-v1": solveTwoRobotFoldBoxL3,
    "TwoRobotGrindFoodL3-v1": solveTwoRobotGrindFoodL3,
    "TwoRobotPlaceMugRackL3-v1": solveTwoRobotPlaceMugRackL3,
    "TwoRobotPlaceBurgerTrayL3-v1": solveTwoRobotPlaceBurgerTrayL3,
    "TwoRobotOpenBoxL3-v1": solveTwoRobotOpenBoxL3,
    "TwoRobotPourLiquidMugL3-v1": solveTwoRobotPourLiquidMugL3,
    "TwoRobotPourLiquidFilterL3-v1": solveTwoRobotPourLiquidFilterL3,
    "TwoRobotLiftLidFromSkilletL3-v1": solveTwoRobotLiftLidFromSkilletL3,
    "TwoRobotFoldTowelL3-v1": solveTwoRobotFoldTowelL3,
    "TwoRobotPlacePillBoxL3-v1": solveTwoRobotPlacePillBoxL3,
    "TwoRobotPlaceShoeBoxL3-v1": solveTwoRobotPlaceShoeBoxL3,
    "TwoRobotPlaceCupPlateL3-v1": solveTwoRobotPlaceCupPlateL3,
    "TwoRobotPlaceScrewdriverL3-v1": solveTwoRobotPlaceScrewdriverL3,
    "TwoRobotCleanCupL3-v1": solveTwoRobotCleanCupL3,
    "TwoRobotOpenLiquidCapL3-v1": solveTwoRobotOpenLiquidCapL3,
    "TwoRobotPressJuicerL3-v1": solveTwoRobotPressJuicerL3,
    "TwoRobotPickWashL3-v1": solveTwoRobotPickWashL3,
    "TwoRobotCleanDeskL3-v1": solveTwoRobotCleanDeskL3,
    "TwoRobotTransFoodL3-v1": solveTwoRobotTransFoodL3,
    "TwoRobotPutBoxL3-v1": solveTwoRobotPutBoxL3,
}
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help=f"Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS.keys())}")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgbd", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=10, help="Number of trajectories to generate.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--traj-name", type=str, help="The name of the trajectory .h5 file that will be created.")
    parser.add_argument("--shader", default="rt-fast", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="demos", help="where to save the recorded trajectories")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--L0", "--l0", dest="l0", action="store_true", help="Enable L0 scene augmentation (offset target objects).")
    parser.add_argument("--L1", "--l1", dest="l1", action="store_true", help="Enable L1 scene augmentation (offset target objects).")
    parser.add_argument("--L2", "--l2", dest="l2", action="store_true", help="Enable L2 scene augmentation (swap object model ids).")
    parser.add_argument("--L3", "--l3", dest="l3", action="store_true", help="Enable L3 scene augmentation (swap container model ids).")
    parser.add_argument(
        "--mirror-robot-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When LR mirror is enabled, also mirror robot root poses. "
            "Use --no-mirror-robot-pose to keep robots on original sides."
        ),
    )
    return parser.parse_args()

def _main(args, proc_id: int = 0, start_seed: int = 0) -> str:
    env_id = args.env_id
    # Reset per-process L-level / mirror overrides to avoid leaking global state across runs.
    L0_L3_utils.set_l1_enabled(False)
    L0_L3_utils.set_l2_enabled(False)
    L0_L3_utils.set_l3_enabled(False)
    # Mirror override:
    #   None  -> follow L2/L3 (default)
    #   False -> force-disable all mirror helper functions while keeping L2/L3 replacements
    #            (useful when BaseEnv mirror hook is commented out).
    L0_L3_utils.set_lr_mirror_enabled(None)
    # Robot pose mirror override (only used when LR mirror is enabled):
    #   True  -> legacy behavior (swap robot sides)
    #   False -> keep robot sides fixed while still mirroring scene objects.
    L0_L3_utils.set_lr_mirror_robot_pose_enabled(args.mirror_robot_pose)
    if args.l1:
        L0_L3_utils.set_l1_enabled(True)
    if args.l2:
        L0_L3_utils.set_l2_enabled(True)
    if args.l3:
        L0_L3_utils.set_l3_enabled(True)
    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend
    )
    if env_id not in MP_SOLUTIONS:
        raise RuntimeError(f"No already written motion planning solutions for {env_id}. Available options are {list(MP_SOLUTIONS.keys())}")

    if not args.traj_name:
        #new_traj_name = time.strftime("%Y%m%d_%H%M%S")
        if args.l0:
            new_traj_name = "l0_" + env_id
        elif args.l1:
            new_traj_name = "l1_" + env_id
        elif args.l2:
            new_traj_name = "l2_" + env_id
        elif args.l3:
            new_traj_name = "l3_" + env_id
    else:
        new_traj_name = args.traj_name

    if args.num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)
    env = RecordEpisode(
        env,
        output_dir=osp.join(args.record_dir, env_id, "motionplanning"),
        trajectory_name=new_traj_name, save_video=args.save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
        info_on_video=True,
    )
    output_h5_path = env._h5_file.filename
    solve = MP_SOLUTIONS[env_id]
    print(f"Motion Planning Running on {env_id}")
    pbar = tqdm(range(args.num_traj), desc=f"proc_id: {proc_id}")
    seed = start_seed
    left_successes = []
    right_successes = []
    left_solution_episode_lengths = []
    right_solution_episode_lengths = []
    failed_motion_plans = 0
    passed = 0
    while True:
        # try:
        left_res, right_res = solve(env, seed=seed, debug=False, vis=True if args.vis else False)
        eval_info = env.unwrapped.evaluate()
        task_success = eval_info["success"]
        if hasattr(task_success, "item"):
            task_success = task_success.item()
        print(f"Task success: {task_success}")
        print("Eval info:", env.unwrapped.evaluate())
        # except Exception as e:
        #     print(f"Cannot find valid solution because of an error in motion planning solution: {e}")
        #     left_res, right_res = (-1, -1)

        if left_res == -1 or right_res == -1:
            all_success = left_success = right_success = False
            failed_motion_plans += 1
        else:
            left_success = left_res[-1]["success"].item()
            right_success = right_res[-1]["success"].item()
            all_success = True
            left_elapsed_steps = left_res[-1]["elapsed_steps"].item()
            right_elapsed_steps = right_res[-1]["elapsed_steps"].item()
            left_solution_episode_lengths.append(left_elapsed_steps)
            right_solution_episode_lengths.append(right_elapsed_steps)
        left_successes.append(left_success)
        right_successes.append(right_success)
        if args.only_count_success and not task_success:
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
                    left_success_rate=np.mean(left_successes),
                    right_success_rate=np.mean(right_successes),
                    failed_motion_plan_rate=failed_motion_plans / (seed + 1),
                    left_avg_episode_length=np.mean(left_solution_episode_lengths),
                    right_avg_episode_length=np.mean(right_solution_episode_lengths),
                    left_max_episode_length=np.max(left_solution_episode_lengths),
                    right_max_episode_length=np.max(right_solution_episode_lengths),
                    # min_episode_length=np.min(solution_episode_lengths)
                )
            )
            seed += 1
            passed += 1
            if passed == args.num_traj:
                break
    env.close()
    return output_h5_path

def main(args):
    if args.num_procs > 1 and args.num_procs < args.num_traj:
        if args.num_traj < args.num_procs:
            raise ValueError("Number of trajectories should be greater than or equal to number of processes")
        args.num_traj = args.num_traj // args.num_procs
        seeds = [*range(0, args.num_procs * args.num_traj, args.num_traj)]
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), i, seeds[i]) for i in range(args.num_procs)]
        res = pool.starmap(_main, proc_args)
        pool.close()
        # Merge trajectory files
        output_path = res[0][: -len("0.h5")] + "h5"
        merge_trajectories(output_path, res)
        for h5_path in res:
            tqdm.write(f"Remove {h5_path}")
            os.remove(h5_path)
            json_path = h5_path.replace(".h5", ".json")
            tqdm.write(f"Remove {json_path}")
            os.remove(json_path)
    else:
        _main(args)

if __name__ == "__main__":
    # start = time.time()
    mp.set_start_method("spawn")
    main(parse_args())
    # print(f"Total time taken: {time.time() - start}")
