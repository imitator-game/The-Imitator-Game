"""
XSkill Evaluation Utilities
Contains evaluation functions for XSkill models to make the package self-contained.
"""

import os
import pickle
import json
import random
import time
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import queue

import h5py
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
import omegaconf
import hydra

# ManiSkill imports
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils import common
from mani_skill.envs.tasks.tabletop import L0_L3_utils

prompt2task_dict: Dict[str, str] = {
    "pick red cube and place on plate.": "human_pick_red_cube_place_plate",
    "pick blue cube and place on plate.": "human_pick_blue_cube_place_plate",
    "pick yellow cup and place on plate.": "human_pick_cup_place_plate",
    "stack red cube on blue cube.": "human_stack_red_cube_on_blue_cube",
    "stack blue cube on red cube.": "human_stack_blue_cube_on_red_cube",
    "pick red cube and place on yellow cup.": "human_pick_red_cube_place_cup",
    "pick blue cube and place on yellow cup .": "human_pick_blue_cube_place_cup",
    "pick yellow cup and pour and place on plate.": "human_pour_cup",
    "": "human_pick_red_cube_place_plate"
}


def extract_base_env_name(env_id: str) -> str:
    """Extract base environment name from env_id.

    L0/L1/L2: 'L{n}_TwoRobotFoo-v1' -> 'TwoRobotFoo-v1'  (base env + flags)
    L3:        'L3_TwoRobotFoo-v1'   -> 'TwoRobotFooL3-v1' (separate registered env)
    """
    if env_id.startswith("L") and "_" in env_id:
        parts = env_id.split("_", 1)
        if len(parts) == 2 and parts[0] in ["L0", "L1", "L2", "L3"]:
            level, base = parts[0], parts[1]
            if level == "L3":
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


_L_ENV_VARS = {
    "L1": "MANI_SKILL_L1",
    "L2": "MANI_SKILL_L2",
    "L3": "MANI_SKILL_L3",
}


def set_l_level(level: str):
    """Set the L-level both via module globals AND via env var.

    The env var set here only affects the main process. For subprocess
    workers (forkserver), the level is propagated via the l_level parameter
    passed to make_eval_envs.
    """
    for env_var in _L_ENV_VARS.values():
        os.environ.pop(env_var, None)
    if level in _L_ENV_VARS:
        os.environ[_L_ENV_VARS[level]] = "1"

    L0_L3_utils.set_l1_enabled(False)
    L0_L3_utils.set_l2_enabled(False)
    L0_L3_utils.set_l3_enabled(False)
    if level == "L1":
        L0_L3_utils.set_l1_enabled(True)
    elif level == "L2":
        L0_L3_utils.set_l2_enabled(True)
    # L3: intentionally no flag — handled by the separate env class

def load_videos(video_path: str, task_name: Optional[str] = None) -> Dict:
    """Load video demonstrations from HDF5 files"""
    videos = {}
    if not os.path.exists(video_path):
        print(f"Warning: Video path not found: {video_path}")
        return videos
        
    video_files = [f for f in os.listdir(video_path) if f.endswith(".h5")]
    
    for video_file in video_files:
        t_name = video_file.replace(".h5", "")
        if task_name and t_name != task_name:
            continue
            
        hdf5_path = os.path.join(video_path, video_file)
        with h5py.File(hdf5_path, 'r') as h5f:
            num_videos = len(h5f)
            videos[t_name] = []
            for i in range(min(num_videos, 20)):
                try:
                    group = h5f[str(i)]
                    video = group["obs"][:]
                    videos[t_name].append(torch.tensor(video, dtype=torch.uint8))
                except (KeyError, AttributeError, TypeError):
                    continue
    
    return videos


def repeat_last_proto(encode_protos: torch.Tensor, eps_len: int) -> torch.Tensor:
    """Repeat last prototype to match episode length"""
    if len(encode_protos) >= eps_len:
        return encode_protos[:eps_len]
    
    rep_proto = encode_protos[-1].unsqueeze(0).repeat(eps_len - len(encode_protos), 1)
    return torch.cat([encode_protos, rep_proto])


def load_pretrain_xskill_model(pretrain_path: str, device: torch.device):
    """Load pretrained XSkill model for prototype extraction"""
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(f"XSkill checkpoint not found: {pretrain_path}")
    
    # Get the directory containing the checkpoint to find config
    pretrain_dir = os.path.dirname(pretrain_path)
    config_path = os.path.join(pretrain_dir, ".hydra/config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"XSkill config not found: {config_path}")
    
    pretrain_cfg = omegaconf.OmegaConf.load(config_path)
    model = hydra.utils.instantiate(pretrain_cfg.Model).to(device)
    
    checkpoint = torch.load(pretrain_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    print(f"✓ Loaded pretrained XSkill model from {pretrain_path}")
    return model


def make_eval_envs(
    env_id,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    other_kwargs: dict,
    video_dir: Optional[str] = None,
    wrappers: list = [],
    l_level: str = "L0",
):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "physx_cpu" or "physx_gpu"
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        other_kwargs: other kwargs including obs_horizon
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    # Import here to avoid circular dependencies
    from mani_skill.utils import gym_utils
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode
    
    base_env_name = extract_base_env_name(env_id)
    level = extract_level(env_id)
    set_l_level(level)

    if sim_backend == "physx_cpu":
        def cpu_make_env(env_id, seed, video_dir=None, env_kwargs=dict(), other_kwargs=dict(), l_level="L0"):
            def thunk():
                # Set l_level inside subprocess worker (forkserver fix)
                set_l_level(l_level)
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = FrameStack(env, num_stack=other_kwargs.get("obs_horizon", 2))
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
                        source_type="xskill",
                        source_desc="xskill evaluation rollout",
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
        envs = vector_cls(
            [
                cpu_make_env(
                    base_env_name,
                    seed,
                    video_dir if seed == 0 else None,
                    env_kwargs,
                    other_kwargs,
                    l_level=level,
                )
                for seed in range(num_envs)
            ]
        )
    else:
        # GPU backend
        envs = gym.make(
            base_env_name,
            num_envs=num_envs, 
            sim_backend=sim_backend, 
            reconfiguration_freq=1,
            **env_kwargs
        )
        max_episode_steps = gym_utils.find_max_episode_steps_value(envs)
        for wrapper in wrappers:
            envs = wrapper(envs)
        envs = FrameStack(envs, num_stack=other_kwargs.get("obs_horizon", 2))
        if video_dir:
            envs = RecordEpisode(
                envs,
                output_dir=video_dir,
                save_trajectory=False,
                save_video=True,
                source_type="xskill",
                source_desc="xskill evaluation rollout",
                max_steps_per_video=max_episode_steps,
            )
        envs = ManiSkillVectorEnv(envs, ignore_terminations=True, record_metrics=True)
    
    return envs


def evaluate_xskill_model(
    evaluator,
    eval_envs,
    num_episodes: int,
    progress_bar: bool = True
) -> Tuple[Dict, Dict]:
    """Main evaluation function for XSkill models"""
    if evaluator.nets is None:
        raise RuntimeError("Model not properly initialized")
        
    evaluator.nets.eval()
    
    pbar = None
    if progress_bar:
        pbar = tqdm(total=num_episodes)
    
    eval_metrics = defaultdict(list)
    task_metrics = defaultdict(lambda: defaultdict(list))
    eps_count = 0
    
    action_space = eval_envs.action_space
    dual_arm_action = isinstance(action_space, gym.spaces.Dict) and "panda_wristcam-0" in action_space.spaces

    with torch.no_grad():
        obs, info = eval_envs.reset()
        
        # Calculate prototypes once per episode before the action loop
        episode_proto_snaps = None
        task_descriptions = []  # Initialize to avoid unbound variable error
        
        while eps_count < num_episodes:
            # Calculate prototypes only at the start of each episode
            if episode_proto_snaps is None:
                # Get task descriptions from environment info
                task_descriptions = info.get('prompt', [''] * eval_envs.num_envs)
                if isinstance(task_descriptions, str):
                    task_descriptions = [task_descriptions] * eval_envs.num_envs
                
                # For each environment, get corresponding demo video and calculate prototypes
                proto_snaps = []
                for env_idx, task_desc in enumerate(task_descriptions):
                    # Map task description to video
                    task_name = prompt2task_dict[task_desc]
                    
                    if (evaluator.videos is not None and 
                        task_name in evaluator.videos and 
                        len(evaluator.videos[task_name]) > 0):
                        # Select a random demo video
                        demo_video = random.choice(evaluator.videos[task_name])

                        # Calculate prototypes from demo
                        _, traj_representation = evaluator.calculate_prototypes_from_demo(demo_video)

                        # Sample snapshot
                        proto_snap = evaluator.sample_snap(traj_representation)
                        proto_snaps.append(proto_snap)
                    else:
                        raise NotImplementedError(f'Task {task_name} not found in videos {evaluator.videos}')
                
                # Stack proto snaps for batch processing
                episode_proto_snaps = torch.stack([ps.unsqueeze(0) for ps in proto_snaps])
                episode_proto_snaps = episode_proto_snaps.reshape(eval_envs.num_envs, evaluator.args.snap_frames, -1)
            
            # Convert observations to tensor format
            obs_tensor = common.to_tensor(obs, evaluator.device)
            
            # Get action sequence using pre-calculated prototypes
            action_seq = evaluator.get_action_sequence(obs_tensor, episode_proto_snaps)
            
            # Convert to numpy for environment
            if evaluator.args.sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            
            # Execute action sequence
            truncated = torch.zeros(eval_envs.num_envs, dtype=torch.bool)
            for i in range(action_seq.shape[1]):
                step_action = action_seq[:, i]
                if dual_arm_action:
                    arm0_dim = action_space["panda_wristcam-0"].shape[-1]
                    arm1_dim = action_space["panda_wristcam-1"].shape[-1]
                    step_action = {
                        "panda_wristcam-0": step_action[:, :arm0_dim],
                        "panda_wristcam-1": step_action[:, arm0_dim:arm0_dim + arm1_dim],
                    }
                obs, rew, terminated, truncated, info = eval_envs.step(step_action)
                if truncated.any():
                    break
            
            # Process episode completion
            if truncated.any():
                assert truncated.all() == truncated.any(), \
                    "all episodes should truncate at the same time for fair evaluation with other algorithms"
                
                # Collect metrics
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].extend(v.float().cpu().numpy())
                        
                        # Also collect per-task metrics
                        for env_idx, task_desc in enumerate(task_descriptions):
                            task_metrics[task_desc][k].append(v[env_idx].float().cpu().numpy())
                else:
                    for env_idx, final_info in enumerate(info["final_info"]):
                        task_desc = task_descriptions[env_idx]
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                            task_metrics[task_desc][k].append(v)
                
                eps_count += eval_envs.num_envs
                if pbar is not None:
                    pbar.update(eval_envs.num_envs)
                
                # Reset environment and clear episode prototypes for next episode
                obs, info = eval_envs.reset()
                episode_proto_snaps = None  # Reset prototypes for next episode
    
    if pbar is not None:
        pbar.close()
    
    # Convert to numpy arrays and stack metrics
    final_eval_metrics = {}
    for k in eval_metrics.keys():
        if eval_metrics[k]:
            # Convert list of arrays to single stacked array
            stacked_data = []
            for item in eval_metrics[k]:
                if isinstance(item, np.ndarray):
                    stacked_data.extend(item.tolist() if item.ndim > 0 else [item.item()])
                else:
                    stacked_data.append(item)
            final_eval_metrics[k] = np.array(stacked_data)
        else:
            final_eval_metrics[k] = np.array([])
    
    # Convert task metrics
    final_task_metrics = {}
    for task_name in task_metrics:
        final_task_metrics[task_name] = {}
        for k in task_metrics[task_name]:
            if task_metrics[task_name][k]:
                # Convert list to numpy array
                final_task_metrics[task_name][k] = np.array(task_metrics[task_name][k])
            else:
                final_task_metrics[task_name][k] = np.array([])
    
    return final_eval_metrics, final_task_metrics
