"""
Updated evaluation script with language embedding support
Supports both precomputed and real-time language encoding
"""

from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from mani_skill.utils import common
from typing import Optional, Dict, List
import json
import os
import random

# Task to prompt mapping for evaluation
prompt2task_dict = {
    "pick red cube and place on plate.": "human_pick_red_cube_place_plate",
    "pick blue cube and place on plate.": "human_pick_blue_cube_place_plate",
    "pick yellow cup and place on plate.": "human_pick_cup_place_plate",
    "stack red cube on blue cube.": "human_stack_red_cube_on_blue_cube",
    "stack blue cube on red cube.": "human_stack_blue_cube_on_red_cube",
    "pick red cube and place on yellow cup.": "human_pick_red_cube_place_cup",
    "pick blue cube and place on yellow cup.": "human_pick_blue_cube_place_cup",
    "pick yellow cup and pour and place on plate.": "human_pour_cup",
    "": "human_pick_red_cube_place_plate"
}


def load_lang_descriptions(lang_desc_path: str) -> Dict:
    """Load language descriptions from JSON file"""
    if not os.path.exists(lang_desc_path):
        print(f"Warning: Language descriptions file not found: {lang_desc_path}")
        return {}

    with open(lang_desc_path, 'r') as f:
        data = json.load(f)
    return data.get('descriptions', {})


def get_test_language_for_prompt(prompt: str, lang_descriptions: Dict, use_robot_prompt: bool = False) -> str:
    """Get test language description for a given prompt

    Args:
        prompt: Environment prompt
        lang_descriptions: Loaded language descriptions
        use_robot_prompt: Whether to use robot_prompt if available

    Returns:
        Test language description
    """
    # Map prompt to task name
    task_name = prompt2task_dict.get(prompt, list(prompt2task_dict.values())[0])

    if task_name in lang_descriptions:
        task_videos = lang_descriptions[task_name]

        if len(task_videos) > 0:
            # Use first video's test description
            video_data = task_videos[0]

            # Try robot_prompt first if requested
            if use_robot_prompt and 'robot_prompt' in video_data:
                return video_data['robot_prompt']

            # Use test description
            if 'test' in video_data and len(video_data['test']) > 0:
                return video_data['test'][random.randint(0, len(video_data['test'])-1)]

            # Fallback to train description
            if 'train' in video_data and len(video_data['train']) > 0:
                return video_data['train'][random.randint(0, len(video_data['train'])-1)]

    # Final fallback to original prompt
    return prompt if prompt else "pick red cube and place on plate."


def prepare_language_embeddings(
        prompts: List[str],
        lang_descriptions: Dict,
        precomputed_lang,
        agent,
        use_robot_prompt: bool = True,
        use_test_descriptions: bool = True
) -> tuple:
    """Prepare language embeddings for evaluation

    Args:
        prompts: List of environment prompts
        lang_descriptions: Language descriptions dict
        precomputed_lang: Precomputed language embedding loader (can be None)
        agent: Agent with language encoding capability
        use_robot_prompt: Whether to prefer robot_prompt over test descriptions
        use_test_descriptions: Whether to use test descriptions (vs train)

    Returns:
        Tuple of (lang_embeds, lang_masks, used_texts)
    """
    # Get appropriate language texts for each prompt
    lang_texts = []
    for prompt in prompts:
        if use_test_descriptions:
            lang_text = get_test_language_for_prompt(prompt, lang_descriptions, use_robot_prompt)
        else:
            # Use training descriptions
            task_name = prompt2task_dict.get(prompt, list(prompt2task_dict.values())[0])
            if task_name in lang_descriptions and len(lang_descriptions[task_name]) > 0:
                video_data = lang_descriptions[task_name][0]
                if 'train' in video_data and len(video_data['train']) > 0:
                    lang_text = video_data['train'][random.randint(0, len(video_data['train'])-1)]
                else:
                    lang_text = prompt if prompt else "pick red cube and place on plate."
            else:
                lang_text = prompt if prompt else "pick red cube and place on plate."

        lang_texts.append(lang_text)

    # Encode language
    if precomputed_lang:
        # Try precomputed embeddings first
        try:
            lang_embeds, lang_masks = precomputed_lang.get_text_embeddings(lang_texts)
            # print(f"Using precomputed embeddings for {len(lang_texts)} texts")
            return lang_embeds, lang_masks, lang_texts
        except Exception as e:
            print(f"Warning: Precomputed embedding failed: {e}, falling back to real-time encoding")

    # Fallback to agent's real-time encoding
    if hasattr(agent, 'encode_language'):
        with torch.no_grad():
            lang_embeds, lang_masks = agent.encode_language(lang_texts)
            return lang_embeds, lang_masks, lang_texts
    else:
        # Last resort: use agent's text embedder directly
        with torch.no_grad():
            lang_embeds, lang_masks = agent.text_embedder.get_text_embeddings(lang_texts)
            return lang_embeds, lang_masks, lang_texts


def evaluate(
        n: int,
        agent,
        eval_envs,
        device,
        sim_backend: str,
        progress_bar: bool = True,
        val_videos=None,
        precomputed_lang=None,
        lang_desc_path: Optional[str] = None,
        use_test_descriptions: bool = True,
        use_robot_prompt: bool = True
):
    """Evaluate agent with language embedding support

    Args:
        n: Number of episodes to evaluate
        agent: Agent to evaluate
        eval_envs: Evaluation environments
        device: Device for computation
        sim_backend: Simulation backend
        progress_bar: Whether to show progress bar
        val_videos: Validation videos (for video-conditioned models)
        precomputed_lang: Precomputed language embedding loader
        lang_desc_path: Path to language descriptions JSON
        use_test_descriptions: Whether to use test descriptions for evaluation
        use_robot_prompt: Whether to prefer robot_prompt over test descriptions
    """
    agent.eval()

    # Load language descriptions
    lang_descriptions = {}
    if lang_desc_path:
        lang_descriptions = load_lang_descriptions(lang_desc_path)

    if progress_bar:
        pbar = tqdm(total=n)

    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0

        while eps_count < n:
            obs = common.to_tensor(obs, device)

            # Get environment prompts
            prompts = info.get('prompt', [''] * eval_envs.num_envs)
            if isinstance(prompts, str):
                prompts = [prompts] * eval_envs.num_envs

            # Prepare language embeddings for current prompts
            try:
                lang_embeds, lang_masks, used_texts = prepare_language_embeddings(
                    prompts,
                    lang_descriptions,
                    precomputed_lang,
                    agent,
                    use_robot_prompt=use_robot_prompt,
                    use_test_descriptions=use_test_descriptions
                )

                # Store language embeddings in agent for action generation
                # This is a temporary storage for the current evaluation step
                if hasattr(agent, '_eval_lang_cache'):
                    agent._eval_lang_cache = {
                        'embeddings': lang_embeds,
                        'masks': lang_masks,
                        'texts': used_texts
                    }

            except Exception as e:
                print(f"Warning: Language embedding preparation failed: {e}")
                # Continue with default language if embedding fails
                lang_embeds = None
                lang_masks = None
                used_texts = prompts

            # Get action sequence from agent
            if hasattr(agent, 'get_action_with_lang'):
                # Agent supports direct language input
                action_seq = agent.get_action_with_lang(obs, lang_embeds, lang_masks)
            elif hasattr(agent, 'get_action'):
                # Standard get_action method
                action_seq = agent.get_action(obs)
            else:
                raise AttributeError("Agent must have either get_action_with_lang or get_action method")

            # Convert to appropriate format for the simulation backend
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()

            # Execute action sequence
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break

            # Process episode completion
            if truncated.any():
                assert truncated.all() == truncated.any(), \
                    "all episodes should truncate at the same time for fair evaluation with other algorithms"

                # Collect metrics
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)

                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)

                # Reset environment for next episode
                obs, info = eval_envs.reset()

                # Update prompts for the new episodes
                prompts = info.get('prompt', [''] * eval_envs.num_envs)
                if isinstance(prompts, str):
                    prompts = [prompts] * eval_envs.num_envs

                # Call environment-specific setup if needed
                for env_idx, prompt in enumerate(prompts):
                    args = (prompt,)
                    eval_envs.unwrapped.call_async('get_objs_from_prompt', *args)
                    eval_envs.unwrapped.call_wait()

    agent.train()
    if progress_bar:
        pbar.close()

    # Convert metrics to numpy arrays
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])

    return eval_metrics


def evaluate_and_save_best(
        iteration: int,
        agent,
        eval_envs,
        device,
        sim_backend: str,
        val_videos=None,
        precomputed_lang=None,
        lang_desc_path: Optional[str] = None,
        num_eval_episodes: int = 50,
        eval_freq: int = 5000,
        best_eval_metrics=None,
        save_checkpoint_fn=None,
        run_name: str = "",
        writer=None,
        use_test_descriptions: bool = True
):
    """Evaluate agent and save best checkpoints during training

    Args:
        iteration: Current training iteration
        agent: Agent to evaluate
        eval_envs: Evaluation environments
        device: Device for computation
        sim_backend: Simulation backend
        val_videos: Validation videos
        precomputed_lang: Precomputed language embedding loader
        lang_desc_path: Path to language descriptions
        num_eval_episodes: Number of episodes to evaluate
        eval_freq: Evaluation frequency
        best_eval_metrics: Dict to track best metrics
        save_checkpoint_fn: Function to save checkpoints
        run_name: Experiment run name
        writer: Tensorboard writer
        use_test_descriptions: Whether to use test descriptions

    Returns:
        Updated best_eval_metrics dict
    """
    if best_eval_metrics is None:
        best_eval_metrics = defaultdict(float)

    if iteration % eval_freq == 0 and iteration > 0:
        print(f"\n{'=' * 60}")
        print(f"Evaluating at iteration {iteration}")
        print(f"{'=' * 60}")

        # Quick evaluation first to check if worth doing full eval
        quick_eval_metrics = evaluate(
            n=min(10, num_eval_episodes),
            agent=agent,
            eval_envs=eval_envs,
            device=device,
            sim_backend=sim_backend,
            progress_bar=True,
            val_videos=val_videos,
            precomputed_lang=precomputed_lang,
            lang_desc_path=lang_desc_path,
            use_test_descriptions=use_test_descriptions
        )

        # If quick eval shows promise (>30% success), do full evaluation
        quick_success = np.mean(quick_eval_metrics.get('success_at_end', [0]))
        if quick_success >= 0.3:
            eval_metrics = evaluate(
                n=num_eval_episodes,
                agent=agent,
                eval_envs=eval_envs,
                device=device,
                sim_backend=sim_backend,
                progress_bar=True,
                val_videos=val_videos,
                precomputed_lang=precomputed_lang,
                lang_desc_path=lang_desc_path,
                use_test_descriptions=use_test_descriptions
            )
        else:
            eval_metrics = quick_eval_metrics

        # Process and log metrics
        print(f"\nEvaluated {len(eval_metrics['success_at_end'])} episodes")

        metric_summary = {}
        for k in eval_metrics.keys():
            metric_value = np.mean(eval_metrics[k])
            metric_summary[k] = metric_value

            if writer:
                writer.add_scalar(f"eval/{k}", metric_value, iteration)

            print(f"  {k}: {metric_value:.4f}")

        # Save checkpoints for best metrics
        save_on_best_metrics = ["success_once", "success_at_end"]
        for metric_name in save_on_best_metrics:
            if metric_name in metric_summary:
                current_value = metric_summary[metric_name]

                if current_value > best_eval_metrics[metric_name]:
                    best_eval_metrics[metric_name] = current_value

                    if save_checkpoint_fn:
                        save_checkpoint_fn(run_name, f"best_eval_{metric_name}", iteration, best_eval_metrics)

                    print(f"  ✓ New best {metric_name}: {current_value:.4f} (saved checkpoint)")

        print(f"{'=' * 60}\n")

    return best_eval_metrics