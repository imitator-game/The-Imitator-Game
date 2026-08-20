# Finetune config used for single node post-training.
from dataclasses import dataclass
from typing import Optional

from gr00t.data.embodiment_tags import EmbodimentTag


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    dataset_path: str
    """Path to the dataset root directory containing trajectory data for fine-tuning."""

    embodiment_tag: EmbodimentTag
    """Identifier specifying which embodiment (robot configuration) this fine-tuning run targets."""

    base_model_path: str = "models/GR00T-N1.6-3B"
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """

    lerobot_version: str = "auto"
    """LeRobot dataset layout version: 'auto', 'v2', or 'v3'."""

    language_source: str = "task"
    """Language source: 'task', 'human_desc', or 'sim_desc'."""

    task_mapping_path: Optional[str] = None
    """Optional override for task_mapping.json."""

    human_desc_path: Optional[str] = None
    """Optional override for human_desc.json."""

    sim_desc_path: Optional[str] = None
    """Optional override for sim_desc.json."""

    human_config_path: Optional[str] = None
    """Optional path to a human-task config JSON used to expand a parent dataset directory."""

    sim_config_path: Optional[str] = None
    """Optional path to a sim-task config JSON used to expand a parent dataset directory."""

    robot_config_path: Optional[str] = None
    """Optional path to a robot-task config JSON used to expand a parent dataset directory."""

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_top_llm_layers: int = 4
    """Number of top Eagle LLM layers to full-finetune. Set 0 to strictly freeze the LLM."""

    use_backbone_lora: bool = False
    """If True, apply LoRA adapters to the Eagle vision backbone."""

    backbone_lora_rank: int = 64
    """LoRA rank for the Eagle vision backbone when use_backbone_lora is enabled."""

    use_llm_lora: bool = False
    """If True, apply LoRA adapters to the Eagle language model."""

    llm_lora_rank: int = 64
    """LoRA rank for the Eagle language model when use_llm_lora is enabled."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    use_action_head_diffusion_lora: bool = False
    """If True, apply LoRA adapters to the action head diffusion model."""

    action_head_diffusion_lora_rank: int = 64
    """LoRA rank for the action head diffusion model when use_action_head_diffusion_lora is enabled."""

    state_dropout_prob: float = 0.0
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """
    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    shortest_image_edge: int | None = None
    """Resize the shortest image edge before VLM processing. None keeps the model default."""

    crop_fraction: float | None = None
    """Fractional center/random crop used by albumentations image transforms. None keeps the model default."""

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total effective batch size across all GPUs and accumulation steps."""

    batch_size: Optional[int] = None
    """Per-device batch size. If set, overrides global_batch_size."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    gradient_accumulation_steps: int = 1
    """Number of forward passes to accumulate before performing a backward/update step."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    resume_from_checkpoint: Optional[str] = None
    """Checkpoint directory to resume optimizer/trainer state from. If None, start a fresh run."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 30
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d6`.
    You need to login to wandb to view the logs.
    """

    logging_steps: int = 100
    """Frequency, in optimizer steps, for trainer logging and wandb metric updates."""

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    epoch_based_training: bool = False
    """If True, interpret training length and save cadence in epochs instead of optimizer steps."""

    num_epochs: int = 1
    """Number of epochs to train when epoch_based_training is enabled."""

    save_epochs: int = 1
    """Save a checkpoint every N epochs when epoch_based_training is enabled."""

    fp16: bool = False
    """Enable fp16 training."""

    bf16: bool = True
    """Enable bf16 training."""

    gradient_checkpointing: bool = False
    """Enable gradient checkpointing to reduce activation memory."""

    enable_online_eval: bool = False
    """Enable ManiSkill online rollout evaluation during training."""

    online_eval_env_id: Optional[str] = None
    """Single task environment id used for online evaluation."""

    online_eval_steps: int = 0
    """Run online evaluation every N optimizer steps when greater than 0."""

    online_eval_epochs: int = 1
    """Run online evaluation every N epochs when epoch_based_training is enabled."""

    online_eval_num_episodes: int = 10
    """Number of episodes per online evaluation run."""

    online_eval_num_envs: int = 1
    """Number of parallel environments to use for online evaluation."""

    online_eval_max_episode_steps: int = 400
    """Maximum episode length for online evaluation."""

    online_eval_sim_backend: str = "physx_cpu"
    """Simulation backend for online evaluation."""

    online_eval_control_mode: str = "pd_joint_pos"
    """Control mode for online evaluation."""

    online_eval_obs_mode: str = "rgb"
    """Observation mode for online evaluation."""

    online_eval_reward_mode: str = "sparse"
    """Reward mode for online evaluation."""

    online_eval_shader: str = "rt-fast"
    """Shader pack for online evaluation rendering."""

    online_eval_capture_video: bool = False
    """If True, record videos during online evaluation."""
