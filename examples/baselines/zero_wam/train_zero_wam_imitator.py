"""Train Zero-WAM. Edit DEFAULTS below or override them on the command line."""

import os

from .zero_wam.config import ZeroWAMConfig

# Common editable defaults. All paths refer to files OUTSIDE this Git repository.
# CLI flags take precedence; model/data/cache paths also accept environment variables.
DEFAULTS = ZeroWAMConfig(
    protocol="scratch",  # scratch / c15 / c30 / c45
    train_scope="joint",
    batch_size=1,  # one COMPLETE episode on ONE GPU
    effective_batch_size=4,  # four GPUs, no accumulation
    total_epochs=60,
    learning_rate=1e-4,
    lr_schedule="cosine",
    min_learning_rate=1e-6,
    sampling_mode="episode",
    action_per_frame=16,
    robot_frame_stride=4,
    enable_mcp=True,
    activation_checkpointing=False,
    task=None,  # full protocol; use --task for a single-task pilot
    episode_count=None,  # use the episode split declared by the protocol
    save_epoch_freq=10,
    model_path=os.environ.get("ZERO_WAM_MODEL", ""),
    empty_text_embedding=os.environ.get("ZERO_WAM_EMPTY_TEXT", ""),
    human_root=os.environ.get("HUMAN_ROOT", ""),
    sim_root=os.environ.get("SIM_ROOT", ""),
    cache_dir=os.environ.get("ZERO_WAM_CACHE", ""),
    run_dir=os.environ.get("ZERO_WAM_RUN_DIR", ""),
    resume_from=None,  # full-state continuation
    initialize_from=None,  # new stage: adapter init, fresh optimizer
)


def main():
    import tyro

    config = tyro.cli(ZeroWAMConfig, default=DEFAULTS)
    from .zero_wam.training import run

    run(config)


if __name__ == "__main__":
    main()
