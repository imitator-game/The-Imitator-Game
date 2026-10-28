"""Evaluate Zero-WAM. Edit DEFAULTS or use CLI overrides."""

from .zero_wam.evaluation import EvalArgs

# Runtime paths must be outside the public source repository.
# None selects the checkpoint's model/data settings and the standard IG eval split.
DEFAULTS = EvalArgs(
    checkpoint="",  # external checkpoints/final.pt
    output_dir="",  # external metrics/diagnostics directory
    video_dir=None,  # optional external flat video directory; defaults to output_dir/videos
    eval_config=None,  # optional environment-list .txt
    human_config=None,
    sim_config=None,
    task_mapping_file=None,
    human_task_description_file=None,
    sim_task_description_file=None,
    num_episodes=10,
    max_episode_steps=500,
    seed=100000,
    icl_guidance_scale=5.0,  # --icl-guidance-scale5 selects native human-video CFG.
    capture_video=False,  # ordinary evaluation: metrics only; --capture-video enables MP4
    trajectory_metrics=False,
)


def main():
    import tyro

    args = tyro.cli(EvalArgs, default=DEFAULTS)
    if not args.checkpoint or not args.output_dir:
        raise ValueError("Set checkpoint and output_dir in DEFAULTS or via CLI")
    from .zero_wam.evaluation import run

    run(args)


if __name__ == "__main__":
    main()
