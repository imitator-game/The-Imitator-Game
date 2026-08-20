from __future__ import annotations


def build_precomputed_vl_expected_metadata(args) -> dict:
    expected = {
        "format_version": 1,
        "vision_encoder": args.vision_encoder,
        "text_encoder": args.text_encoder,
        "t5_version": args.t5_version,
        "max_lang_len": args.max_lang_len,
        "obs_horizon": args.obs_horizon,
        "pred_horizon": args.pred_horizon,
        "use_lerobot": args.use_lerobot,
        "lerobot_use_paired_dataset": args.lerobot_use_paired_dataset,
        "lerobot_image_size": list(args.lerobot_image_size) if args.use_lerobot else None,
        "lerobot_video_backend": args.lerobot_video_backend if args.use_lerobot else None,
        "lerobot_state_type": args.lerobot_state_type if args.use_lerobot else None,
        "lerobot_human_dataset_file": args.lerobot_human_dataset_file if args.lerobot_use_paired_dataset else None,
        "lerobot_sim_dataset_file": args.lerobot_sim_dataset_file if args.use_lerobot else None,
        "lerobot_task_mapping_file": args.lerobot_task_mapping_file if args.lerobot_use_paired_dataset else None,
        "lerobot_human_task_description_file": (
            args.lerobot_human_task_description_file if args.lerobot_use_paired_dataset else None
        ),
        "lerobot_sim_task_description_file": args.lerobot_sim_task_description_file if args.use_lerobot else None,
        "demo_path": args.demo_path if not args.use_lerobot else None,
    }

    expected_mode = getattr(args, "expected_precomputed_vl_mode", None)
    if expected_mode is None:
        expected_mode = getattr(args, "expected_precomputed_img_mode", None)

    if expected_mode == "language_only":
        # Language-only caches are keyed by description text and can be reused by
        # smaller dataset configs as long as they are generated from the same
        # text encoder and description source. The exact train config and sim
        # config may legitimately differ when using a superset cache. Vision
        # encoder config is also irrelevant in this mode because img tokens are
        # still computed online at train/eval time.
        for key in (
            "vision_encoder",
            "obs_horizon",
            "pred_horizon",
            "lerobot_image_size",
            "lerobot_video_backend",
            "lerobot_state_type",
            "lerobot_human_dataset_file",
            "lerobot_sim_dataset_file",
            "lerobot_task_mapping_file",
            "lerobot_sim_task_description_file",
            "demo_path",
        ):
            expected.pop(key, None)
    elif expected_mode == "image_only":
        # Image-only caches are keyed by sample_id and do not depend on T5 or
        # description text. Dataset configs may be subsets of a larger cache
        # (e.g. 30/15 reuse the 45-task image cache); missing sample_ids are
        # checked at batch load time. Keep only feature-shape/preprocessing
        # fields strict here.
        for key in (
            "text_encoder",
            "t5_version",
            "max_lang_len",
            "lerobot_human_dataset_file",
            "lerobot_sim_dataset_file",
            "lerobot_task_mapping_file",
            "lerobot_human_task_description_file",
            "lerobot_sim_task_description_file",
            "demo_path",
        ):
            expected.pop(key, None)

    return expected
