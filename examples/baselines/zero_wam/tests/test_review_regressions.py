"""Regression checks for cache identity, checkpoint lifecycle and video export."""

from contextlib import ExitStack
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from easydict import EasyDict
import numpy as np
import torch

from examples.baselines.zero_wam.zero_wam import cache
from examples.baselines.zero_wam.zero_wam import evaluation
from examples.baselines.zero_wam.zero_wam import training
from examples.baselines.zero_wam.zero_wam.config import ZeroWAMConfig


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_manifest_tracks_fps_and_data_roots(self):
        config = ZeroWAMConfig(
            task="L0_TwoRobotPickRemoteControl-v1",
            episode_count=2,
            model_path="/external/foundation",
            human_root="/external/human",
            sim_root="/external/sim",
            cache_dir=str(self.root / "cache"),
        )

        def records(contract, root):
            return {k: [{"source": {"episode_length": 17}, "expected_latent_frames": 2}] for k in ["sim", "human"]}, {}

        with (
            patch.object(cache, "encoder_identity", side_effect=lambda path: {"path": path}),
            patch.object(cache, "make_records", side_effect=records),
            patch.object(cache, "validate_sources"),
        ):
            cache.build_manifest(config)
            for field, value in [
                ("human_fps", 6.0),
                ("human_root", "/external/different_human"),
                ("sim_root", "/external/different_sim"),
            ]:
                with self.subTest(field=field):
                    changed = deepcopy(config)
                    setattr(changed, field, value)
                    path = cache.build_manifest(changed)
                    actual = cache.read_json(cache.read_json(path)["contract_path"])
                    observed = (
                        actual["video"]["human"]["target_fps"] if field == "human_fps" else actual["paths"][field]
                    )
                    self.assertEqual(observed, value)
            changed = deepcopy(config)
            changed.manifest = str(cache.build_manifest(config))
            changed.human_fps = 6.0
            with self.assertRaisesRegex(ValueError, "encoding|configuration"):
                cache.build_manifest(changed)

    def test_action_tail_policy_invalidates_existing_episode_manifest(self):
        config = ZeroWAMConfig(task="L0_TwoRobotPickRemoteControl-v1",episode_count=2,
            model_path="/external/foundation",human_root="/external/human",sim_root="/external/sim",
            cache_dir=str(self.root/"cache"))
        request = config.cache_request_identity()
        old = dict(selection_identity=config.selection_identity(),cache_request_identity="pre_tail_fix")
        path = self.root/"old_manifest.json"
        path.write_text(json.dumps(old))
        config.manifest = str(path)
        with self.assertRaisesRegex(ValueError,"encoding settings"):
            cache.require_complete(config)
        with patch.object(cache,"encoder_identity",return_value={"model":"fixture"}), \
             patch.object(cache,"make_records",return_value=({k:[{"source":{"episode_length":169},"expected_latent_frames":23}] for k in ["sim","human"]},{})):
            config.manifest = None
            path = cache.build_manifest(config)
            manifest = cache.read_json(path)
            contract = cache.read_json(manifest["contract_path"])
        self.assertEqual(contract["video"]["sim"]["action_tail_policy"],"all_actions_v1")
        self.assertEqual(manifest["cache_request_identity"],request)
        # The contract is part of the payload namespace/fingerprint, so old
        # latent files cannot be accepted as new complete-action-tail caches.
        old_video = deepcopy(contract["video"])
        del old_video["sim"]["action_tail_policy"]
        self.assertNotEqual(cache.digest_json(old_video),manifest["data_contract_sha256"])

    def test_initialized_run_can_resume_without_reinitializing(self):
        settings = EasyDict(
            ig_resume="/resume.pt", ig_initialization=None, world_size=1, rank=0, save_root=str(self.root),
            batch_size=1, ig_effective_batch=1, ig_total_epochs=2, ig_sampling_mode="episode"
        )
        old = dict(settings)
        old["ig_resume"] = None
        old["ig_initialization"] = "/original_stage.pt"
        scheduler = Mock()
        scheduler.state_dict.return_value = {"last_epoch": 1}
        trainer = SimpleNamespace(
            config=settings,
            transformer=Mock(),
            optimizer=Mock(),
            lr_scheduler=scheduler,
            ig_sampler=Mock(),
            train_loader=SimpleNamespace(dataset=list(range(32))),
            device="cpu",
            ig_adaptation={},
        )
        state = {
            "native_config": old,
            "adapter_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {"last_epoch": 1},
            "global_step": 1,
            "progress": {"epoch": 0, "next_batch": 1},
            "rank_rng_states": [{}],
            "adaptation": {"initialized_from": "/original_stage.pt"},
        }
        with (
            patch.object(training.torch, "load", return_value=state) as load,
            patch.object(training, "set_model_state_dict"),
            patch.object(training, "set_optimizer_state_dict") as optimizer,
            patch.object(training, "restore_rng") as rng,
        ):
            training.resume(trainer)
        self.assertEqual(trainer.step, 1)
        self.assertEqual(trainer.ig_progress, {"epoch": 0, "next_batch": 1, "sample_offset": 1})
        self.assertEqual(trainer.config.ig_initialization, "/original_stage.pt")
        self.assertEqual(load.call_count, 1)
        self.assertEqual(load.call_args.args[0], "/resume.pt")
        optimizer.assert_called_once()
        rng.assert_called_once()

    def test_extended_training_refreshes_final_without_mutating_old_snapshot(self):
        (self.root / "checkpoints").mkdir()
        trainer = SimpleNamespace(
            config=EasyDict(world_size=1, rank=0, save_root=str(self.root), num_steps=1, save_epoch_freq=5),
            transformer=Mock(),
            optimizer=Mock(),
            device="cpu",
            ig_trainable_names=["w"],
            ig_adaptation={},
            lr_scheduler=Mock(),
            step=1,
            ig_progress={"epoch": 0, "next_batch": 1},
            ig_last_saved_epoch=0,
        )
        trainer.lr_scheduler.state_dict.return_value = {}

        def gather(states, local):
            states[0] = local

        with (
            patch.object(training, "get_model_state_dict", return_value={"w": torch.ones(1)}),
            patch.object(training, "get_optimizer_state_dict", return_value={}),
            patch.object(training, "rng_state", return_value={}),
            patch.object(training.dist, "all_gather_object", side_effect=gather),
            patch.object(training.dist, "barrier"),
        ):
            training.save_checkpoint(trainer)
            os.link(self.root / "checkpoints/final.pt", self.root / "checkpoints/historical.pt")
            trainer.config.num_steps = 2
            trainer.step = 2
            training.save_checkpoint(trainer)
        read = lambda name: torch.load(self.root / "checkpoints" / name, weights_only=False)["global_step"]
        self.assertEqual(read("latest.pt"), 2)
        self.assertEqual(read("final.pt"), 2)
        self.assertEqual(read("historical.pt"), 1)

    def evaluate_fixture(
        self,
        count,
        capture=True,
        stored_overrides=None,
        saved_resources=None,
        args_overrides=None,
        probe_resources=False,
    ):
        output = self.root / f"eval_{count}_{capture}"
        checkpoint_file = self.root / "checkpoint.pt"
        if not checkpoint_file.exists():
            checkpoint_file.write_bytes(b"fixture checkpoint identity")
        envlist = self.root / "envs.txt"
        envlist.write_text("L0_TwoRobotPickRemoteControl-v1\n")
        mapping = self.root / "mapping.json"
        mapping.write_text(
            json.dumps(
                {"task_mappings": [{"human_task_id": "human_H4", "sim_task_id": ["L0_TwoRobotPickRemoteControl-v1"]}]}
            )
        )
        config = ZeroWAMConfig(task_mapping_file=str(mapping))
        for name, value in (stored_overrides or {}).items():
            setattr(config, name, value)
        checkpoint = {
            "format_version": 3,
            "native_trainer": True,
            "native_config": {"ig_app_config": asdict(config)},
            "adaptation": {"trainable_shapes": {}},
            "adapter_state_dict": {},
        }
        if saved_resources is not None:
            checkpoint["native_config"]["ig_code_resources"] = saved_resources

        class Agent:
            adaptation_report = {"trainable_shapes": {}}
            vae = None
            action_space = None

            def __init__(self, *args, **kwargs):
                self.gripper_history_corrections = torch.zeros(2, dtype=torch.long)
                self.gripper_history_max_correction = torch.zeros(2)
                self.total_executed_steps = 0

            def load_adapter(self, *args):
                pass

            def eval(self):
                return self

            def requires_grad_(self, *args):
                return self

            def clear_cache(self):
                pass

        class Recorder:
            def __init__(self, output, *args):
                self.output = Path(output)

            def finish(self, *args):
                self.output.mkdir(parents=True, exist_ok=True)
                np.savez(self.output / "actual_rgb.npz", frames=np.zeros((17, 8, 8, 3), dtype=np.uint8))
                np.savez(self.output / "predicted_chunk_000.npz", frames=np.ones((4, 8, 8, 3), dtype=np.uint8) * 100)
                (self.output / "manifest.json").write_text(
                    json.dumps(
                        {
                            "executed_steps": 16,
                            "fps": 30,
                            "robot_frame_stride": 4,
                            "chunks": [{"chunk": 0, "start_step": 0, "action_steps": 16}],
                        }
                    )
                )

        from examples.baselines.diffusion_policy import eval_dp_imitator as shared
        from examples.baselines.lerobot_dataset import evaluate_processor as processors
        from examples.baselines.zero_wam.zero_wam import policy

        dataset_args = {}
        humans, robots = ZeroWAMConfig(task="L0_TwoRobotPickRemoteControl-v1").selections()
        for kind, rows in [("human", humans), ("sim", robots)]:
            path = self.root / (kind + "_selection.json")
            path.write_text(json.dumps(rows))
            dataset_args[kind + "_config"] = str(path)

        args = evaluation.EvalArgs(
            checkpoint=str(checkpoint_file),
            output_dir=str(output),
            video_dir=str(output / "videos"),
            eval_config=str(envlist),
            num_episodes=count,
            capture_video=capture,
            trajectory_metrics=False,
            **dataset_args,
            **(args_overrides or {}),
        )

        def prepare(config, **kwargs):
            self.prepared_config = config
            if probe_resources:
                config.selections()
                for name in ["task_mapping_file", "human_task_description_file", "sim_task_description_file"]:
                    self.assertTrue(Path(getattr(config, name)).is_file())
            return self.root / "manifest.json"

        def load(path, **kwargs):
            return checkpoint if str(path) == str(checkpoint_file) else {"latent": torch.zeros(48, 5, 14, 14)}

        with ExitStack() as stack:
            stack.enter_context(patch.object(cache, "prepare", side_effect=prepare))
            stack.enter_context(
                patch.object(
                    cache,
                    "read_json",
                    return_value={"human_episodes": [{"repo_id": "human_H4", "cache_path": "/human.pt"}]},
                )
            )
            stack.enter_context(patch.object(policy, "ZeroWAMAgent", Agent))
            stack.enter_context(patch.object(evaluation, "PreviewDiagnostics", Recorder))
            stack.enter_context(patch.object(processors, "HumanVideoSimEvaluateProcessor", return_value=Mock()))
            stack.enter_context(
                patch.object(processors, "HumanVideoSimEvaluateProcessorConfig", side_effect=lambda **kw: None)
            )
            stack.enter_context(patch.object(shared, "make_eval_envs_with_level", return_value=Mock()))
            stack.enter_context(
                patch.object(
                    __import__("examples.baselines.zero_wam.zero_wam.rollout", fromlist=["evaluate_with_task_encoder"]),
                    "evaluate_with_task_encoder",
                    return_value={"success_once": [True], "success_at_end": [True], "return": [1.0]},
                )
            )
            stack.enter_context(patch.object(shared, "clear_l_level"))
            stack.enter_context(patch.object(evaluation.torch, "load", side_effect=load))
            stack.enter_context(patch.object(evaluation.torch, "autocast", side_effect=lambda *a, **kw: nullcontext()))
            stack.enter_context(patch("examples.baselines.zero_wam.zero_wam.runtime.setup_runtime"))
            stack.enter_context(patch("examples.baselines.zero_wam.zero_wam.runtime.start_log"))
            evaluation.run(args)
        return output

    def test_metrics_only_does_not_export_video(self):
        output = self.evaluate_fixture(1, capture=False)
        self.assertFalse((output / "videos").exists())
        self.assertTrue((output / "summary.json").is_file())
        self.assertEqual(evaluation.EvalArgs(checkpoint="/a", output_dir="/b").capture_video, False)

    def test_parallel_rollout_start_preserves_global_seed_and_output_index(self):
        output = self.evaluate_fixture(1, capture=False, args_overrides={"episode_start": 7})
        rows = json.loads((output / "metrics.json").read_text())["results"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 100007)
        self.assertTrue((output / "L0_TwoRobotPickRemoteControl-v1/episode_007/metrics.json").exists())
        self.assertEqual(json.loads((output / "protocol.json").read_text())["episode_start"], 7)

    def test_one_rollout_exports_video(self):
        output = self.evaluate_fixture(1)
        movies = list((output / "videos").glob("*.mp4"))
        self.assertEqual(len(movies), 1)
        self.assertGreater(movies[0].stat().st_size, 0)

    def test_odd_rollout_exports_final_unpaired_video(self):
        output = self.evaluate_fixture(3)
        movies = list((output / "videos").glob("*.mp4"))
        self.assertEqual(len(movies), 2)
        self.assertTrue(all(p.stat().st_size > 0 for p in movies))

    def test_eval_worker_uses_private_manifest_and_preserves_payload_cache(self):
        paths = []
        for worker in range(2):
            runtime = self.root / f"local_worker{worker}"
            with patch.dict(os.environ, {"IG_RUNTIME_CACHE_ROOT": str(runtime)}):
                self.evaluate_fixture(
                    1, capture=False, probe_resources=True,
                    args_overrides={"cache_dir": str(self.root / "shared_payloads")},
                )
            config = self.prepared_config
            path = Path(config.manifest)
            self.assertTrue(path.is_relative_to(runtime / "vae_manifests"))
            self.assertEqual(config.cache_dir, str(self.root / "shared_payloads"))
            self.assertEqual(path.name, f"{config.protocol}_{config.cache_request_identity()[:16]}.json")
            paths.append(path)
        self.assertNotEqual(paths[0], paths[1])

    def relocated_resources(self):
        return {
            "task_mapping_file": "/previous_checkout/examples/baselines/lerobot_dataset/task_mapping.json",
            "human_task_description_file": "/previous_checkout/examples/baselines/lerobot_dataset/task_desc/human_desc.json",
            "sim_task_description_file": "/previous_checkout/examples/baselines/lerobot_dataset/task_desc/sim_desc.json",
        }

    def test_eval_resolves_legacy_default_resources_in_current_checkout(self):
        self.evaluate_fixture(1, capture=False, stored_overrides=self.relocated_resources(), probe_resources=True)
        current = ZeroWAMConfig()
        for name in self.relocated_resources():
            self.assertEqual(getattr(self.prepared_config, name), getattr(current, name))

    def test_eval_restores_custom_mapping_snapshot_after_original_path_disappears(self):
        stored = self.relocated_resources()
        stored["task_mapping_file"] = "/previous_checkout/custom_mapping.json"
        content = json.dumps(
            {"task_mappings": [{"human_task_id": "human_H4", "sim_task_id": ["L0_TwoRobotPickRemoteControl-v1"]}]}
        )
        resources = {
            "task_mapping_file": {
                "kind": "custom",
                "text": content,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
            }
        }
        output = self.evaluate_fixture(
            1, capture=False, stored_overrides=stored, saved_resources=resources, probe_resources=True
        )
        path = Path(self.prepared_config.task_mapping_file)
        self.assertTrue(path.is_relative_to(output))
        self.assertEqual(path.read_text(), content)

    def test_eval_allows_explicit_custom_resource_paths(self):
        stored = {name: "/previous_checkout/custom_" + name + ".json" for name in self.relocated_resources()}
        overrides = {}
        for name, value in self.relocated_resources().items():
            path = self.root / (name + ".json")
            path.write_text(
                json.dumps(
                    {
                        "task_mappings": [
                            {"human_task_id": "human_H4", "sim_task_id": ["L0_TwoRobotPickRemoteControl-v1"]}
                        ]
                    }
                )
                if name == "task_mapping_file"
                else "{}"
            )
            overrides[name] = str(path)
        self.evaluate_fixture(1, capture=False, stored_overrides=stored, args_overrides=overrides, probe_resources=True)
        for name, path in overrides.items():
            self.assertEqual(getattr(self.prepared_config, name), path)


if __name__ == "__main__":
    unittest.main()
