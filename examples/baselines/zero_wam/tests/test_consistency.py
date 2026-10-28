"""CPU regression checks for public settings and checkpoint rollback consistency."""

from contextlib import ExitStack
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from easydict import EasyDict
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from examples.baselines.zero_wam.zero_wam import cache
from examples.baselines.zero_wam.zero_wam import training
from examples.baselines.zero_wam.zero_wam.config import ZeroWAMConfig


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_json(self, path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def make_episode_metadata(self, kind, repo):
        root = self.root / kind / repo
        image = cache.IMAGE_KEY
        self.write_json(
            root / "meta/info.json",
            {
                "fps": 30,
                "features": {
                    image: {"shape": [3, 224, 224], "info": {"video.fps": 30}},
                    cache.ACTION_KEY: {"shape": [16]},
                },
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            },
        )
        self.write_json(root / "meta/stats.json", {})
        metadata = root / "meta/episodes/chunk-000/file-000.parquet"
        metadata.parent.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "episode_index": 0,
                        "length": 17,
                        "data/chunk_index": 0,
                        "data/file_index": 0,
                        f"videos/{image}/chunk_index": 0,
                        f"videos/{image}/file_index": 0,
                        f"videos/{image}/from_timestamp": 0.0,
                        f"videos/{image}/to_timestamp": 17 / 30,
                    }
                ]
            ),
            metadata,
        )
        for relative in [f"videos/{image}/chunk-000/file-000.mp4", "data/chunk-000/file-000.parquet"]:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"metadata-only fixture")

    def test_custom_mapping_reaches_real_record_construction(self):
        task = "L0_TwoRobotPickRemoteControl-v1"
        mapping = self.root / "custom_mapping.json"
        self.write_json(mapping, {"task_mappings": [{"human_task_id": "human_H13", "sim_task_id": [task]}]})
        self.make_episode_metadata("sim", task)
        self.make_episode_metadata("human", "human_H13")
        for kind, names in [("sim", [task]), ("human", ["human_H4", "human_H13"])]:
            self.write_json(
                self.root / f"{kind}.json", [{"repo_id": repo, "root": repo, "train": "0:1"} for repo in names]
            )
        config = ZeroWAMConfig(
            task=task,
            task_mapping_file=str(mapping),
            human_dataset_file=str(self.root / "human.json"),
            sim_dataset_file=str(self.root / "sim.json"),
            model_path=str(self.root / "model"),
            human_root=str(self.root / "human"),
            sim_root=str(self.root / "sim"),
            cache_dir=str(self.root / "cache"),
        )
        self.assertEqual([row["repo_id"] for row in config.selections()[0]], ["human_H13"])
        with patch.object(cache, "encoder_identity", return_value={"fixture": True}):
            manifest = cache.read_json(cache.build_manifest(config))
        self.assertEqual(manifest["sim_episodes"][0]["human_repo_id"], "human_H13")
        contract = cache.read_json(manifest["contract_path"])
        self.assertEqual(contract["paths"]["task_mapping"], str(mapping.resolve()))
        self.assertEqual(manifest["config_identities"]["task_mapping"], cache.file_identity(mapping, True))
        native = training.native_config(config, manifest)
        self.assertEqual(native.ig_code_resources["task_mapping_file"]["text"], mapping.read_text())
        self.assertEqual(native.ig_code_resources["task_mapping_file"]["kind"], "custom")

    def test_training_receives_public_conditioning_settings(self):
        config = ZeroWAMConfig(video_snr_shift=2.0, action_snr_shift=3.0, icl_rope_h=12)
        manifest = {"sim_episodes": [{"source": {"episode_length": 17}}], "records_sha256": "fixture"}
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            native = training.native_config(config, manifest)
        self.assertEqual((native.snr_shift, native.action_snr_shift, native.icl_rope_h), (2.0, 3.0, 12))
        stored = native.ig_app_config
        self.assertEqual(
            (native.snr_shift, native.action_snr_shift, native.icl_rope_h),
            (stored["video_snr_shift"], stored["action_snr_shift"], stored["icl_rope_h"]),
        )

    def test_pairing_change_reuses_same_episode_encoding_without_overwrite(self):
        task = "L0_TwoRobotPickRemoteControl-v1"
        for kind, names in [("sim", [task]), ("human", ["human_H4", "human_H13"])]:
            for repo in names:
                self.make_episode_metadata(kind, repo)
            self.write_json(
                self.root / f"{kind}.json", [{"repo_id": repo, "root": repo, "train": "0:1"} for repo in names]
            )
        model = self.root / "model"
        self.write_json(model / "vae/config.json", {})
        (model / "vae/model.safetensors").write_bytes(b"metadata-only weights")
        configs, manifests = [], []
        for human in ["human_H4", "human_H13"]:
            mapping = self.root / (human + ".json")
            self.write_json(mapping, {"task_mappings": [{"human_task_id": human, "sim_task_id": [task]}]})
            config = ZeroWAMConfig(
                task=task,
                task_mapping_file=str(mapping),
                model_path=str(model),
                human_root=str(self.root / "human"),
                sim_root=str(self.root / "sim"),
                cache_dir=str(self.root / "cache"),
                human_dataset_file=str(self.root / "human.json"),
                sim_dataset_file=str(self.root / "sim.json"),
            )
            configs.append(config)
            manifests.append(cache.read_json(cache.build_manifest(config)))
        first, second = [m["sim_episodes"][0] for m in manifests]
        self.assertEqual(first["cache_path"], second["cache_path"])
        self.assertNotEqual(first["record_sha256"], second["record_sha256"])
        # Legacy payload has source identity but no dedicated encoding hash.
        payload = {k: first[k] for k in ["kind", "repo_id", "episode_index", "source", "record_sha256"]}
        payload.update(
            data_contract_sha256=manifests[0]["data_contract_sha256"],
            encoder_fingerprint=cache.digest_json(manifests[0]["encoder_identity"]),
        )
        destination = Path(first["cache_path"])
        destination.parent.mkdir(parents=True)
        torch.save(payload, destination)
        from examples.baselines.zero_wam.zero_wam.wan_va.modules import utils

        vae = Mock()
        vae.eval.return_value = vae
        vae.config = SimpleNamespace(
            scale_factor_spatial=16, scale_factor_temporal=4, latents_mean=[0] * 48, latents_std=[1] * 48
        )
        wrapper = Mock()
        args = SimpleNamespace(
            contract=manifests[1]["contract_path"],
            manifest=str(configs[1].manifest_path()),
            model_path=str(model),
            kind="sim",
            repo_id=None,
            shard_index=0,
            num_shards=1,
            limit=None,
            dtype="bfloat16",
            device="cpu",
            overwrite=False,
            decode_threads=1,
        )
        with (
            patch.object(utils, "load_vae", return_value=vae),
            patch.object(utils, "WanVAEStreamingWrapper", return_value=wrapper),
        ):
            cache.encode(args)
            progress = cache.read_json(configs[1].manifest_path().parent / "sim_rank0/progress.json")
            self.assertEqual((progress["reused"], progress["completed"]), (1, 0))
            wrapper.encode_chunk.assert_not_called()
            payload["source"]["fps"] = 31
            torch.save(payload, destination)
            with self.assertRaisesRegex(RuntimeError, "stale/incompatible"):
                cache.encode(args)

    def resume_fixture(self, rank=0, restore_rng_effect=None):
        settings = EasyDict(
            ig_resume="/restore.pt", ig_initialization=None, world_size=1, rank=rank, save_root=str(self.root),
            batch_size=1, ig_effective_batch=1, ig_total_epochs=2, ig_sampling_mode="episode"
        )
        old = dict(settings, ig_resume=None)
        trainer = SimpleNamespace(
            config=settings,
            transformer=Mock(),
            optimizer=Mock(),
            lr_scheduler=Mock(),
            ig_sampler=Mock(),
            train_loader=SimpleNamespace(dataset=list(range(32))),
            device="cpu",
            ig_adaptation={},
        )
        trainer.lr_scheduler.state_dict.return_value = {}
        checkpoint = {
            "native_config": old,
            "adapter_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "global_step": 1,
            "progress": {"epoch": 0, "next_batch": 1},
            "rank_rng_states": [{}, {}],
        }
        real_load = torch.load

        def load(path, **kwargs):
            return deepcopy(checkpoint) if str(path) == "/restore.pt" else real_load(path, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(training.torch, "load", side_effect=load))
            for name in ["set_model_state_dict", "set_optimizer_state_dict"]:
                stack.enter_context(patch.object(training, name))
            stack.enter_context(patch.object(training, "restore_rng", side_effect=restore_rng_effect))
            stack.enter_context(patch.object(training.dist, "barrier"))
            training.resume(trainer)
        return trainer

    def loss_row(self, step):
        return dict(
            event="optimizer_update",
            step=step,
            epoch=0,
            samples=32,
            loss=0.3,
            video_loss=0.2,
            action_loss=0.1,
            lr=1e-4,
            grad_norm=1.0,
            update_seconds=1.0,
        )

    def test_resume_archives_future_rows_reports_and_checkpoint_links(self):
        rows = [self.loss_row(step) for step in [1, 2, 3]]
        original = b"".join((json.dumps(row) + "\n").encode() for row in rows)
        (self.root / "loss.jsonl").write_bytes(original)
        self.write_json(self.root / "latest_metrics.json", rows[-1])
        self.write_json(self.root / "checkpoint_status.json", {"step": 3})
        (self.root / "loss.csv").write_text("stale report")
        (self.root / "loss.png").write_bytes(b"stale plot")
        folder = self.root / "checkpoints"
        folder.mkdir()
        for name, step in [("latest.pt", 3), ("final.pt", 3), ("epoch_005.pt", 2), ("epoch_001.pt", 1)]:
            torch.save({"global_step": step}, folder / name)
        self.resume_fixture()
        canonical = [json.loads(line) for line in (self.root / "loss.jsonl").read_text().splitlines()]
        self.assertEqual([row["step"] for row in canonical], [1])
        archives = list((self.root / "recovery").glob("resume_*"))
        self.assertEqual(len(archives), 1)
        self.assertEqual((archives[0] / "loss.jsonl").read_bytes(), original)
        self.assertEqual(cache.read_json(self.root / "latest_metrics.json")["step"], 1)
        with (self.root / "loss.csv").open() as handle:
            self.assertEqual([int(row["step"]) for row in csv.DictReader(handle)], [1])
        for name in ["latest.pt", "final.pt", "epoch_005.pt"]:
            self.assertFalse((folder / name).exists())
            self.assertTrue((archives[0] / "checkpoints" / name).is_file())
        self.assertTrue((folder / "epoch_001.pt").is_file())
        self.assertFalse((self.root / "checkpoint_status.json").exists())
        self.assertEqual(torch.load(archives[0] / "checkpoints/epoch_005.pt", weights_only=False)["global_step"], 2)
        with (self.root / "loss.jsonl").open("a") as handle:
            handle.write(json.dumps(self.loss_row(2)) + "\n")
        self.assertEqual(
            [json.loads(line)["step"] for line in (self.root / "loss.jsonl").read_text().splitlines()], [1, 2]
        )

    def test_resume_preserves_and_removes_incomplete_final_line(self):
        raw = (json.dumps(self.loss_row(1)) + '\n{"step":2,').encode()
        (self.root / "loss.jsonl").write_bytes(raw)
        self.resume_fixture()
        self.assertEqual((self.root / "loss.jsonl").read_text(), json.dumps(self.loss_row(1)) + "\n")
        self.assertEqual(next((self.root / "recovery").glob("resume_*/loss.jsonl")).read_bytes(), raw)

    def test_resume_rejects_corrupt_middle_line_without_rewriting(self):
        raw = (json.dumps(self.loss_row(1)) + '\n{"broken"\n' + json.dumps(self.loss_row(2)) + "\n").encode()
        (self.root / "loss.jsonl").write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "line 2"):
            self.resume_fixture()
        self.assertEqual((self.root / "loss.jsonl").read_bytes(), raw)

    def test_clean_resume_does_not_archive_committed_history(self):
        raw = (json.dumps(self.loss_row(1)) + "\n").encode()
        (self.root / "loss.jsonl").write_bytes(raw)
        self.resume_fixture()
        self.assertEqual((self.root / "loss.jsonl").read_bytes(), raw)
        self.assertFalse((self.root / "recovery").exists())

    def test_resume_retains_complete_final_row_without_newline(self):
        raw = json.dumps(self.loss_row(1)).encode()
        (self.root / "loss.jsonl").write_bytes(raw)
        self.resume_fixture()
        self.assertEqual((self.root / "loss.jsonl").read_bytes(), raw + b"\n")
        self.assertEqual(next((self.root / "recovery").glob("resume_*/loss.jsonl")).read_bytes(), raw)

    def test_resume_rejects_ambiguous_committed_duplicate_steps(self):
        raw = (json.dumps(self.loss_row(1)) + "\n") * 2
        (self.root / "loss.jsonl").write_text(raw)
        with self.assertRaisesRegex(ValueError, "Nonmonotonic.*line 2"):
            self.resume_fixture()
        self.assertEqual((self.root / "loss.jsonl").read_text(), raw)

    def test_nonzero_rank_does_not_reconcile_shared_logs(self):
        raw = (json.dumps(self.loss_row(2)) + "\n").encode()
        (self.root / "loss.jsonl").write_bytes(raw)
        self.resume_fixture(rank=1)
        self.assertEqual((self.root / "loss.jsonl").read_bytes(), raw)

    def test_reconciliation_does_not_advance_restored_rng(self):
        self.addCleanup(random.setstate, random.getstate())
        restored = random.Random(9).getstate()
        with patch.object(training, "reconcile_run_outputs", side_effect=lambda *args: random.random()):
            self.resume_fixture(restore_rng_effect=lambda *args: random.setstate(restored))
        self.assertEqual(random.getstate(), restored)


if __name__ == "__main__":
    unittest.main()
