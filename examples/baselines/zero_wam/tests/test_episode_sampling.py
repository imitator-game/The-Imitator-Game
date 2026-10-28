"""Complete-episode cadence, sampler, human pairing and action-tail checks."""
from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from examples.baselines.zero_wam.zero_wam.dataset import IGDataset, EpisodeBatchSampler
from examples.baselines.zero_wam.zero_wam.model import IG_CHANNELS
from examples.baselines.zero_wam.zero_wam.training import resume_sample_progress
from examples.baselines.zero_wam.zero_wam.training import resume
from easydict import EasyDict


class IdentityNormalizer:
    def add_dataset_stats(self, *args):
        pass

    def normalize_action(self, value, index):
        return value

    def normalize_state(self, value, index):
        return value





class EpisodeRecipeTests(unittest.TestCase):
    def test_native_distributed_sampler_padding_and_resume_suffix(self):
        from torch.utils.data.distributed import DistributedSampler
        for size in [1, 50, 200]:
            for epoch in [0, 3]:
                flat = []
                for rank in range(4):
                    sampler = EpisodeBatchSampler(size, 4, rank, seed=17)
                    sampler.set_epoch(epoch)
                    expected = DistributedSampler(range(size), num_replicas=4, rank=rank,
                        shuffle=True, seed=17, drop_last=False)
                    expected.set_epoch(epoch)
                    batches = list(sampler)
                    self.assertEqual([b[0][0] for b in batches], list(expected))
                    self.assertTrue(all(b[0][1] == 1 for b in batches))  # repeats retain native weight
                    sampler.start_batch = min(2, len(batches))
                    self.assertEqual(list(sampler), batches[sampler.start_batch:])
                    flat += [b[0][0] for b in batches]
                self.assertEqual(len(flat), ((size+3)//4)*4)
                self.assertEqual(set(flat), set(range(size)))

    def test_complete_episode_cadence_zero_prefix_tail_and_human_pairing(self):
        from examples.baselines.zero_wam.zero_wam.cache import sample_plan
        pairs = []
        for interval in [16]:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root/"meta").mkdir()
                (root/"meta/stats.json").write_text("{}")
                length = 38
                plan = sample_plan(length, 30, "sim", {"video":{"sim":{"source_frame_stride":interval//4}}})
                anchors = torch.from_numpy(plan["latent_nominal_anchor_indices"])
                robot = dict(latent=anchors[None,:,None,None].expand(48,-1,14,14).float(),
                    latent_nominal_anchor_indices=anchors, action_chunk_size=interval,
                    actions=100+torch.arange(length)[:,None].expand(-1,16).float(),
                    states=torch.arange(length)[:,None].expand(-1,18).float())
                torch.save(robot, root/"robot.pt")
                humans = []
                for i in range(7):
                    path = root/f"human{i}.pt"
                    torch.save(dict(latent=torch.full((48,3+i,14,14),float(i))), path)
                    humans.append(dict(repo_id="human", cache_path=str(path)))
                torch.save(torch.zeros(512,4096), root/"text.pt")
                record = dict(repo_id="robot", human_repo_id="human", cache_path=str(root/"robot.pt"),
                    source=dict(episode_length=length, repo_path=str(root)))
                (root/"manifest.json").write_text(json.dumps(dict(sim_episodes=[record],human_episodes=humans)))
                with patch("examples.baselines.lerobot_dataset.normalizer.ActionNormalizer", IdentityNormalizer):
                    ds = IGDataset(root/"manifest.json", empty_text_path=root/"text.pt",
                        sampling_mode="episode", action_per_frame=interval, action_history_mode="zero_history_supervised_v1")
                self.assertEqual(len(ds), 1)  # not38 controls, not ceil(38/interval) windows
                rows = [ds[(0,1.,e,0,4,1)] for e in range(10)]
                pairs.append([r["ig_human_index"] for r in rows])
                row = rows[0]
                self.assertTrue(torch.equal(row["latents"], robot["latent"]))
                self.assertFalse(bool(row["actions"][:,0].any()))
                self.assertTrue(bool(row["actions_mask"][list(IG_CHANNELS),0].all()))
                self.assertFalse(bool(row["actions_mask"][:14].any()))
                values = row["actions"][IG_CHANNELS[0],1:,:,0].flatten()
                valid = row["actions_mask"][IG_CHANNELS[0],1:,:,0].flatten()
                self.assertTrue(torch.equal(values[valid], torch.arange(length).float()+100))
                self.assertEqual(int(valid.sum()), length)
                for chunk in [1,2,3,4]:
                    # Each chunk sees only the state at its first supervised action.
                    got = row["ig_states"][::chunk,0]
                    starts = (torch.arange(len(anchors))[::chunk]-1).clamp_min(0)*interval
                    self.assertTrue(torch.equal(got, starts.clamp_max(length-1).float()))
                bad = dict(robot, action_chunk_size=8)
                torch.save(bad, root/"robot.pt")
                ds.payloads.clear()
                with self.assertRaisesRegex(ValueError, "cadence"):
                    ds[(0,1.,0,0,4,1)]
        self.assertGreater(len(set(pairs[0])), 1)  # random human pool is not collapsed to one prompt

    def test_all_real_actions_survive_tail_boundaries_in_actual_episode_item(self):
        from examples.baselines.zero_wam.zero_wam.cache import sample_plan
        for interval in [16]:
            for length in [1, interval, interval+1, 2*interval, 2*interval+1, 169]:
                with self.subTest(interval=interval, length=length):
                    contract = {"video":{"sim":{"source_frame_stride":interval//4,
                        "action_tail_policy":"all_actions_v1"}}}
                    plan = sample_plan(length, 30, "sim", contract)
                    anchors = torch.from_numpy(plan["latent_nominal_anchor_indices"])
                    robot = dict(latent=torch.zeros(48,len(anchors),1,1),
                        latent_nominal_anchor_indices=anchors, action_chunk_size=interval,
                        actions=100+torch.arange(length)[:,None].expand(-1,16).float(),
                        states=torch.arange(length)[:,None].expand(-1,18).float())
                    ds = IGDataset.__new__(IGDataset)
                    ds.action_space, ds.action_channels = None, list(IG_CHANNELS)
                    ds.action_per_frame, ds.seed = interval, 1
                    ds.records = [dict(repo_id="robot",human_repo_id="human",cache_path="robot")]
                    ds.repo_indices = {"robot":0}
                    ds.normalizer = IdentityNormalizer()
                    ds.humans_by_repo = {"human":[0]}
                    ds.humans = [dict(cache_path="human")]
                    ds.empty_text = torch.zeros(1)
                    ds.payload = lambda path: robot if path == "robot" else dict(latent=torch.zeros(48,2,1,1))
                    row = ds.episode_item((0,1.,0,0,4,1))
                    values = row["actions"][IG_CHANNELS[0],1:,:,0].flatten()
                    valid = row["actions_mask"][IG_CHANNELS[0],1:,:,0].flatten()
                    self.assertTrue(torch.equal(values[valid],torch.arange(length).float()+100))
                    self.assertEqual(int(valid.sum()),length)
                    self.assertEqual(len(values)-int(valid.sum()),(-length)%interval)
                    self.assertFalse(bool(row["actions"][:,0].any()))
                    self.assertTrue(bool(row["actions_mask"][list(IG_CHANNELS),0].all()))
                    self.assertEqual(len(anchors),1+(length+interval-1)//interval)
                    # Tail repeats never request a nonexistent/future RGB frame.
                    self.assertLessEqual(int(plan["sampled_frame_indices"].max()),length-1)
                    if length == 169:
                        self.assertEqual(float(values[valid][-1]),268.)

    def test_padded_epoch_cursor_resume_never_replays_committed_episode(self):
        config = dict(ig_sampling_mode="episode", world_size=4, ig_total_epochs=10)
        ckpt = dict(native_config=config, global_step=18,
            progress=dict(epoch=1,next_batch=5,sample_offset=20))
        progress, target = resume_sample_progress(ckpt, config, 50)
        self.assertEqual(progress, dict(epoch=1,next_batch=5,sample_offset=20))
        self.assertEqual(target, 130)
        ckpt["native_config"] = dict(ig_sampling_mode="control_step")
        with self.assertRaisesRegex(ValueError,"legacy"):
            resume_sample_progress(ckpt, config, 50)


if __name__ == "__main__":
    unittest.main()
