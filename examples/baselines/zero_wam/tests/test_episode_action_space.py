"""Coordinate, mapping and rollout dispatch checks without GPU allocations."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from examples.baselines.zero_wam.zero_wam.action_space import (
    ACTION_JOINTS, STATE_JOINTS, EpisodeActionSpace, EpisodeEvaluationNormalizer,
    episode_coordinates, fit_statistics, channel_ids,
)
from examples.baselines.zero_wam.zero_wam.dataset import IGDataset
from examples.baselines.zero_wam.zero_wam.dispatch import RolloutQueue
from examples.baselines.zero_wam.zero_wam.model import pack_actions, unpack_actions
from examples.baselines.zero_wam.zero_wam.policy import ZeroWAMAgent


def stats(representation="relative_episode"):
    return dict(representation=representation, clip=2.,
                action=dict(q01=[-10.]*16,q99=[10.]*16),
                state=dict(q01=[-10.]*18,q99=[10.]*18))


class EpisodeCoordinateTests(unittest.TestCase):
    def test_origin_is_measured_initial_state_and_grippers_stay_absolute(self):
        states = torch.arange(54).reshape(3,18).float()/10
        actions = torch.arange(48).reshape(3,16).float()/10
        a,s = episode_coordinates(actions,states,"relative_episode")
        torch.testing.assert_close(s[0,list(STATE_JOINTS)],torch.zeros(14))
        torch.testing.assert_close(a[:,list(ACTION_JOINTS)],actions[:,list(ACTION_JOINTS)]-states[0,list(STATE_JOINTS)])
        torch.testing.assert_close(a[:,[7,15]],actions[:,[7,15]])
        torch.testing.assert_close(s[:,[7,8,16,17]],states[:,[7,8,16,17]])

    def test_training_and_online_coordinates_round_trip_with_fixed_origin(self):
        space=EpisodeActionSpace(stats())
        states=torch.arange(54).reshape(3,18).float()/20
        actions=torch.arange(48).reshape(3,16).float()/20
        a,s=space.normalize_episode(actions,states)
        for i in range(3):
            torch.testing.assert_close(space.encode_state(states[i:i+1],states[:1]),s[i:i+1])
            torch.testing.assert_close(space.decode_action(a[i:i+1],states[:1]),actions[i:i+1],atol=2e-6,rtol=1e-5)
        agent=SimpleNamespace(action_space=space,episode_origin=states[:1])
        boundary=EpisodeEvaluationNormalizer(None,agent)
        self.assertIs(boundary.normalize_state(states,0),states)
        torch.testing.assert_close(boundary.denormalize_action(a[None],0),actions[None],atol=2e-6,rtol=1e-5)

    def test_global_statistics_use_all_real_rows_and_distinct_coordinates(self):
        records=[dict(i=0),dict(i=1)]
        episodes=[dict(actions=torch.ones(3,16)*v,states=torch.ones(3,18)*(v+1)) for v in [1.,5.]]
        manifest=dict(sim_episodes=records,selection_identity="selected",records_sha256="records")
        with tempfile.TemporaryDirectory() as temp, patch("examples.baselines.zero_wam.zero_wam.cache.read_episode",side_effect=lambda r:episodes[r["i"]]):
            relative=fit_statistics(manifest,"relative_episode",Path(temp)/"rel.json")
        self.assertEqual(relative["episodes"],2)
        self.assertEqual(relative["rows"],6)
        self.assertEqual(relative["action"]["q01"][0],-1.)
        self.assertEqual(relative["state"]["q99"][0],0.)
        self.assertEqual(relative["action"]["q99"][7],5.)

    def test_mapping_changes_only_selected_channels_and_masks(self):
        actions=torch.randn(1,3,16)
        for offset in [14]:
            packed=pack_actions(actions,offset)
            torch.testing.assert_close(unpack_actions(packed,offset),actions)
            inactive=sorted(set(range(30))-set(channel_ids(offset)))
            self.assertEqual(packed[...,inactive].count_nonzero().item(),0)
        dataset=IGDataset.__new__(IGDataset)
        dataset.records=[dict(repo_id="robot",human_repo_id="human",cache_path="r")]
        dataset.humans=[dict(cache_path="h")]
        dataset.humans_by_repo={"human":[0]};dataset.repo_indices={"robot":0}
        dataset.seed=1;dataset.action_per_frame=16
        dataset.empty_text=torch.zeros(512,4096)
        dataset.action_space=EpisodeActionSpace(stats())
        payload=dict(latent=torch.ones(48,3,2,2),latent_nominal_anchor_indices=[0,16,32],action_chunk_size=16,
                     actions=torch.randn(17,16),states=torch.randn(17,18))
        dataset.payload=lambda path: payload if path=="r" else dict(latent=torch.ones(48,2,2,2))
        results=[]
        for offset in [14]:
            dataset.action_channels=list(channel_ids(offset))
            results.append(dataset.episode_item((0,1.,0,0,4,1)))
        row = results[0]
        self.assertTrue(row["actions_mask"][list(channel_ids()),0].all())
        self.assertEqual(row["actions"][:,0].count_nonzero().item(),0)
        self.assertEqual(int(row["actions_mask"].sum()),(16+17)*16)

    def test_online_robot_and_human_resize_are_distinct(self):
        agent=SimpleNamespace(args=SimpleNamespace(image_size=224,robot_width=288,human_height=320,human_width=480),
                              device=torch.device("cpu"),dtype=torch.float32,_video=ZeroWAMAgent._video)
        video=torch.rand(1,5,20,30,3)
        robot=ZeroWAMAgent._pixels(agent,video)
        human=ZeroWAMAgent._pixels(agent,video,human=True)
        self.assertEqual(robot.shape,(1,3,5,224,288))
        self.assertEqual(human.shape,(1,3,5,320,480))
        torch.testing.assert_close(ZeroWAMAgent._pixels(agent,ZeroWAMAgent._video(video)),robot)

    def test_complete_cache_accepts_requested_nonsquare_latent_layout(self):
        from contextlib import ExitStack
        from examples.baselines.zero_wam.zero_wam import cache
        contract=dict(video=dict(sim=dict(source_frame_stride=4,resize_hw=[224,288],action_tail_policy="all_actions_v1"),
                                 human=dict(target_fps=12,resize_hw=[320,480]),vae_spatial_stride=16))
        manifest=dict(selection_identity="selection",cache_request_identity="request",contract_path="contract",
                      data_contract_sha256=cache.digest_json(contract["video"]),encoder_identity={"model":"fixture"})
        payloads={}
        for kind,(h,w) in [("sim",(14,18)),("human",(20,30))]:
            plan=cache.sample_plan(17,30,kind,contract);frames=len(plan["latent_anchor_indices"])
            manifest[kind+"_episodes"]=[dict(source=dict(episode_length=17,fps=30),expected_latent_frames=frames,cache_path=kind)]
            payloads[kind]=dict(latent=torch.zeros(48,frames,h,w),sampled_frame_indices=torch.from_numpy(plan["sampled_frame_indices"]))
        config=SimpleNamespace(manifest_path=lambda:"manifest",selection_identity=lambda:"selection",
                               cache_request_identity=lambda:"request",model_path="model",sampling_mode="episode")
        with ExitStack() as stack:
            stack.enter_context(patch.object(cache,"read_json",side_effect=lambda p:manifest if str(p)=="manifest" else contract))
            stack.enter_context(patch.object(cache,"load_task_mapping"))
            stack.enter_context(patch.object(cache,"validate_sources"))
            stack.enter_context(patch.object(cache,"validate_cached_payload"))
            stack.enter_context(patch.object(cache,"encoder_identity",return_value={"model":"fixture"}))
            stack.enter_context(patch.object(torch,"load",side_effect=lambda p,**kw:payloads[p]))
            self.assertIs(cache.require_complete(config),manifest)
            payloads["human"]["latent"]=torch.zeros(48,frames,14,14)
            with self.assertRaisesRegex(ValueError,"Invalid VAE layout"):
                cache.require_complete(config)


class RolloutDispatchTests(unittest.TestCase):
    def test_individual_cases_resume_and_no_duplicate_assignments(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            queue=RolloutQueue(root,["env_a","env_b"],3,100000,500,4)
            first=queue.claim("worker0")
            second=queue.claim("worker1")
            self.assertNotEqual(first,second)
            path=queue.metrics_path(first);path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(env_id=first["env_id"],seed=100000+first["episode"],max_episode_steps=500)))
            queue.complete("worker0")
            resumed=RolloutQueue(root,["env_a","env_b"],3,100000,500,4)
            cases=[]
            for worker in range(5):
                case=resumed.claim(str(worker));cases.append((case["env_id"],case["episode"]))
            self.assertEqual(len(set(cases)),5)
            self.assertNotIn((first["env_id"],first["episode"]),cases)
            self.assertIn((second["env_id"],second["episode"]),cases)
            self.assertIsNone(resumed.claim("idle"))
            self.assertEqual(resumed.completed,1)


if __name__=="__main__":
    unittest.main()
