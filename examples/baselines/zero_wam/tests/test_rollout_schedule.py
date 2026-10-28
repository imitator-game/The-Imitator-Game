"""Exercise the final rollout loop on CPU, including its last partial chunk."""
from types import SimpleNamespace
import unittest

import gymnasium
import numpy as np
import torch

from examples.baselines.lerobot_dataset.lerobot_paired_dataset import InputMode
from examples.baselines.zero_wam.zero_wam.rollout import evaluate_with_task_encoder


class RolloutScheduleTests(unittest.TestCase):
    def test_first16_then32_history_and_exact500_controls(self):
        class Environment:
            num_envs = 1
            single_observation_space = gymnasium.spaces.Box(-1, 1, (18,))
            action_space = gymnasium.spaces.Box(-1, 1, (16,))

            def __init__(self):
                self.executed = []

            def reset(self):
                self.t = 0
                return self.obs(), {}

            def obs(self):
                return dict(state=torch.full((1, 18), float(self.t)), rgb=torch.zeros(1, 3, 2, 2))

            def step(self, action):
                self.executed.append(np.concatenate([action['panda_wristcam-0'], action['panda_wristcam-1']], axis=-1))
                self.t += 1
                done = torch.tensor([self.t == 500])
                info = {'final_info': {'episode': {'episode_len': torch.tensor([self.t]),
                                                  'success_at_end': torch.tensor([True])}}}
                return self.obs(), torch.ones(1), torch.tensor([False]), done, info

        class Agent:
            def __init__(self):
                self.queries, self.observed, self.history = [], [], []

            def eval(self):
                pass

            def train(self):
                pass

            def prepare_for_eval(self, **kwargs):
                self.prepared = kwargs

            def observe_step(self, obs):
                self.observed.append(int(obs['state'][0, 0]))

            def get_action(self, obs):
                start = int(obs['state'][0, 0])
                size = 16 if not self.queries else 32
                self.queries.append(start)
                return torch.arange(start, start + size).float()[None, :, None].expand(1, size, 16)

            def record_executed_action(self, action):
                self.history.append(action.clone())

            def clear_cache(self):
                self.cleared = True

        env, agent = Environment(), Agent()
        processor = SimpleNamespace(get_video=lambda *args: torch.zeros(1, 1),
                                    normalize_state_rgb=lambda state, rgb, env: (state, rgb),
                                    denormalize_action=lambda action, env: action)
        metrics = evaluate_with_task_encoder(1, agent, env,
            dict(env_id='fixture', delta_control=False, pred_horizon=32, temporal_agg=False,
                 dynamic_action_chunks=True, max_timesteps=500, device='cpu', sim_backend='physx_cpu'),
            processor, InputMode.VIDEO_ONLY, progress_bar=False)
        self.assertEqual(agent.queries, [0, *range(16, 500, 32)])
        self.assertEqual(agent.observed, list(range(1, 500)))
        self.assertEqual(len(agent.history), 500)
        np.testing.assert_array_equal(np.concatenate(env.executed)[:, 0], np.arange(500))
        torch.testing.assert_close(torch.cat(agent.history)[:, 0], torch.arange(500).float())
        self.assertEqual(int(metrics['episode_len'].item()), 500)
        self.assertTrue(agent.cleared)


if __name__ == '__main__':
    unittest.main()
