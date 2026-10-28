"""Controller-limited command history, using the real policy callback on CPU."""
from types import SimpleNamespace
import unittest

import torch

from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer
from examples.baselines.zero_wam.zero_wam.action_space import EpisodeActionSpace, EpisodeEvaluationNormalizer
from examples.baselines.zero_wam.zero_wam.policy import ZeroWAMAgent


def agent_for(denormalizer):
    return SimpleNamespace(
        history_action_denormalizer=denormalizer,
        history_gripper_affine=None, executed_actions=[], total_executed_steps=0,
        gripper_history_corrections=torch.zeros(2, dtype=torch.long),
        gripper_history_max_correction=torch.zeros(2),
    )


class ExecutedGripperHistoryTests(unittest.TestCase):
    def test_absolute_and_relative_history_match_limited_commands(self):
        for representation in ['absolute', 'relative_episode']:
            with self.subTest(representation=representation):
                stats = dict(representation=representation, clip=2.,
                             action=dict(q01=[-.8]*16, q99=[.9]*16))
                space = EpisodeActionSpace(stats)
                policy = SimpleNamespace(action_space=space, episode_origin=torch.arange(18)[None].float()/10)
                boundary = EpisodeEvaluationNormalizer(None, policy)
                decode = lambda value: boundary.denormalize_action(value, 'task')
                agent = agent_for(decode)
                predicted = torch.linspace(-3, 3, 16)[None]
                predicted[:, [7, 15]] = torch.tensor([[2., -2.]])
                original = predicted.clone()
                raw = decode(predicted)
                expected = raw.clone()
                expected[:, [7, 15]] = expected[:, [7, 15]].clamp(-1, 1)
                ZeroWAMAgent.record_executed_action(agent, predicted)
                cached = agent.executed_actions[0]
                torch.testing.assert_close(decode(cached), expected, atol=3e-6, rtol=1e-6)
                joints = [i for i in range(16) if i not in [7, 15]]
                self.assertTrue(torch.equal(cached[:, joints], original[:, joints]))
                self.assertTrue(torch.equal(predicted, original))
                self.assertEqual(agent.gripper_history_corrections.tolist(), [1, 1])

    def test_inside_bounds_is_bitwise_unchanged_and_history_owns_its_copy(self):
        space = EpisodeActionSpace(dict(representation='absolute', clip=2.,
                                        action=dict(q01=[-1.]*16, q99=[1.]*16)))
        agent = agent_for(lambda x: space.denormalize(x, 'action'))
        predicted = torch.linspace(-.7, .7, 16)[None]
        original = predicted.clone()
        ZeroWAMAgent.record_executed_action(agent, predicted)
        self.assertTrue(torch.equal(agent.executed_actions[0], original))
        predicted.zero_()
        self.assertTrue(torch.equal(agent.executed_actions[0], original))
        self.assertEqual(agent.gripper_history_corrections.tolist(), [0, 0])

    def test_controller_limit_is_not_reclipped_to_training_normalized_range(self):
        lo, hi = [-1.]*16, [1.]*16
        for i in [7, 15]:
            lo[i], hi[i] = -.1, .1
        space = EpisodeActionSpace(dict(representation='absolute', clip=2.,
                                        action=dict(q01=lo, q99=hi)))
        agent = agent_for(lambda x: space.denormalize(x, 'action'))
        predicted = torch.zeros(1, 16)
        predicted[:, [7, 15]] = torch.tensor([[20., -20.]])
        ZeroWAMAgent.record_executed_action(agent, predicted)
        cached = agent.executed_actions[0]
        self.assertTrue(bool((cached[:, [7, 15]].abs() > 2).all()))
        torch.testing.assert_close(space.denormalize(cached, 'action')[:, [7, 15]],
                                   torch.tensor([[1., -1.]]), atol=2e-6, rtol=0)

    def test_legacy_per_dataset_inverse_and_constant_gripper(self):
        normalizer = ActionNormalizer()
        lo, hi = [-.5]*16, [.7]*16
        lo[15] = hi[15] = 1.  # A constant channel has no normalization slope.
        normalizer.add_dataset_stats(0, 'robot', {'action': dict(q01=lo, q99=hi)}, 'state', 'action')
        decode = lambda x: normalizer.denormalize_action(x, 0)
        agent = agent_for(decode)
        predicted = torch.zeros(1, 16)
        predicted[0, 7] = 3.
        predicted[0, 15] = -7.
        ZeroWAMAgent.record_executed_action(agent, predicted)
        cached = agent.executed_actions[0]
        torch.testing.assert_close(decode(cached)[:, [7, 15]], torch.ones(1, 2), atol=1e-6, rtol=0)
        self.assertTrue(bool(torch.isfinite(cached).all()))
        self.assertEqual(cached[0, 15], predicted[0, 15])
        self.assertEqual(agent.gripper_history_corrections.tolist(), [1, 0])


if __name__ == '__main__':
    unittest.main()
