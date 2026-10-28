"""CPU checks for protocol selection, distributed tail weighting and action mapping."""

import unittest

import torch

from examples.baselines.zero_wam.zero_wam.config import REPO_ROOT
from examples.baselines.zero_wam.zero_wam.config import ZeroWAMConfig
from examples.baselines.zero_wam.zero_wam.config import outside_repository
from examples.baselines.zero_wam.zero_wam.model import pack_actions
from examples.baselines.zero_wam.zero_wam.model import unpack_actions


class ProtocolTests(unittest.TestCase):
    def test_repository_protocols(self):
        expected = {
            "scratch": (5, 20, 50, 200),
            "c15": (15, 60, 750, 3000),
            "c30": (30, 120, 1500, 6000),
            "c45": (48, 180, 2400, 9000),
        }
        for name, counts in expected.items():
            config = ZeroWAMConfig(protocol=name)
            human, robot = config.selections()
            count = lambda rows: sum(int(r["train"].split(":")[1]) - int(r["train"].split(":")[0]) for r in rows)
            self.assertEqual((len(human), len(robot), count(human), count(robot)), counts)

    def test_single_task_filter_preserves_pairing(self):
        config = ZeroWAMConfig(task="L0_TwoRobotPickRemoteControl-v1", episode_count=10)
        human, robot = config.selections()
        self.assertEqual([r["repo_id"] for r in human], ["human_H4"])
        self.assertEqual([r["repo_id"] for r in robot], [config.task])
        self.assertEqual(human[0]["train"], "0:10")

    def test_explicit_fifty_episode_pilot_does_not_expand_full_protocol(self):
        config = ZeroWAMConfig(task="L0_TwoRobotPickRemoteControl-v1", episode_count=50)
        human, robot = config.selections()
        self.assertEqual([r["train"] for r in human+robot], ["0:50", "0:50"])
        with self.assertRaisesRegex(ValueError, "training split"):
            ZeroWAMConfig(episode_count=50).selections()

    def test_no_artifacts_in_public_repository(self):
        for path in [REPO_ROOT / "runs", REPO_ROOT / "examples/baselines/zero_wam/cache"]:
            with self.assertRaises(ValueError):
                outside_repository(str(path), "output")


class SamplingTests(unittest.TestCase):


    def test_channel_mapping_round_trip(self):
        actions = torch.arange(2 * 32 * 16, dtype=torch.float32).reshape(2, 32, 16)
        self.assertTrue(torch.equal(unpack_actions(pack_actions(actions)), actions))


if __name__ == "__main__":
    unittest.main()
