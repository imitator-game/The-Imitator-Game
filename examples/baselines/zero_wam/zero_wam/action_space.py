"""Episode-referenced joint coordinates and train-split quantile statistics."""

import json
from pathlib import Path

import torch

ACTION_JOINTS = (*range(7), *range(8, 15))
STATE_JOINTS = (*range(7), *range(9, 16))


def channel_ids(offset=14):
    return (*range(offset, offset + 7), 28, *range(offset + 7, offset + 14), 29)


def episode_coordinates(actions, states, representation):
    """Use the initial measured state, preserving both gripper conventions."""
    actions, states = actions.clone(), states.clone()
    if representation == "relative_episode":
        origin = states[0, list(STATE_JOINTS)].clone()
        actions[..., list(ACTION_JOINTS)] -= origin
        states[..., list(STATE_JOINTS)] -= origin
    return actions, states


def fit_statistics(manifest, representation, output):
    """Fit each dimension over all selected real rows, before VAE padding."""
    from .cache import read_episode

    actions, states = [], []
    for record in manifest["sim_episodes"]:
        raw = read_episode(record)
        a, s = episode_coordinates(torch.as_tensor(raw["actions"]), torch.as_tensor(raw["states"]), representation)
        actions.append(a)
        states.append(s)
    result = dict(schema="ig_episode_joint_quantiles_v1", representation=representation,
                  selection_identity=manifest["selection_identity"],
                  records_sha256=manifest["records_sha256"], episodes=len(actions),
                  rows=sum(len(x) for x in actions), origin="initial_measured_joint_state",
                  gripper="absolute", clip=2.0)
    for name, values in [("action", actions), ("state", states)]:
        q = torch.quantile(torch.cat(values).float(), torch.tensor([0.01, 0.99]), dim=0)
        result[name] = dict(q01=q[0].tolist(), q99=q[1].tolist())
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2))
    return result


class EpisodeActionSpace:
    def __init__(self, statistics):
        self.statistics = statistics
        self.relative = statistics["representation"] == "relative_episode"
        self.clip = float(statistics["clip"])

    def bounds(self, value, name):
        stats = self.statistics[name]
        return value.new_tensor(stats["q01"]), value.new_tensor(stats["q99"])

    def normalize(self, value, name):
        lo, hi = self.bounds(value, name)
        return (2 * (value - lo) / (hi - lo + 1e-6) - 1).clamp(-self.clip, self.clip)

    def denormalize(self, value, name):
        lo, hi = self.bounds(value, name)
        return (value + 1) / 2 * (hi - lo + 1e-6) + lo

    def normalize_episode(self, actions, states):
        actions, states = episode_coordinates(actions, states, self.statistics["representation"])
        return self.normalize(actions, "action"), self.normalize(states, "state")

    def encode_state(self, state, origin):
        value = state.clone()
        if self.relative:
            value[..., list(STATE_JOINTS)] -= origin[..., list(STATE_JOINTS)]
        return self.normalize(value, "state")

    def decode_action(self, normalized, origin):
        value = self.denormalize(normalized, "action")
        if self.relative:
            reference = origin.to(device=value.device, dtype=value.dtype)
            value[..., list(ACTION_JOINTS)] += reference[..., list(STATE_JOINTS)]
        return value


class EpisodeEvaluationNormalizer:
    """Pass measured states to the agent; decode its commands in the same frame."""
    def __init__(self, original, agent):
        self.original, self.agent = original, agent

    def normalize_state(self, state, dataset_idx, method="bounds_q99"):
        return state

    def denormalize_action(self, action, dataset_idx, method="bounds_q99"):
        return self.agent.action_space.decode_action(action, self.agent.episode_origin)

    def __getattr__(self, name):
        return getattr(self.original, name)
