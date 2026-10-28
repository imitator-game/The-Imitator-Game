"""Complete-episode sampling over causal streaming VAE caches."""

from collections import OrderedDict
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from .model import IG_CHANNELS

ACTION_HISTORY_MODE = "zero_history_supervised_v1"


class IGDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        empty_text_path=None,
        seed=1,
        sampling_mode="episode",
        action_history_mode=ACTION_HISTORY_MODE,
        action_per_frame=16,
        normalization_path=None,
        action_representation="relative_episode",
        action_channel_offset=14,
    ):
        from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer

        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.records = self.manifest["sim_episodes"]
        self.humans = self.manifest["human_episodes"]
        self.action_per_frame = int(action_per_frame)
        self.seed = seed
        self.sampling_mode = sampling_mode
        self.action_history_mode = action_history_mode
        self.payloads = OrderedDict()
        self.empty_text = torch.load(empty_text_path, weights_only=True, map_location="cpu")
        self.normalizer = ActionNormalizer()
        from .action_space import EpisodeActionSpace, channel_ids
        self.action_channels = list(channel_ids(action_channel_offset))
        self.action_space = EpisodeActionSpace(json.loads(Path(normalization_path).read_text())) if normalization_path else None
        if self.action_space and self.action_space.statistics["representation"] != action_representation:
            raise ValueError("Normalization statistics use a different action representation")
        self.repo_indices = {}
        self.humans_by_repo = {}
        for i, record in enumerate(self.humans):
            self.humans_by_repo.setdefault(record["repo_id"], []).append(i)
        for record in self.records:
            repo_id = record["repo_id"]
            if record["human_repo_id"] not in self.humans_by_repo:
                raise ValueError(f"Missing human pairing for {repo_id}")
            if repo_id not in self.repo_indices:
                index = len(self.repo_indices)
                self.repo_indices[repo_id] = index
                stats = json.loads((Path(record["source"]["repo_path"]) / "meta/stats.json").read_text())
                self.normalizer.add_dataset_stats(
                    index, repo_id, stats, "observation.qpos_gripper_states", "action.qpos_gripper_actions"
                )

    def __len__(self):
        return len(self.records)

    def payload(self, path):
        if path not in self.payloads:
            self.payloads[path] = torch.load(path, weights_only=False, mmap=True, map_location="cpu")
            if len(self.payloads) > 8:
                self.payloads.popitem(last=False)
        self.payloads.move_to_end(path)
        return self.payloads[path]

    def __getitem__(self, key):
        return self.episode_item(key)

    def episode_item(self, key):
        """One complete episode, with native normalized-zero initial action group.

        IG pairs a robot task with a uniformly random same-task human episode on
        each read. Use a draw seed independent of cadence, worker scheduling and
        prefetch, preserving reproducible human draws after resume.
        """
        index, weight, epoch, batch_index, group_count, group_microsteps = key
        record = self.records[index]
        payload = self.payload(record["cache_path"])
        latent = payload["latent"].clone()
        frames, interval = latent.shape[1], self.action_per_frame
        anchors = torch.as_tensor(payload["latent_nominal_anchor_indices"])
        expected = torch.arange(frames) * interval
        if not torch.equal(anchors, expected) or payload.get("action_chunk_size") != interval:
            raise ValueError("Episode cache cadence differs from requested actions/latent")
        raw_actions = torch.as_tensor(payload["actions"]).float()
        raw_states = torch.as_tensor(payload["states"]).float()
        normalizer_index = self.repo_indices[record["repo_id"]]
        action_ids = (torch.arange(frames)[:, None] - 1) * interval + torch.arange(interval)[None]
        valid = (action_ids >= 0) & (action_ids < len(raw_actions))
        if self.action_space:
            normalized_actions, normalized_states = self.action_space.normalize_episode(raw_actions, raw_states)
            values = normalized_actions[action_ids.clamp(0, len(raw_actions)-1)]
        else:
            values = self.normalizer.normalize_action(raw_actions[action_ids.clamp(0, len(raw_actions)-1)], normalizer_index)
        values[0] = 0  # Native boundary padding AFTER normalization, supervised.
        valid[0] = True
        actions = torch.zeros(30, frames, interval, 1)
        mask = torch.zeros_like(actions, dtype=torch.bool)
        actions[self.action_channels, :, :, 0] = values.permute(2, 0, 1)
        mask[self.action_channels, :, :, 0] = valid[None]
        # prepare_episode selects the state at each chunk's causal START, not at
        # the end of the current action group (which would leak future state).
        state_ids = (torch.arange(frames)-1).clamp_min(0) * interval
        states = (normalized_states[state_ids.clamp_max(len(raw_states)-1)] if self.action_space else
                  self.normalizer.normalize_state(raw_states[state_ids.clamp_max(len(raw_states)-1)], normalizer_index))
        candidates = self.humans_by_repo[record["human_repo_id"]]
        draw = random.Random(f"{self.seed}:{epoch}:{index}:{batch_index}")
        human_index = candidates[draw.randrange(len(candidates))]
        human = self.payload(self.humans[human_index]["cache_path"])["latent"].clone()
        return dict(latents=latent, actions=actions, actions_mask=mask,
            text_emb=self.empty_text, icl_latents=human, icl_text_emb=self.empty_text,
            ig_states=states, ig_weight=weight, ig_video_length=frames,
            ig_human_length=human.shape[1], ig_epoch=epoch, ig_batch_index=batch_index,
            ig_group_count=group_count, ig_group_microsteps=group_microsteps,
            ig_sample_index=index, ig_episode=index, ig_control_time=0,
            ig_latent_start=0, ig_action_start=0, ig_human_index=human_index,
            ig_mcp_source_latents=latent, ig_mcp_source_length=frames)


class EpisodeBatchSampler(Sampler):
    """Metadata wrapper over the UNMODIFIED PyTorch DistributedSampler order.

    Native padding is intentional:50 episodes/4 ranks gives52 presentations.
    start_batch skips committed local batches, leaving the permutation intact.
    """
    def __init__(self, length, world, rank, seed=1):
        from torch.utils.data.distributed import DistributedSampler
        self.native = DistributedSampler(range(length), num_replicas=world,
            rank=rank, shuffle=True, seed=seed, drop_last=False)
        self.world, self.rank, self.epoch, self.start_batch = world, rank, 0, 0

    def __len__(self):
        return len(self.native)

    @property
    def total_size(self):
        return self.native.total_size

    def set_epoch(self, epoch):
        self.epoch, self.start_batch = epoch, 0
        self.native.set_epoch(epoch)

    def sample_offset_after(self, batch_index):
        return (batch_index + 1) * self.world

    def __iter__(self):
        for batch_index, index in enumerate(self.native):
            if batch_index >= self.start_batch:
                yield [(index, 1., self.epoch, batch_index, self.world, 1)]




def collate(samples):
    max_video = max(s["ig_video_length"] for s in samples)
    max_human = max(s["ig_human_length"] for s in samples)
    result = {}
    for key in ["latents", "actions", "actions_mask", "icl_latents"]:
        maximum = max_human if key == "icl_latents" else max_video
        padded = []
        for s in samples:
            value = s[key]
            length = value.shape[1]
            if length < maximum:
                if key == "latents":
                    value = torch.cat([value, value[:, -1:].expand(-1, maximum - length, -1, -1)], dim=1)
                else:
                    value = F.pad(value, (0, 0, 0, 0, 0, maximum - length))
            padded.append(value)
        result[key] = torch.stack(padded)
    result["ig_states"] = torch.stack(
        [torch.cat([s["ig_states"], s["ig_states"][-1:].expand(max_video - s["ig_video_length"], -1)]) for s in samples]
    )
    for key in ["text_emb", "icl_text_emb", "ig_mcp_source_latents"]:
        result[key] = torch.stack([s[key] for s in samples])
    for key in samples[0]:
        if key.startswith("ig_") and key not in ["ig_states", "ig_mcp_source_latents"]:
            result[key] = torch.tensor([s[key] for s in samples])
    return result


"""Exclude per-sample padding, retaining the native temporal mask predicate."""


def valid_seq_ids(model, seq_ids, blocks):
    if not hasattr(model, "ig_batch_lengths"):
        return seq_ids
    video_lengths, human_lengths, video_max, human_max = model.ig_batch_lengths
    batch = video_lengths.shape[0]
    device = seq_ids.device
    # The IG zed2i latent is14x14; native spatial patch is2x2.
    spatial = 49
    video = torch.arange(video_max * spatial, device=device)[None] < video_lengths[:, None] * spatial
    action = torch.arange(video_max * 16, device=device)[None] < video_lengths[:, None] * 16
    parts = [video, video]
    if blocks is model.blocks and human_max:
        human = torch.arange(human_max * spatial, device=device)[None] < human_lengths[:, None] * spatial
        parts.append(human)
    parts += [action, action]
    valid = torch.cat(parts, dim=1)
    padding = seq_ids.shape[0] - valid.shape[1]
    assert padding >= 0
    valid = torch.nn.functional.pad(valid, (0, padding))
    return seq_ids[None].expand(batch, -1).masked_fill(~valid, -1)
