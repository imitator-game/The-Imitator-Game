"""Prepare streaming VAE caches. Edit DEFAULTS or use CLI overrides."""

from dataclasses import dataclass
import os
from typing import Literal

from .zero_wam.config import ZeroWAMConfig


@dataclass
class Args(ZeroWAMConfig):
    kind: Literal["all", "sim", "human"] = "all"
    num_shards: int = 1
    shard_index: int = 0
    encode_frames: int = 16
    build_only: bool = False
    verify_only: bool = False


# Model/data/cache paths inherit ZERO_WAM_MODEL, HUMAN_ROOT, SIM_ROOT,
# ZERO_WAM_CACHE. They can also be set explicitly here or on the command line.
DEFAULTS = Args(
    model_path=os.environ.get("ZERO_WAM_MODEL", ""),
    human_root=os.environ.get("HUMAN_ROOT", ""),
    sim_root=os.environ.get("SIM_ROOT", ""),
    cache_dir=os.environ.get("ZERO_WAM_CACHE", ""),
    protocol="scratch",  # use the same protocol as training
    kind="all",  # all / sim / human
    num_shards=1,
    shard_index=0,
    device="cuda",
    human_fps=12.0,
    encode_frames=16,
)


def main():
    import tyro

    args = tyro.cli(Args, default=DEFAULTS)
    args.validate()
    from .zero_wam.runtime import setup_runtime
    from .zero_wam.runtime import start_log

    setup_runtime(args.cache_dir)
    start_log(args.cache_dir, f"prepare_{args.kind}_rank{args.shard_index}")
    from .zero_wam.cache import prepare

    path = prepare(
        args,
        kind=args.kind,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        build_only=args.build_only,
        verify_only=args.verify_only,
        encode_frames=args.encode_frames,
    )
    print(f"Manifest: {path}", flush=True)


if __name__ == "__main__":
    main()
