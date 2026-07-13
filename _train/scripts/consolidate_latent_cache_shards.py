#!/usr/bin/env python3
"""Consolidate per-GPU Qwen-Image latent shards into flat cache directories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DATASET_ROOT = Path("/mnt/images/a.gainetdionov/Datasets/es_hunt_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate completed Qwen-Image latent-cache shards.")
    parser.add_argument(
        "--shards-root",
        type=Path,
        default=DATASET_ROOT / "cache_qwen_image_sft_real_edit_short_latent_shards",
    )
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--expected-per-resolution", type=int, default=42_984)
    parser.add_argument(
        "--output-1328",
        type=Path,
        default=DATASET_ROOT / "cache_qwen_image_sft_real_edit_short_1328",
    )
    parser.add_argument(
        "--output-640",
        type=Path,
        default=DATASET_ROOT / "cache_qwen_image_sft_real_edit_short_640",
    )
    return parser.parse_args()


def consolidate_resolution(
    shards_root: Path,
    resolution: str,
    output_dir: Path,
    num_shards: int,
    expected: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name: path for path in output_dir.glob("*_qie.safetensors")}
    sources: dict[str, Path] = {}

    for shard_idx in range(num_shards):
        shard_dir = shards_root / f"shard_{shard_idx:02d}" / resolution
        for path in shard_dir.glob("*_qie.safetensors"):
            if path.name in sources:
                raise RuntimeError(f"Duplicate source cache filename: {path.name}")
            sources[path.name] = path

    all_names = set(existing) | set(sources)
    if len(all_names) != expected:
        raise RuntimeError(
            f"{resolution}: expected {expected} unique cache files, "
            f"found {len(existing)} consolidated + {len(sources)} sharded "
            f"({len(all_names)} unique)"
        )

    conflicts = set(existing) & set(sources)
    if conflicts:
        raise RuntimeError(
            f"{resolution}: {len(conflicts)} files exist in both source and destination; "
            "resolve them before consolidation"
        )

    for name, source in sources.items():
        os.replace(source, output_dir / name)

    final_count = sum(1 for _ in output_dir.glob("*_qie.safetensors"))
    if final_count != expected:
        raise RuntimeError(f"{resolution}: expected {expected} final files, found {final_count}")
    print(f"{resolution}: consolidated {len(sources)} files; final count={final_count}")


def main() -> None:
    args = parse_args()
    consolidate_resolution(
        args.shards_root,
        "1328",
        args.output_1328,
        args.num_shards,
        args.expected_per_resolution,
    )
    consolidate_resolution(
        args.shards_root,
        "640",
        args.output_640,
        args.num_shards,
        args.expected_per_resolution,
    )


if __name__ == "__main__":
    main()
