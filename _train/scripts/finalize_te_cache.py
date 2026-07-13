#!/usr/bin/env python3
"""Assemble final short/full-JSON Qwen edit caches after TE shard jobs finish."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


LOCAL_DATASET_ROOT = Path("/mnt/images/a.gainetdionov/Datasets/es_hunt_only")
NFS_CACHE_ROOT = Path(
    "/mnt/images-not-ha/a.gainetdinov/qwen_sft_cache/"
    "sft_lork_lr_3e-4_1328_640_img_1_0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize sharded Qwen edit TE caches.")
    parser.add_argument("--te-shards-root", type=Path, default=NFS_CACHE_ROOT / "te_shards")
    parser.add_argument("--output-root", type=Path, default=NFS_CACHE_ROOT / "final")
    parser.add_argument(
        "--local-latents-1328",
        type=Path,
        default=LOCAL_DATASET_ROOT / "cache_qwen_image_sft_real_edit_short_1328",
    )
    parser.add_argument(
        "--local-latents-640",
        type=Path,
        default=LOCAL_DATASET_ROOT / "cache_qwen_image_sft_real_edit_short_640",
    )
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--expected-samples", type=int, default=42_984)
    return parser.parse_args()


def ensure_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"Symlink target mismatch: {link} -> {os.readlink(link)}, expected {target}")
        return
    if link.exists():
        raise FileExistsError(f"Cannot create symlink; path exists: {link}")
    link.symlink_to(target)


def collect_te_sources(shards_root: Path, variant: str, num_shards: int) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for shard_idx in range(num_shards):
        shard_dir = shards_root / f"shard_{shard_idx:02d}" / variant
        for path in shard_dir.glob("*_qie_te.safetensors"):
            if path.name in sources:
                raise RuntimeError(f"Duplicate {variant} TE filename: {path.name}")
            sources[path.name] = path
    return sources


def finalize_variant(
    variant: str,
    te_shards_root: Path,
    output_root: Path,
    local_latents: dict[str, dict[str, Path]],
    num_shards: int,
    expected: int,
) -> None:
    output_1328 = output_root / f"{variant}_1328"
    output_640 = output_root / f"{variant}_640"
    output_1328.mkdir(parents=True, exist_ok=True)
    output_640.mkdir(parents=True, exist_ok=True)

    existing_te = {path.name: path for path in output_1328.glob("*_qie_te.safetensors")}
    source_te = collect_te_sources(te_shards_root, variant, num_shards)
    if set(existing_te) & set(source_te):
        raise RuntimeError(f"{variant}: TE files exist in both shard and final directories")
    if len(set(existing_te) | set(source_te)) != expected:
        raise RuntimeError(
            f"{variant}: expected {expected} TE files, "
            f"found final={len(existing_te)}, sharded={len(source_te)}"
        )

    for name, source in source_te.items():
        os.replace(source, output_1328 / name)

    final_te = {path.name: path for path in output_1328.glob("*_qie_te.safetensors")}
    if len(final_te) != expected:
        raise RuntimeError(f"{variant}: expected {expected} consolidated TE files, found {len(final_te)}")

    for latent_name, latent_path in local_latents["1328"].items():
        ensure_symlink(output_1328 / latent_name, latent_path)
    for latent_name, latent_path in local_latents["640"].items():
        ensure_symlink(output_640 / latent_name, latent_path)
    for te_name, te_path in final_te.items():
        ensure_symlink(output_640 / te_name, te_path)

    print(
        f"{variant}: TE={len(final_te)}, "
        f"latent1328={len(local_latents['1328'])}, latent640={len(local_latents['640'])}"
    )


def main() -> None:
    args = parse_args()
    local_latents = {
        "1328": {path.name: path for path in args.local_latents_1328.glob("*_qie.safetensors")},
        "640": {path.name: path for path in args.local_latents_640.glob("*_qie.safetensors")},
    }
    for resolution, paths in local_latents.items():
        if len(paths) != args.expected_samples:
            raise RuntimeError(
                f"Expected {args.expected_samples} local {resolution} latents, found {len(paths)}"
            )

    finalize_variant(
        "short",
        args.te_shards_root,
        args.output_root,
        local_latents,
        args.num_shards,
        args.expected_samples,
    )
    finalize_variant(
        "json",
        args.te_shards_root,
        args.output_root,
        local_latents,
        args.num_shards,
        args.expected_samples,
    )


if __name__ == "__main__":
    main()
