#!/usr/bin/env python3
"""Build a metadata JSONL file for musubi-tuner image (+control) datasets.

Files are matched across the input folders by a shared key: the part of the
filename before the FIRST dot. Filenames may contain several dots, so only the
first one is used as the separator, e.g.::

    0151561.raw.jpg  -> key "0151561"
    0151561.png      -> key "0151561"
    0151561.txt      -> key "0151561"

For every target image, an example is emitted only if the caption folder and
every requested control folder also contain a file with the same key. If any of
them is missing for that key, the example is skipped.

The caption file content is written verbatim into the ``caption`` field (after
stripping surrounding whitespace), so JSON-text captions are kept as-is.

By default every target/control image is opened and decoded with PIL up front;
examples whose images are unreadable/corrupt are excluded so they cannot crash
latent caching later. Pass ``--no-validate-images`` to skip this check.

Example::

    python _train/scripts/build_jsonl.py \
        --image-dir   /data/es_hunt_only/img_normalized_for_qwen_image \
        --control-dir /data/es_hunt_only/ref_person \
        --caption-dir /data/es_hunt_only/caption_grok_fused_json \
        --output-dir  /data/es_hunt_only \
        --output-name es_hunt_qwen_edit.jsonl
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional, TypeVar

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to periodic logging
    tqdm = None

logger = logging.getLogger("build_jsonl")

IMAGE_EXTS_DEFAULT = "png,jpg,jpeg,webp,bmp"

T = TypeVar("T")


def progress(iterable: Iterable[T], total: Optional[int], desc: str) -> Iterable[T]:
    """Wrap ``iterable`` with a tqdm bar, or periodic log lines if tqdm is absent."""
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit="file")

    def _logged() -> Iterable[T]:
        step = max(1, (total or 0) // 20)  # ~5% increments
        for index, item in enumerate(iterable, start=1):
            yield item
            if index % step == 0 or index == total:
                if total:
                    logger.info("%s: %d/%d (%d%%)", desc, index, total, index * 100 // total)
                else:
                    logger.info("%s: %d", desc, index)

    return _logged()


def file_key(filename: str) -> str:
    """Return the key for a filename: the basename text before the first dot."""
    return os.path.basename(filename).split(".", 1)[0]


def index_folder(folder: str, allowed_exts: Optional[set[str]]) -> dict[str, str]:
    """Map ``key -> absolute path`` for files in ``folder``.

    ``allowed_exts`` is a set of lowercase extensions without the dot, or None to
    accept any file. Duplicate keys are reported and the first (sorted) file wins.
    """
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Not a directory: {folder}")

    mapping: dict[str, str] = {}
    duplicates: dict[str, int] = defaultdict(int)
    names = sorted(os.listdir(folder))
    for name in progress(names, total=len(names), desc=f"Indexing {os.path.basename(os.path.normpath(folder))}"):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if allowed_exts is not None:
            ext = os.path.splitext(name)[1].lower().lstrip(".")
            if ext not in allowed_exts:
                continue
        key = file_key(name)
        if not key:
            continue
        if key in mapping:
            duplicates[key] += 1
            continue
        mapping[key] = os.path.abspath(path)

    for key, extra in duplicates.items():
        logger.warning(
            "Duplicate key '%s' in %s: %d extra file(s) ignored, kept '%s'",
            key,
            folder,
            extra,
            os.path.basename(mapping[key]),
        )
    return mapping


def find_broken_images(paths: set[str], workers: int) -> set[str]:
    """Return the subset of ``paths`` that PIL cannot fully decode.

    Each image is opened and loaded (the decode that the caching/training step
    performs), so truncated or corrupt files are detected here instead of later.
    """
    try:
        from PIL import Image, ImageFile
    except ImportError:
        raise SystemExit("Pillow is required for image validation. Install it or pass --no-validate-images.")

    ImageFile.LOAD_TRUNCATED_IMAGES = False

    def check(path: str) -> Optional[str]:
        try:
            with Image.open(path) as image:
                image.load()
            return None
        except Exception:  # OSError, SyntaxError, etc.
            return path

    path_list = list(paths)
    broken: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in progress(executor.map(check, path_list), total=len(path_list), desc="Validating images"):
            if result is not None:
                broken.add(result)
    return broken


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a metadata JSONL by matching files on the key before the first dot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-dir", required=True, help="Folder with target (generation) images.")
    parser.add_argument("--caption-dir", required=True, help="Folder with caption files (content used as the prompt).")
    parser.add_argument(
        "--control-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Folder with control images. Repeat the flag for multiple control images per example.",
    )
    parser.add_argument("--output-dir", required=True, help="Output folder for the JSONL file (created if missing).")
    parser.add_argument("--output-name", default="dataset.jsonl", help="Output JSONL filename.")
    parser.add_argument("--caption-ext", default=".txt", help="Caption file extension to match.")
    parser.add_argument(
        "--image-exts",
        default=IMAGE_EXTS_DEFAULT,
        help="Comma-separated image extensions for image/control folders.",
    )
    parser.add_argument(
        "--keep-empty-captions",
        action="store_true",
        help="Keep examples whose caption file is empty (skipped by default).",
    )
    parser.add_argument(
        "--validate-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open each target/control image with PIL and skip examples with unreadable images.",
    )
    parser.add_argument("--validate-workers", type=int, default=32, help="Parallel worker threads for image validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    image_exts = {e.strip().lower().lstrip(".") for e in args.image_exts.split(",") if e.strip()}
    caption_ext = args.caption_ext.strip().lower().lstrip(".")
    caption_exts = {caption_ext} if caption_ext else None

    image_map = index_folder(args.image_dir, image_exts)
    caption_map = index_folder(args.caption_dir, caption_exts)
    control_maps = [index_folder(d, image_exts) for d in args.control_dir]

    logger.info(
        "Indexed %d images, %d captions, control folders: [%s]",
        len(image_map),
        len(caption_map),
        ", ".join(str(len(m)) for m in control_maps) or "none",
    )

    # Validate images up front (in parallel) so broken targets/controls are
    # excluded from the JSONL instead of crashing latent caching later.
    broken_images: set[str] = set()
    if args.validate_images:
        paths_to_check: set[str] = set(image_map.values())
        for control_map in control_maps:
            paths_to_check.update(control_map.values())
        broken_images = find_broken_images(paths_to_check, args.validate_workers)
        logger.info("Validated %d image(s), %d broken (excluded).", len(paths_to_check), len(broken_images))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)

    written = 0
    skipped_caption = 0
    skipped_control = 0
    skipped_broken = 0
    skipped_empty = 0
    skipped_read = 0

    with open(out_path, "w", encoding="utf-8") as out_file:
        for key in progress(sorted(image_map), total=len(image_map), desc="Building JSONL"):
            caption_path = caption_map.get(key)
            if caption_path is None:
                skipped_caption += 1
                continue

            control_paths: list[str] = []
            missing_control = False
            for control_map in control_maps:
                control_path = control_map.get(key)
                if control_path is None:
                    missing_control = True
                    break
                control_paths.append(control_path)
            if missing_control:
                skipped_control += 1
                continue

            if image_map[key] in broken_images or any(c in broken_images for c in control_paths):
                skipped_broken += 1
                continue

            try:
                with open(caption_path, encoding="utf-8") as caption_file:
                    caption = caption_file.read().strip()
            except OSError as exc:
                logger.warning("Could not read caption '%s': %s", caption_path, exc)
                skipped_read += 1
                continue

            if not caption and not args.keep_empty_captions:
                skipped_empty += 1
                continue

            record: dict[str, str] = {"image_path": image_map[key]}
            if len(control_paths) == 1:
                record["control_path"] = control_paths[0]
            else:
                for i, control_path in enumerate(control_paths):
                    record[f"control_path_{i}"] = control_path
            record["caption"] = caption

            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Wrote %d example(s) -> %s", written, out_path)
    logger.info(
        "Skipped: %d no-caption, %d missing-control, %d broken-image, %d empty-caption, %d unreadable-caption",
        skipped_caption,
        skipped_control,
        skipped_broken,
        skipped_empty,
        skipped_read,
    )


if __name__ == "__main__":
    main()
