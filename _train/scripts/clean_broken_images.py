#!/usr/bin/env python3
"""Check folders for unreadable / corrupt images and optionally delete them.

An image is considered broken if PIL cannot fully decode it (``Image.open().load()``
raises). This matches how the caching / training pipeline fails on such files
(the decode happens lazily during resize).

SAFETY: by default this runs as a DRY RUN and only reports what it would delete.
Pass ``--delete`` to actually remove the broken files.

Examples::

    # dry run: just list broken files
    python _train/scripts/clean_broken_images.py \
        /data/es_hunt_only/ref_person \
        /data/es_hunt_only/img_normalized_for_qwen_image

    # actually delete broken files (recursively)
    python _train/scripts/clean_broken_images.py /data/babesource/ref_person -r --delete
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from PIL import Image, ImageFile

logger = logging.getLogger("clean_broken_images")

IMAGE_EXTS_DEFAULT = "png,jpg,jpeg,webp,bmp"


def iter_image_files(folder: str, exts: set[str], recursive: bool):
    """Yield image file paths in ``folder`` filtered by extension."""
    if recursive:
        for root, _dirs, names in os.walk(folder):
            for name in names:
                if os.path.splitext(name)[1].lower().lstrip(".") in exts:
                    yield os.path.join(root, name)
    else:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower().lstrip(".") in exts:
                yield path


def check_image(path: str) -> Optional[str]:
    """Return an error string if the image cannot be decoded, else None."""
    try:
        with Image.open(path) as image:
            image.load()
        return None
    except Exception as exc:  # OSError, SyntaxError, etc.
        return f"{type(exc).__name__}: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and (optionally) delete unreadable/corrupt images in folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dirs", nargs="+", help="One or more folders to scan.")
    parser.add_argument("--delete", action="store_true", help="Actually delete broken files (default: dry run).")
    parser.add_argument("-r", "--recursive", action="store_true", help="Scan subdirectories recursively.")
    parser.add_argument("--exts", default=IMAGE_EXTS_DEFAULT, help="Comma-separated image extensions to check.")
    parser.add_argument("--workers", type=int, default=32, help="Parallel worker threads (I/O bound).")
    parser.add_argument(
        "--allow-truncated",
        action="store_true",
        help="Treat truncated images as valid (sets PIL LOAD_TRUNCATED_IMAGES).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ImageFile.LOAD_TRUNCATED_IMAGES = args.allow_truncated

    exts = {e.strip().lower().lstrip(".") for e in args.exts.split(",") if e.strip()}

    total_checked = 0
    total_broken = 0
    total_deleted = 0
    total_failed = 0

    for folder in args.dirs:
        if not os.path.isdir(folder):
            logger.warning("Not a directory, skipping: %s", folder)
            continue

        files = sorted(iter_image_files(folder, exts, args.recursive))
        broken: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, (path, err) in enumerate(zip(files, executor.map(check_image, files)), start=1):
                if err is not None:
                    broken.append((path, err))
                if index % 3000 == 0:
                    logger.info("%s: scanned %d/%d", folder, index, len(files))

        total_checked += len(files)
        total_broken += len(broken)
        logger.info("%s: %d files, %d broken", folder, len(files), len(broken))

        for path, err in broken:
            if args.delete:
                try:
                    os.remove(path)
                    total_deleted += 1
                    logger.info("DELETED %s (%s)", path, err)
                except OSError as exc:
                    total_failed += 1
                    logger.error("FAILED to delete %s: %s", path, exc)
            else:
                logger.info("BROKEN  %s (%s)", path, err)

    logger.info("---")
    logger.info("Checked %d image(s), found %d broken.", total_checked, total_broken)
    if args.delete:
        logger.info("Deleted %d, failed to delete %d.", total_deleted, total_failed)
    elif total_broken:
        logger.info("DRY RUN: nothing deleted. Re-run with --delete to remove the %d broken file(s).", total_broken)


if __name__ == "__main__":
    main()
