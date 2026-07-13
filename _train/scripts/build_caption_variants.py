#!/usr/bin/env python3
"""Build real-only edit JSONLs matching eva-diffusion's caption distribution.

For the es_hunt real-image subset, ``caption_joy`` is empty for every sample
in LanceDB (42,987/42,987 checked).  With eva's CaptionFormatMixer settings
(``plain_prob=0.5``, ``json_prob=0.5``, ``grok_vs_joy_prob=0.5``), both plain
lanes therefore resolve to ``short_description`` from the fused JSON.  The
actual distribution is:

  * 50% fused JSON ``short_description``
  * 50% full fused JSON

Both variants receive the same identity-reference edit prefix used by the eva
edit SFT experiment.  The validated source JSONL supplies normalized image
paths and fused JSON captions; current control paths are built from
``ref_person/<sample_id>.png``.  ``real_images.json`` filters by filename stem.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TextIO


DEFAULT_DATASET_ROOT = Path("/mnt/images/a.gainetdionov/Datasets/es_hunt_only")
EDIT_PREFIX = (
    "Use the person from Picture 1 as the exact identity reference.\n"
    "Preserve their facial features, hairstyle, eye shape and color, nose, lips,\n"
    "jawline, skin tone, natural skin texture, body type, proportions, and overall build.\n"
    "Keep the face unchanged and photorealistic.\n\n"
    "SCENE:\n"
)


def file_key(name: str) -> str:
    """Match build_jsonl.py semantics: use the basename before the first dot."""
    return os.path.basename(name).split(".", 1)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build short/full-JSON edit manifests for real es_hunt images."
    )
    parser.add_argument(
        "--real-images",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "real_images.json",
        help="JSON list of real-image filenames.",
    )
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "es_hunt_qwen_image_reference.jsonl",
        help="Validated source manifest containing image_path and fused JSON caption.",
    )
    parser.add_argument(
        "--control-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "ref_person",
        help="Directory containing current identity-reference PNGs keyed by sample ID.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory for generated manifests.",
    )
    parser.add_argument(
        "--short-output-name",
        default="es_hunt_real_edit_short.jsonl",
        help="Filename for the short_description variant.",
    )
    parser.add_argument(
        "--json-output-name",
        default="es_hunt_real_edit_json.jsonl",
        help="Filename for the full fused-JSON variant.",
    )
    parser.add_argument(
        "--no-validate-paths",
        action="store_true",
        help="Do not check that image_path and control_path exist and decode successfully with Pillow.",
    )
    parser.add_argument(
        "--validate-workers",
        type=int,
        default=32,
        help="Number of threads used to decode target and control images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output manifests atomically.",
    )
    return parser.parse_args()


def load_real_keys(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as f:
        values = json.load(f)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{path} must contain a JSON list of filenames")

    keys = {file_key(value) for value in values}
    if "" in keys:
        raise ValueError(f"{path} contains an empty filename key")
    if len(keys) != len(values):
        raise ValueError(
            f"{path} contains duplicate filename keys: {len(values)} entries, {len(keys)} unique keys"
        )
    return keys


def open_temp_output(final_path: Path, overwrite: bool) -> tuple[Path, TextIO]:
    if final_path.exists() and not overwrite:
        raise FileExistsError(f"{final_path} already exists; pass --overwrite to replace it")
    temp_path = final_path.with_name(f".{final_path.name}.tmp")
    temp_path.unlink(missing_ok=True)
    return temp_path, temp_path.open("w", encoding="utf-8")


def write_record(handle: TextIO, image_path: str, control_path: str, caption: str) -> None:
    record = {
        "image_path": image_path,
        "control_path": control_path,
        "caption": caption,
    }
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_broken_images(paths: set[str], workers: int) -> list[tuple[str, str]]:
    """Decode every image and return ``(path, error)`` entries for failures."""
    if workers < 1:
        raise ValueError(f"--validate-workers must be >= 1, got {workers}")
    try:
        from PIL import Image, ImageFile
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for image validation; install it or pass --no-validate-paths"
        ) from exc

    ImageFile.LOAD_TRUNCATED_IMAGES = False

    def check(path: str) -> tuple[str, str] | None:
        try:
            with Image.open(path) as image:
                image.load()
            return None
        except Exception as exc:
            return path, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return sorted(result for result in executor.map(check, paths) if result is not None)


def main() -> None:
    args = parse_args()
    real_keys = load_real_keys(args.real_images)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    short_path = args.output_dir / args.short_output_name
    json_path = args.output_dir / args.json_output_name
    if short_path.resolve() == json_path.resolve():
        raise ValueError("short and full-JSON output paths must differ")

    short_temp, short_file = open_temp_output(short_path, args.overwrite)
    json_temp: Path | None = None
    json_file: TextIO | None = None

    source_rows = 0
    written = 0
    seen_real_keys: set[str] = set()
    image_paths_to_validate: set[str] = set()

    try:
        json_temp, json_file = open_temp_output(json_path, args.overwrite)

        with args.source_jsonl.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                source_rows += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {args.source_jsonl}:{line_number}: {exc}") from exc

                image_path = record.get("image_path")
                full_caption = record.get("caption")
                if not isinstance(image_path, str):
                    raise ValueError(f"Missing image_path at {args.source_jsonl}:{line_number}")

                key = file_key(image_path)
                if key not in real_keys:
                    continue
                if key in seen_real_keys:
                    raise ValueError(f"Duplicate real-image key {key!r} at line {line_number}")
                seen_real_keys.add(key)

                control_path = str(args.control_dir / f"{key}.png")
                if not isinstance(full_caption, str) or not full_caption.strip():
                    raise ValueError(f"Missing fused JSON caption for real key {key!r} at line {line_number}")
                if not args.no_validate_paths:
                    if not Path(image_path).is_file():
                        raise FileNotFoundError(f"Image for real key {key!r} does not exist: {image_path}")
                    if not Path(control_path).is_file():
                        raise FileNotFoundError(f"Control image for real key {key!r} does not exist: {control_path}")
                    image_paths_to_validate.update((image_path, control_path))

                try:
                    caption_object = json.loads(full_caption)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid fused JSON caption for real key {key!r}: {exc}") from exc
                if not isinstance(caption_object, dict):
                    raise ValueError(f"Fused JSON caption for real key {key!r} is not an object")

                short_description = caption_object.get("short_description")
                if not isinstance(short_description, str) or not short_description.strip():
                    raise ValueError(f"Missing short_description for real key {key!r}")

                write_record(short_file, image_path, control_path, EDIT_PREFIX + short_description.strip())
                write_record(json_file, image_path, control_path, EDIT_PREFIX + full_caption.strip())
                written += 1

        if not args.no_validate_paths:
            print(
                f"Decoding {len(image_paths_to_validate)} target/control images "
                f"with {args.validate_workers} Pillow workers..."
            )
            broken_images = find_broken_images(image_paths_to_validate, args.validate_workers)
            if broken_images:
                preview = "\n".join(f"  {path}: {error}" for path, error in broken_images[:20])
                remainder = len(broken_images) - 20
                suffix = f"\n  ... and {remainder} more" if remainder > 0 else ""
                raise RuntimeError(
                    f"Pillow failed to decode {len(broken_images)} selected images:\n{preview}{suffix}"
                )

        short_file.flush()
        json_file.flush()
        os.fsync(short_file.fileno())
        os.fsync(json_file.fileno())
        short_file.close()
        json_file.close()

        os.replace(short_temp, short_path)
        os.replace(json_temp, json_path)
    except BaseException:
        short_file.close()
        if json_file is not None:
            json_file.close()
        short_temp.unlink(missing_ok=True)
        if json_temp is not None:
            json_temp.unlink(missing_ok=True)
        raise

    missing_from_source = sorted(real_keys - seen_real_keys, key=int)
    print(f"Real-image keys: {len(real_keys)}")
    print(f"Source rows scanned: {source_rows}")
    print(f"Rows written per variant: {written}")
    print(f"Real keys absent from validated source: {len(missing_from_source)}")
    if missing_from_source:
        print(f"Absent keys: {', '.join(missing_from_source)}")
    print(f"Short manifest: {short_path}")
    print(f"Full-JSON manifest: {json_path}")


if __name__ == "__main__":
    main()
