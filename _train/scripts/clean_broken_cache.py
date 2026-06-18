#!/usr/bin/env python3
"""Find (and optionally delete) broken .safetensors cache files.

A "broken" file is one that cannot be loaded during training (it raises
safetensors_rust.SafetensorError inside a dataloader worker, which then shows
up as a confusing multiprocessing PicklingError). Common causes: 0-byte /
truncated / partially-written cache files (interrupted caching, disk full) or
dangling symlinks.

By default this is a DRY RUN (only reports). Pass --delete to remove them.

Detection (fast, header-only by default):
  - size 0 / smaller than the 8-byte header
  - unreadable / invalid header length
  - header JSON not parseable
  - truncated: actual size < 8 + header_len + max(tensor data_offsets end)
Pass --full to additionally fully load each structurally-valid file (slower,
but reads every tensor exactly like training does).

Transient I/O safety: each file that fails is retried (--retries, default 3)
with a pause (--retry-delay) before being flagged, so a momentary NFS blip does
NOT cause a healthy file to be reported (and, with --delete, removed).

Example:
  python _train/scripts/clean_broken_cache.py \
    /mnt/images/a.gainetdionov/Datasets/es_hunt_only/cache_qwen_image_reference_1328 \
    /mnt/images/a.gainetdionov/Datasets/es_hunt_only/cache_qwen_image_reference_640 \
    /mnt/images/a.gainetdionov/Datasets/es_hunt_only/cache_qwen_image_reference_320 \
    --full --delete --log /tmp/broken_cache.txt
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def structural_error(path: str):
    """Return an error string if the file is structurally broken, else None."""
    try:
        size = os.path.getsize(path)  # follows symlinks; dangling -> OSError
    except OSError as e:
        return f"stat failed ({e})"
    if size == 0:
        return "0 bytes"
    if size < 8:
        return f"smaller than 8-byte header ({size} bytes)"
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if n <= 0 or 8 + n > size:
                return f"invalid header length ({n}); file size {size}"
            header = f.read(n)
    except OSError as e:
        return f"read failed ({e})"
    try:
        meta = json.loads(header)
    except Exception as e:
        return f"header JSON parse error ({e})"
    max_end = 0
    for k, v in meta.items():
        if k == "__metadata__":
            continue
        if not isinstance(v, dict):
            return f"bad header entry for key {k!r}"
        off = v.get("data_offsets")
        if off and len(off) == 2:
            max_end = max(max_end, off[1])
    expected = 8 + n + max_end
    if size < expected:
        return f"truncated: size {size} < expected {expected}"
    return None


def full_load_error(path: str):
    try:
        from safetensors.torch import load_file

        load_file(path)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def check(path: str, full: bool, retries: int, retry_delay: float):
    """Check one file, retrying on failure to ride out transient I/O (NFS) blips.

    Only files that fail EVERY attempt are reported as broken, so --delete will
    not remove a healthy file because of a momentary read error.
    """
    err = None
    for attempt in range(retries + 1):
        err = structural_error(path)
        if err is None and full:
            err = full_load_error(path)
        if err is None:
            return path, None
        if attempt < retries:
            time.sleep(retry_delay)
    return path, f"{err} (failed {retries + 1} attempts)"


def main():
    ap = argparse.ArgumentParser(description="Find/delete broken .safetensors cache files")
    ap.add_argument("dirs", nargs="+", help="cache directories to scan")
    ap.add_argument("--glob", default="*.safetensors", help="glob within each dir (default: *.safetensors)")
    ap.add_argument("--recursive", action="store_true", help="recurse into subdirectories")
    ap.add_argument("--full", action="store_true", help="also fully load structurally-valid files (slow)")
    ap.add_argument("--workers", type=int, default=32, help="parallel I/O workers (default: 32)")
    ap.add_argument("--retries", type=int, default=3, help="retry a failing file N times before flagging (default: 3)")
    ap.add_argument("--retry-delay", type=float, default=1.0, help="seconds between retries (default: 1.0)")
    ap.add_argument("--delete", action="store_true", help="delete broken files (default: dry run)")
    ap.add_argument("--log", default=None, help="write broken file paths to this log file")
    args = ap.parse_args()

    files = []
    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"WARN: not a directory, skipping: {d}", file=sys.stderr)
            continue
        if args.recursive:
            found = glob.glob(os.path.join(d, "**", args.glob), recursive=True)
        else:
            found = glob.glob(os.path.join(d, args.glob))
        print(f"{d}: {len(found)} files", flush=True)
        files.extend(found)

    if not files:
        print("No files found to scan.")
        return

    mode = "full load" if args.full else "structural"
    print(
        f"\nScanning {len(files)} files with {args.workers} workers "
        f"({mode} check, up to {args.retries} retries x {args.retry_delay}s)...\n",
        flush=True,
    )

    broken = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(check, p, args.full, args.retries, args.retry_delay) for p in files]
        for fut in as_completed(futs):
            path, err = fut.result()
            done += 1
            if err:
                broken.append((path, err))
                print(f"BROKEN: {path}  ->  {err}", flush=True)
            if done % 5000 == 0:
                print(f"... {done}/{len(files)} checked, {len(broken)} broken so far", flush=True)

    print(f"\n=== {len(broken)} broken file(s) out of {len(files)} ===", flush=True)
    for path, err in broken:
        print(f"  {os.path.basename(path)}  ->  {err}")

    if args.log and broken:
        with open(args.log, "w") as f:
            for path, err in broken:
                f.write(f"{path}\t{err}\n")
        print(f"\nWrote broken list to {args.log}")

    if not broken:
        print("No broken files found.")
        return

    if args.delete:
        print("\nDeleting broken files...", flush=True)
        deleted = 0
        for path, _ in broken:
            try:
                os.remove(path)
                deleted += 1
                print(f"DELETED: {path}", flush=True)
            except OSError as e:
                print(f"FAILED to delete {path}: {e}", file=sys.stderr)
        print(f"\nDeleted {deleted}/{len(broken)} files.")
    else:
        print("\nDry run (no files deleted). Re-run with --delete to remove them.")


if __name__ == "__main__":
    main()
