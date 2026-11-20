"""Rename tile caches made before `canny_cutoff` was part of the cache key.

The tile cache key used to leave out `canny_cutoff`, so a cache made with one value
was silently reused after changing it. The key now includes it. Adding it changed
tiling.py, and the file's hash is also part of the key, so every existing cache
zip has a stale name and would be re-tiled.

The tiling itself did not change, so old caches are still correct, *provided you
know which canny_cutoff they were made with*: old caches don't record it. This
script renames them to the new key using the value you give.

    python -m stamp.preprocessing.slides.migrate_cache --cache-dir /path/to/cache \
        --canny-cutoff 0.02 --dry-run

0.02 is STAMP's default; use whatever your config had when the caches were made
(`none` if canny rejection was disabled). Run without --dry-run to rename.
"""

import argparse
import json
import os
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from stamp.preprocessing import tiling

# tiling.py hashes from before canny_cutoff was in the cache key.
# Caches from any other code version are left alone unless --any-code-hash.
KNOWN_OLD_CODE_HASHES = {
    "a82fe89cacd1c919006fd63e2853e3804935ddd1dbaafb471e943ebdb75a9c61",
}


def migrate(
    cache_dir: Path, canny_cutoff: float | None, dry_run: bool, any_code_hash: bool
) -> dict[str, int]:
    counts = {"renamed": 0, "already_current": 0, "target_exists": 0, "skipped": 0}
    for zip_path in sorted(cache_dir.glob("*.zip")):
        try:
            with ZipFile(zip_path) as zf:
                params = json.loads(zf.read("tiler_params.json").decode())
        except (BadZipFile, KeyError, json.JSONDecodeError):
            print(f"skip (not a STAMP tile cache): {zip_path.name}")
            counts["skipped"] += 1
            continue

        if "canny_cutoff" in params:
            counts["already_current"] += 1
            continue
        if params.get("code_sha256") not in KNOWN_OLD_CODE_HASHES and not any_code_hash:
            print(f"skip (made by an unknown tiling.py version): {zip_path.name}")
            counts["skipped"] += 1
            continue

        slide_path = Path(params["slide_path"])
        new_params = tiling.make_tiler_params(
            slide_path,
            cache_tiles_ext=params.get("tile_ext", "jpg"),
            tile_size_um=params["tile_size_um"],
            tile_size_px=params["tile_size_px"],
            max_supertile_size_slide_px=params["max_supertile_size_slide_px"],
            brightness_cutoff=params["brightness_cutoff"],
            canny_cutoff=canny_cutoff,
        )
        target = tiling.cache_file_for(cache_dir, slide_path, new_params)
        if target == zip_path:
            counts["already_current"] += 1
            continue
        if target.exists():
            counts["target_exists"] += 1
            continue
        print(
            f"{'would rename' if dry_run else 'rename'}: "
            f"{zip_path.name} -> {target.name}"
        )
        if not dry_run:
            os.replace(zip_path, target)
        counts["renamed"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--canny-cutoff",
        required=True,
        help="canny_cutoff the existing caches were made with (e.g. 0.02, or 'none')",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--any-code-hash",
        action="store_true",
        help="also migrate caches from tiling.py versions not known to this script",
    )
    args = parser.parse_args()
    canny = None if args.canny_cutoff.lower() == "none" else float(args.canny_cutoff)
    counts = migrate(args.cache_dir, canny, args.dry_run, args.any_code_hash)
    print(", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
