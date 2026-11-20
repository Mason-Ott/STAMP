"""Slides and their tiles, on top of STAMP's tiler (stamp.preprocessing.tiling).

wsi.py            slide metadata (resolution, size)
geometry.py       tile order and which tiles lie inside the scan
tile_cache.py     reading STAMP's tile cache
migrate_cache.py  renaming tile caches made before canny_cutoff was in the key
"""
