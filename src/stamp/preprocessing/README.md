# Preprocessing

Turns WSIs into what training needs. Two ways to run it:

* **Per-slide (STAMP's original):** `stamp preprocess` with only a
  `preprocessing:` section writes one feature file per slide.
* **Patient-level pipeline:** add a `pipeline:` section and `stamp preprocess`
  instead writes one tile bag per patient, with optional sampling and an
  optional slide encoder (e.g. TITAN). The rest of this file is about this.

```
sample_then_extract:  tile -> combine -> sample -> extract [-> encode]
extract_then_sample:  tile -> extract_slides -> combine -> sample [-> encode]
                      (with slide_feat_dir: import_slides -> combine -> sample)
```

## Config

```yaml
preprocessing:            # unchanged: tiling + extractor settings
  output_dir: /data/out
  wsi_dir: /data/wsis
  cache_dir: /data/out/tile_cache
  extractor: conch1_5
  device: cuda

pipeline:
  clinical_table: /data/clinical.csv   # patient_id, slide_id, label, hospital
  order: sample_then_extract
  sampling: {n_tiles: 8192, strategy: first_n, pad_mode: none}
  encoder: titan                        # optional
```

`slide_id` is the WSI file name without its extension. Patients are keyed by
`hospital_patientid`, the same `group_key` the split pipeline uses. All
settings are documented in `pipeline/config.py` and `distributed/config.py`.

## Outputs (under `preprocessing.output_dir`)

Folders are named after the settings behind their contents, so **one
`output_dir` holds any number of runs** (other extractors, sampling settings,
orders, encoders) and runs reuse what they share. T = tiling settings (e.g.
`256um_224px_3fa91c`), G = `T_gap1` (T plus the combine settings), E = the
extractor, S = the sampling settings (`first_n-8192`, `random-8192-seed0`,
`full` without sampling):

| folder | contents |
|---|---|
| `tile_positions/T/` | tile positions, one file per slide |
| `patient_positions/G/` | all tile positions of a patient |
| `sampled_positions/G/S/` | the sampled tile positions |
| `slide_features/E/T/` | `extract_then_sample`: per-slide features |
| `patient_features/E/G/` | `extract_then_sample`: patient features, all tiles |
| **`features/E/G/S/`** | **training bags**: one per patient, sampled tiles with features |
| `embeddings/<encoder>/E/G/S/` | one embedding per patient (only with `encoder`) |
| `overviews/tiles/T/`, `overviews/sampled/G/S/` | thumbnails, if `save_visualisations` |

Both orders end in the same `features/E/G/S/` folder, because they produce the
same training bags: if one order has run, the other finds the work done. With
`slide_feat_dir`, T is `imported_<hash>`. Each results folder has a
`settings.json` with the full settings it was made with.

What a second run reuses: another `n_tiles`/`strategy` redoes only sampling and
extraction; another `extractor` redoes only extraction (the tile positions and
the STAMP tile cache are shared); adding an `encoder` redoes only encoding;
changing tile size or cutoffs starts from the slides (the tile cache is keyed
the same way).

**Encoding needs a compatible extractor.** An `encoder` only accepts features
from certain extractors (TITAN: `conch1_5`). With any other extractor the run
logs a warning and skips the `encode` step; asking for `--steps encode`
explicitly is an error.

**Intermediate files are kept** (`keep_intermediates: true`, the default), which
is what makes that reuse possible. Set it to `false` to save disk: a stage's
file is then deleted once the next stage has written its own, so only the latest
finished stage remains (plus training bags, embeddings and overviews), and a run
with other settings starts again from the slides. Rerunning the same
configuration still skips everything done. In `extract_then_sample` the
slide features are the big files.

Every `.h5` above (except the embeddings) is a *bag*; `data/bags.py` documents
the format. In short: `feats (N, D)`; `coords (N, 2)` in µm, with the patient's
slides laid side by side; `coords_slide` (position on the tile's own WSI);
`slide_idx`; `tile_idx`; `mask` (False on zero-padding rows); attributes
`label`, `group_key`, `patient_id`, `hospital`, `slides`, `sampling`,
`extractor`, `tile_size_um`, `tile_size_px`. They are also valid STAMP feature
files, so `stamp encode_slides` and STAMP training can read them.

## Sampling

`first_n` sorts each slide's tiles in raster order (top-to-bottom, then
left-to-right) and takes the first `n_tiles // n_slides` from each slide. With
`redistribute: true`, quota a small slide can't use goes to the patient's other
slides. To add a strategy, see `steps/sample.py`.

## Running on several machines

List the machines in the config:

```yaml
pipeline:
  workers:
    hosts: [lab01, lab02, lab03]
    activate: source /home/mason/masters/STAMP/.venv/bin/activate
```

Then, from your own machine:

```bash
stamp -c config.yaml workers start                  # a worker on every host
stamp -c config.yaml workers start --hosts lab04    # add a machine later
stamp -c config.yaml workers start --steps tile     # e.g. only tile for now
stamp -c config.yaml workers status                 # progress per step, workers per host
stamp -c config.yaml workers stop                   # stop cleanly (or --force)
```

Each host runs `stamp worker` in a detached `screen` session, so it keeps going
after you disconnect; `screen -r` on the host shows it, and its log is in
`<output_dir>/logs/<host>.log`. Starting a host that already has a worker for
this run does nothing.

There is no queue or server. All workers run the same steps and share the work
through lock files: each slide or patient is claimed with `<output>.lock`
(refreshed while its owner works), so no two machines do the same item. A step
whose inputs are still being produced elsewhere waits for them. A stopped
worker releases its lock; a crashed one's lock goes stale after
`coordination.lock_stale_after_s` and another worker takes the item over.
Errors leave `<output>.failed` with the traceback; delete it, or set
`coordination.retry_failed: true`, to retry.

On shared lab machines a worker won't start while someone else is logged in,
and stops cleanly if someone logs in later (`yield_to_other_users`, using
`who`). Its unfinished item is left for the other workers.

Requirements: passwordless SSH (keys) to every host, `screen` on every host,
and the same paths everywhere (config file, `wsi_dir`, `output_dir`,
`cache_dir`), e.g. a network-mounted home folder. `stamp worker` can also be
run by hand, or by any other job launcher.

## Code structure

```
stamp/preprocessing/
├── __init__.py, config.py, tiling.py, extractor/   STAMP's per-slide preprocessing
├── pipeline/   how the steps fit together
│   ├── config.py     the `pipeline:` config section; which steps are valid
│   ├── plan.py       which files each step reads and writes, for each order
│   ├── run.py        entry points: run_pipeline_, run_pipeline_worker, pipeline_progress
│   └── cli.py        `stamp workers start|status|stop`
├── steps/      one module per step, plus models.py (loading extractors/encoders)
│   └── tile, import_slides, extract, combine, sample, encode
├── data/       what the steps exchange
│   ├── bags.py       the tile bag (HDF5 format documented at the top)
│   ├── clinical.py   clinical table -> patients and slides
│   └── files.py      atomic file writing
└── slides/     slides and tiles, on top of STAMP's tiler
    ├── wsi.py, geometry.py, tile_cache.py
    └── migrate_cache.py

stamp/distributed/   running work on several machines (not preprocessing-specific)
├── workqueue.py      runs a step: lock files, waiting for inputs, failure markers
├── progress.py       how far a step has got, from its outputs
├── worker.py         a job as one of several workers; clean stop; leaving shared machines
├── presence.py       who else is logged in
├── remote.py         start/check/stop workers over SSH
└── config.py
```

The top-level `.py` files in `stamp/preprocessing/` are STAMP's. STAMP hashes
exactly those files (not subfolders) to name feature folders, so the pipeline
lives in subfolders and can change without renaming your feature folders.

Each step module has the same shape: a settings object, pure functions that
work on `Bag`s in memory (`combine_slide_bags`, `sample_bag`,
`extract_features`, ...), and `run(item, inputs, output)`. Steps don't know about
each other or about other machines; `pipeline/plan.py` connects them and
`distributed/workqueue.py` runs them.

## Upgrading an existing tile cache

`canny_cutoff` is now part of the tile cache key. Before, changing it silently
reused tiles filtered with the old value. Fixing this changed `tiling.py`, so
existing cache zips no longer match their key and would be re-tiled. The tiling
itself is unchanged, so you can rename the old caches instead, giving the
`canny_cutoff` they were made with (STAMP's default is 0.02):

```bash
python -m stamp.preprocessing.slides.migrate_cache --cache-dir /path/to/cache --canny-cutoff 0.02 --dry-run
python -m stamp.preprocessing.slides.migrate_cache --cache-dir /path/to/cache --canny-cutoff 0.02
```

With `generate_hash: true`, plain `stamp preprocess` also writes to a new feature
folder name now, because the folder name includes a hash of
`stamp/preprocessing/`. Existing feature files are still valid; use them with
`pipeline.slide_feat_dir`.

## Tests

```bash
pytest tests/preprocessing tests/distributed
```

Synthetic slides and small fake models, so no weights or GPU are needed. The
SSH tests use stand-in `ssh` and `screen` scripts.
