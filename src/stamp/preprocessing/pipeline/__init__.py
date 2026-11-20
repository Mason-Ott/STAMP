"""The patient-level pipeline: how the steps fit together, and its entry points.

Runs as `stamp preprocess` when the config has a `pipeline:` section, and as
`stamp worker` / `stamp workers start|status|stop` on several machines.

    sample_then_extract:  tile -> combine -> sample -> extract [-> encode]
    extract_then_sample:  tile -> extract_slides -> combine -> sample [-> encode]
                          (or import_slides instead of tile + extract_slides)

    config.py   the `pipeline:` config section, and which steps are valid
    plan.py     which files each step reads and writes, for each order
    run.py      entry points: run_pipeline_, run_pipeline_worker, pipeline_progress
    cli.py      `stamp workers start|status|stop`

The steps themselves are in ../steps, the data they exchange in ../data, and
slide/tile helpers in ../slides. See ../README.md for an overview.
"""
