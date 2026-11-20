"""Running work on several machines that share a filesystem.

Nothing here is specific to preprocessing: a job describes its work as `Step`s
(items, output path per item, inputs per item) and these modules do the rest.

    workqueue.py   runs a Step: claims items with lock files, waits for inputs
                   being produced elsewhere, records failures
    progress.py    how far a Step has got, from its output files
    worker.py      runs a job as one of several workers: clean stopping, and
                   leaving shared machines when someone else logs in
    presence.py    who else is logged in
    remote.py      starts, checks and stops workers on other machines over SSH
    config.py      settings for the above
"""
