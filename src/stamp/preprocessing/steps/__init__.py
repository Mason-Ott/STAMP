"""The patient-level pipeline's steps, one module each.

Every step module has the same shape:
  * a frozen settings dataclass (or a config model) with everything it needs;
  * pure functions that do the work on `Bag`s, without touching the file system
    or knowing about other steps (e.g. `combine_slide_bags`, `sample_bag`);
  * `run(item, inputs, output, *, settings, ...)`: reads `inputs`, calls the pure
    functions, writes `output`. It is all the scheduler
    (stamp.distributed.workqueue) calls.

Which files feed which step is decided in ../pipeline/plan.py, not here. Model
loading for extract and encode is in models.py.
"""
