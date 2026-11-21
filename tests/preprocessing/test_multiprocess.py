"""Several `stamp preprocess` processes sharing one run through the command line."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from stamp.preprocessing.data.bags import read_bag

from .conftest import SLIDES, TILE_SIZE_PX, TILE_SIZE_UM


def test_processes_split_the_work_and_wait_for_each_other(dataset, tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "preprocessing": {
                    "output_dir": str(tmp_path / "out"),
                    "wsi_dir": str(dataset.wsi_dir),
                    "cache_dir": str(tmp_path / "cache"),  # fresh: force real tiling
                    "extractor": "conch1_5",
                    "tile_size_um": TILE_SIZE_UM,
                    "tile_size_px": TILE_SIZE_PX,
                    "device": "cpu",
                    "max_workers": 2,
                },
                "pipeline": {
                    "experiment": "e",
                    "clinical_table": str(dataset.clinical_table),
                    "sampling": {"n_tiles": 50},
                    "coordination": {"poll_interval_s": 0.5},
                },
            }
        )
    )
    command = [sys.executable, "-m", "stamp", "-c", str(config), "preprocess"]
    processes = [
        subprocess.Popen(
            [*command, "--steps", "tile,combine,sample"],
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(3)
    ]
    logs = [process.communicate(timeout=600)[0] for process in processes]
    for process, log in zip(processes, logs, strict=True):
        assert process.returncode == 0, log[-3000:]

    out = tmp_path / "out"
    assert not list(out.rglob("*.failed")) and not list(out.rglob("*.lock"))
    tiled = sum(_count(log, step="tile", key="done") for log in logs)
    assert tiled == len(SLIDES), "each slide should be tiled by exactly one process"

    sampled = {
        path.stem: read_bag(path).sampling.per_slide
        for path in out.glob("e/*/sampled_positions/*.h5")
    }
    assert sampled == {"H1_P1": [25, 25], "H1_P2": [36, 7], "H2_P1": [8]}
    for path in out.glob("e/*/sampled_positions/*.h5"):
        assert np.all(read_bag(path).mask)


def test_unknown_step_is_rejected_by_the_command_line(dataset, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "stamp", "preprocess", "--steps", "tile,nonsense"],
        env=os.environ,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unknown steps ['nonsense']" in result.stderr


def _count(log: str, *, step: str, key: str) -> int:
    """Sum `<key>=<n>` over the summary lines that run_step logs for `step`."""
    pattern = re.compile(rf"\t{step}: .*\b{key}=(\d+)")
    return sum(int(match.group(1)) for match in pattern.finditer(log))
