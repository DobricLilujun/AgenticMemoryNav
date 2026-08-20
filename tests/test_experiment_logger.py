import os
from pathlib import Path

from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun


def test_experiment_run_chdirs_into_run_folder(tmp_path: Path) -> None:
    previous_cwd = Path.cwd()
    run = ExperimentRun(tmp_path, {"foo": "bar"})
    try:
        assert Path.cwd() == run.path
    finally:
        run.close()
        os.chdir(previous_cwd)
