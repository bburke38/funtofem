"""
Tests for SlurmSweepSubmitter: dry-run script content, point.json creation,
skip_completed logic, and extra_directives injection.
"""

import csv
import json
import os
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Import directly from the sweep subpackage to avoid pulling in native TACS/FUN3D
# ---------------------------------------------------------------------------
import importlib.util
import pathlib

_SWEEP_PKG = pathlib.Path(__file__).parents[3] / "funtofem" / "sweep"


def _load_sweep_module(name: str):
    spec = importlib.util.spec_from_file_location(
        f"funtofem.sweep.{name}",
        str(_SWEEP_PKG / f"{name}.py"),
    )
    # No submodule_search_locations: these are plain modules, not packages.
    # module_from_spec then derives __package__ = "funtofem.sweep" from
    # spec.parent, which is what the modules' relative imports need. Setting
    # __package__ by hand while the spec says "package" makes them disagree
    # and Python emits ImportWarning: __package__ != __spec__.parent.
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"funtofem.sweep.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


# Ensure funtofem package stub exists
if "funtofem" not in sys.modules:
    _funtofem_stub = types.ModuleType("funtofem")
    _funtofem_stub.__path__ = [str(pathlib.Path(__file__).parents[3] / "funtofem")]
    _funtofem_stub.__package__ = "funtofem"
    sys.modules["funtofem"] = _funtofem_stub

_strategy_mod = _load_sweep_module("strategy")
_mesh_mode_mod = _load_sweep_module("mesh_mode")

if "funtofem.sweep" not in sys.modules:
    _sweep_pkg = types.ModuleType("funtofem.sweep")
    sys.modules["funtofem.sweep"] = _sweep_pkg

sys.modules["funtofem.sweep"].SweepStrategy = _strategy_mod.SweepStrategy
sys.modules["funtofem.sweep"].CartesianStrategy = _strategy_mod.CartesianStrategy
sys.modules["funtofem.sweep"].MeshMode = _mesh_mode_mod.MeshMode

_param_sweep_mod = _load_sweep_module("parameter_sweep")
_slurm_mod = _load_sweep_module("slurm_submitter")

CartesianStrategy = _strategy_mod.CartesianStrategy
MeshMode = _mesh_mode_mod.MeshMode
ParameterSweep = _param_sweep_mod.ParameterSweep
SlurmSweepSubmitter = _slurm_mod.SlurmSweepSubmitter
_make_key = _param_sweep_mod._make_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sweep(tmp_path, params=None, output_csv=None):
    """Return a minimal ParameterSweep with no callbacks registered."""
    if params is None:
        params = {"alpha": [1.0, 2.0], "beta": [10.0]}
    if output_csv is None:
        output_csv = str(tmp_path / "sweep_results.csv")
    return ParameterSweep(
        params,
        strategy=CartesianStrategy(),
        mesh_mode=MeshMode.NONE,
        cfd_root=str(tmp_path / "cfd"),
        struct_root=str(tmp_path / "struct"),
        output_csv=output_csv,
    )


_BASE_SLURM_CONFIG = {
    "account": "TEST_ACCOUNT",
    "partition": "test_partition",
    "qos": "test_qos",
    "nodes": 2,
    "ntasks_per_node": 64,
    "walltime": "04:00:00",
}


def _make_submitter(sweep, extra_directives=None, preamble="", log_dir="sweep_logs"):
    return SlurmSweepSubmitter(
        sweep,
        _BASE_SLURM_CONFIG,
        sweep_script="my_sweep.py",
        preamble=preamble,
        log_dir=log_dir,
        extra_directives=extra_directives,
    )


# ---------------------------------------------------------------------------
# Test: _build_job_script produces expected content
# ---------------------------------------------------------------------------


class TestBuildJobScript:
    def test_slurm_headers_present(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        submitter = _make_submitter(sweep)
        key = "pt_abc123def456"
        point_json = f"{sweep.cfd_root}/{key}/point.json"
        script = submitter._build_job_script(key, point_json)

        assert "#!/bin/bash" in script
        assert "#SBATCH -A TEST_ACCOUNT" in script
        assert f"#SBATCH --job-name={key}" in script
        assert "#SBATCH -p test_partition" in script
        assert "#SBATCH -q test_qos" in script
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --ntasks-per-node=64" in script
        assert "#SBATCH --time=04:00:00" in script

    def test_log_paths_use_key(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        submitter = _make_submitter(sweep, log_dir="my_logs")
        key = "pt_testkey000001"
        script = submitter._build_job_script(key, "some/path/point.json")

        assert f"#SBATCH --output=my_logs/{key}.out" in script
        assert f"#SBATCH --error=my_logs/{key}.err" in script

    def test_srun_command_with_point_file(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        submitter = _make_submitter(sweep)
        key = "pt_abc123def456"
        point_json_path = f"{sweep.cfd_root}/{key}/point.json"
        script = submitter._build_job_script(key, point_json_path)

        assert f"srun python my_sweep.py --point-file {point_json_path}" in script

    def test_preamble_inserted_before_srun(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        preamble = "source $HOME/.bashrc\nmodule load python"
        submitter = _make_submitter(sweep, preamble=preamble)
        key = "pt_abc123def456"
        script = submitter._build_job_script(key, "some/point.json")

        preamble_pos = script.find(preamble)
        srun_pos = script.find("srun python")
        assert preamble_pos != -1, "Preamble not found in script"
        assert srun_pos != -1, "srun line not found in script"
        assert preamble_pos < srun_pos, "Preamble must appear before srun"

    def test_extra_directives_injected(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        extra = ["--mail-type=END", "--mail-user=me@example.com"]
        submitter = _make_submitter(sweep, extra_directives=extra)
        key = "pt_abc123def456"
        script = submitter._build_job_script(key, "some/point.json")

        assert "#SBATCH --mail-type=END" in script
        assert "#SBATCH --mail-user=me@example.com" in script

    def test_no_extra_directives_by_default(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        submitter = _make_submitter(sweep)
        key = "pt_abc123def456"
        script = submitter._build_job_script(key, "some/point.json")

        # Only standard directives should appear
        standard_prefixes = {
            "#SBATCH -A",
            "#SBATCH --job-name",
            "#SBATCH -p",
            "#SBATCH -q",
            "#SBATCH --nodes",
            "#SBATCH --ntasks-per-node",
            "#SBATCH --time",
            "#SBATCH --output",
            "#SBATCH --error",
        }
        sbatch_lines = [
            line for line in script.splitlines() if line.startswith("#SBATCH")
        ]
        for line in sbatch_lines:
            assert any(
                line.startswith(d) for d in standard_prefixes
            ), f"Unexpected #SBATCH directive: {line}"


# ---------------------------------------------------------------------------
# Test: dry_run=True prints scripts without calling sbatch
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_prints_script(self, tmp_path, capsys):
        sweep = _make_sweep(tmp_path)
        submitter = _make_submitter(sweep)
        submitter.submit(dry_run=True)

        captured = capsys.readouterr()
        assert "srun python my_sweep.py --point-file" in captured.out
        assert "#!/bin/bash" in captured.out
        assert "DRY RUN" in captured.out

    def test_dry_run_writes_point_json(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        submitter = _make_submitter(sweep)
        submitter.submit(dry_run=True)

        # Verify point.json files were created for all design points
        for alpha in [1.0, 2.0]:
            point = {"alpha": alpha, "beta": 10.0}
            key = _make_key(point)
            point_json = tmp_path / "cfd" / key / "point.json"
            assert point_json.exists(), f"Expected {point_json} to exist"
            loaded = json.loads(point_json.read_text())
            assert loaded["alpha"] == alpha
            assert loaded["beta"] == 10.0

    def test_dry_run_creates_summary_file(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        submitter = _make_submitter(sweep)
        submitter.submit(dry_run=True)

        summary_path = tmp_path / "submission_summary.txt"
        assert summary_path.exists()
        content = summary_path.read_text()
        assert "dry_run" in content


# ---------------------------------------------------------------------------
# Test: point.json creation and content
# ---------------------------------------------------------------------------


class TestPointJsonCreation:
    def test_point_json_correct_content(self, tmp_path):
        sweep = _make_sweep(tmp_path, params={"x": [0.5], "y": [1.5]})
        submitter = _make_submitter(sweep)
        submitter.submit(dry_run=True)

        point = {"x": 0.5, "y": 1.5}
        key = _make_key(point)
        point_json_path = tmp_path / "cfd" / key / "point.json"
        assert point_json_path.exists()
        loaded = json.loads(point_json_path.read_text())
        assert loaded == point

    def test_point_json_directory_created(self, tmp_path):
        sweep = _make_sweep(tmp_path, params={"z": [3.14]})
        submitter = _make_submitter(sweep)
        submitter.submit(dry_run=True)

        key = _make_key({"z": 3.14})
        point_dir = tmp_path / "cfd" / key
        assert point_dir.is_dir()

    def test_point_json_path_in_script(self, tmp_path):
        sweep = _make_sweep(tmp_path, params={"v": [7]})
        submitter = _make_submitter(sweep)

        key = _make_key({"v": 7})
        expected_path = os.path.join(sweep.cfd_root, key, "point.json")
        script = submitter._build_job_script(key, expected_path)
        assert f"--point-file {expected_path}" in script

    def test_multiple_points_each_get_point_json(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        submitter = _make_submitter(sweep)
        submitter.submit(dry_run=True)

        for alpha in [1.0, 2.0]:
            key = _make_key({"alpha": alpha, "beta": 10.0})
            assert (tmp_path / "cfd" / key / "point.json").exists()


# ---------------------------------------------------------------------------
# Test: skip_completed logic
# ---------------------------------------------------------------------------


class TestSkipCompleted:
    def _write_csv_with_status(self, csv_path, rows):
        fieldnames = ["key", "status", "alpha", "beta"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_skip_completed_skips_success_rows(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        completed_point = {"alpha": 1.0, "beta": 10.0}
        completed_key = _make_key(completed_point)
        self._write_csv_with_status(
            sweep.output_csv,
            [{"key": completed_key, "status": "success", "alpha": 1.0, "beta": 10.0}],
        )

        submitter = _make_submitter(sweep)
        submitter.submit(dry_run=True, skip_completed=True)

        summary_path = tmp_path / "submission_summary.txt"
        content = summary_path.read_text()
        # The completed key should appear as skipped
        assert completed_key in content
        assert "SKIP" in content

        # The other point (alpha=2.0) should be a dry run
        other_key = _make_key({"alpha": 2.0, "beta": 10.0})
        assert other_key in content
        assert "DRY RUN" in content

    def test_skip_completed_failed_rows_are_rerun(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        failed_point = {"alpha": 1.0, "beta": 10.0}
        failed_key = _make_key(failed_point)
        self._write_csv_with_status(
            sweep.output_csv,
            [{"key": failed_key, "status": "failed", "alpha": 1.0, "beta": 10.0}],
        )

        submitter = _make_submitter(sweep)
        submitter.submit(dry_run=True, skip_completed=True)

        summary_path = tmp_path / "submission_summary.txt"
        content = summary_path.read_text()
        # Failed points should be submitted (DRY RUN), not skipped
        assert failed_key in content
        assert "DRY RUN" in content
        # Should not appear as skipped
        skip_lines = [
            l for l in content.splitlines() if "SKIP" in l and failed_key in l
        ]
        assert not skip_lines, "Failed point should not be skipped"

    def test_skip_completed_no_csv_processes_all(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        submitter = _make_submitter(sweep)
        # No CSV exists — all points should be processed
        submitter.submit(dry_run=True, skip_completed=True)

        summary_path = tmp_path / "submission_summary.txt"
        content = summary_path.read_text()
        # Both design points should appear as DRY RUN
        assert content.count("DRY RUN") >= 2

    def test_skip_completed_false_does_not_skip_success(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        completed_point = {"alpha": 1.0, "beta": 10.0}
        completed_key = _make_key(completed_point)
        self._write_csv_with_status(
            sweep.output_csv,
            [{"key": completed_key, "status": "success", "alpha": 1.0, "beta": 10.0}],
        )

        submitter = _make_submitter(sweep)
        # skip_completed=False (default) — all points should be submitted
        submitter.submit(dry_run=True, skip_completed=False)

        summary_path = tmp_path / "submission_summary.txt"
        content = summary_path.read_text()
        # Both points should appear as DRY RUN (no skipping)
        assert content.count("DRY RUN") >= 2


# ---------------------------------------------------------------------------
# Test: extra_directives injection
# ---------------------------------------------------------------------------


class TestExtraDirectives:
    def test_single_extra_directive(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        submitter = _make_submitter(sweep, extra_directives=["--mail-type=END"])
        key = "pt_testkey"
        script = submitter._build_job_script(key, "some/path.json")
        assert "#SBATCH --mail-type=END" in script

    def test_multiple_extra_directives_all_present(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        directives = [
            "--mail-type=END",
            "--mail-user=user@example.com",
            "--constraint=haswell",
        ]
        submitter = _make_submitter(sweep, extra_directives=directives)
        key = "pt_testkey"
        script = submitter._build_job_script(key, "some/path.json")

        for directive in directives:
            assert f"#SBATCH {directive}" in script

    def test_none_extra_directives_treated_as_empty(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        submitter = SlurmSweepSubmitter(
            sweep,
            {
                "account": "A",
                "partition": "p",
                "qos": "q",
                "nodes": 1,
                "ntasks_per_node": 1,
                "walltime": "1:00:00",
            },
            sweep_script="run.py",
            extra_directives=None,
        )
        assert submitter.extra_directives == []

    def test_extra_directives_appear_before_srun(self, tmp_path):
        """Extra directives must appear before the srun line."""
        sweep = _make_sweep(tmp_path)
        submitter = _make_submitter(
            sweep, preamble="echo start", extra_directives=["--gres=gpu:1"]
        )
        key = "pt_testkey"
        script = submitter._build_job_script(key, "some/path.json")

        extra_pos = script.find("#SBATCH --gres=gpu:1")
        srun_pos = script.find("srun python")

        assert extra_pos != -1, "Extra directive not found"
        assert extra_pos < srun_pos, "Extra directive must appear before srun"


# ---------------------------------------------------------------------------
# Test: submission summary
# ---------------------------------------------------------------------------


class TestSubmissionSummary:
    def test_summary_path_is_alongside_csv(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        submitter = _make_submitter(sweep)
        submitter.submit(dry_run=True)

        expected = tmp_path / "submission_summary.txt"
        assert expected.exists()

    def test_summary_contains_all_keys(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        submitter = _make_submitter(sweep)
        submitter.submit(dry_run=True)

        keys = [
            _make_key({"alpha": 1.0, "beta": 10.0}),
            _make_key({"alpha": 2.0, "beta": 10.0}),
        ]
        content = (tmp_path / "submission_summary.txt").read_text()
        for key in keys:
            assert key in content

    def test_summary_has_header_line(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        submitter = _make_submitter(sweep)
        submitter.submit(dry_run=True)

        content = (tmp_path / "submission_summary.txt").read_text()
        # Summary should contain a timestamped banner and design point info
        assert "Parameter sweep launcher" in content
        assert "Total design points" in content


# ---------------------------------------------------------------------------
# Test: constructor stores attributes correctly
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_attributes_stored(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        slurm_config = {
            "account": "ACC",
            "partition": "part",
            "qos": "q",
            "nodes": 4,
            "ntasks_per_node": 32,
            "walltime": "02:00:00",
        }
        submitter = SlurmSweepSubmitter(
            sweep,
            slurm_config,
            sweep_script="/path/to/script.py",
            preamble="echo hello",
            log_dir="my_logs",
            extra_directives=["--x=y"],
        )
        assert submitter.sweep is sweep
        assert submitter.slurm_config is slurm_config
        assert submitter.sweep_script == "/path/to/script.py"
        assert submitter.preamble == "echo hello"
        assert submitter.log_dir == "my_logs"
        assert submitter.extra_directives == ["--x=y"]

    def test_default_values(self, tmp_path):
        sweep = _make_sweep(tmp_path)
        submitter = SlurmSweepSubmitter(
            sweep,
            {
                "account": "A",
                "partition": "p",
                "qos": "q",
                "nodes": 1,
                "ntasks_per_node": 1,
                "walltime": "1:00:00",
            },
            sweep_script="run.py",
        )
        assert submitter.preamble == ""
        assert submitter.log_dir == "sweep_logs"
        assert submitter.extra_directives == []
