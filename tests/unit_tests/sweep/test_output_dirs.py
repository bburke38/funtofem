"""
Tests for ParameterSweep._make_output_dirs.

Covers:
1. Default construction produces "sweep_output/<key>/cfd" and
   "sweep_output/<key>/struct"
2. Both output directories are siblings under one per-point directory when
   cfd_root and struct_root are equal
3. Distinct roots split the two trees
4. The per-point directory is where SlurmSweepSubmitter writes point.json,
   so with shared roots the point file sits alongside cfd/ and struct/
"""

import os
import sys
import types
import unittest

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

ParameterSweep = _param_sweep_mod.ParameterSweep


class OutputDirLayoutTest(unittest.TestCase):
    """Pin the per-point output directory layout."""

    KEY = "pt_a8f3c2d14e91"

    def test_default_roots(self):
        """Default construction nests both kinds under one per-point directory."""
        sweep = ParameterSweep({"alpha": [1, 2]})

        # Pins the defaults themselves, not just the templates: a silent change
        # to either root default relocates every user's output on disk.
        self.assertEqual(sweep.cfd_root, "sweep_output")
        self.assertEqual(sweep.struct_root, "sweep_output")

        cfd_dir, struct_dir = sweep._make_output_dirs(self.KEY)

        self.assertEqual(cfd_dir, f"sweep_output/{self.KEY}/cfd")
        self.assertEqual(struct_dir, f"sweep_output/{self.KEY}/struct")

    def test_shared_root_makes_siblings(self):
        """Equal roots put cfd/ and struct/ side by side under one point dir."""
        sweep = ParameterSweep({"alpha": [1]}, cfd_root="shared", struct_root="shared")
        cfd_dir, struct_dir = sweep._make_output_dirs(self.KEY)

        self.assertEqual(os.path.dirname(cfd_dir), os.path.dirname(struct_dir))
        self.assertEqual(os.path.dirname(cfd_dir), f"shared/{self.KEY}")

    def test_point_file_is_sibling_of_output_dirs(self):
        """point.json lands beside cfd/ and struct/, not inside either.

        SlurmSweepSubmitter builds the point file path as
        "<cfd_root>/<key>/point.json" independently of _make_output_dirs, so
        this pins the two constructions against drifting apart.
        """
        sweep = ParameterSweep({"alpha": [1]})
        cfd_dir, struct_dir = sweep._make_output_dirs(self.KEY)
        point_json = os.path.join(sweep.cfd_root, self.KEY, "point.json")

        self.assertEqual(os.path.dirname(point_json), os.path.dirname(cfd_dir))
        self.assertEqual(os.path.dirname(point_json), os.path.dirname(struct_dir))

    def test_distinct_roots_split_trees(self):
        """Different roots separate the two trees entirely."""
        sweep = ParameterSweep(
            {"alpha": [1]}, cfd_root="/scratch/cfd", struct_root="/home/struct"
        )
        cfd_dir, struct_dir = sweep._make_output_dirs(self.KEY)

        self.assertEqual(cfd_dir, f"/scratch/cfd/{self.KEY}/cfd")
        self.assertEqual(struct_dir, f"/home/struct/{self.KEY}/struct")
        self.assertNotEqual(os.path.dirname(cfd_dir), os.path.dirname(struct_dir))

    def test_key_is_not_interpreted(self):
        """The key is substituted verbatim, including custom key_fn output."""
        sweep = ParameterSweep({"alpha": [1]}, key_fn=lambda p: "alpha_1.5")
        cfd_dir, struct_dir = sweep._make_output_dirs("alpha_1.5")

        self.assertEqual(cfd_dir, "sweep_output/alpha_1.5/cfd")
        self.assertEqual(struct_dir, "sweep_output/alpha_1.5/struct")


if __name__ == "__main__":
    unittest.main()
