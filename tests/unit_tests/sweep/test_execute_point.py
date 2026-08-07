"""
Tests for ParameterSweep._execute_point (Task 7.1).

Covers:
- model_builder → solver_builder → FUNtoFEMnlbgs-path driver → result_extractor pipeline
- custom sweep_driver path (driver=None passed to extractor)
- per-point error isolation: exception → status="failed", sweep continues
- progress messages printed on rank 0
- _write_csv_row called with correct arguments
- _default_result_extractor returns {} (stub)
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Import directly from the sweep subpackage to avoid pulling in native TACS/FUN3D
# libraries through funtofem/__init__.py. The sweep submodule has no native deps.
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


# Ensure funtofem package stub exists so relative imports from parameter_sweep work
if "funtofem" not in sys.modules:
    _funtofem_stub = types.ModuleType("funtofem")
    _funtofem_stub.__path__ = [str(pathlib.Path(__file__).parents[3] / "funtofem")]
    _funtofem_stub.__package__ = "funtofem"
    sys.modules["funtofem"] = _funtofem_stub

# Load modules in dependency order
_strategy_mod = _load_sweep_module("strategy")
_mesh_mode_mod = _load_sweep_module("mesh_mode")

# parameter_sweep imports from .strategy and .mesh_mode using relative imports;
# register them so the relative imports resolve correctly.
if "funtofem.sweep" not in sys.modules:
    _sweep_pkg = types.ModuleType("funtofem.sweep")
    sys.modules["funtofem.sweep"] = _sweep_pkg

sys.modules["funtofem.sweep"].SweepStrategy = _strategy_mod.SweepStrategy
sys.modules["funtofem.sweep"].CartesianStrategy = _strategy_mod.CartesianStrategy
sys.modules["funtofem.sweep"].MeshMode = _mesh_mode_mod.MeshMode

_param_sweep_mod = _load_sweep_module("parameter_sweep")

ParameterSweep = _param_sweep_mod.ParameterSweep
_default_result_extractor = _param_sweep_mod._default_result_extractor
MeshMode = _mesh_mode_mod.MeshMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_comm(rank: int = 0):
    """Return a minimal mock MPI communicator."""
    comm = MagicMock()
    comm.rank = rank
    return comm


def _make_sweep_none_mesh():
    """Return a ParameterSweep with MeshMode.NONE so mesh callbacks are not required."""
    return ParameterSweep(
        {"alpha": [1, 2]},
        mesh_mode=MeshMode.NONE,
    )


def _noop_write_csv_row(self_ref, *args, **kwargs):
    """No-op replacement for _write_csv_row so it doesn't affect test assertions."""
    pass


# ---------------------------------------------------------------------------
# _default_result_extractor stub
# ---------------------------------------------------------------------------


class TestDefaultResultExtractorStub(unittest.TestCase):
    def test_model_none_driver_none_returns_flow_fields_as_nan(self):
        """When model=None and driver=None, _default_result_extractor returns
        flow diagnostic keys with nan/empty values and no function values."""
        import math

        result = _default_result_extractor(None, None, {}, None, None)
        self.assertIn("flow_resid_norm", result)
        self.assertIn("flow_resid_comps", result)
        self.assertIn("flow_iters", result)
        self.assertTrue(math.isnan(result["flow_resid_norm"]))
        self.assertEqual(result["flow_resid_comps"], "")
        self.assertTrue(math.isnan(result["flow_iters"]))
        # No function-value keys when model is None
        extra_keys = set(result.keys()) - {
            "flow_resid_norm",
            "flow_resid_comps",
            "flow_iters",
        }
        self.assertEqual(extra_keys, set())


# ---------------------------------------------------------------------------
# Success path: custom sweep_driver
# ---------------------------------------------------------------------------


class TestExecutePointCustomDriver(unittest.TestCase):
    def _build_sweep_with_custom_driver(self):
        """Build a ParameterSweep whose execution uses a custom sweep_driver."""
        sweep = _make_sweep_none_mesh()

        mock_model = MagicMock(name="model")
        mock_solvers = MagicMock(name="solvers")
        mock_results = {"lift": 1.23}

        model_builder = MagicMock(return_value=mock_model)
        solver_builder = MagicMock(return_value=mock_solvers)
        sweep_driver = MagicMock()
        result_extractor = MagicMock(return_value=mock_results)

        sweep.set_model_builder(model_builder)
        sweep.set_solver_builder(solver_builder)
        sweep.set_sweep_driver(sweep_driver)
        sweep.set_result_extractor(result_extractor)

        return (
            sweep,
            model_builder,
            solver_builder,
            sweep_driver,
            result_extractor,
            mock_model,
            mock_solvers,
        )

    def test_custom_driver_calls_in_order(self):
        (
            sweep,
            model_builder,
            solver_builder,
            sweep_driver,
            result_extractor,
            mock_model,
            mock_solvers,
        ) = self._build_sweep_with_custom_driver()

        # Patch _write_csv_row so it doesn't interfere
        with patch.object(ParameterSweep, "_write_csv_row", _noop_write_csv_row):
            comm = _make_mock_comm()
            design_point = {"alpha": 1}
            sweep._execute_point(
                comm,
                1,
                2,
                "pt_abc123",
                design_point,
                "cfd/pt_abc123",
                "sw/pt_abc123/struct",
            )

        model_builder.assert_called_once_with(
            comm, design_point, "cfd/pt_abc123", "sw/pt_abc123/struct"
        )
        solver_builder.assert_called_once_with(
            comm, mock_model, design_point, "cfd/pt_abc123", "sw/pt_abc123/struct"
        )
        sweep_driver.assert_called_once_with(
            comm, mock_model, mock_solvers, design_point, False
        )
        # result_extractor receives whatever the sweep_driver callback returned
        result_extractor.assert_called_once_with(
            mock_model,
            sweep_driver.return_value,
            design_point,
            "cfd/pt_abc123",
            "sw/pt_abc123/struct",
        )

    def test_custom_driver_returning_none_passes_none_to_extractor(self):
        (
            sweep,
            model_builder,
            solver_builder,
            sweep_driver,
            result_extractor,
            mock_model,
            mock_solvers,
        ) = self._build_sweep_with_custom_driver()
        sweep_driver.return_value = None

        with patch.object(ParameterSweep, "_write_csv_row", _noop_write_csv_row):
            comm = _make_mock_comm()
            design_point = {"alpha": 1}
            sweep._execute_point(
                comm,
                1,
                1,
                "pt_abc123",
                design_point,
                "cfd/pt_abc123",
                "sw/pt_abc123/struct",
            )

        result_extractor.assert_called_once_with(
            mock_model, None, design_point, "cfd/pt_abc123", "sw/pt_abc123/struct"
        )

    def test_custom_driver_with_adjoint_passed(self):
        (
            sweep,
            model_builder,
            solver_builder,
            sweep_driver,
            result_extractor,
            mock_model,
            mock_solvers,
        ) = self._build_sweep_with_custom_driver()

        with patch.object(ParameterSweep, "_write_csv_row", _noop_write_csv_row):
            comm = _make_mock_comm()
            design_point = {"alpha": 2}
            sweep._execute_point(
                comm,
                1,
                1,
                "pt_xyz789",
                design_point,
                "cfd/pt_xyz789",
                "sw/pt_xyz789/struct",
                compute_adjoint=True,
            )

        # compute_adjoint=True must be forwarded to the custom driver
        sweep_driver.assert_called_once_with(
            comm, mock_model, mock_solvers, design_point, True
        )


# ---------------------------------------------------------------------------
# Success path: default FUNtoFEMnlbgs driver (mocked)
# ---------------------------------------------------------------------------


def _make_nlbgs_mock():
    """Create a mock FUNtoFEMnlbgs class and driver instance."""
    mock_driver_instance = MagicMock(name="driver_instance")
    mock_nlbgs_cls = MagicMock(return_value=mock_driver_instance)
    return mock_nlbgs_cls, mock_driver_instance


def _mock_nlbgs_import(mock_nlbgs_cls):
    """Context manager that makes the lazy import inside _execute_point return mock_nlbgs_cls.

    The implementation does ``from ..driver import FUNtoFEMnlbgs`` inside a
    try/except block.  We patch ``funtofem.sweep.parameter_sweep``'s import
    machinery by temporarily placing the mock into sys.modules.
    """
    fake_driver_mod = types.ModuleType("funtofem.driver")
    fake_driver_mod.FUNtoFEMnlbgs = mock_nlbgs_cls

    # Also need funtofem.sweep to have __package__ set correctly for the
    # relative import to resolve. Ensure funtofem.driver is reachable.
    return patch.dict(sys.modules, {"funtofem.driver": fake_driver_mod})


class TestExecutePointDefaultDriver(unittest.TestCase):
    def test_default_driver_path_calls_solve_forward(self):
        """When no sweep_driver is set, FUNtoFEMnlbgs.solve_forward() must be called."""
        sweep = _make_sweep_none_mesh()

        mock_model = MagicMock(name="model")
        mock_solvers = MagicMock(name="solvers")
        mock_results = {"drag": 0.05}

        sweep.set_model_builder(MagicMock(return_value=mock_model))
        sweep.set_solver_builder(MagicMock(return_value=mock_solvers))
        sweep.set_result_extractor(MagicMock(return_value=mock_results))

        mock_nlbgs_cls, mock_driver_instance = _make_nlbgs_mock()

        with _mock_nlbgs_import(mock_nlbgs_cls):
            with patch.object(ParameterSweep, "_write_csv_row", _noop_write_csv_row):
                comm = _make_mock_comm()
                design_point = {"alpha": 1}
                sweep._execute_point(
                    comm, 1, 1, "pt_def", design_point, "cfd/pt_def", "sw/pt_def/struct"
                )

        # model/transfer_settings must be keywords: the second positional
        # argument of FUNtoFEMnlbgs is comm_manager, not model.
        mock_nlbgs_cls.assert_called_once_with(
            mock_solvers, model=mock_model, transfer_settings=None
        )
        mock_driver_instance.solve_forward.assert_called_once()
        mock_driver_instance.solve_adjoint.assert_not_called()

    def test_default_driver_calls_solve_adjoint_when_requested(self):
        """When compute_adjoint=True and no sweep_driver, solve_adjoint() must be called."""
        sweep = _make_sweep_none_mesh()

        mock_model = MagicMock(name="model")
        mock_solvers = MagicMock(name="solvers")

        sweep.set_model_builder(MagicMock(return_value=mock_model))
        sweep.set_solver_builder(MagicMock(return_value=mock_solvers))
        sweep.set_result_extractor(MagicMock(return_value={}))

        mock_nlbgs_cls, mock_driver_instance = _make_nlbgs_mock()

        with _mock_nlbgs_import(mock_nlbgs_cls):
            with patch.object(ParameterSweep, "_write_csv_row", _noop_write_csv_row):
                comm = _make_mock_comm()
                design_point = {"alpha": 1}
                sweep._execute_point(
                    comm,
                    1,
                    1,
                    "pt_adj",
                    design_point,
                    "cfd/pt_adj",
                    "sw/pt_adj/struct",
                    compute_adjoint=True,
                )

        mock_driver_instance.solve_forward.assert_called_once()
        mock_driver_instance.solve_adjoint.assert_called_once()

    def test_default_extractor_used_when_none_registered(self):
        """Falls back to _default_result_extractor when result_extractor is None."""
        sweep = _make_sweep_none_mesh()

        mock_model = MagicMock(name="model")
        mock_model.scenarios = []  # no scenarios → no func values
        mock_solvers = MagicMock(name="solvers")

        sweep.set_model_builder(MagicMock(return_value=mock_model))
        sweep.set_solver_builder(MagicMock(return_value=mock_solvers))
        # No result_extractor registered

        mock_nlbgs_cls, mock_driver_instance = _make_nlbgs_mock()
        # No flow solver on the driver → flow fields degrade to nan/""
        mock_driver_instance.solvers = MagicMock()
        mock_driver_instance.solvers.flow = None

        write_csv_calls = []

        def capturing_write(
            self_ref,
            comm,
            case_idx,
            key,
            design_point,
            status,
            results,
            error_msg,
            *,
            compute_adjoint=False,
        ):
            write_csv_calls.append({"status": status, "results": results})

        import math

        with _mock_nlbgs_import(mock_nlbgs_cls):
            with patch.object(ParameterSweep, "_write_csv_row", capturing_write):
                comm = _make_mock_comm()
                design_point = {"alpha": 1}
                sweep._execute_point(
                    comm,
                    1,
                    1,
                    "pt_def2",
                    design_point,
                    "cfd/pt_def2",
                    "sw/pt_def2/struct",
                )

        self.assertEqual(len(write_csv_calls), 1)
        self.assertEqual(write_csv_calls[0]["status"], "success")
        results = write_csv_calls[0]["results"]
        # Default extractor always produces flow diagnostic keys
        self.assertIn("flow_resid_norm", results)
        self.assertIn("flow_resid_comps", results)
        self.assertIn("flow_iters", results)
        self.assertTrue(math.isnan(results["flow_resid_norm"]))
        self.assertEqual(results["flow_resid_comps"], "")
        self.assertTrue(math.isnan(results["flow_iters"]))


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


class TestExecutePointErrorIsolation(unittest.TestCase):
    def test_model_builder_exception_sets_failed_status(self):
        """Exception in model_builder must not propagate; status must be 'failed'."""
        sweep = _make_sweep_none_mesh()

        def bad_model_builder(comm, design_point, cfd_dir, struct_dir):
            raise RuntimeError("simulated model build failure")

        sweep.set_model_builder(bad_model_builder)
        sweep.set_solver_builder(MagicMock(return_value=MagicMock()))

        write_csv_calls = []

        def capturing_write(
            self_ref,
            comm,
            case_idx,
            key,
            design_point,
            status,
            results,
            error_msg,
            *,
            compute_adjoint=False,
        ):
            write_csv_calls.append(
                {
                    "case_idx": case_idx,
                    "status": status,
                    "results": results,
                    "error_msg": error_msg,
                }
            )

        with patch.object(ParameterSweep, "_write_csv_row", capturing_write):
            comm = _make_mock_comm()
            design_point = {"alpha": 1}
            # Must NOT raise
            sweep._execute_point(
                comm, 1, 1, "pt_bad", design_point, "cfd/pt_bad", "sw/pt_bad/struct"
            )

        self.assertEqual(len(write_csv_calls), 1)
        self.assertEqual(write_csv_calls[0]["status"], "failed")
        self.assertEqual(write_csv_calls[0]["results"], {})
        self.assertIn("simulated model build failure", write_csv_calls[0]["error_msg"])

    def test_solver_builder_exception_sets_failed_status(self):
        """Exception in solver_builder must not propagate."""
        sweep = _make_sweep_none_mesh()

        sweep.set_model_builder(MagicMock(return_value=MagicMock()))

        def bad_solver_builder(comm, model, design_point, cfd_dir, struct_dir):
            raise ValueError("solver build error")

        sweep.set_solver_builder(bad_solver_builder)

        write_csv_calls = []

        def capturing_write(
            self_ref,
            comm,
            case_idx,
            key,
            design_point,
            status,
            results,
            error_msg,
            *,
            compute_adjoint=False,
        ):
            write_csv_calls.append({"status": status, "error_msg": error_msg})

        with patch.object(ParameterSweep, "_write_csv_row", capturing_write):
            comm = _make_mock_comm()
            sweep._execute_point(comm, 2, 5, "pt_sv_err", {"alpha": 1}, "cfd", "sw")

        self.assertEqual(write_csv_calls[0]["status"], "failed")
        self.assertIn("solver build error", write_csv_calls[0]["error_msg"])

    def test_exception_in_one_point_does_not_stop_sweep(self):
        """A failure in one point must not prevent subsequent points from running."""
        sweep = ParameterSweep(
            {"alpha": [1, 2, 3]},
            mesh_mode=MeshMode.NONE,
        )

        call_log = []

        def flaky_model_builder(comm, design_point, cfd_dir, struct_dir):
            call_log.append(design_point["alpha"])
            if design_point["alpha"] == 2:
                raise RuntimeError("point 2 explodes")
            return MagicMock()

        mock_nlbgs_cls, mock_driver_instance = _make_nlbgs_mock()

        sweep.set_model_builder(flaky_model_builder)
        sweep.set_solver_builder(MagicMock(return_value=MagicMock()))
        sweep.set_result_extractor(MagicMock(return_value={}))

        with _mock_nlbgs_import(mock_nlbgs_cls):
            with patch.object(ParameterSweep, "_write_csv_row", _noop_write_csv_row):
                comm = _make_mock_comm()
                sweep.run(comm)

        # All three points must have been attempted
        self.assertEqual(call_log, [1, 2, 3])


# ---------------------------------------------------------------------------
# Progress messages
# ---------------------------------------------------------------------------


class TestExecutePointProgressMessages(unittest.TestCase):
    def test_starting_message_printed_on_rank_0(self):
        """Rank-0 must print the starting progress message."""
        sweep = _make_sweep_none_mesh()

        sweep.set_model_builder(MagicMock(return_value=MagicMock()))
        sweep.set_solver_builder(MagicMock(return_value=MagicMock()))
        sweep.set_result_extractor(MagicMock(return_value={}))
        sweep.set_sweep_driver(MagicMock())

        import io
        import contextlib

        output = io.StringIO()
        comm = _make_mock_comm(rank=0)
        design_point = {"alpha": 7}

        with patch.object(ParameterSweep, "_write_csv_row", _noop_write_csv_row):
            with contextlib.redirect_stdout(output):
                sweep._execute_point(comm, 3, 10, "pt_prog", design_point, "cfd", "sw")

        printed = output.getvalue()
        self.assertIn("[sweep] case 3/10 starting:", printed)
        self.assertIn("pt_prog", printed)

    def test_no_output_on_non_rank_0(self):
        """Non-rank-0 processes must not print progress messages."""
        sweep = _make_sweep_none_mesh()

        sweep.set_model_builder(MagicMock(return_value=MagicMock()))
        sweep.set_solver_builder(MagicMock(return_value=MagicMock()))
        sweep.set_result_extractor(MagicMock(return_value={}))
        sweep.set_sweep_driver(MagicMock())

        import io
        import contextlib

        output = io.StringIO()
        comm = _make_mock_comm(rank=1)  # non-zero rank

        with patch.object(ParameterSweep, "_write_csv_row", _noop_write_csv_row):
            with contextlib.redirect_stdout(output):
                sweep._execute_point(comm, 3, 10, "pt_quiet", {"alpha": 7}, "cfd", "sw")

        self.assertEqual(output.getvalue(), "")

    def test_completion_message_printed_on_success(self):
        """Rank-0 must print the 'done' message after a successful point."""
        sweep = _make_sweep_none_mesh()

        sweep.set_model_builder(MagicMock(return_value=MagicMock()))
        sweep.set_solver_builder(MagicMock(return_value=MagicMock()))
        sweep.set_result_extractor(MagicMock(return_value={"lift": 9.9}))
        sweep.set_sweep_driver(MagicMock())

        import io
        import contextlib

        output = io.StringIO()
        comm = _make_mock_comm(rank=0)

        with patch.object(ParameterSweep, "_write_csv_row", _noop_write_csv_row):
            with contextlib.redirect_stdout(output):
                sweep._execute_point(comm, 1, 1, "pt_done", {"alpha": 1}, "cfd", "sw")

        self.assertIn("status=success", output.getvalue())

    def test_failed_message_printed_on_exception(self):
        """Rank-0 must print the FAILED message when a point raises."""
        sweep = _make_sweep_none_mesh()

        def bad_builder(comm, dp, cfd, struct):
            raise RuntimeError("boom")

        sweep.set_model_builder(bad_builder)
        sweep.set_solver_builder(MagicMock(return_value=MagicMock()))

        import io
        import contextlib

        output = io.StringIO()
        comm = _make_mock_comm(rank=0)

        with patch.object(ParameterSweep, "_write_csv_row", _noop_write_csv_row):
            with contextlib.redirect_stdout(output):
                sweep._execute_point(comm, 1, 1, "pt_fail", {"alpha": 1}, "cfd", "sw")

        self.assertIn("FAILED", output.getvalue())
        self.assertIn("boom", output.getvalue())


# ---------------------------------------------------------------------------
# _write_csv_row stub
# ---------------------------------------------------------------------------


class TestWriteCsvRowStub(unittest.TestCase):
    def test_write_csv_row_stub_does_not_raise(self):
        """_write_csv_row now writes to CSV on rank 0 without raising."""
        import tempfile
        import os

        sweep = _make_sweep_none_mesh()
        comm = _make_mock_comm(rank=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            sweep.output_csv = os.path.join(tmpdir, "test_output.csv")
            result = sweep._write_csv_row(
                comm, 1, "pt_abc", {"alpha": 1}, "success", {}, None
            )
            self.assertIsNone(result)

    def test_write_csv_row_called_on_success(self):
        """_write_csv_row must be called once on a successful point with correct args."""
        sweep = _make_sweep_none_mesh()

        sweep.set_model_builder(MagicMock(return_value=MagicMock()))
        sweep.set_solver_builder(MagicMock(return_value=MagicMock()))
        sweep.set_result_extractor(MagicMock(return_value={"x": 1}))
        sweep.set_sweep_driver(MagicMock())

        write_calls = []

        def spy_write(
            self_ref,
            comm,
            case_idx,
            key,
            design_point,
            status,
            results,
            error_msg,
            *,
            compute_adjoint=False,
        ):
            write_calls.append(
                {
                    "status": status,
                    "results": results,
                    "error_msg": error_msg,
                    "compute_adjoint": compute_adjoint,
                }
            )

        with patch.object(ParameterSweep, "_write_csv_row", spy_write):
            comm = _make_mock_comm()
            sweep._execute_point(
                comm, 1, 1, "pt_wr", {"alpha": 1}, "cfd", "sw", compute_adjoint=True
            )

        self.assertEqual(len(write_calls), 1)
        self.assertEqual(write_calls[0]["status"], "success")
        self.assertEqual(write_calls[0]["results"], {"x": 1})
        self.assertIsNone(write_calls[0]["error_msg"])
        self.assertTrue(write_calls[0]["compute_adjoint"])

    def test_write_csv_row_called_on_failure(self):
        """_write_csv_row must be called once with error_msg on failure."""
        sweep = _make_sweep_none_mesh()

        def bad_builder(comm, dp, cfd, struct):
            raise KeyError("missing key")

        sweep.set_model_builder(bad_builder)
        sweep.set_solver_builder(MagicMock(return_value=MagicMock()))

        write_calls = []

        def spy_write(
            self_ref,
            comm,
            case_idx,
            key,
            design_point,
            status,
            results,
            error_msg,
            *,
            compute_adjoint=False,
        ):
            write_calls.append(
                {"status": status, "results": results, "error_msg": error_msg}
            )

        with patch.object(ParameterSweep, "_write_csv_row", spy_write):
            comm = _make_mock_comm()
            sweep._execute_point(comm, 1, 1, "pt_wr_fail", {"alpha": 1}, "cfd", "sw")

        self.assertEqual(len(write_calls), 1)
        self.assertEqual(write_calls[0]["status"], "failed")
        self.assertEqual(write_calls[0]["results"], {})
        self.assertIsNotNone(write_calls[0]["error_msg"])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Task 7.3: Adjoint gradient collection tests
# ---------------------------------------------------------------------------


def _make_mock_model_with_gradients(func_names, var_names, grad_values):
    """Build a mock FUNtoFEMmodel where func.derivatives holds gradient values.

    Parameters
    ----------
    func_names : list[str]
        Names of functions (e.g. ["lift", "drag"]).
    var_names : list[str]
        Names of design variables (e.g. ["alpha", "beta"]).
    grad_values : dict[(str, str), float]
        Mapping from (func_name, var_name) to gradient value.
    """
    variables = []
    for vname in var_names:
        var = MagicMock(name=f"var_{vname}")
        var.name = vname
        variables.append(var)

    funcs = []
    for fname in func_names:
        func = MagicMock(name=f"func_{fname}")
        func.name = fname
        # Build derivatives dict keyed by variable mock objects
        derivs = {}
        for var in variables:
            derivs[var] = grad_values.get((fname, var.name), float("nan"))
        func.derivatives = derivs
        funcs.append(func)

    scenario = MagicMock()
    scenario.functions = funcs

    model = MagicMock(name="model")
    model.scenarios = [scenario]
    model.get_variables = MagicMock(return_value=variables)

    return model, funcs, variables


class TestCollectGradients(unittest.TestCase):
    """Tests for the _collect_gradients helper method."""

    def test_returns_empty_when_model_is_none(self):
        """_collect_gradients returns {} when model is None."""
        sweep = _make_sweep_none_mesh()
        sweep.set_model_builder(MagicMock(return_value=MagicMock()))
        sweep.set_solver_builder(MagicMock(return_value=MagicMock()))
        result = sweep._collect_gradients(None)
        self.assertEqual(result, {})

    def test_collects_correct_gradient_columns(self):
        """_collect_gradients returns d<func>_d<var> keys for each (func, var) pair."""
        sweep = _make_sweep_none_mesh()

        model, funcs, variables = _make_mock_model_with_gradients(
            func_names=["lift", "drag"],
            var_names=["alpha", "beta"],
            grad_values={
                ("lift", "alpha"): 1.5,
                ("lift", "beta"): 2.0,
                ("drag", "alpha"): -0.3,
                ("drag", "beta"): 0.7,
            },
        )

        result = sweep._collect_gradients(model)

        self.assertIn("dlift_dalpha", result)
        self.assertIn("dlift_dbeta", result)
        self.assertIn("ddrag_dalpha", result)
        self.assertIn("ddrag_dbeta", result)
        self.assertAlmostEqual(result["dlift_dalpha"], 1.5)
        self.assertAlmostEqual(result["dlift_dbeta"], 2.0)
        self.assertAlmostEqual(result["ddrag_dalpha"], -0.3)
        self.assertAlmostEqual(result["ddrag_dbeta"], 0.7)

    def test_missing_gradient_returns_nan(self):
        """_collect_gradients uses nan when a variable key is absent from derivatives."""
        import math

        sweep = _make_sweep_none_mesh()

        model, _, _ = _make_mock_model_with_gradients(
            func_names=["lift"],
            var_names=["alpha"],
            grad_values={},  # No gradients stored → nan expected
        )

        result = sweep._collect_gradients(model)
        self.assertIn("dlift_dalpha", result)
        self.assertTrue(math.isnan(result["dlift_dalpha"]))

    def test_no_variables_returns_empty(self):
        """_collect_gradients returns {} when model has no variables."""
        sweep = _make_sweep_none_mesh()

        model = MagicMock()
        model.scenarios = []
        model.get_variables = MagicMock(return_value=[])

        result = sweep._collect_gradients(model)
        self.assertEqual(result, {})

    def test_get_variables_raises_returns_empty(self):
        """_collect_gradients handles model.get_variables() raising gracefully."""
        sweep = _make_sweep_none_mesh()

        model = MagicMock()
        model.scenarios = []
        model.get_variables = MagicMock(side_effect=AttributeError("no get_variables"))

        result = sweep._collect_gradients(model)
        self.assertEqual(result, {})


class TestAdjointGradientCollectionDefaultDriver(unittest.TestCase):
    """Tests that gradient columns appear in results when compute_adjoint=True (default driver path)."""

    def test_gradients_added_to_results_on_default_driver_path(self):
        """With default driver and compute_adjoint=True, gradient columns appear in write_csv_row."""
        sweep = _make_sweep_none_mesh()

        model, funcs, variables = _make_mock_model_with_gradients(
            func_names=["lift"],
            var_names=["alpha"],
            grad_values={("lift", "alpha"): 3.14},
        )

        sweep.set_model_builder(MagicMock(return_value=model))
        sweep.set_solver_builder(MagicMock(return_value=MagicMock()))
        sweep.set_result_extractor(MagicMock(return_value={"lift": 9.9}))

        mock_nlbgs_cls, mock_driver_instance = _make_nlbgs_mock()

        write_csv_calls = []

        def capturing_write(
            self_ref,
            comm,
            case_idx,
            key,
            design_point,
            status,
            results,
            error_msg,
            *,
            compute_adjoint=False,
        ):
            write_csv_calls.append({"status": status, "results": results})

        with _mock_nlbgs_import(mock_nlbgs_cls):
            with patch.object(ParameterSweep, "_write_csv_row", capturing_write):
                comm = _make_mock_comm()
                design_point = {"alpha": 1}
                sweep._execute_point(
                    comm,
                    1,
                    1,
                    "pt_grad",
                    {"alpha": 1},
                    "cfd/pt_grad",
                    "sw/pt_grad/struct",
                    compute_adjoint=True,
                )

        self.assertEqual(len(write_csv_calls), 1)
        self.assertEqual(write_csv_calls[0]["status"], "success")
        results = write_csv_calls[0]["results"]
        # Gradient column must be present
        self.assertIn("dlift_dalpha", results)
        self.assertAlmostEqual(results["dlift_dalpha"], 3.14)
        # Extractor result must also be present
        self.assertIn("lift", results)

    def test_no_gradient_columns_when_compute_adjoint_false(self):
        """With compute_adjoint=False, no gradient columns appear in results."""
        sweep = _make_sweep_none_mesh()

        model, funcs, variables = _make_mock_model_with_gradients(
            func_names=["lift"],
            var_names=["alpha"],
            grad_values={("lift", "alpha"): 99.0},
        )

        sweep.set_model_builder(MagicMock(return_value=model))
        sweep.set_solver_builder(MagicMock(return_value=MagicMock()))
        sweep.set_result_extractor(MagicMock(return_value={"lift": 5.0}))

        mock_nlbgs_cls, mock_driver_instance = _make_nlbgs_mock()

        write_csv_calls = []

        def capturing_write(
            self_ref,
            comm,
            case_idx,
            key,
            design_point,
            status,
            results,
            error_msg,
            *,
            compute_adjoint=False,
        ):
            write_csv_calls.append({"status": status, "results": results})

        with _mock_nlbgs_import(mock_nlbgs_cls):
            with patch.object(ParameterSweep, "_write_csv_row", capturing_write):
                comm = _make_mock_comm()
                sweep._execute_point(
                    comm,
                    1,
                    1,
                    "pt_nograd",
                    {"alpha": 1},
                    "cfd/pt_nograd",
                    "sw/pt_nograd/struct",
                    compute_adjoint=False,
                )

        results = write_csv_calls[0]["results"]
        gradient_keys = [k for k in results if k.startswith("d") and "_d" in k[1:]]
        self.assertEqual(
            gradient_keys, [], f"Expected no gradient columns, found: {gradient_keys}"
        )


class TestAdjointGradientCollectionCustomDriver(unittest.TestCase):
    """Tests gradient collection when a custom sweep_driver is used with compute_adjoint=True."""

    def test_gradients_added_to_results_on_custom_driver_path(self):
        """With custom driver and compute_adjoint=True, gradient columns appear in results."""
        sweep = _make_sweep_none_mesh()

        model, funcs, variables = _make_mock_model_with_gradients(
            func_names=["drag"],
            var_names=["beta"],
            grad_values={("drag", "beta"): -0.5},
        )

        sweep.set_model_builder(MagicMock(return_value=model))
        sweep.set_solver_builder(MagicMock(return_value=MagicMock()))
        sweep.set_sweep_driver(MagicMock())
        sweep.set_result_extractor(MagicMock(return_value={"drag": 0.1}))

        write_csv_calls = []

        def capturing_write(
            self_ref,
            comm,
            case_idx,
            key,
            design_point,
            status,
            results,
            error_msg,
            *,
            compute_adjoint=False,
        ):
            write_csv_calls.append({"status": status, "results": results})

        with patch.object(ParameterSweep, "_write_csv_row", capturing_write):
            comm = _make_mock_comm()
            sweep._execute_point(
                comm,
                1,
                1,
                "pt_cust_grad",
                {"alpha": 1},
                "cfd/pt_cust_grad",
                "sw/pt_cust_grad/struct",
                compute_adjoint=True,
            )

        self.assertEqual(len(write_csv_calls), 1)
        self.assertEqual(write_csv_calls[0]["status"], "success")
        results = write_csv_calls[0]["results"]
        self.assertIn("ddrag_dbeta", results)
        self.assertAlmostEqual(results["ddrag_dbeta"], -0.5)

    def test_no_gradients_on_custom_driver_when_compute_adjoint_false(self):
        """Custom driver + compute_adjoint=False: no gradient columns."""
        sweep = _make_sweep_none_mesh()

        model, funcs, variables = _make_mock_model_with_gradients(
            func_names=["drag"],
            var_names=["beta"],
            grad_values={("drag", "beta"): 99.0},
        )

        sweep.set_model_builder(MagicMock(return_value=model))
        sweep.set_solver_builder(MagicMock(return_value=MagicMock()))
        sweep.set_sweep_driver(MagicMock())
        sweep.set_result_extractor(MagicMock(return_value={"drag": 0.1}))

        write_csv_calls = []

        def capturing_write(
            self_ref,
            comm,
            case_idx,
            key,
            design_point,
            status,
            results,
            error_msg,
            *,
            compute_adjoint=False,
        ):
            write_csv_calls.append({"status": status, "results": results})

        with patch.object(ParameterSweep, "_write_csv_row", capturing_write):
            comm = _make_mock_comm()
            sweep._execute_point(
                comm,
                1,
                1,
                "pt_cust_nograd",
                {"alpha": 1},
                "cfd/pt_cust_nograd",
                "sw/pt_cust_nograd/struct",
                compute_adjoint=False,
            )

        results = write_csv_calls[0]["results"]
        gradient_keys = [k for k in results if k.startswith("d") and "_d" in k[1:]]
        self.assertEqual(gradient_keys, [])
