__all__ = ["OnewayStructTrimDriver"]

from ..oneway_struct_driver import OnewayStructDriver
import numpy as np
from mpi4py import MPI


class OnewayStructTrimDriver(OnewayStructDriver):
    """
    Oneway-coupled structural sizing driver with AOA-based trim.

    Runs oneway-coupled structural sizing while trimming the wing by
    scaling aero/struct loads linearly with changes in AOA. This is
    valid in the small-perturbation (linearized aerodynamics) regime.

    All load scaling applies only to uncoupled scenarios
    (scenario.coupled = False).

    The user should set up a load-factor-based composite function such as:
        lift - load_factor * weight = 0
    where load_factor is user-specified for each scenario.

    Parameters
    ----------
    solvers : SolverManager
    model : FUNtoFEMmodel
    initial_trim_dict : dict
        Initial trim state for each uncoupled scenario, keyed by scenario name.
        Each entry must contain:
          - 'AOA'  : initial angle of attack (must be non-zero)
          - 'cl'   : initial lift coefficient (C_L normalized by area and qinf)
        Example: {scenario_name: {'cl': cl_0, 'AOA': AOA_0}}
    transfer_settings : TransferSettings, optional
    nprocs : int, optional
    fun3d_dir : path, optional
    external_shape : bool, optional
    timing_file : str or path, optional
    """

    def __init__(
        self,
        solvers,
        model,
        initial_trim_dict: dict,
        transfer_settings=None,
        nprocs=None,
        fun3d_dir=None,
        external_shape=False,
        timing_file=None,
    ):

        # create base class OnewayStructDriver
        super(OnewayStructTrimDriver, self).__init__(
            solvers,
            model,
            transfer_settings,
            nprocs,
            fun3d_dir,
            external_shape,
            timing_file,
        )

        # Validate the trim dict before storing it: AOA=0 would cause
        # divide-by-zero in load scaling and derivative computation.
        for scenario in self.uncoupled_scenarios:
            if scenario.name not in initial_trim_dict:
                raise KeyError(
                    f"OnewayStructTrimDriver: scenario '{scenario.name}' is uncoupled "
                    f"but has no entry in initial_trim_dict."
                )
            orig_AOA = initial_trim_dict[scenario.name].get("AOA")
            if orig_AOA is None:
                raise KeyError(
                    f"OnewayStructTrimDriver: initial_trim_dict['{scenario.name}'] "
                    f"is missing required key 'AOA'."
                )
            if orig_AOA == 0.0:
                raise ValueError(
                    f"OnewayStructTrimDriver: initial_trim_dict['{scenario.name}']['AOA'] "
                    f"must be non-zero (got 0). Load scaling requires a non-zero reference AOA."
                )

        # get data from scenario initial trim dict
        # {scenario_name : {'cl' : cl_0, 'AOA' : AOA_0}}
        # only required to put scenarios which are uncoupled here
        # with scenario.coupled = False boolean
        self.initial_trim_dict = initial_trim_dict

        # save initial struct loads vectors for each scenario
        self._orig_struct_loads = {}
        for scenario in self.uncoupled_scenarios:
            self._orig_struct_loads[scenario.name] = {}
            for body in model.bodies:
                struct_loads = body.struct_loads[scenario.id]
                self._orig_struct_loads[scenario.name][body.name] = struct_loads * 1.0

    @classmethod
    def prime_loads_from_file(
        cls,
        filename,
        solvers,
        initial_trim_dict,
        model,
        nprocs,
        transfer_settings,
        external_shape=False,
        init_transfer=False,
        timing_file=None,
    ):
        # same as base class prime_loads_from_file but with extra input argument
        # aka initial_trim_dict
        return super().prime_loads_from_file(
            filename,
            solvers,
            model,
            nprocs,
            transfer_settings,
            external_shape=external_shape,
            init_transfer=init_transfer,
            timing_file=timing_file,
            initial_trim_dict=initial_trim_dict,
        )

    def solve_forward(self):

        # scale up the loads by new AOA vs previous AOA
        # note this only works for steady-state case
        for scenario in self.uncoupled_scenarios:
            orig_AOA = self.initial_trim_dict[scenario.name]["AOA"]
            new_AOA = scenario.get_variable("AOA").value.real
            for body in self.model.bodies:
                orig_struct_loads = self._orig_struct_loads[scenario.name][body.name]
                body.struct_loads[scenario.id][:] = (
                    orig_struct_loads * new_AOA / orig_AOA
                )[:]

        # now do super class solve_forward which will include
        # transferring fixed aero loads to the new struct loads and then linear static solve
        super(OnewayStructTrimDriver, self).solve_forward()

        # compute new lift values, for function name cl
        for scenario in self.uncoupled_scenarios:
            orig_cl = self.initial_trim_dict[scenario.name]["cl"]
            orig_AOA = self.initial_trim_dict[scenario.name]["AOA"]
            new_AOA = scenario.get_variable("AOA").value.real

            for func in scenario.functions:
                if func.name == "cl":
                    func.value = orig_cl * new_AOA / orig_AOA

        # composite functions are evaluated in the OptimizationManager FYI and will also be updated after this..

    def _solve_steady_adjoint(self, scenario, bodies):
        super()._solve_steady_adjoint(scenario, bodies)

        # get additional derivative terms for custom
        self._get_custom_derivatives(scenario)

    def _solve_unsteady_adjoint(self, scenario, bodies):
        raise NotImplementedError(
            "OnewayStructTrimDriver does not support unsteady adjoint. "
            "The trim load-scaling derivatives assume a steady-state AOA relationship "
            "and are not valid for time-varying loads."
        )

    def _get_custom_derivatives(self, scenario):
        """
        Compute trim-specific derivatives of functions w.r.t. AOA.

        For the 'cl' function: d(cl)/d(AOA) = cl_0 / AOA_0  (linear scaling).
        For all other adjoint functions: accounts for the change in structural
        loads with AOA via the dot product of the load adjoint with the
        reference load sensitivity d(f_S)/d(AOA) = f_S_0 / AOA_0.
        """

        orig_cl = self.initial_trim_dict[scenario.name]["cl"]
        orig_AOA = self.initial_trim_dict[scenario.name]["AOA"]
        aoa_var = scenario.get_variable("AOA")

        # since mass not adjoint function only iterate over these guys
        adjoint_functions = [func for func in scenario.functions if func.adjoint]
        for ifunc, func in enumerate(adjoint_functions):
            if func.name == "cl":
                func.derivatives[aoa_var] = orig_cl / orig_AOA
                continue

            # account for changing loads terms in AOA:
            # d(func)/d(AOA) = psi_fS^T * d(fS)/d(AOA) = psi_fS^T * (fS_0 / AOA_0)
            AOA_deriv = 0.0
            for body in self.model.bodies:
                struct_loads_ajp = body.get_struct_loads_ajp(scenario)
                func_fs_ajp = struct_loads_ajp[:, ifunc]
                # I am removing an implicit deep copy (before was doing * 1.0), might need it
                orig_struct_loads = self._orig_struct_loads[scenario.name][body.name]

                AOA_deriv += np.dot(func_fs_ajp, orig_struct_loads / orig_AOA)

            # add across all processors then reduce
            global_derivative = self.comm.reduce(AOA_deriv, op=MPI.SUM, root=0)
            global_derivative = self.comm.bcast(global_derivative)

            func.derivatives[aoa_var] = global_derivative
