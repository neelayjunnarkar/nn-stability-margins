# Dissipative code using the condition that is a BMI in controller parameters and storage function.
import time

import cvxpy as cp
import numpy as np

from variable_structs import ControllerThetaParameters, PlantParameters


def is_positive_semidefinite(X):
    if not np.allclose(X, X.T):
        return False
    eigvals, _eigvecs = np.linalg.eigh(X)
    if np.min(eigvals) < 0:
        return False, f"Minimum eigenvalue {np.min(eigvals)} < 0"
    return True


def is_positive_definite(X):
    # Check symmetric.
    if not np.allclose(X, X.T):
        return False
    # Check PD (np.linalg.cholesky does not check for symmetry)
    try:
        np.linalg.cholesky(X)
    except Exception as _e:
        return False
    return True


def construct_dissipativity_matrix(
    A, Bw, Bd, Cv, Dvw, Dvd, Ce, Dew, Ded, P, LDelta, Mvw, Mww, Xdd, Xde, LX, stacker
):
    if stacker == "numpy":
        stacker = np.bmat
    elif stacker == "cvxpy":
        stacker = cp.bmat
    else:
        raise ValueError(f"Stacker {stacker} must be 'numpy' or 'cvxpy'.")

    # F = F1 + F2 + F3
    # where F1 has A^T P + P A, F2 is term with M, and F3 is term with X

    # fmt: off
    F1 = stacker([
        [A.T @ P + P @ A, P @ Bw, P @ Bd],
        [Bw.T @ P, np.zeros((Bw.shape[1], Bw.shape[1] + Bd.shape[1]))],
        [Bd.T @ P, np.zeros((Bd.shape[1], Bw.shape[1] + Bd.shape[1]))]
    ])

    F2 = stacker([
        [np.zeros((Cv.shape[1], Cv.shape[1])), Cv.T @ Mvw, np.zeros((Cv.shape[1], Dvd.shape[1]))],
        [Mvw.T @ Cv, Mvw.T @ Dvw + Dvw.T @ Mvw + Mww, Mvw.T @ Dvd],
        [np.zeros((Dvd.shape[1], Cv.shape[1])), Dvd.T @ Mvw, np.zeros((Dvd.shape[1], Dvd.shape[1]))]
    ])

    F3 = -stacker([
        [np.zeros((Ce.shape[1], Ce.shape[1] + Dew.shape[1])), Ce.T @ Xde.T],
        [np.zeros((Dew.shape[1], Ce.shape[1] + Dew.shape[1])), Dew.T @ Xde.T],
        [Xde @ Ce, Xde @ Dew, Xde @ Ded + Ded.T @ Xde.T + Xdd]
    ])

    F = F1 + F2 + F3

    # mat = [F, H^T; H, -I]
    H = stacker([
        [LDelta @ Cv, LDelta @ Dvw, LDelta @ Dvd],
        [LX @ Ce, LX @ Dew, LX @ Ded]
    ])

    mat = stacker([
        [F, H.T],
        [H, -np.eye(H.shape[0])]
    ])

    # fmt: on
    return mat


def construct_closed_loop(
    plant_params: PlantParameters,
    LDeltap,
    controller_params: ControllerThetaParameters,
    stacker,
):
    Ap = plant_params.Ap
    Bpw = plant_params.Bpw
    Bpd = plant_params.Bpd
    Bpu = plant_params.Bpu
    Cpv = plant_params.Cpv
    Dpvw = plant_params.Dpvw
    Dpvd = plant_params.Dpvd
    Dpvu = plant_params.Dpvu
    Cpe = plant_params.Cpe
    Dpew = plant_params.Dpew
    Dped = plant_params.Dped
    Dpeu = plant_params.Dpeu
    Cpy = plant_params.Cpy
    Dpyw = plant_params.Dpyw
    Dpyd = plant_params.Dpyd
    MDeltapvv = plant_params.MDeltapvv
    MDeltapvw = plant_params.MDeltapvw
    MDeltapww = plant_params.MDeltapww
    Ak = controller_params.Ak
    Bkw = controller_params.Bkw
    Bky = controller_params.Bky
    Ckv = controller_params.Ckv
    Dkvw = controller_params.Dkvw
    Dkvy = controller_params.Dkvy
    Cku = controller_params.Cku
    Dkuw = controller_params.Dkuw
    Dkuy = controller_params.Dkuy
    Lambda = controller_params.Lambda

    if stacker == "numpy":
        stacker = np.bmat
    elif stacker == "cvxpy":
        stacker = cp.bmat
    else:
        raise ValueError(f"Stacker {stacker} must be 'numpy' or 'cvxpy'.")

    # fmt: off
    A = stacker([
        [Ap + Bpu @ Dkuy @ Cpy, Bpu @ Cku],
        [Bky @ Cpy, Ak]
    ])
    Bw = stacker([
        [Bpw + Bpu @ Dkuy @ Dpyw, Bpu @ Dkuw],
        [Bky @ Dpyw, Bkw]
    ])
    Bd = stacker([
        [Bpd + Bpu @ Dkuy @ Dpyd],
        [Bky @ Dpyd]
    ])
    Cv = stacker([
        [Cpv + Dpvu @ Dkuy @ Cpy, Dpvu @ Cku],
        [Dkvy @ Cpy, Ckv]
    ])
    Dvw = stacker([
        [Dpvw + Dpvu @ Dkuy @ Dpyw, Dpvu @ Dkuw],
        [Dkvy @ Dpyw, Dkvw]
    ])
    Dvd = stacker([
        [Dpvd + Dpvu @ Dkuy @ Dpyd],
        [Dkvy @ Dpyd]
    ])
    Ce = stacker([
        [Cpe + Dpeu @ Dkuy @ Cpy, Dpeu @ Cku]
    ])
    Dew = stacker([
        [Dpew + Dpeu @ Dkuy @ Dpyw, Dpeu @ Dkuw]
    ])
    Ded = stacker([
        [Dped + Dpeu @ Dkuy @ Dpyd]
    ])

    LDelta = np.bmat([
        [LDeltap, np.zeros((LDeltap.shape[0], Lambda.shape[1]))],
        [np.zeros((Lambda.shape[0], LDeltap.shape[1] + Lambda.shape[1]))]
    ])

    Mvw = stacker([
        [MDeltapvw, np.zeros((MDeltapvv.shape[0], Lambda.shape[1]))],
        [np.zeros((Lambda.shape[0], MDeltapvw.shape[1])), Lambda]
    ])
    Mww = stacker([
        [MDeltapww, np.zeros((MDeltapww.shape[0], Lambda.shape[1]))],
        [np.zeros((Lambda.shape[0], MDeltapww.shape[1])), -2*Lambda]
    ])
    # fmt: on

    return A, Bw, Bd, Cv, Dvw, Dvd, Ce, Dew, Ded, LDelta, Mvw, Mww


class Projector:
    """Projection and verification related to dissipativity BMI."""

    def __init__(
        self,
        plant_params: PlantParameters,
        # Epsilon to be used in enforcing definiteness of conditions
        eps,
        # Dimensions of variables for controller
        nonlin_size,
        output_size,
        state_size,
        input_size,
    ):
        self.plant_params = plant_params
        self.eps = eps
        self.nonlin_size = nonlin_size
        self.output_size = output_size
        self.state_size = state_size
        self.input_size = input_size

        assert is_positive_semidefinite(plant_params.MDeltapvv)
        Dm, Vm = np.linalg.eigh(plant_params.MDeltapvv)
        self.LDeltap = np.diag(np.sqrt(Dm)) @ Vm.T

        assert is_positive_semidefinite(-plant_params.Xee)
        Dx, Vx = np.linalg.eigh(-plant_params.Xee)
        self.LX = np.diag(np.sqrt(Dx)) @ Vx.T

        self._construct_projection_problem()
        self._construct_check_dissipativity_problem()

    def _construct_projection_problem(self):
        # Define projection problem: Projecting theta into BMI parameterized by P and Lambda.
        # Parameters
        plant_state_size = self.plant_params.Ap.shape[0]
        P_size = plant_state_size + self.state_size
        self.pprojP = cp.Parameter((P_size, P_size), PSD=True)
        pprojLambda = cp.Parameter((self.nonlin_size, self.nonlin_size), diag=True)
        pprojAk = cp.Parameter((self.state_size, self.state_size))
        pprojBkw = cp.Parameter((self.state_size, self.nonlin_size))
        pprojBky = cp.Parameter((self.state_size, self.input_size))
        pprojCkv = cp.Parameter((self.nonlin_size, self.state_size))
        pprojDkvw = cp.Parameter((self.nonlin_size, self.nonlin_size))
        pprojDkvy = cp.Parameter((self.nonlin_size, self.input_size))
        pprojCku = cp.Parameter((self.output_size, self.state_size))
        pprojDkuw = cp.Parameter((self.output_size, self.nonlin_size))
        pprojDkuy = cp.Parameter((self.output_size, self.input_size))
        # fmt: off
        self.pproj_k = ControllerThetaParameters(
            pprojAk, pprojBkw, pprojBky, pprojCkv, pprojDkvw, pprojDkvy,
            pprojCku, pprojDkuw, pprojDkuy, pprojLambda
        )
        # fmt: on

        # Variables
        vprojAk = cp.Variable((self.state_size, self.state_size))
        vprojBkw = cp.Variable((self.state_size, self.nonlin_size))
        vprojBky = cp.Variable((self.state_size, self.input_size))
        vprojCkv = cp.Variable((self.nonlin_size, self.state_size))
        vprojDkvw = cp.Variable((self.nonlin_size, self.nonlin_size))
        vprojDkvy = cp.Variable((self.nonlin_size, self.input_size))
        vprojCku = cp.Variable((self.output_size, self.state_size))
        vprojDkuw = cp.Variable((self.output_size, self.nonlin_size))
        vprojDkuy = cp.Variable((self.output_size, self.input_size))
        # fmt: off
        self.vproj_k = ControllerThetaParameters(
            vprojAk, vprojBkw, vprojBky, vprojCkv, vprojDkvw, vprojDkvy,
            vprojCku, vprojDkuw, vprojDkuy, None,
        )


        controller_params = ControllerThetaParameters(
            vprojAk, vprojBkw, vprojBky, vprojCkv, vprojDkvw, vprojDkvy,
            vprojCku, vprojDkuw, vprojDkuy, self.pproj_k.Lambda
        )
        A, Bw, Bd, Cv, Dvw, Dvd, Ce, Dew, Ded, LDelta, Mvw, Mww = construct_closed_loop(
            self.plant_params, self.LDeltap, controller_params, "cvxpy"
        )
        mat = construct_dissipativity_matrix(
            A, Bw, Bd, Cv, Dvw, Dvd, Ce, Dew, Ded, self.pprojP,
            LDelta, Mvw, Mww,
            self.plant_params.Xdd, self.plant_params.Xde, self.LX,
            "cvxpy"
        )
        # fmt: on

        constraints = [
            self.pprojP >> self.eps * np.eye(self.pprojP.shape[0]),
            self.pproj_k.Lambda >> 0,
            mat << 0,
        ]

        # fmt: off
        cost_projection_error = sum([
            cp.sum_squares(pprojAk - vprojAk),
            cp.sum_squares(pprojBkw - vprojBkw),
            cp.sum_squares(pprojBky - vprojBky),
            cp.sum_squares(pprojCkv - vprojCkv),
            cp.sum_squares(pprojDkvw - vprojDkvw),
            cp.sum_squares(pprojDkvy - vprojDkvy),
            cp.sum_squares(pprojCku - vprojCku),
            cp.sum_squares(pprojDkuw - vprojDkuw),
            cp.sum_squares(pprojDkuy - vprojDkuy),
        ])
        cost_size = sum([
            cp.sum_squares(vprojAk),
            cp.sum_squares(vprojBkw),
            cp.sum_squares(vprojBky),
            cp.sum_squares(vprojCkv),
            cp.sum_squares(vprojDkvw),
            cp.sum_squares(vprojDkvy),
            cp.sum_squares(vprojCku),
            cp.sum_squares(vprojDkuw),
            cp.sum_squares(vprojDkuy),
        ])
        # fmt: on
        objective = cost_projection_error

        self.proj_problem = cp.Problem(cp.Minimize(objective), constraints)

    def project(
        self,
        controller_params: ControllerThetaParameters,
        P,
        solver=cp.MOSEK,
        **kwargs,
    ):
        """Projects input variables to set corresponding to dissipative controllers."""
        self.pproj_k.Ak.value = controller_params.Ak
        self.pproj_k.Bkw.value = controller_params.Bkw
        self.pproj_k.Bky.value = controller_params.Bky
        self.pproj_k.Ckv.value = controller_params.Ckv
        self.pproj_k.Dkvw.value = controller_params.Dkvw
        self.pproj_k.Dkvy.value = controller_params.Dkvy
        self.pproj_k.Cku.value = controller_params.Cku
        self.pproj_k.Dkuw.value = controller_params.Dkuw
        self.pproj_k.Dkuy.value = controller_params.Dkuy
        self.pproj_k.Lambda.value = controller_params.Lambda
        self.pprojP.value = P

        try:
            t0 = time.perf_counter()
            self.proj_problem.solve(enforce_dpp=True, solver=solver, **kwargs)
            t1 = time.perf_counter()
            # print(f"Projection solving took {t1-t0} seconds.")
        except Exception as e:
            print(f"Failed to solve: {e}")
            raise e

        feas_stats = [
            cp.OPTIMAL,
            cp.UNBOUNDED,
            cp.OPTIMAL_INACCURATE,
            cp.UNBOUNDED_INACCURATE,
        ]
        if self.proj_problem.status not in feas_stats:
            print(f"Failed to solve with status {self.proj_problem.status}")
            raise Exception()
        print(f"Projection objective: {self.proj_problem.value}")

        # fmt: off
        new_controller_params = ControllerThetaParameters(
            self.vproj_k.Ak.value, self.vproj_k.Bkw.value, self.vproj_k.Bky.value,
            self.vproj_k.Ckv.value, self.vproj_k.Dkvw.value, self.vproj_k.Dkvy.value,
            self.vproj_k.Cku.value, self.vproj_k.Dkuw.value, self.vproj_k.Dkuy.value,
            None
        )
        # fmt: on

        return new_controller_params

    def _construct_check_dissipativity_problem(self):
        # Define dissipativity verification problem: find if there exist P and Lambda
        # that certify a given controller is dissipative.
        # Parameters
        plant_state_size = self.plant_params.Ap.shape[0]
        P_size = plant_state_size + self.state_size
        self.pcheckP = cp.Parameter((P_size, P_size), PSD=True)
        pcheckLambda = cp.Parameter((self.nonlin_size, self.nonlin_size), diag=True)
        pcheckAk = cp.Parameter((self.state_size, self.state_size))
        pcheckBkw = cp.Parameter((self.state_size, self.nonlin_size))
        pcheckBky = cp.Parameter((self.state_size, self.input_size))
        pcheckCkv = cp.Parameter((self.nonlin_size, self.state_size))
        pcheckDkvw = cp.Parameter((self.nonlin_size, self.nonlin_size))
        pcheckDkvy = cp.Parameter((self.nonlin_size, self.input_size))
        pcheckCku = cp.Parameter((self.output_size, self.state_size))
        pcheckDkuw = cp.Parameter((self.output_size, self.nonlin_size))
        pcheckDkuy = cp.Parameter((self.output_size, self.input_size))
        # fmt: off
        self.pcheck_k = ControllerThetaParameters(
            pcheckAk, pcheckBkw, pcheckBky, pcheckCkv, pcheckDkvw, pcheckDkvy,
            pcheckCku, pcheckDkuw, pcheckDkuy, pcheckLambda
        )
        # fmt: on

        # Variables
        self.vcheckP = cp.Variable((P_size, P_size), PSD=True)
        self.vcheckLambda = cp.Variable((self.nonlin_size, self.nonlin_size), diag=True)
        self.vcheckEps = cp.Variable(nonneg=True)

        # fmt: off
        controller_params = ControllerThetaParameters(
            pcheckAk, pcheckBkw, pcheckBky, pcheckCkv, pcheckDkvw, pcheckDkvy,
            pcheckCku, pcheckDkuw, pcheckDkuy, self.vcheckLambda
        )
        A, Bw, Bd, Cv, Dvw, Dvd, Ce, Dew, Ded, LDelta, Mvw, Mww = construct_closed_loop(
            self.plant_params, self.LDeltap, controller_params, "cvxpy"
        )
        mat = construct_dissipativity_matrix(
            A, Bw, Bd, Cv, Dvw, Dvd, Ce, Dew, Ded, self.vcheckP,
            LDelta, Mvw, Mww, 
            self.plant_params.Xdd, self.plant_params.Xde, self.LX,
            "cvxpy"
        )
        # fmt: on

        constraints = [
            self.vcheckP >> self.eps * np.eye(self.vcheckP.shape[0]),
            self.vcheckLambda >> self.eps * np.eye(self.vcheckLambda.shape[0]),
            mat << -self.vcheckEps,
            self.vcheckEps >= 0,
        ]

        # fmt: off
        cost_projection_error = sum([
            cp.sum_squares(self.vcheckP - self.pcheckP),
            cp.sum_squares(self.vcheckLambda - self.pcheck_k.Lambda),
        ])
        cost_size = sum([
            cp.sum_squares(self.vcheckP),
            cp.sum_squares(self.vcheckLambda),
        ])
        # fmt: on
        objective = -self.vcheckEps

        self.check_problem = cp.Problem(cp.Minimize(objective), constraints)

    def is_dissipative(
        self,
        controller_params: ControllerThetaParameters,
        P,
        solver=cp.MOSEK,
        **kwargs,
    ):
        """Checks if there exist P and Lambda that certify the controller is dissipative."""
        self.pcheck_k.Ak.value = controller_params.Ak
        self.pcheck_k.Bkw.value = controller_params.Bkw
        self.pcheck_k.Bky.value = controller_params.Bky
        self.pcheck_k.Ckv.value = controller_params.Ckv
        self.pcheck_k.Dkvw.value = controller_params.Dkvw
        self.pcheck_k.Dkvy.value = controller_params.Dkvy
        self.pcheck_k.Cku.value = controller_params.Cku
        self.pcheck_k.Dkuw.value = controller_params.Dkuw
        self.pcheck_k.Dkuy.value = controller_params.Dkuy
        self.pcheck_k.Lambda.value = controller_params.Lambda
        print(P)
        self.pcheckP.value = P

        try:
            t0 = time.perf_counter()
            self.check_problem.solve(enforce_dpp=True, solver=solver, **kwargs)
            t1 = time.perf_counter()
            # print(f"Projection solving took {t1-t0} seconds.")
        except Exception as e:
            print(f"Failed to solve: {e}")
            return False, None, None

        feas_stats = [
            cp.OPTIMAL,
            cp.UNBOUNDED,
            cp.OPTIMAL_INACCURATE,
            cp.UNBOUNDED_INACCURATE,
        ]
        if self.check_problem.status not in feas_stats:
            print(f"Failed to solve with status {self.check_problem.status}")
            return False, None, None
        print(f"Projection objective: {self.check_problem.value}")

        newP = self.vcheckP.value
        newLambda = self.vcheckLambda.value.toarray()

        return True, newP, newLambda
