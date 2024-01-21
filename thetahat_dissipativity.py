import time

import cvxpy as cp
import numpy as np

from variable_structs import (
    ControllerLTIThetahatParameters,
    ControllerThetahatParameters,
    PlantParameters,
)


def is_positive_semidefinite(X):
    if not np.allclose(X, X.T):
        return False
    eigvals, _eigvecs = np.linalg.eigh(X)
    if np.min(eigvals) < 0:
        return False
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
    plant_params: PlantParameters,
    LDeltap,
    LX,
    controller_params: ControllerThetahatParameters,
    stacker,
):
    if stacker == "numpy":
        stacker = np.bmat
    elif stacker == "cvxpy":
        stacker = cp.bmat
    else:
        raise ValueError(f"Stacker {stacker} must be 'numpy' or 'cvxpy'.")

    K = controller_params
    P = plant_params

    ytpay11 = P.Ap @ K.R + P.Bpu @ K.NA21
    ytpay12 = P.Ap + P.Bpu @ K.NA22 @ P.Cpy
    ytpay21 = K.NA11
    ytpay22 = K.S @ P.Ap + K.NA12 @ P.Cpy
    ytpay = stacker([[ytpay11, ytpay12], [ytpay21, ytpay22]])

    ytpbw11 = P.Bpw + P.Bpu @ K.NA22 @ P.Dpyw
    ytpbw12 = P.Bpu @ K.Dkuw
    ytpbw21 = K.S @ P.Bpw + K.NA12 @ P.Dpyw
    ytpbw22 = K.NB
    ytpbw = stacker([[ytpbw11, ytpbw12], [ytpbw21, ytpbw22]])

    ytpbd1 = P.Bpd + P.Bpu @ K.NA22 @ P.Dpyd
    ytpbd2 = K.S @ P.Bpd + K.NA12 @ P.Dpyd
    ytpbd = stacker([[ytpbd1], [ytpbd2]])

    mvwtcvy11 = P.MDeltapvw.T @ P.Cpv @ K.R + P.MDeltapvw.T @ P.Dpvu @ K.NA21
    mvwtcvy12 = P.MDeltapvw.T @ P.Cpv + P.MDeltapvw.T @ P.Dpvu @ K.NA22 @ P.Cpy
    mvwtcvy21 = K.NC
    mvwtcvy22 = K.Dkvyhat @ P.Cpy
    mvwtcvy = stacker([[mvwtcvy11, mvwtcvy12], [mvwtcvy21, mvwtcvy22]])

    nxdecey1 = -P.Xde @ (P.Cpe @ K.R + P.Dpeu @ K.NA21)
    nxdecey2 = -P.Xde @ (P.Cpe + P.Dpeu @ K.NA22 @ P.Cpy)
    nxdecey = stacker([[nxdecey1, nxdecey2]])

    ldeltacvy11 = LDeltap @ (P.Cpv @ K.R + P.Dpvu @ K.NA21)
    ldeltacvy12 = LDeltap @ (P.Cpv + P.Dpvu @ K.NA22 @ P.Cpy)
    ldeltacvy1 = stacker([[ldeltacvy11, ldeltacvy12]])

    lxcey1 = LX @ (P.Cpe @ K.R + P.Dpeu @ K.NA21)
    lxcey2 = LX @ (P.Cpe + P.Dpeu @ K.NA22 @ P.Cpy)
    lxcey = stacker([[lxcey1, lxcey2]])

    mvwtdvw11 = P.MDeltapvw.T @ (P.Dpvw + P.Dpvu @ K.NA22 @ P.Dpyw)
    mvwtdvw12 = P.MDeltapvw.T @ P.Dpvu @ K.Dkuw
    mvwtdvw21 = K.Dkvyhat @ P.Dpyw
    mvwtdvw22 = K.Dkvwhat
    mvwtdvw = stacker([[mvwtdvw11, mvwtdvw12], [mvwtdvw21, mvwtdvw22]])

    mvwtdvd1 = P.MDeltapvw.T @ (P.Dpvd + P.Dpvu @ K.NA22 @ P.Dpyd)
    mvwtdvd2 = K.Dkvyhat @ P.Dpyd
    mvwtdvd = stacker([[mvwtdvd1], [mvwtdvd2]])

    # fmt: off
    Mww = stacker([
        [P.MDeltapww, np.zeros((P.MDeltapww.shape[0], K.Lambda.shape[1]))],
        [np.zeros((K.Lambda.shape[0], P.MDeltapww.shape[1])), -2 * K.Lambda],
    ])
    # fmt: on

    Dew = stacker([[P.Dpew + P.Dpeu @ K.NA22 @ P.Dpyw, P.Dpew @ K.Dkuw]])

    Ded = P.Dped + P.Dpeu @ K.NA22 @ P.Dpyd

    # fmt: off
    ldeltadvw1 = stacker([
        [LDeltap @ (P.Dpvw + P.Dpvu @ K.NA22 @ P.Dpyw), LDeltap @ P.Dpvu @ K.Dkuw]
    ])
    # fmt: on

    ldeltadvd1 = LDeltap @ (P.Dpvd + P.Dpvu @ K.NA22 @ P.Dpyd)

    # Define half the matrix and then add it to its transpose
    # Ensure Mww is symmetric. It needs to be for the method overall anyway.
    assert np.allclose(P.MDeltapww, P.MDeltapww.T)
    # Ensure Xdd is symmetric. It needs to be for the method overall anyway.
    assert np.allclose(P.Xdd, P.Xdd.T)
    # fmt: off
    row1 = stacker([[
        ytpay.T,
        np.zeros((
            ytpay.T.shape[0],
            mvwtdvw.shape[1]
            + P.Xdd.shape[1]
            + LDeltap.shape[0]
            + LX.shape[0],
        )),
    ]])
    row2 = stacker([[
        ytpbw.T + mvwtcvy,
        mvwtdvw + 0.5 * Mww,
        np.zeros(
            (ytpbw.T.shape[0], P.Xdd.shape[1] + LDeltap.shape[0] + LX.shape[0])
        ),
    ]])
    row3 = stacker([[
        ytpbd.T + nxdecey,
        mvwtdvd.T - P.Xde @ Dew,
        -P.Xde @ Ded - 0.5 * P.Xdd,
        np.zeros((ytpbd.T.shape[0], LDeltap.shape[0] + LX.shape[0])),
    ]])
    row4 = stacker([[
        ldeltacvy1,
        ldeltadvw1,
        ldeltadvd1,
        -0.5 * np.eye(ldeltacvy1.shape[0]),
        np.zeros((ldeltacvy1.shape[0], LX.shape[0])),
    ]])
    row5 = stacker([[
        lxcey,
        LX @ Dew,
        LX @ Ded,
        np.zeros((lxcey.shape[0], ldeltacvy1.shape[0])),
        -0.5 * np.eye(lxcey.shape[0]),
    ]])
    mat = stacker([
        [row1],
        [row2],
        [row3],
        [row4],
        [row5],
    ])
    # fmt: on
    mat = mat + mat.T

    return mat


class Projector:
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
        # Parameters for tuning condition number of I - RS,
        trs_mode,  # Either "fixed" or "variable"
        min_trs,  # Used as the trs value when trs_mode="fixed"
    ):
        self.plant_params = plant_params
        self.eps = eps
        self.nonlin_size = nonlin_size
        self.output_size = output_size
        self.state_size = state_size
        self.input_size = input_size
        self.trs_mode = trs_mode
        self.min_trs = min_trs

        assert is_positive_semidefinite(plant_params.MDeltapvv)
        Dm, Vm = np.linalg.eigh(plant_params.MDeltapvv)
        self.LDeltap = np.diag(np.sqrt(Dm)) @ Vm.T

        assert is_positive_semidefinite(-plant_params.Xee)
        Dx, Vx = np.linalg.eigh(-plant_params.Xee)
        self.LX = np.diag(np.sqrt(Dx)) @ Vx.T

        self._construct_projection_problem()

    def _construct_projection_problem(self):
        # Parameters: This is the thetahat to be projected into the stabilizing set.
        self.pThetahat = ControllerThetahatParameters(
            S=cp.Parameter((self.state_size, self.state_size), PSD=True),
            R=cp.Parameter((self.state_size, self.state_size), PSD=True),
            NA11=cp.Parameter((self.state_size, self.state_size)),
            NA12=cp.Parameter((self.state_size, self.input_size)),
            NA21=cp.Parameter((self.output_size, self.state_size)),
            NA22=cp.Parameter((self.output_size, self.input_size)),
            NB=cp.Parameter((self.state_size, self.nonlin_size)),
            NC=cp.Parameter((self.nonlin_size, self.state_size)),
            Dkuw=cp.Parameter((self.output_size, self.nonlin_size)),
            Dkvyhat=cp.Parameter((self.nonlin_size, self.input_size)),
            Dkvwhat=cp.Parameter((self.nonlin_size, self.nonlin_size)),
            Lambda=cp.Parameter((self.nonlin_size, self.nonlin_size), diag=True),
        )

        # Variables: This will be the solution of the projection.
        self.vThetahat = ControllerThetahatParameters(
            S=cp.Variable((self.state_size, self.state_size), PSD=True),
            R=cp.Variable((self.state_size, self.state_size), PSD=True),
            NA11=cp.Variable((self.state_size, self.state_size)),
            NA12=cp.Variable((self.state_size, self.input_size)),
            NA21=cp.Variable((self.output_size, self.state_size)),
            NA22=cp.Variable((self.output_size, self.input_size)),
            NB=cp.Variable((self.state_size, self.nonlin_size)),
            NC=cp.Variable((self.nonlin_size, self.state_size)),
            Dkuw=cp.Variable((self.output_size, self.nonlin_size)),
            Dkvyhat=cp.Variable((self.nonlin_size, self.input_size)),
            Dkvwhat=cp.Variable((self.nonlin_size, self.nonlin_size)),
            Lambda=cp.Variable((self.nonlin_size, self.nonlin_size), diag=True),
        )

        mat = construct_dissipativity_matrix(
            plant_params=self.plant_params,
            LDeltap=self.LDeltap,
            LX=self.LX,
            controller_params=self.vThetahat,
            stacker="cvxpy",
        )

        # Used for conditioning I - RS
        if self.trs_mode == "variable":
            self.vtrs = cp.Variable(nonneg=True)
            cost_ill_conditioning = -self.vtrs
        elif self.trs_mode == "fixed":
            self.vtrs = self.min_trs
            cost_ill_conditioning = 0
        else:
            raise ValueError(f"Unexpected trs_mode value of {self.trs_mode}.")

        # fmt: off
        constraints = [
            # self.vtrs >= self.min_trs,
            self.vThetahat.S >> self.eps * np.eye(self.vThetahat.S.shape[0]),
            self.vThetahat.R >> self.eps * np.eye(self.vThetahat.R.shape[0]),
            self.vThetahat.Lambda >> self.eps * np.eye(self.vThetahat.Lambda.shape[0]),
            cp.bmat([
                [self.vThetahat.R, self.vtrs * np.eye(self.vThetahat.R.shape[0])],
                [self.vtrs * np.eye(self.vThetahat.S.shape[0]), self.vThetahat.S],
            ]) >> self.eps * np.eye(self.vThetahat.R.shape[0] + self.vThetahat.S.shape[0]),
            mat << 0,
        ]
        if self.trs_mode == "variable":
            constraints.append(self.vtrs >= self.min_trs)


        cost_projection_error = sum([
            cp.sum_squares(self.pThetahat.Dkuw - self.vThetahat.Dkuw),
            cp.sum_squares(self.pThetahat.S - self.vThetahat.S),
            cp.sum_squares(self.pThetahat.R - self.vThetahat.R),
            cp.sum_squares(self.pThetahat.Lambda - self.vThetahat.Lambda),
            cp.sum_squares(self.pThetahat.NA11 - self.vThetahat.NA11),
            cp.sum_squares(self.pThetahat.NA12 - self.vThetahat.NA12),
            cp.sum_squares(self.pThetahat.NA21 - self.vThetahat.NA21),
            cp.sum_squares(self.pThetahat.NA22 - self.vThetahat.NA22),
            cp.sum_squares(self.pThetahat.NB - self.vThetahat.NB),
            cp.sum_squares(self.pThetahat.NC - self.vThetahat.NC),
            cp.sum_squares(self.pThetahat.Dkvyhat - self.vThetahat.Dkvyhat),
            cp.sum_squares(self.pThetahat.Dkvwhat - self.vThetahat.Dkvwhat),
        ])
        cost_size = sum([
            cp.sum_squares(self.vThetahat.Dkuw),
            cp.sum_squares(self.vThetahat.S),
            cp.sum_squares(self.vThetahat.R),
            cp.sum_squares(self.vThetahat.Lambda),
            cp.sum_squares(self.vThetahat.NA11),
            cp.sum_squares(self.vThetahat.NA12),
            cp.sum_squares(self.vThetahat.NA21),
            cp.sum_squares(self.vThetahat.NA22),
            cp.sum_squares(self.vThetahat.NB),
            cp.sum_squares(self.vThetahat.NC),
            cp.sum_squares(self.vThetahat.Dkvyhat),
            cp.sum_squares(self.vThetahat.Dkvwhat),
        ])
        # fmt: on
        objective = cost_projection_error + cost_ill_conditioning # + cost_size

        self.problem = cp.Problem(cp.Minimize(objective), constraints)

    def project(self, controller_params: ControllerThetahatParameters, solver=cp.MOSEK, **kwargs):
        """Projects input variables to set corresponding to dissipative controllers."""
        K = controller_params
        self.pThetahat.Dkuw.value = K.Dkuw
        self.pThetahat.S.value = K.S
        self.pThetahat.R.value = K.R
        self.pThetahat.Lambda.value = K.Lambda
        self.pThetahat.NA11.value = K.NA11
        self.pThetahat.NA12.value = K.NA12
        self.pThetahat.NA21.value = K.NA21
        self.pThetahat.NA22.value = K.NA22
        self.pThetahat.NB.value = K.NB
        self.pThetahat.NC.value = K.NC
        self.pThetahat.Dkvyhat.value = K.Dkvyhat
        self.pThetahat.Dkvwhat.value = K.Dkvwhat

        try:
            # t0 = time.perf_counter()
            self.problem.solve(enforce_dpp=True, solver=solver, **kwargs)
            # t1 = time.perf_counter()
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
        if self.problem.status not in feas_stats:
            print(f"Failed to solve with status {self.problem.status}")
            raise Exception()
        # print(f"Projection objective: {self.problem.value}")

        # fmt: off
        new_controller_params = ControllerThetahatParameters(
            self.vThetahat.S.value, self.vThetahat.R.value, self.vThetahat.NA11.value,
            self.vThetahat.NA12.value, self.vThetahat.NA21.value, self.vThetahat.NA22.value,
            self.vThetahat.NB.value, self.vThetahat.NC.value, self.vThetahat.Dkuw.value,
            self.vThetahat.Dkvyhat.value, self.vThetahat.Dkvwhat.value, self.vThetahat.Lambda.value.toarray()
        )
        # fmt: on
        return new_controller_params

    def is_dissipative(self, controller_params: ControllerThetahatParameters):
        """Check whether given variables already satisfy dissipativity condition."""
        # All inputs must be numpy 2d arrays.

        # Check S, R, and Lambda are positive definite
        if not is_positive_definite(controller_params.S):
            print("S is not PD.")
            return False
        if not is_positive_definite(controller_params.R):
            print("R is not PD.")
            return False
        if not is_positive_definite(controller_params.Lambda):
            print("Lambda is not PD.")
            return False

        # Check [R, I; I, S] is positive definite.
        # fmt: off
        mat = np.asarray(np.bmat([
            [controller_params.R, np.eye(controller_params.R.shape[0])],
            [np.eye(controller_params.R.shape[0]), controller_params.S]
        ]))
        # fmt: on
        if not is_positive_definite(mat):
            print("[R, I; I, S] is not PD.")
            return False

        # Check main dissipativity condition.
        mat = construct_dissipativity_matrix(
            plant_params=self.plant_params,
            LDeltap=self.LDeltap,
            LX=self.LX,
            controller_params=controller_params,
            stacker="numpy",
        )
        # Check condition mat <= 0
        return is_positive_semidefinite(-mat)


class LTIProjector:
    def __init__(
        self,
        plant_params: PlantParameters,
        # Epsilon to be used in enforcing definiteness of conditions
        eps,
        # Dimensions of variables for controller
        output_size,
        state_size,
        input_size,
        # Parameters for tuning condition number of I - RS,
        trs_mode,  # Either "fixed" or "variable"
        min_trs,  # Used as the trs value when trs_mode="fixed"
    ):
        self.plant_params = plant_params
        self.eps = eps
        self.output_size = output_size
        self.state_size = state_size
        self.input_size = input_size
        self.trs_mode = trs_mode
        self.min_trs = min_trs
        self.nonlin_size = 1  # placeholder nonlin size used for creating zeros

        assert is_positive_semidefinite(plant_params.MDeltapvv)
        Dm, Vm = np.linalg.eigh(plant_params.MDeltapvv)
        self.LDeltap = np.diag(np.sqrt(Dm)) @ Vm.T

        assert is_positive_semidefinite(-plant_params.Xee)
        Dx, Vx = np.linalg.eigh(-plant_params.Xee)
        self.LX = np.diag(np.sqrt(Dx)) @ Vx.T

        self._construct_projection_problem()

    def _construct_projection_problem(self):
        # Parameters: This is the thetahat to be projected into the stabilizing set.
        self.pThetahat = ControllerLTIThetahatParameters(
            S=cp.Parameter((self.state_size, self.state_size), PSD=True),
            R=cp.Parameter((self.state_size, self.state_size), PSD=True),
            NA11=cp.Parameter((self.state_size, self.state_size)),
            NA12=cp.Parameter((self.state_size, self.input_size)),
            NA21=cp.Parameter((self.output_size, self.state_size)),
            NA22=cp.Parameter((self.output_size, self.input_size)),
        )

        # Variables: This will be the solution of the projection.
        self.vThetahat = ControllerLTIThetahatParameters(
            S=cp.Variable((self.state_size, self.state_size), PSD=True),
            R=cp.Variable((self.state_size, self.state_size), PSD=True),
            NA11=cp.Variable((self.state_size, self.state_size)),
            NA12=cp.Variable((self.state_size, self.input_size)),
            NA21=cp.Variable((self.output_size, self.state_size)),
            NA22=cp.Variable((self.output_size, self.input_size)),
        )

        controller_params = ControllerThetahatParameters(
            S=self.vThetahat.S,
            R=self.vThetahat.R,
            NA11=self.vThetahat.NA11,
            NA12=self.vThetahat.NA12,
            NA21=self.vThetahat.NA21,
            NA22=self.vThetahat.NA22,
            NB=np.zeros((self.state_size, self.nonlin_size)),
            NC=np.zeros((self.nonlin_size, self.state_size)),
            Dkuw=np.zeros((self.output_size, self.nonlin_size)),
            Dkvyhat=np.zeros((self.nonlin_size, self.input_size)),
            Dkvwhat=np.zeros((self.nonlin_size, self.nonlin_size)),
            Lambda=np.zeros((self.nonlin_size, self.nonlin_size)),
        )
        mat = construct_dissipativity_matrix(
            plant_params=self.plant_params,
            LDeltap=self.LDeltap,
            LX=self.LX,
            controller_params=controller_params,
            stacker="cvxpy",
        )

        # Used for conditioning I - RS
        if self.trs_mode == "variable":
            self.vtrs = cp.Variable(nonneg=True)
            cost_ill_conditioning = -self.vtrs
        elif self.trs_mode == "fixed":
            self.vtrs = self.min_trs
            cost_ill_conditioning = 0
        else:
            raise ValueError(f"Unexpected trs_mode value of {self.trs_mode}.")

        # fmt: off
        constraints = [
            self.vtrs >= self.min_trs,
            self.vThetahat.S >> self.eps * np.eye(self.vThetahat.S.shape[0]),
            self.vThetahat.R >> self.eps * np.eye(self.vThetahat.R.shape[0]),
            cp.bmat([
                [self.vThetahat.R, self.vtrs * np.eye(self.vThetahat.R.shape[0])],
                [self.vtrs * np.eye(self.vThetahat.S.shape[0]), self.vThetahat.S],
            ]) >> self.eps * np.eye(self.vThetahat.R.shape[0] + self.vThetahat.S.shape[0]),
            mat << 0,
        ]

        cost_projection_error = sum([
            cp.sum_squares(self.pThetahat.S - self.vThetahat.S),
            cp.sum_squares(self.pThetahat.R - self.vThetahat.R),
            cp.sum_squares(self.pThetahat.NA11 - self.vThetahat.NA11),
            cp.sum_squares(self.pThetahat.NA12 - self.vThetahat.NA12),
            cp.sum_squares(self.pThetahat.NA21 - self.vThetahat.NA21),
            cp.sum_squares(self.pThetahat.NA22 - self.vThetahat.NA22),
        ])
        cost_size = sum([
            cp.sum_squares(self.vThetahat.S),
            cp.sum_squares(self.vThetahat.R),
            cp.sum_squares(self.vThetahat.NA11),
            cp.sum_squares(self.vThetahat.NA12),
            cp.sum_squares(self.vThetahat.NA21),
            cp.sum_squares(self.vThetahat.NA22),
        ])
        # fmt: on
        objective = cost_projection_error + cost_ill_conditioning # + cost_size

        self.problem = cp.Problem(cp.Minimize(objective), constraints)

    def project(
        self, controller_params: ControllerLTIThetahatParameters, solver=cp.MOSEK, **kwargs
    ):
        """Projects input variables to set corresponding to dissipative controllers."""
        K = controller_params
        self.pThetahat.S.value = K.S
        self.pThetahat.R.value = K.R
        self.pThetahat.NA11.value = K.NA11
        self.pThetahat.NA12.value = K.NA12
        self.pThetahat.NA21.value = K.NA21
        self.pThetahat.NA22.value = K.NA22

        try:
            # t0 = time.perf_counter()
            self.problem.solve(enforce_dpp=True, solver=solver, **kwargs)
            # t1 = time.perf_counter()
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
        if self.problem.status not in feas_stats:
            print(f"Failed to solve with status {self.problem.status}")
            raise Exception()
        print(f"Projection objective: {self.problem.value}")

        new_controller_params = ControllerLTIThetahatParameters(
            S=self.vThetahat.S.value,
            R=self.vThetahat.R.value,
            NA11=self.vThetahat.NA11.value,
            NA12=self.vThetahat.NA12.value,
            NA21=self.vThetahat.NA21.value,
            NA22=self.vThetahat.NA22.value,
        )
        return new_controller_params

    def is_dissipative(self, controller_params: ControllerLTIThetahatParameters):
        """Check whether given variables already satisfy dissipativity condition."""
        # All inputs must be numpy 2d arrays.

        # Check S, R, and Lambda are positive definite
        if not is_positive_definite(controller_params.S):
            print("S is not PD.")
            return False
        if not is_positive_definite(controller_params.R):
            print("R is not PD.")
            return False

        # Check [R, I; I, S] is positive definite.
        # fmt: off
        mat = np.asarray(np.bmat([
            [controller_params.R, np.eye(controller_params.R.shape[0])],
            [np.eye(controller_params.R.shape[0]), controller_params.S]
        ]))
        # fmt: on
        if not is_positive_definite(mat):
            print("[R, I; I, S] is not PD.")
            return False

        # Check main dissipativity condition.
        K = controller_params
        controller_params = ControllerThetahatParameters(
            S=K.S,
            R=K.R,
            NA11=K.NA11,
            NA12=K.NA12,
            NA21=K.NA21,
            NA22=K.NA22,
            NB=np.zeros((self.state_size, self.nonlin_size)),
            NC=np.zeros((self.nonlin_size, self.state_size)),
            Dkuw=np.zeros((self.output_size, self.nonlin_size)),
            Dkvyhat=np.zeros((self.nonlin_size, self.input_size)),
            Dkvwhat=np.zeros((self.nonlin_size, self.nonlin_size)),
            Lambda=np.zeros((self.nonlin_size, self.nonlin_size)),
        )
        mat = construct_dissipativity_matrix(
            plant_params=self.plant_params,
            LDeltap=self.LDeltap,
            LX=self.LX,
            controller_params=controller_params,
            stacker="numpy",
        )
        # Check condition mat <= 0
        return is_positive_semidefinite(-mat)
