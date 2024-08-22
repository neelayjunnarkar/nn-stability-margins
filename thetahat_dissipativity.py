import time
import copy
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

    Dew = stacker([[P.Dpew + P.Dpeu @ K.NA22 @ P.Dpyw, P.Dpeu @ K.Dkuw]])

    Ded = P.Dped + P.Dpeu @ K.NA22 @ P.Dpyd

    # fmt: off
    ldeltadvw1 = stacker([
        [LDeltap @ (P.Dpvw + P.Dpvu @ K.NA22 @ P.Dpyw), LDeltap @ P.Dpvu @ K.Dkuw]
    ])
    # fmt: on

    ldeltadvd1 = LDeltap @ (P.Dpvd + P.Dpvu @ K.NA22 @ P.Dpyd)

    # Define half the matrix and then add it to its transpose
    # Ensure Mww is symmetric. It needs to be for the method overall anyway.
    # Note it might be a cvxpy Parameter
    if isinstance(P.MDeltapww, np.ndarray):
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
        backoff_factor=1.1,  # Multiplier for bound on suboptimality
    ):
        self.plant_params = plant_params
        self.eps = eps
        self.nonlin_size = nonlin_size
        self.output_size = output_size
        self.state_size = state_size
        self.input_size = input_size
        self.trs_mode = trs_mode
        self.min_trs = min_trs
        assert self.trs_mode == "fixed", "trs_mode variable deprecated"
        self.backoff_factor = backoff_factor

        assert is_positive_semidefinite(plant_params.MDeltapvv)
        Dm, Vm = np.linalg.eigh(plant_params.MDeltapvv)
        self.LDeltap = np.diag(np.sqrt(Dm)) @ Vm.T

        assert is_positive_semidefinite(-plant_params.Xee)
        Dx, Vx = np.linalg.eigh(-plant_params.Xee)
        self.LX = np.diag(np.sqrt(Dx)) @ Vx.T

        self._construct_projection_problem()
        self._construct_backoff_problem()

    def _construct_projection_problem(self):
        # Parameters: This is the thetahat to be projected into the stabilizing set.
        self.proj_pThetahat = ControllerThetahatParameters(
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
        # Enable using the most up-to-date MDeltap during each projection
        # TODO: is the symmetric specification here a numerical problem?
        self.proj_pLDeltap = cp.Parameter((self.LDeltap.shape[0], self.LDeltap.shape[1]))
        self.proj_pMDeltapvv = cp.Parameter((self.plant_params.MDeltapvv.shape[0], self.plant_params.MDeltapvv.shape[1]), symmetric=True)
        self.proj_pMDeltapvw = cp.Parameter((self.plant_params.MDeltapvw.shape[0], self.plant_params.MDeltapvw.shape[1]))
        self.proj_pMDeltapww = cp.Parameter((self.plant_params.MDeltapww.shape[0], self.plant_params.MDeltapww.shape[1]), symmetric=True)
        plant_params = copy.copy(self.plant_params)
        plant_params.MDeltapvv = self.proj_pMDeltapvv
        plant_params.MDeltapvw = self.proj_pMDeltapvw
        plant_params.MDeltapww = self.proj_pMDeltapww

        # Variables: This will be the solution of the projection.
        self.proj_vThetahat = ControllerThetahatParameters(
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
            plant_params=plant_params, # Use the copy
            # LDeltap=self.LDeltap,
            LDeltap=self.proj_pLDeltap,
            LX=self.LX,
            controller_params=self.proj_vThetahat,
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
            self.proj_vThetahat.S >> self.eps * np.eye(self.proj_vThetahat.S.shape[0]),
            self.proj_vThetahat.R >> self.eps * np.eye(self.proj_vThetahat.R.shape[0]),
            self.proj_vThetahat.Lambda >> self.eps * np.eye(self.proj_vThetahat.Lambda.shape[0]),
            cp.bmat([
                [self.proj_vThetahat.R, self.vtrs * np.eye(self.proj_vThetahat.R.shape[0])],
                [self.vtrs * np.eye(self.proj_vThetahat.S.shape[0]), self.proj_vThetahat.S],
            ]) >> self.eps * np.eye(self.proj_vThetahat.R.shape[0] + self.proj_vThetahat.S.shape[0]),
            # Well-posedness condition Lambda Dkvw + Dkvw^T Lambda - 2 Lambda < 0
            self.proj_vThetahat.Dkvwhat + self.proj_vThetahat.Dkvwhat.T - 2*self.proj_vThetahat.Lambda << -self.eps * np.eye(self.proj_vThetahat.Lambda.shape[0]),
            # Dissipativity condition
            mat << 0,
        ]
        if self.trs_mode == "variable":
            constraints.append(self.vtrs >= self.min_trs)

        cost_projection_error = sum([
            cp.sum_squares(self.proj_pThetahat.Dkuw - self.proj_vThetahat.Dkuw),
            cp.sum_squares(self.proj_pThetahat.S - self.proj_vThetahat.S),
            cp.sum_squares(self.proj_pThetahat.R - self.proj_vThetahat.R),
            cp.sum_squares(self.proj_pThetahat.Lambda - self.proj_vThetahat.Lambda),
            cp.sum_squares(self.proj_pThetahat.NA11 - self.proj_vThetahat.NA11),
            cp.sum_squares(self.proj_pThetahat.NA12 - self.proj_vThetahat.NA12),
            cp.sum_squares(self.proj_pThetahat.NA21 - self.proj_vThetahat.NA21),
            cp.sum_squares(self.proj_pThetahat.NA22 - self.proj_vThetahat.NA22),
            cp.sum_squares(self.proj_pThetahat.NB - self.proj_vThetahat.NB),
            cp.sum_squares(self.proj_pThetahat.NC - self.proj_vThetahat.NC),
            cp.sum_squares(self.proj_pThetahat.Dkvyhat - self.proj_vThetahat.Dkvyhat),
            cp.sum_squares(self.proj_pThetahat.Dkvwhat - self.proj_vThetahat.Dkvwhat),
        ])
        # cost_size = sum([
        #     cp.sum_squares(self.proj_vThetahat.Dkuw),
        #     cp.sum_squares(self.proj_vThetahat.S),
        #     cp.sum_squares(self.proj_vThetahat.R),
        #     cp.sum_squares(self.proj_vThetahat.Lambda),
        #     cp.sum_squares(self.proj_vThetahat.NA11),
        #     cp.sum_squares(self.proj_vThetahat.NA12),
        #     cp.sum_squares(self.proj_vThetahat.NA21),
        #     cp.sum_squares(self.proj_vThetahat.NA22),
        #     cp.sum_squares(self.proj_vThetahat.NB),
        #     cp.sum_squares(self.proj_vThetahat.NC),
        #     cp.sum_squares(self.proj_vThetahat.Dkvyhat),
        #     cp.sum_squares(self.proj_vThetahat.Dkvwhat),
        # ])
        # fmt: on
        # Must be only projection error for the backoff step
        # + cost_ill_conditioning  # + cost_size
        objective = cost_projection_error

        self.proj_problem = cp.Problem(cp.Minimize(objective), constraints)

    def _construct_backoff_problem(self):
        # Parameters: This is the thetahat to be projected into the stabilizing set.
        self.backoff_pThetahat = ControllerThetahatParameters(
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
        # Enable using the most up-to-date MDeltap during each projection
        self.backoff_pLDeltap = cp.Parameter((self.LDeltap.shape[0], self.LDeltap.shape[1]))
        # TODO: is the symmetric specification here creating a numerical problem?
        self.backoff_pMDeltapvv = cp.Parameter((self.plant_params.MDeltapvv.shape[0], self.plant_params.MDeltapvv.shape[1]), symmetric=True)
        self.backoff_pMDeltapvw = cp.Parameter((self.plant_params.MDeltapvw.shape[0], self.plant_params.MDeltapvw.shape[1]))
        self.backoff_pMDeltapww = cp.Parameter((self.plant_params.MDeltapww.shape[0], self.plant_params.MDeltapww.shape[1]), symmetric=True)
        plant_params = copy.copy(self.plant_params)
        plant_params.MDeltapvv = self.backoff_pMDeltapvv
        plant_params.MDeltapvw = self.backoff_pMDeltapvw
        plant_params.MDeltapww = self.backoff_pMDeltapww
        # Squared projection error
        self.backoff_optimal_projection_error = cp.Parameter(nonneg=True)

        # Variables: This will be the solution of the projection.
        self.backoff_vThetahat = ControllerThetahatParameters(
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
        self.backoff_veps = cp.Variable(pos=True)

        mat = construct_dissipativity_matrix(
            plant_params=plant_params, # Use copy
            # LDeltap=self.LDeltap,
            LDeltap=self.backoff_pLDeltap,
            LX=self.LX,
            controller_params=self.backoff_vThetahat,
            stacker="cvxpy",
        )

        # fmt: off
        cost_projection_error = sum([
            cp.sum_squares(self.backoff_pThetahat.Dkuw    - self.backoff_vThetahat.Dkuw),
            cp.sum_squares(self.backoff_pThetahat.S       - self.backoff_vThetahat.S),
            cp.sum_squares(self.backoff_pThetahat.R       - self.backoff_vThetahat.R),
            cp.sum_squares(self.backoff_pThetahat.Lambda  - self.backoff_vThetahat.Lambda),
            cp.sum_squares(self.backoff_pThetahat.NA11    - self.backoff_vThetahat.NA11),
            cp.sum_squares(self.backoff_pThetahat.NA12    - self.backoff_vThetahat.NA12),
            cp.sum_squares(self.backoff_pThetahat.NA21    - self.backoff_vThetahat.NA21),
            cp.sum_squares(self.backoff_pThetahat.NA22    - self.backoff_vThetahat.NA22),
            cp.sum_squares(self.backoff_pThetahat.NB      - self.backoff_vThetahat.NB),
            cp.sum_squares(self.backoff_pThetahat.NC      - self.backoff_vThetahat.NC),
            cp.sum_squares(self.backoff_pThetahat.Dkvyhat - self.backoff_vThetahat.Dkvyhat),
            cp.sum_squares(self.backoff_pThetahat.Dkvwhat - self.backoff_vThetahat.Dkvwhat),
        ])
        # fmt: on

        # fmt: off
        constraints = [
            self.backoff_vThetahat.S >> self.eps * np.eye(self.backoff_vThetahat.S.shape[0]),
            self.backoff_vThetahat.R >> self.eps * np.eye(self.backoff_vThetahat.R.shape[0]),
            self.backoff_vThetahat.Lambda >> self.eps * np.eye(self.backoff_vThetahat.Lambda.shape[0]),
            cp.bmat([
                [self.backoff_vThetahat.R, np.eye(self.backoff_vThetahat.R.shape[0])],
                [np.eye(self.backoff_vThetahat.S.shape[0]), self.backoff_vThetahat.S],
            ]) >> self.backoff_veps * np.eye(self.backoff_vThetahat.R.shape[0] + self.backoff_vThetahat.S.shape[0]),
            # Well-posedness condition Lambda Dkvw + Dkvw^T Lambda - 2 Lambda < 0
            self.backoff_vThetahat.Dkvwhat + self.backoff_vThetahat.Dkvwhat.T - 2*self.backoff_vThetahat.Lambda << -self.eps * np.eye(self.backoff_vThetahat.Lambda.shape[0]),
            # Dissipativity condition
            mat << 0,
            # Backoff
            cost_projection_error <= self.backoff_factor**2 * self.backoff_optimal_projection_error
        ]
        # fmt: on

        objective = self.backoff_veps
        self.backoff_problem = cp.Problem(cp.Maximize(objective), constraints)

    def base_project(
        self, controller_params: ControllerThetahatParameters, LDeltap, MDeltapvv, MDeltapvw, MDeltapww, solver=cp.MOSEK, **kwargs
    ):
        """Projects input variables to set corresponding to dissipative controllers."""
        K = controller_params
        self.proj_pThetahat.Dkuw.value = K.Dkuw
        self.proj_pThetahat.S.value = K.S
        self.proj_pThetahat.R.value = K.R
        self.proj_pThetahat.Lambda.value = K.Lambda
        self.proj_pThetahat.NA11.value = K.NA11
        self.proj_pThetahat.NA12.value = K.NA12
        self.proj_pThetahat.NA21.value = K.NA21
        self.proj_pThetahat.NA22.value = K.NA22
        self.proj_pThetahat.NB.value = K.NB
        self.proj_pThetahat.NC.value = K.NC
        self.proj_pThetahat.Dkvyhat.value = K.Dkvyhat
        self.proj_pThetahat.Dkvwhat.value = K.Dkvwhat
        self.proj_pLDeltap.value = LDeltap
        self.proj_pMDeltapvv.value = MDeltapvv
        self.proj_pMDeltapvw.value = MDeltapvw
        self.proj_pMDeltapww.value = MDeltapww

        print("\n\n")
        print(f"Thetahat project MDeltapvv: {MDeltapvv}")
        print("\n\n")

        try:
            # t0 = time.perf_counter()
            self.proj_problem.solve(enforce_dpp=True, solver=solver, **kwargs)
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
        if self.proj_problem.status not in feas_stats:
            print(f"Failed to solve with status {self.proj_problem.status}")
            raise Exception()
        # print(f"Projection objective: {self.proj_problem.value}")

        # fmt: off
        new_controller_params = ControllerThetahatParameters(
            self.proj_vThetahat.S.value, self.proj_vThetahat.R.value, self.proj_vThetahat.NA11.value,
            self.proj_vThetahat.NA12.value, self.proj_vThetahat.NA21.value, self.proj_vThetahat.NA22.value,
            self.proj_vThetahat.NB.value, self.proj_vThetahat.NC.value, self.proj_vThetahat.Dkuw.value,
            self.proj_vThetahat.Dkvyhat.value, self.proj_vThetahat.Dkvwhat.value, self.proj_vThetahat.Lambda.value.toarray()
        )
        # fmt: on
        return new_controller_params, {"value": self.proj_problem.value}

    def project(self, controller_params: ControllerThetahatParameters, LDeltap, MDeltapvv, MDeltapvw, MDeltapww, solver=cp.MOSEK, **kwargs):
        """Projects input variables to set corresponding to dissipative controllers, allowing some suboptimality to improve conditioning."""
        # First solve projection to get optimal projection error
        _, info = self.base_project(controller_params, LDeltap, MDeltapvv, MDeltapvw, MDeltapww, solver=solver, **kwargs)
        self.backoff_optimal_projection_error.value = info["value"]

        # Then solve backoff problem which allows some suboptimality in projection,
        # but should improve conditioning of thetahat->theta reconstruction.
        K = controller_params
        self.backoff_pThetahat.Dkuw.value = K.Dkuw
        self.backoff_pThetahat.S.value = K.S
        self.backoff_pThetahat.R.value = K.R
        self.backoff_pThetahat.Lambda.value = K.Lambda
        self.backoff_pThetahat.NA11.value = K.NA11
        self.backoff_pThetahat.NA12.value = K.NA12
        self.backoff_pThetahat.NA21.value = K.NA21
        self.backoff_pThetahat.NA22.value = K.NA22
        self.backoff_pThetahat.NB.value = K.NB
        self.backoff_pThetahat.NC.value = K.NC
        self.backoff_pThetahat.Dkvyhat.value = K.Dkvyhat
        self.backoff_pThetahat.Dkvwhat.value = K.Dkvwhat
        self.backoff_pLDeltap.value = LDeltap
        self.backoff_pMDeltapvv.value = MDeltapvv
        self.backoff_pMDeltapvw.value = MDeltapvw
        self.backoff_pMDeltapww.value = MDeltapww

        try:
            # t0 = time.perf_counter()
            self.backoff_problem.solve(enforce_dpp=True, solver=solver, **kwargs)
            # t1 = time.perf_counter()
            # print(f"Backoff solving took {t1-t0} seconds.")
        except Exception as e:
            print(f"Failed to solve: {e}")
            raise e

        feas_stats = [
            cp.OPTIMAL,
            cp.UNBOUNDED,
            cp.OPTIMAL_INACCURATE,
            cp.UNBOUNDED_INACCURATE,
        ]
        if self.backoff_problem.status not in feas_stats:
            print(f"Failed to solve with status {self.backoff_problem.status}")
            raise Exception()

        # fmt: off
        new_controller_params = ControllerThetahatParameters(
            self.backoff_vThetahat.S.value, self.backoff_vThetahat.R.value, self.backoff_vThetahat.NA11.value,
            self.backoff_vThetahat.NA12.value, self.backoff_vThetahat.NA21.value, self.backoff_vThetahat.NA22.value,
            self.backoff_vThetahat.NB.value, self.backoff_vThetahat.NC.value, self.backoff_vThetahat.Dkuw.value,
            self.backoff_vThetahat.Dkvyhat.value, self.backoff_vThetahat.Dkvwhat.value, self.backoff_vThetahat.Lambda.value.toarray()
        )
        # fmt: on

        # Testing
        print(f"Backoff eps value: {self.backoff_veps.value}")
        # fmt: off
        cost_projection_error = np.sqrt(np.sum([
            np.sum(np.square(controller_params.Dkuw    - new_controller_params.Dkuw)),
            np.sum(np.square(controller_params.S       - new_controller_params.S)),
            np.sum(np.square(controller_params.R       - new_controller_params.R)),
            np.sum(np.square(controller_params.Lambda  - new_controller_params.Lambda)),
            np.sum(np.square(controller_params.NA11    - new_controller_params.NA11)),
            np.sum(np.square(controller_params.NA12    - new_controller_params.NA12)),
            np.sum(np.square(controller_params.NA21    - new_controller_params.NA21)),
            np.sum(np.square(controller_params.NA22    - new_controller_params.NA22)),
            np.sum(np.square(controller_params.NB      - new_controller_params.NB)),
            np.sum(np.square(controller_params.NC      - new_controller_params.NC)),
            np.sum(np.square(controller_params.Dkvyhat - new_controller_params.Dkvyhat)),
            np.sum(np.square(controller_params.Dkvwhat - new_controller_params.Dkvwhat)),
        ]))
        print(f"Projection error before vs after backoff: {np.sqrt(info['value'])} -> {cost_projection_error}")
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
        riis = np.asarray(np.bmat([
            [controller_params.R, np.eye(controller_params.R.shape[0])],
            [np.eye(controller_params.R.shape[0]), controller_params.S]
        ]))
        # fmt: on
        if not is_positive_definite(riis):
            print("[R, I; I, S] is not PD.")
            return False

        # Check well-posedness condition
        if not is_positive_definite(
            2 * controller_params.Lambda - controller_params.Dkvwhat - controller_params.Dkvwhat.T
        ):
            print("Not well-posed.")
            return False

        # Check main dissipativity condition.
        mat = construct_dissipativity_matrix(
            plant_params=self.plant_params,
            LDeltap=self.LDeltap,
            LX=self.LX,
            controller_params=controller_params,
            stacker="numpy",
        )
        # Check dissipativity condition mat <= 0
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
        backoff_factor=1.1,  # Multiplier for bound on suboptimality
    ):
        self.plant_params = plant_params
        self.eps = eps
        self.output_size = output_size
        self.state_size = state_size
        self.input_size = input_size
        self.trs_mode = trs_mode
        self.min_trs = min_trs
        assert self.trs_mode == "fixed", "trs_mode variable deprecated"
        self.backoff_factor = backoff_factor

        self.nonlin_size = 1  # placeholder nonlin size used for creating zeros

        assert is_positive_semidefinite(plant_params.MDeltapvv)
        Dm, Vm = np.linalg.eigh(plant_params.MDeltapvv)
        self.LDeltap = np.diag(np.sqrt(Dm)) @ Vm.T

        assert is_positive_semidefinite(-plant_params.Xee)
        Dx, Vx = np.linalg.eigh(-plant_params.Xee)
        self.LX = np.diag(np.sqrt(Dx)) @ Vx.T

        self._construct_projection_problem()
        self._construct_backoff_problem()

    def _construct_projection_problem(self):
        # Parameters: This is the thetahat to be projected into the stabilizing set.
        self.proj_pThetahat = ControllerLTIThetahatParameters(
            S=cp.Parameter((self.state_size, self.state_size), PSD=True),
            R=cp.Parameter((self.state_size, self.state_size), PSD=True),
            NA11=cp.Parameter((self.state_size, self.state_size)),
            NA12=cp.Parameter((self.state_size, self.input_size)),
            NA21=cp.Parameter((self.output_size, self.state_size)),
            NA22=cp.Parameter((self.output_size, self.input_size)),
        )
        # Enable using the most up-to-date MDeltap during each projection
        # TODO: is the symmetric specification here a numerical problem?
        self.proj_pLDeltap = cp.Parameter((self.LDeltap.shape[0], self.LDeltap.shape[1]))
        self.proj_pMDeltapvv = cp.Parameter((self.plant_params.MDeltapvv.shape[0], self.plant_params.MDeltapvv.shape[1]), symmetric=True)
        self.proj_pMDeltapvw = cp.Parameter((self.plant_params.MDeltapvw.shape[0], self.plant_params.MDeltapvw.shape[1]))
        self.proj_pMDeltapww = cp.Parameter((self.plant_params.MDeltapww.shape[0], self.plant_params.MDeltapww.shape[1]), symmetric=True)
        plant_params = copy.copy(self.plant_params)
        plant_params.MDeltapvv = self.proj_pMDeltapvv
        plant_params.MDeltapvw = self.proj_pMDeltapvw
        plant_params.MDeltapww = self.proj_pMDeltapww

        # Variables: This will be the solution of the projection.
        self.proj_vThetahat = ControllerLTIThetahatParameters(
            S=cp.Variable((self.state_size, self.state_size), PSD=True),
            R=cp.Variable((self.state_size, self.state_size), PSD=True),
            NA11=cp.Variable((self.state_size, self.state_size)),
            NA12=cp.Variable((self.state_size, self.input_size)),
            NA21=cp.Variable((self.output_size, self.state_size)),
            NA22=cp.Variable((self.output_size, self.input_size)),
        )

        controller_params = ControllerThetahatParameters(
            S=self.proj_vThetahat.S,
            R=self.proj_vThetahat.R,
            NA11=self.proj_vThetahat.NA11,
            NA12=self.proj_vThetahat.NA12,
            NA21=self.proj_vThetahat.NA21,
            NA22=self.proj_vThetahat.NA22,
            NB=np.zeros((self.state_size, self.nonlin_size)),
            NC=np.zeros((self.nonlin_size, self.state_size)),
            Dkuw=np.zeros((self.output_size, self.nonlin_size)),
            Dkvyhat=np.zeros((self.nonlin_size, self.input_size)),
            Dkvwhat=np.zeros((self.nonlin_size, self.nonlin_size)),
            Lambda=np.zeros((self.nonlin_size, self.nonlin_size)),
        )
        mat = construct_dissipativity_matrix(
            plant_params=plant_params, # Use the copy
            # LDeltap=self.LDeltap,
            LDeltap=self.proj_pLDeltap,
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
            # self.vtrs >= self.min_trs,
            self.proj_vThetahat.S >> self.eps * np.eye(self.proj_vThetahat.S.shape[0]),
            self.proj_vThetahat.R >> self.eps * np.eye(self.proj_vThetahat.R.shape[0]),
            cp.bmat([
                [self.proj_vThetahat.R, self.vtrs * np.eye(self.proj_vThetahat.R.shape[0])],
                [self.vtrs * np.eye(self.proj_vThetahat.S.shape[0]), self.proj_vThetahat.S],
            ]) >> self.eps * np.eye(self.proj_vThetahat.R.shape[0] + self.proj_vThetahat.S.shape[0]),
            mat << 0,
        ]

        cost_projection_error = sum([
            cp.sum_squares(self.proj_pThetahat.S - self.proj_vThetahat.S),
            cp.sum_squares(self.proj_pThetahat.R - self.proj_vThetahat.R),
            cp.sum_squares(self.proj_pThetahat.NA11 - self.proj_vThetahat.NA11),
            cp.sum_squares(self.proj_pThetahat.NA12 - self.proj_vThetahat.NA12),
            cp.sum_squares(self.proj_pThetahat.NA21 - self.proj_vThetahat.NA21),
            cp.sum_squares(self.proj_pThetahat.NA22 - self.proj_vThetahat.NA22),
        ])
        # cost_size = sum([
        #     cp.sum_squares(self.proj_vThetahat.S),
        #     cp.sum_squares(self.proj_vThetahat.R),
        #     cp.sum_squares(self.proj_vThetahat.NA11),
        #     cp.sum_squares(self.proj_vThetahat.NA12),
        #     cp.sum_squares(self.proj_vThetahat.NA21),
        #     cp.sum_squares(self.proj_vThetahat.NA22),
        # ])
        # fmt: on
        objective = cost_projection_error  # + cost_ill_conditioning  # + cost_size

        self.proj_problem = cp.Problem(cp.Minimize(objective), constraints)

    def _construct_backoff_problem(self):
        # Parameters: This is the thetahat to be projected into the stabilizing set.
        self.backoff_pThetahat = ControllerLTIThetahatParameters(
            S=cp.Parameter((self.state_size, self.state_size), PSD=True),
            R=cp.Parameter((self.state_size, self.state_size), PSD=True),
            NA11=cp.Parameter((self.state_size, self.state_size)),
            NA12=cp.Parameter((self.state_size, self.input_size)),
            NA21=cp.Parameter((self.output_size, self.state_size)),
            NA22=cp.Parameter((self.output_size, self.input_size)),
        )
        # Enable using the most up-to-date MDeltap during each projection
        self.backoff_pLDeltap = cp.Parameter((self.LDeltap.shape[0], self.LDeltap.shape[1]))
        # TODO: is the symmetric specification here creating a numerical problem?
        self.backoff_pMDeltapvv = cp.Parameter((self.plant_params.MDeltapvv.shape[0], self.plant_params.MDeltapvv.shape[1]), symmetric=True)
        self.backoff_pMDeltapvw = cp.Parameter((self.plant_params.MDeltapvw.shape[0], self.plant_params.MDeltapvw.shape[1]))
        self.backoff_pMDeltapww = cp.Parameter((self.plant_params.MDeltapww.shape[0], self.plant_params.MDeltapww.shape[1]), symmetric=True)
        plant_params = copy.copy(self.plant_params)
        plant_params.MDeltapvv = self.backoff_pMDeltapvv
        plant_params.MDeltapvw = self.backoff_pMDeltapvw
        plant_params.MDeltapww = self.backoff_pMDeltapww
        # Squared projection error
        self.backoff_optimal_projection_error = cp.Parameter(nonneg=True)

        # Variables: This will be the solution of the projection.
        self.backoff_vThetahat = ControllerLTIThetahatParameters(
            S=cp.Variable((self.state_size, self.state_size), PSD=True),
            R=cp.Variable((self.state_size, self.state_size), PSD=True),
            NA11=cp.Variable((self.state_size, self.state_size)),
            NA12=cp.Variable((self.state_size, self.input_size)),
            NA21=cp.Variable((self.output_size, self.state_size)),
            NA22=cp.Variable((self.output_size, self.input_size)),
        )
        self.backoff_veps = cp.Variable(pos=True)

        controller_params = ControllerThetahatParameters(
            S=self.backoff_vThetahat.S,
            R=self.backoff_vThetahat.R,
            NA11=self.backoff_vThetahat.NA11,
            NA12=self.backoff_vThetahat.NA12,
            NA21=self.backoff_vThetahat.NA21,
            NA22=self.backoff_vThetahat.NA22,
            NB=np.zeros((self.state_size, self.nonlin_size)),
            NC=np.zeros((self.nonlin_size, self.state_size)),
            Dkuw=np.zeros((self.output_size, self.nonlin_size)),
            Dkvyhat=np.zeros((self.nonlin_size, self.input_size)),
            Dkvwhat=np.zeros((self.nonlin_size, self.nonlin_size)),
            Lambda=np.zeros((self.nonlin_size, self.nonlin_size)),
        )
        mat = construct_dissipativity_matrix(
            plant_params=plant_params, # Use copy
            # LDeltap=self.LDeltap,
            LDeltap=self.backoff_pLDeltap,
            LX=self.LX,
            controller_params=controller_params,
            stacker="cvxpy",
        )

        # fmt: off
        cost_projection_error = sum([
            cp.sum_squares(self.backoff_pThetahat.S - self.backoff_vThetahat.S),
            cp.sum_squares(self.backoff_pThetahat.R - self.backoff_vThetahat.R),
            cp.sum_squares(self.backoff_pThetahat.NA11 - self.backoff_vThetahat.NA11),
            cp.sum_squares(self.backoff_pThetahat.NA12 - self.backoff_vThetahat.NA12),
            cp.sum_squares(self.backoff_pThetahat.NA21 - self.backoff_vThetahat.NA21),
            cp.sum_squares(self.backoff_pThetahat.NA22 - self.backoff_vThetahat.NA22),
        ])
        constraints = [
            self.backoff_vThetahat.S >> self.eps * np.eye(self.backoff_vThetahat.S.shape[0]),
            self.backoff_vThetahat.R >> self.eps * np.eye(self.backoff_vThetahat.R.shape[0]),
            cp.bmat([
                [self.backoff_vThetahat.R, np.eye(self.backoff_vThetahat.R.shape[0])],
                [np.eye(self.backoff_vThetahat.S.shape[0]), self.backoff_vThetahat.S],
            ]) >> self.backoff_veps * np.eye(self.backoff_vThetahat.R.shape[0] + self.backoff_vThetahat.S.shape[0]),
            # Dissipativity condition
            mat << 0,
            # Backoff
            cost_projection_error <= self.backoff_factor**2 * self.backoff_optimal_projection_error
        ]
        # fmt: on

        objective = self.backoff_veps
        self.backoff_problem = cp.Problem(cp.Maximize(objective), constraints)

    def base_project(
        self, controller_params: ControllerLTIThetahatParameters, LDeltap, MDeltapvv, MDeltapvw, MDeltapww, solver=cp.MOSEK, **kwargs
    ):
        """Projects input variables to set corresponding to dissipative controllers."""
        K = controller_params
        self.proj_pThetahat.S.value = K.S
        self.proj_pThetahat.R.value = K.R
        self.proj_pThetahat.NA11.value = K.NA11
        self.proj_pThetahat.NA12.value = K.NA12
        self.proj_pThetahat.NA21.value = K.NA21
        self.proj_pThetahat.NA22.value = K.NA22
        self.proj_pLDeltap.value = LDeltap
        self.proj_pMDeltapvv.value = MDeltapvv
        self.proj_pMDeltapvw.value = MDeltapvw
        self.proj_pMDeltapww.value = MDeltapww

        try:
            # t0 = time.perf_counter()
            self.proj_problem.solve(enforce_dpp=True, solver=solver, **kwargs)
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
        if self.proj_problem.status not in feas_stats:
            print(f"Failed to solve with status {self.proj_problem.status}")
            raise Exception()
        # print(f"Projection objective: {self.proj_problem.value}")

        new_controller_params = ControllerLTIThetahatParameters(
            S=self.proj_vThetahat.S.value,
            R=self.proj_vThetahat.R.value,
            NA11=self.proj_vThetahat.NA11.value,
            NA12=self.proj_vThetahat.NA12.value,
            NA21=self.proj_vThetahat.NA21.value,
            NA22=self.proj_vThetahat.NA22.value,
        )
        return new_controller_params, {"value": self.proj_problem.value}

    def project(
        self, controller_params: ControllerLTIThetahatParameters, LDeltap, MDeltapvv, MDeltapvw, MDeltapww, solver=cp.MOSEK, **kwargs
    ):
        """Projects input variables to set corresponding to dissipative controllers, allowing some suboptimality to improve conditioning."""
        # First solve projection to get optimal projection error
        _, info = self.base_project(controller_params, LDeltap, MDeltapvv, MDeltapvw, MDeltapww, solver=solver, **kwargs)
        self.backoff_optimal_projection_error.value = info["value"]

        # Then solve backoff problem which allows some suboptimality in projection,
        # but should improve conditioning of thetahat->theta reconstruction.
        K = controller_params
        self.backoff_pThetahat.S.value = K.S
        self.backoff_pThetahat.R.value = K.R
        self.backoff_pThetahat.NA11.value = K.NA11
        self.backoff_pThetahat.NA12.value = K.NA12
        self.backoff_pThetahat.NA21.value = K.NA21
        self.backoff_pThetahat.NA22.value = K.NA22
        self.backoff_pLDeltap.value = LDeltap
        self.backoff_pMDeltapvv.value = MDeltapvv
        self.backoff_pMDeltapvw.value = MDeltapvw
        self.backoff_pMDeltapww.value = MDeltapww

        try:
            # t0 = time.perf_counter()
            self.backoff_problem.solve(enforce_dpp=True, solver=solver, **kwargs)
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
        if self.backoff_problem.status not in feas_stats:
            print(f"Failed to solve with status {self.backoff_problem.status}")
            raise Exception()
        # print(f"Projection objective: {self.backoff_problem.value}")

        new_controller_params = ControllerLTIThetahatParameters(
            S=self.backoff_vThetahat.S.value,
            R=self.backoff_vThetahat.R.value,
            NA11=self.backoff_vThetahat.NA11.value,
            NA12=self.backoff_vThetahat.NA12.value,
            NA21=self.backoff_vThetahat.NA21.value,
            NA22=self.backoff_vThetahat.NA22.value,
        )

        # Testing
        print(f"Backoff eps value: {self.backoff_veps.value}")
        # fmt: off
        cost_projection_error = np.sqrt(np.sum([
            np.sum(np.square(controller_params.S    - new_controller_params.S)),
            np.sum(np.square(controller_params.R    - new_controller_params.R)),
            np.sum(np.square(controller_params.NA11 - new_controller_params.NA11)),
            np.sum(np.square(controller_params.NA12 - new_controller_params.NA12)),
            np.sum(np.square(controller_params.NA21 - new_controller_params.NA21)),
            np.sum(np.square(controller_params.NA22 - new_controller_params.NA22)),
        ]))
        print(f"Projection error before vs after backoff: {np.sqrt(info['value'])} -> {cost_projection_error}")
        # fmt: on

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
