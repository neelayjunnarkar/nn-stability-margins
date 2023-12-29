import numpy as np
import cvxpy as cp
import time


def construct_dissipativity_matrix(
    Dkuw, S, R, Lambda, NA11, NA12, NA21, NA22, NB, NC, Dkvyhat, Dkvwhat,
    Ap, Bpw, Bpd, Bpu, Cpv, Dpvw, Dpvd, Dpvu, Cpe, Dpew, Dped, Dpeu, Cpy,
    Dpyw, Dpyd, LDeltap, MDeltapvw, MDeltapww, Xdd, Xde, LX, stacker,
):
    if stacker == "numpy":
        stacker = np.bmat
    elif stacker == "cvxpy":
        stacker = cp.bmat
    else:
        raise ValueError(f"Stacker {stacker} must be 'numpy' or 'cvxpy'.")

    ytpay11 = Ap @ R + Bpu @ NA21
    ytpay12 = Ap + Bpu @ NA22 @ Cpy
    ytpay21 = NA11
    ytpay22 = S @ Ap + NA12 @ Cpy
    ytpay = stacker([[ytpay11, ytpay12], [ytpay21, ytpay22]])

    ytpbw11 = Bpw + Bpu @ NA22 @ Dpyw
    ytpbw12 = Bpu @ Dkuw
    ytpbw21 = S @ Bpw + NA12 @ Dpyw
    ytpbw22 = NB
    ytpbw = stacker([[ytpbw11, ytpbw12], [ytpbw21, ytpbw22]])

    ytpbd1 = Bpd + Bpu @ NA22 @ Dpyd
    ytpbd2 = S @ Bpd + NA12 @ Dpyd
    ytpbd = stacker([[ytpbd1], [ytpbd2]])

    mvwtcvy11 = MDeltapvw.T @ Cpv @ R + MDeltapvw.T @ Dpvu @ NA21
    mvwtcvy12 = MDeltapvw.T @ Cpv + MDeltapvw.T @ Dpvu @ NA22 @ Cpy
    mvwtcvy21 = NC
    mvwtcvy22 = Dkvyhat @ Cpy
    mvwtcvy = stacker([[mvwtcvy11, mvwtcvy12], [mvwtcvy21, mvwtcvy22]])

    nxdecey1 = -Xde @ (Cpe @ R + Dpeu @ NA21)
    nxdecey2 = -Xde @ (Cpe + Dpeu @ NA22 @ Cpy)
    nxdecey = stacker([[nxdecey1, nxdecey2]])

    ldeltacvy11 = LDeltap @ (Cpv @ R + Dpvu @ NA21)
    ldeltacvy12 = LDeltap @ (Cpv + Dpvu @ NA22 @ Cpy)
    ldeltacvy1 = stacker([[ldeltacvy11, ldeltacvy12]])

    lxcey1 = LX @ (Cpe @ R + Dpeu @ NA21)
    lxcey2 = LX @ (Cpe + Dpeu @ NA22 @ Cpy)
    lxcey = stacker([[lxcey1, lxcey2]])

    mvwtdvw11 = MDeltapvw.T @ (Dpvw + Dpvu @ NA22 @ Dpyw)
    mvwtdvw12 = MDeltapvw.T @ Dpvu @ Dkuw
    mvwtdvw21 = Dkvyhat @ Dpyw
    mvwtdvw22 = Dkvwhat
    mvwtdvw = stacker([[mvwtdvw11, mvwtdvw12], [mvwtdvw21, mvwtdvw22]])

    mvwtdvd1 = MDeltapvw.T @ (Dpvd + Dpvu @ NA22 @ Dpyd)
    mvwtdvd2 = Dkvyhat @ Dpyd
    mvwtdvd = stacker([[mvwtdvd1], [mvwtdvd2]])

    Mww = stacker([
        [MDeltapww, np.zeros((MDeltapww.shape[0], Lambda.shape[1]))],
        [np.zeros((Lambda.shape[0], MDeltapww.shape[1])), -2 * Lambda],
    ])

    Dew = stacker([[Dpew + Dpeu @ NA22 @ Dpyw, Dpew @ Dkuw]])

    Ded = Dped + Dpeu @ NA22 @ Dpyd

    ldeltadvw1 = stacker(
        [[LDeltap @ (Dpvw + Dpvu @ NA22 @ Dpyw), LDeltap @ Dpvu @ Dkuw]]
    )

    ldeltadvd1 = LDeltap @ (Dpvd + Dpvu @ NA22 @ Dpyd)

    # Define half the matrix and then add it to its transpose
    # Ensure Mww is symmetric. It needs to be for the method overall anyway.
    assert np.allclose(MDeltapww, MDeltapww.T)
    # Ensure Xdd is symmetric. It needs to be for the method overall anyway.
    assert np.allclose(Xdd, Xdd.T)
    row1 = stacker([[
        ytpay.T,
        np.zeros((
            ytpay.T.shape[0],
            mvwtdvw.shape[1]
            + Xdd.shape[1]
            + LDeltap.shape[0]
            + LX.shape[0],
        )),
    ]])
    row2 = stacker([[
        ytpbw.T + mvwtcvy,
        mvwtdvw + 0.5 * Mww,
        np.zeros(
            (ytpbw.T.shape[0], Xdd.shape[1] + LDeltap.shape[0] + LX.shape[0])
        ),
    ]])
    row3 = stacker([[
        ytpbd.T + nxdecey,
        mvwtdvd.T - Xde @ Dew,
        -Xde @ Ded - 0.5 * Xdd,
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
    mat = mat + mat.T

    return mat

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


class Projector:
    # Uses the Disciplined Parameterized Programming feature of CVXPY to construct
    # an optimization problem once, and then resolve it, for (hopefull) a speed up.

    def __init__(
        self,
        # Plant parameters
        Ap, Bpw, Bpd, Bpu, Cpv, Dpvw, Dpvd, Dpvu,
        Cpe, Dpew, Dped, Dpeu, Cpy, Dpyw, Dpyd,
        # Plant uncertainty Delta static IQC multiplier
        MDeltapvv, MDeltapvw, MDeltapww,
        # Supply rate
        Xdd, Xde, Xee,
        # Epsilon to be used in enforcing definiteness of conditions
        eps,
        # Dimensions of variables for controller
        nonlin_size, output_size, state_size, input_size,
        # Parameters for tuning condition number of I - RS,
        trs, min_trs,
    ):
        self.Ap = Ap
        self.Bpw = Bpw
        self.Bpd = Bpd
        self.Bpu = Bpu
        self.Cpv = Cpv
        self.Dpvw = Dpvw
        self.Dpvd = Dpvd
        self.Dpvu = Dpvu
        self.Cpe = Cpe
        self.Dpew = Dpew
        self.Dped = Dped
        self.Dpeu = Dpeu
        self.Cpy = Cpy
        self.Dpyw = Dpyw
        self.Dpyd = Dpyd
        self.MDeltapvv = MDeltapvv
        self.MDeltapvw = MDeltapvw
        self.MDeltapww = MDeltapww
        self.Xdd = Xdd
        self.Xde = Xde
        self.Xee = Xee
        self.eps = eps
        self.nonlin_size = nonlin_size
        self.output_size = output_size
        self.state_size = state_size
        self.input_size = input_size

        assert np.allclose(self.MDeltapvv, self.MDeltapvv.T)
        # LDeltap is defined as transpose of Cholesky decomposition matrix of a PSD.
        Dm, Vm = np.linalg.eigh(self.MDeltapvv)
        assert np.min(Dm) >= 0 , f"min(Dm): {np.min(Dm)}" # - eps
        self.LDeltap = np.diag(np.sqrt(Dm)) @ Vm.T

        assert np.allclose(self.Xee, self.Xee.T)
        # LX is defined as the transpose of the Cholesky decomposition matrix of a NSD.
        Dx, Vx = np.linalg.eigh(-self.Xee)
        assert np.min(Dx) >= 0  # -eps
        self.LX = np.diag(np.sqrt(Dx)) @ Vx.T

        # Define parameters and variables of LMI
        # Those with _bar suffix indicate ones that will be used to construct the decision variables.
        # The _T suffix indicates transpose

        # Parameters: This is the thetahat to be projected into the stabilizing set.
        self.pDkuw = cp.Parameter((self.output_size, self.nonlin_size))
        self.pS = cp.Parameter((self.state_size, self.state_size), PSD=True)
        self.pR = cp.Parameter((self.state_size, self.state_size), PSD=True)
        self.pLambda = cp.Parameter((self.nonlin_size, self.nonlin_size), diag=True)
        self.pNA11 = cp.Parameter((self.state_size, self.state_size))
        self.pNA12 = cp.Parameter((self.state_size, self.input_size))
        self.pNA21 = cp.Parameter((self.output_size, self.state_size))
        self.pNA22 = cp.Parameter((self.output_size, self.input_size))
        self.pNB = cp.Parameter((self.state_size, self.nonlin_size))
        self.pNC = cp.Parameter((self.nonlin_size, self.state_size))
        self.pDkvyhat = cp.Parameter((self.nonlin_size, self.input_size))
        self.pDkvwhat = cp.Parameter((self.nonlin_size, self.nonlin_size))

        # Variables: This will be the solution of the projection.
        self.vDkuw = cp.Variable((self.output_size, self.nonlin_size))
        self.vS = cp.Variable((self.state_size, self.state_size), PSD=True)
        self.vR = cp.Variable((self.state_size, self.state_size), PSD=True)
        self.vLambda = cp.Variable((self.nonlin_size, self.nonlin_size), diag=True)
        self.vNA11 = cp.Variable((self.state_size, self.state_size))
        self.vNA12 = cp.Variable((self.state_size, self.input_size))
        self.vNA21 = cp.Variable((self.output_size, self.state_size))
        self.vNA22 = cp.Variable((self.output_size, self.input_size))
        self.vNB = cp.Variable((self.state_size, self.nonlin_size))
        self.vNC = cp.Variable((self.nonlin_size, self.state_size))
        self.vDkvyhat = cp.Variable((self.nonlin_size, self.input_size))
        self.vDkvwhat = cp.Variable((self.nonlin_size, self.nonlin_size))

        mat = construct_dissipativity_matrix(
            Dkuw=self.vDkuw,
            S=self.vS,
            R=self.vR,
            Lambda=self.vLambda,
            NA11=self.vNA11,
            NA12=self.vNA12,
            NA21=self.vNA21,
            NA22=self.vNA22,
            NB=self.vNB,
            NC=self.vNC,
            Dkvyhat=self.vDkvyhat,
            Dkvwhat=self.vDkvwhat,
            Ap=self.Ap,
            Bpw=self.Bpw,
            Bpd=self.Bpd,
            Bpu=self.Bpu,
            Cpv=self.Cpv,
            Dpvw=self.Dpvw,
            Dpvd=self.Dpvd,
            Dpvu=self.Dpvu,
            Cpe=self.Cpe,
            Dpew=self.Dpew,
            Dped=self.Dped,
            Dpeu=self.Dpeu,
            Cpy=self.Cpy,
            Dpyw=self.Dpyw,
            Dpyd=self.Dpyd,
            LDeltap=self.LDeltap,
            MDeltapvw=self.MDeltapvw,
            MDeltapww=self.MDeltapww,
            Xdd=self.Xdd,
            Xde=self.Xde,
            LX=self.LX,
            stacker="cvxpy",
        )

        # Used for conditioning I - RS
        self.min_tRS = min_trs
        if trs is None:
            self.tRS = cp.Variable(nonneg=True)
            cost_ill_conditioning = -self.tRS
        else:
            self.tRS = trs
            cost_ill_conditioning = 0
        
        # self.tR = cp.Variable(nonneg=True)
        # self.tS = cp.Variable(nonneg=True)

        constraints = [
            self.vS >> self.eps * np.eye(self.vS.shape[0]),
            self.vR >> self.eps * np.eye(self.vR.shape[0]),
            # (1-self.tS) * np.eye(self.vS.shape[0]) >> self.vS,
            # (1-self.tR) * np.eye(self.vR.shape[0]) >> self.vR,
            self.tRS >= self.min_tRS + self.eps,
            self.vLambda >> self.eps * np.eye(self.vLambda.shape[0]),
            cp.bmat(
                [
                    [self.vR, self.tRS * np.eye(self.vR.shape[0])],
                    [self.tRS * np.eye(self.vS.shape[0]), self.vS],
                ]
            )
            >> self.eps * np.eye(self.vR.shape[0] + self.vS.shape[0]),
            mat << 0, # -self.eps * np.eye(mat.shape[0]),
        ]

        cost_projection_error = sum([
                cp.sum_squares(self.pDkuw - self.vDkuw),
                cp.sum_squares(self.pS - self.vS),
                cp.sum_squares(self.pR - self.vR),
                cp.sum_squares(self.pLambda - self.vLambda),
                cp.sum_squares(self.pNA11 - self.vNA11),
                cp.sum_squares(self.pNA12 - self.vNA12),
                cp.sum_squares(self.pNA21 - self.vNA21),
                cp.sum_squares(self.pNA22 - self.vNA22),
                cp.sum_squares(self.pNB - self.vNB),
                cp.sum_squares(self.pNC - self.vNC),
                cp.sum_squares(self.pDkvyhat - self.vDkvyhat),
                cp.sum_squares(self.pDkvwhat - self.vDkvwhat),
        ])
        cost_size = sum([
                cp.sum_squares(self.vDkuw),
                cp.sum_squares(self.vS),
                cp.sum_squares(self.vR),
                cp.sum_squares(self.vLambda),
                cp.sum_squares(self.vNA11),
                cp.sum_squares(self.vNA12),
                cp.sum_squares(self.vNA21),
                cp.sum_squares(self.vNA22),
                cp.sum_squares(self.vNB),
                cp.sum_squares(self.vNC),
                cp.sum_squares(self.vDkvyhat),
                cp.sum_squares(self.vDkvwhat),
        ])
        
        # objective = cost_projection_error + cost_ill_conditioning + cost_size
        objective = cost_projection_error + cost_ill_conditioning + cost_size
        # objective = 0 # cost_ill_conditioning

        self.problem = cp.Problem(cp.Minimize(objective), constraints)

    def project(
        self, Dkuw, S, R, Lambda, NA11, NA12, NA21, NA22,
        NB, NC, Dkvyhat, Dkvwhat, solver=cp.MOSEK, **kwargs,
    ):
        """Projects input variables to set corresponding to dissipative controllers."""
        self.pDkuw.value = Dkuw
        self.pS.value = S
        self.pR.value = R
        self.pLambda.value = Lambda
        self.pNA11.value = NA11
        self.pNA12.value = NA12
        self.pNA21.value = NA21
        self.pNA22.value = NA22
        self.pNB.value = NB
        self.pNC.value = NC
        self.pDkvyhat.value = Dkvyhat
        self.pDkvwhat.value = Dkvwhat

        try:
            t0 = time.perf_counter()
            # TODO(Neelay) test various solvers
            # self.problem.solve(enforce_dpp=True, solver=solver, **kwargs)
            self.problem.solve(solver=solver, **kwargs)
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
        if self.problem.status not in feas_stats:
            print(f"Failed to solve with status {self.problem.status}")
            raise Exception()
        print(f"Projection objective: {self.problem.value}")

        oDkuw = self.vDkuw.value
        oS = self.vS.value
        oR = self.vR.value
        # Since diag=True makes it vLambda.value sparse
        oLambda = self.vLambda.value.toarray()
        oNA11 = self.vNA11.value
        oNA12 = self.vNA12.value
        oNA21 = self.vNA21.value
        oNA22 = self.vNA22.value
        oNB = self.vNB.value
        oNC = self.vNC.value
        oDkvyhat = self.vDkvyhat.value
        oDkvwhat = self.vDkvwhat.value

        # print(f"tRS: {self.tRS.value}")

        assert not np.any(np.isnan(oDkuw))
        assert not np.any(np.isnan(oS))
        assert not np.any(np.isnan(oR))
        assert not np.any(np.isnan(oLambda))
        assert not np.any(np.isnan(oNA11))
        assert not np.any(np.isnan(oNA12))
        assert not np.any(np.isnan(oNA21))
        assert not np.any(np.isnan(oNA22))
        assert not np.any(np.isnan(oNB))
        assert not np.any(np.isnan(oNC))
        assert not np.any(np.isnan(oDkvyhat))
        assert not np.any(np.isnan(oDkvwhat))

        return (
            oDkuw,
            oS,
            oR,
            oLambda,
            oNA11,
            oNA12,
            oNA21,
            oNA22,
            oNB,
            oNC,
            oDkvyhat,
            oDkvwhat,
        )

    def is_dissipative(
        self,
        Dkuw,
        S,
        R,
        Lambda,
        NA11,
        NA12,
        NA21,
        NA22,
        NB,
        NC,
        Dkvyhat,
        Dkvwhat,
    ):
        """Check whether given variables already satisfy dissipativity condition."""
        # All inputs must be numpy 2d arrays.

        # Check S, R, and Lambda are positive definite
        if not is_positive_definite(S):
            print("S is not PD.")
            return False
        if not is_positive_definite(R):
            print("R is not PD.")
            return False
        if not is_positive_definite(Lambda):
            print("Lambda is not PD.")
            return False

        # Check [R, I; I, S] is positive definite.
        mat = np.asarray(np.bmat([[R, np.eye(R.shape[0])], [np.eye(R.shape[0]), S]]))
        if not is_positive_definite(mat):
            print("[R, I; I, S] is not PD.")
            return False

        # Check main dissipativity condition.
        mat = construct_dissipativity_matrix(
            Dkuw=Dkuw,
            S=S,
            R=R,
            Lambda=Lambda,
            NA11=NA11,
            NA12=NA12,
            NA21=NA21,
            NA22=NA22,
            NB=NB,
            NC=NC,
            Dkvyhat=Dkvyhat,
            Dkvwhat=Dkvwhat,
            Ap=self.Ap,
            Bpw=self.Bpw,
            Bpd=self.Bpd,
            Bpu=self.Bpu,
            Cpv=self.Cpv,
            Dpvw=self.Dpvw,
            Dpvd=self.Dpvd,
            Dpvu=self.Dpvu,
            Cpe=self.Cpe,
            Dpew=self.Dpew,
            Dped=self.Dped,
            Dpeu=self.Dpeu,
            Cpy=self.Cpy,
            Dpyw=self.Dpyw,
            Dpyd=self.Dpyd,
            LDeltap=self.LDeltap,
            MDeltapvw=self.MDeltapvw,
            MDeltapww=self.MDeltapww,
            Xdd=self.Xdd,
            Xde=self.Xde,
            LX=self.LX,
            stacker="numpy",
        )
        # Check condition mat <= 0
        return is_positive_semidefinite(-mat)
    
    def verify_dissipativity(self, Ak, Bkw, Bky, Ckv, Dkvw, Dkvy, Cku, Dkuw, Dkuy):

        # Form closed-loop
        A = np.bmat([
            [self.Ap + self.Bpu @ Dkuy @ self.Cpy, self.Bpu @ Cku],
            [Bky @ self.Cpy, Ak]
        ])
        Bw = np.bmat([
            [self.Bpw + self.Bpu @ Dkuy @ self.Dpyw, self.Bpu @ Dkuw],
            [Bky @ self.Dpyw, Bkw]
        ])
        Bd = np.bmat([
            [self.Bpd + self.Bpu @ Dkuy @ self.Dpyd],
            [Bky @ self.Dpyd]
        ])
        Cv = np.bmat([
            [self.Cpv + self.Dpvu @ Dkuy @ self.Cpy, self.Dpvu @ Cku],
            [Dkvy @ self.Cpy, Ckv]
        ])
        Dvw = np.bmat([
            [self.Dpvw + self.Dpvu @ Dkuy @ self.Dpyw, self.Dpvu @ Dkuw],
            [Dkvy @ self.Dpyw, Dkvw]
        ])
        Dvd = np.bmat([
            [self.Dpvd + self.Dpvu @ Dkuy @ self.Dpyd],
            [Dkvy @ self.Dpyd]
        ])
        Ce = np.bmat([
            [self.Cpe + self.Dpeu @ Dkuy @ self.Cpy, self.Dpeu @ Cku]
        ])
        Dew = np.bmat([
            [self.Dpew + self.Dpeu @ Dkuy @ self.Dpyw, self.Dpeu @ Dkuw]
        ])
        Ded = np.bmat([
            [self.Dped + self.Dpeu @ Dkuy @ self.Dpyd]
        ])

        P = cp.Variable((2*self.state_size, 2*self.state_size), PSD=True)
        Lambda = cp.Variable((self.nonlin_size, self.nonlin_size), diag=True)

        F1 = cp.bmat([
            [A.T @ P + P @ A, P @ Bw, P @ Bd],
            [Bw.T @ P, np.zeros((Bw.shape[1], Bw.shape[1] + Bd.shape[1]))],
            [Bd.T @ P, np.zeros((Bd.shape[1], Bw.shape[1] + Bd.shape[1]))]
        ])
        
        Mright = np.bmat([
            [Cv, Dvw, Dvd],
            [np.zeros((Dvw.shape[1], Cv.shape[1])), np.eye(Dvw.shape[1]), np.zeros((Dvw.shape[1], Dvd.shape[1]))]
        ])
        M = cp.bmat([
            [self.MDeltapvv, np.zeros((self.MDeltapvv.shape[0], Lambda.shape[1])), self.MDeltapvw, np.zeros((self.MDeltapvv.shape[0], Lambda.shape[1]))],
            [np.zeros((Lambda.shape[0], self.MDeltapvv.shape[1] + Lambda.shape[1] + self.MDeltapvw.shape[1])), Lambda],
            [self.MDeltapvw.T, np.zeros((self.MDeltapww.shape[0], Lambda.shape[1])), self.MDeltapww, np.zeros((self.MDeltapww.shape[0], Lambda.shape[1]))],
            [np.zeros((Lambda.shape[0], self.MDeltapvv.shape[1])), Lambda, np.zeros((Lambda.shape[0], self.MDeltapww.shape[1])), -2*Lambda]
        ])

        Xright = np.bmat([
            [np.zeros((Ded.shape[1], Ce.shape[1] + Dew.shape[1])), np.eye(Ded.shape[1])],
            [Ce, Dew, Ded]
        ])
        X = np.bmat([
            [self.Xdd, self.Xde],
            [self.Xde.T, self.Xee]
        ])

        mat = F1 + Mright.T @ M @ Mright - Xright.T @ X @ Xright

        constraints = [
            P >> self.eps * np.eye(P.shape[0]),
            Lambda >> self.eps * np.eye(Lambda.shape[0]),
            mat << 0, # -self.eps * np.eye(mat.shape[0])
        ]

        problem = cp.Problem(cp.Minimize(cp.sum_squares(P) + cp.sum_squares(Lambda)), constraints)

        try:
            problem.solve(solver=cp.MOSEK) 
        except Exception as e:
            print(f"Verification: Failed to solve with exception: {e}")
            return False

        feas_stats = [
            cp.OPTIMAL,
            cp.UNBOUNDED,
            cp.OPTIMAL_INACCURATE,
            cp.UNBOUNDED_INACCURATE,
        ]
        if self.problem.status not in feas_stats:
            print(f"Verification: Failed with status {self.problem.status}")
            return False
        
        return True
