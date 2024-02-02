"""Functions for designing controllers for LTI systems"""

import control as ct
import numpy as np

from theta_dissipativity import LTIProjector as ThetaLTIProjector
from thetahat_dissipativity import LTIProjector as ThetahatLTIProjector
from variable_structs import (
    ControllerLTIThetahatParameters,
    ControllerLTIThetaParameters,
    PlantParameters,
)


def lqr(
    plant_params: PlantParameters,
    output_size=None,
    state_size=None,
    input_size=None,
    Q=None,
    R=None,
    N=None,
    **kwargs,
):
    assert state_size is not None
    Ap = plant_params.Ap
    Bpu = plant_params.Bpu
    # Only works for state-feedback.
    assert Q is not None
    assert R is not None
    # N can be None
    if N is None:
        K, S, E = ct.lqr(Ap, Bpu, Q, R)
    else:
        K, S, E = ct.lqr(Ap, Bpu, Q, R, N)

    # Construct LTI controller
    nx = state_size
    ny = input_size
    nu = output_size
    controller = ControllerLTIThetaParameters(
        Ak=np.zeros((nx, nx)), Bky=np.zeros((nx, ny)), Cku=np.zeros((nu, nx)), Dkuy=-K
    )

    return controller, {}


def construct_theta(thetahat: ControllerLTIThetahatParameters, plant_params: PlantParameters):
    """Construct theta, the parameters of the controller, from thetahat,
    the decision variables for the dissipativity condition."""
    state_size = thetahat.S.shape[0]

    # np.linalg.solve(A, B) solves for X in AX = B, and assumes A is invertible

    # Construct V and U by solving VU^T = I - RS
    svdU, svdSigma, svdV_T = np.linalg.svd(np.eye(state_size) - thetahat.R @ thetahat.S)
    sqrt_svdSigma = np.diag(np.sqrt(svdSigma))
    V = svdU @ sqrt_svdSigma
    U = svdV_T.T @ sqrt_svdSigma

    # Construct P via P = [I, S; 0, U^T] Y^-1
    # fmt: off
    Y = np.vstack((
        np.hstack((thetahat.R, np.eye(thetahat.R.shape[0]))),
        np.hstack((V.T, np.zeros((V.T.shape[0], thetahat.R.shape[0]))))
    ))
    Y2 = np.vstack((
        np.hstack((np.eye(thetahat.S.shape[0]), thetahat.S)),
        np.hstack((np.zeros((U.T.shape[0], thetahat.S.shape[0])), U.T))
    ))
    # fmt: on
    P = np.linalg.solve(Y.T, Y2.T)
    P = 0.5 * (P + P.T)

    Ap = plant_params.Ap
    Bpu = plant_params.Bpu
    Cpy = plant_params.Cpy

    Duy = thetahat.NA22
    By = np.linalg.solve(U, thetahat.NA12 - thetahat.S @ Bpu @ Duy)
    Cu = np.linalg.solve(V, thetahat.NA21.T - thetahat.R @ Cpy.T @ Duy.T).T
    AVT = np.linalg.solve(
        U,
        thetahat.NA11
        - U @ By @ Cpy @ thetahat.R
        - thetahat.S @ Bpu @ (Cu @ V.T + Duy @ Cpy @ thetahat.R)
        - thetahat.S @ Ap @ thetahat.R,
    )
    A = np.linalg.solve(V, AVT.T).T

    controller = ControllerLTIThetaParameters(Ak=A, Bky=By, Cku=Cu, Dkuy=Duy)
    return controller, P


def dissipative_thetahat(
    plant_params: PlantParameters,
    # Epsilon to be used in enforcing definiteness of conditions
    eps=1e-3,
    # Dimensions of variables for controller
    output_size=None,
    state_size=None,
    input_size=None,
    trs_mode="fixed",
    min_trs=1.0,
    backoff_factor=1.1,
    **kwargs,
):
    """Synthesize dissipative LTI controller using convex condition."""
    assert output_size is not None
    assert state_size is not None
    assert input_size is not None

    projector = ThetahatLTIProjector(
        plant_params,
        eps,
        output_size,
        state_size,
        input_size,
        trs_mode,
        min_trs,
        backoff_factor=backoff_factor,
    )

    # Construct thetahat0 as thetahat when theta=0 and P=I
    thetahat0 = ControllerLTIThetahatParameters(
        S=np.eye(state_size),
        R=np.eye(state_size),
        NA11=plant_params.Ap,
        NA12=np.zeros((state_size, input_size)),
        NA21=np.zeros((output_size, state_size)),
        NA22=np.zeros((output_size, input_size)),
    )
    thetahat = projector.project(thetahat0)
    controller, P = construct_theta(thetahat, plant_params)

    return controller, {"P": P, "thetahat": thetahat}


def dissipative_theta(
    plant_params: PlantParameters,
    # Epsilon to be used in enforcing definiteness of conditions
    eps=1e-3,
    # Dimensions of variables for controller
    output_size=None,
    state_size=None,
    input_size=None,
    P0=None,
    **kwargs,
):
    """Synthesize dissipative LTI controller."""
    assert output_size is not None
    assert state_size is not None
    assert input_size is not None

    projector = ThetaLTIProjector(plant_params, eps, output_size, state_size, input_size)

    k0 = ControllerLTIThetaParameters(
        Ak=np.zeros((state_size, state_size)),
        Bky=np.zeros((state_size, input_size)),
        Cku=np.zeros((output_size, state_size)),
        Dkuy=np.zeros((output_size, input_size)),
    )
    if P0 is None:
        P0 = np.eye(plant_params.Ap.shape[0] + state_size)

    controller = projector.project(k0, P0)
    return controller, {"P": P0}


controller_map = {
    "lqr": lqr,
    "dissipative_theta": dissipative_theta,
    "dissipative_thetahat": dissipative_thetahat,
}
