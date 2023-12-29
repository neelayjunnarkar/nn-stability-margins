"""Functions for designing controllers for LTI systems"""

import control as ct
import numpy as np


def lqr(Ap=None, Bpu=None, Q=None, R=None, N=None, **_kwargs):
    # Only works for state-feedback.
    assert Ap is not None
    assert Bpu is not None
    assert Q is not None
    assert R is not None
    # N can be None
    if N is None:
        K, S, E = ct.lqr(Ap, Bpu, Q, R)
    else:
        K, S, E = ct.lqr(Ap, Bpu, Q, R, N)

    # Construct LTI controller
    nx = 1
    ny = Ap.shape[0]
    nu = Bpu.shape[1]
    Ak = np.zeros((nx, nx))
    Bky = np.zeros((nx, ny))
    Cku = np.zeros((nu, nx))
    Dkuy = -K

    return (Ak, Bky, Cku, Dkuy)


controller_map = {"lqr": lqr}
