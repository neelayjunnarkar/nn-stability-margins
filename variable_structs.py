from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from utils import from_numpy, to_numpy

NDArrayF32 = npt.NDArray[np.float32]


@dataclass
class PlantParameters:
    Ap: Any  # NDArrayF32
    Bpw: Any  # NDArrayF32
    Bpd: Any  # NDArrayF32
    Bpu: Any  # NDArrayF32
    Cpv: Any  # NDArrayF32
    Dpvw: Any  # NDArrayF32
    Dpvd: Any  # NDArrayF32
    Dpvu: Any  # NDArrayF32
    Cpe: Any  # NDArrayF32
    Dpew: Any  # NDArrayF32
    Dped: Any  # NDArrayF32
    Dpeu: Any  # NDArrayF32
    Cpy: Any  # NDArrayF32
    Dpyw: Any  # NDArrayF32
    Dpyd: Any  # NDArrayF32
    MDeltapvv: Any = None  # NDArrayF32
    MDeltapvw: Any = None  # NDArrayF32
    MDeltapww: Any = None  # NDArrayF32
    Xdd: Any = None
    Xde: Any = None
    Xee: Any = None

    def np_to_torch(self, device):
        Ap = from_numpy(self.Ap, device=device)
        Bpw = from_numpy(self.Bpw, device=device)
        Bpd = from_numpy(self.Bpd, device=device)
        Bpu = from_numpy(self.Bpu, device=device)
        Cpv = from_numpy(self.Cpv, device=device)
        Dpvw = from_numpy(self.Dpvw, device=device)
        Dpvd = from_numpy(self.Dpvd, device=device)
        Dpvu = from_numpy(self.Dpvu, device=device)
        Cpe = from_numpy(self.Cpe, device=device)
        Dpew = from_numpy(self.Dpew, device=device)
        Dped = from_numpy(self.Dped, device=device)
        Dpeu = from_numpy(self.Dpeu, device=device)
        Cpy = from_numpy(self.Cpy, device=device)
        Dpyw = from_numpy(self.Dpyw, device=device)
        Dpyd = from_numpy(self.Dpyd, device=device)
        # fmt: off
        MDeltapvv = from_numpy(self.MDeltapvv, device=device) if self.MDeltapvv is not None else None
        MDeltapvw = from_numpy(self.MDeltapvw, device=device) if self.MDeltapvw is not None else None
        MDeltapww = from_numpy(self.MDeltapww, device=device) if self.MDeltapww is not None else None
        Xdd = from_numpy(self.Xdd, device=device) if self.Xdd is not None else None
        Xde = from_numpy(self.Xde, device=device) if self.Xde is not None else None
        Xee = from_numpy(self.Xee, device=device) if self.Xee is not None else None
        return PlantParameters(
            Ap, Bpw, Bpd, Bpu, Cpv, Dpvw, Dpvd, Dpvu,
            Cpe, Dpew, Dped, Dpeu, Cpy, Dpyw, Dpyd,
            MDeltapvv, MDeltapvw, MDeltapww,
            Xdd, Xde, Xee
        )
        # fmt: on


@dataclass
class ControllerThetaParameters:
    Ak: Any  # NDArrayF32
    Bkw: Any  # NDArrayF32
    Bky: Any  # NDArrayF32
    Ckv: Any  # NDArrayF32
    Dkvw: Any  # NDArrayF32
    Dkvy: Any  # NDArrayF32
    Cku: Any  # NDArrayF32
    Dkuw: Any  # NDArrayF32
    Dkuy: Any  # NDArrayF32
    Lambda: Any  # NDArrayF32

    def torch_to_np(self):
        Ak = to_numpy(self.Ak)
        Bkw = to_numpy(self.Bkw)
        Bky = to_numpy(self.Bky)
        Ckv = to_numpy(self.Ckv)
        Dkvw = to_numpy(self.Dkvw)
        Dkvy = to_numpy(self.Dkvy)
        Cku = to_numpy(self.Cku)
        Dkuw = to_numpy(self.Dkuw)
        Dkuy = to_numpy(self.Dkuy)
        Lambda = to_numpy(self.Lambda) if self.Lambda is not None else None
        return ControllerThetaParameters(Ak, Bkw, Bky, Ckv, Dkvw, Dkvy, Cku, Dkuw, Dkuy, Lambda)

    def np_to_torch(self, device):
        Ak = from_numpy(self.Ak, device=device)
        Bkw = from_numpy(self.Bkw, device=device)
        Bky = from_numpy(self.Bky, device=device)
        Ckv = from_numpy(self.Ckv, device=device)
        Dkvw = from_numpy(self.Dkvw, device=device)
        Dkvy = from_numpy(self.Dkvy, device=device)
        Cku = from_numpy(self.Cku, device=device)
        Dkuw = from_numpy(self.Dkuw, device=device)
        Dkuy = from_numpy(self.Dkuy, device=device)
        Lambda = from_numpy(self.Lambda, device=device) if self.Lambda is not None else None
        return ControllerThetaParameters(Ak, Bkw, Bky, Ckv, Dkvw, Dkvy, Cku, Dkuw, Dkuy, Lambda)

    def matrix(self, type="np"):
        if type == "np":
            stacker = np
        elif type == "torch":
            stacker = torch
        else:
            raise ValueError(f"Unexpected type: {type}.")
        # fmt: off
        theta = stacker.vstack((
            stacker.hstack((self.Ak,  self.Bkw, self.Bky)),
            stacker.hstack((self.Ckv, self.Dkvw, self.Dkvy)),
            stacker.hstack((self.Cku, self.Dkuw, self.Dkuy))
        ))
        # fmt: on
        return theta

    def torch_construct_thetahat(self, P, plant_params, eps=1e-3):
        """Construct thetahat parameters from self and P"""
        state_size = self.Ak.shape[0]
        assert state_size == plant_params.Ap.shape[0]

        S = P[:state_size, :state_size]
        S = 0.5 * (S + S.t())
        U = P[:state_size, state_size:]
        assert torch.allclose(S, S.t()), "S is not symmetric"
        assert (
            torch.linalg.eigvalsh(S).min() > 0
        ), f"S min eigval is {torch.linalg.eigvalsh(S).min()}"

        # Pinv = self.P.inverse()
        # R = Pinv[:state_size, :state_size]
        # V = Pinv[:state_size, state_size:]
        # if (not torch.allclose(R, R.t())) and  (R - R.t()).abs().max() < eps:
        #     print(f"R: {torch.allclose(R, R.t())}, {(R - R.t()).abs().max()}, {torch.linalg.eigvalsh(R).min()}")
        #     R = (R + R.t())/2.0
        #     print(f"R: {torch.allclose(R, R.t())}, {(R - R.t()).abs().max()}, {torch.linalg.eigvalsh(R).min()}")
        # assert torch.allclose(R, R.t()), "R is not symmetric"
        # assert torch.linalg.eigvalsh(R).min() > eps, f"P min eigval is {torch.linalg.eigvalsh(R).min()}"
        # Solve P Y = Y2 for Y
        # fmt: off
        Y2 = torch.vstack((
            torch.hstack((torch.eye(S.shape[0], device=S.device), S)),
            torch.hstack((S.new_zeros((U.t().shape[0], S.shape[0])), U.t()))
        ))
        # fmt: on
        Y = torch.linalg.solve(P, Y2)
        R = Y[:state_size, :state_size]
        R = 0.5 * (R + R.t())
        V = Y[state_size:, :state_size].t()
        # fmt: off
        # if (not torch.allclose(R, R.t())) and  (R - R.t()).abs().max() < eps:
        #     print(f"R: {torch.allclose(R, R.t())}, {(R - R.t()).abs().max()}, {torch.linalg.eigvalsh(R).min()}")
        #     R = (R + R.t())/2.0
        #     print(f"R: {torch.allclose(R, R.t())}, {(R - R.t()).abs().max()}, {torch.linalg.eigvalsh(R).min()}")
        # fmt: on
        assert torch.allclose(R, R.t()), "R is not symmetric"
        assert (
            torch.linalg.eigvalsh(R).min() > 0
        ), f"P min eigval is {torch.linalg.eigvalsh(R).min()}"

        Lambda = self.Lambda
        Lambda = 0.5 * (Lambda + Lambda.t())

        # NA = NA1 + NA2 NA3 NA4

        Ap = plant_params.Ap
        Bpu = plant_params.Bpu
        Cpy = plant_params.Cpy

        NA111 = torch.mm(S, torch.mm(Ap, R))
        NA112 = V.new_zeros((state_size, Cpy.shape[0]))
        NA121 = V.new_zeros((Bpu.shape[1], state_size))
        NA122 = V.new_zeros((Bpu.shape[1], Cpy.shape[0]))
        NA1 = torch.vstack((torch.hstack((NA111, NA112)), torch.hstack((NA121, NA122))))

        NA211 = U
        NA212 = torch.mm(S, Bpu)
        NA221 = V.new_zeros((Bpu.shape[1], U.shape[1]))
        NA222 = torch.eye(Bpu.shape[1], device=V.device)
        NA2 = torch.vstack((torch.hstack((NA211, NA212)), torch.hstack((NA221, NA222))))

        # fmt: off
        NA3 = torch.vstack((
            torch.hstack((self.Ak, self.Bky)), 
            torch.hstack((self.Cku, self.Dkuy))
        ))
        # fmt: on

        NA411 = V.t()
        NA412 = V.new_zeros(V.shape[1], Cpy.shape[0])
        NA421 = torch.mm(Cpy, R)
        NA422 = torch.eye(Cpy.shape[0], device=V.device)
        NA4 = torch.vstack((torch.hstack((NA411, NA412)), torch.hstack((NA421, NA422))))

        NA = NA1 + torch.mm(NA2, torch.mm(NA3, NA4))
        NA11 = NA[:state_size, :state_size]
        NA12 = NA[:state_size, state_size:]
        NA21 = NA[state_size:, :state_size]
        NA22 = NA[state_size:, state_size:]

        NB = torch.mm(S, torch.mm(Bpu, self.Dkuw)) + torch.mm(U, self.Bkw)
        # fmt: off
        NC = torch.mm(Lambda, torch.mm(self.Dkvy, torch.mm(Cpy, R))) \
            + torch.mm(Lambda, torch.mm(self.Ckv, V.t()))
        # fmt: on

        Dvyhat = torch.mm(Lambda, self.Dkvy)
        Dvwhat = torch.mm(Lambda, self.Dkvw)

        return ControllerThetahatParameters(
            S, R, NA11, NA12, NA21, NA22, NB, NC, self.Dkuw, Dvyhat, Dvwhat, Lambda
        )


@dataclass
class ControllerThetahatParameters:
    S: Any
    R: Any
    NA11: Any
    NA12: Any
    NA21: Any
    NA22: Any
    NB: Any
    NC: Any
    Dkuw: Any
    Dkvyhat: Any
    Dkvwhat: Any
    Lambda: Any

    def torch_to_np(self):
        S = to_numpy(self.S)
        R = to_numpy(self.R)
        NA11 = to_numpy(self.NA11)
        NA12 = to_numpy(self.NA12)
        NA21 = to_numpy(self.NA21)
        NA22 = to_numpy(self.NA22)
        NB = to_numpy(self.NB)
        NC = to_numpy(self.NC)
        Dkuw = to_numpy(self.Dkuw)
        Dkvyhat = to_numpy(self.Dkvyhat)
        Dkvwhat = to_numpy(self.Dkvwhat)
        Lambda = to_numpy(self.Lambda)
        return ControllerThetahatParameters(
            S, R, NA11, NA12, NA21, NA22, NB, NC, Dkuw, Dkvyhat, Dkvwhat, Lambda
        )

    def np_to_torch(self, device):
        S = from_numpy(self.S, device=device)
        R = from_numpy(self.R, device=device)
        NA11 = from_numpy(self.NA11, device=device)
        NA12 = from_numpy(self.NA12, device=device)
        NA21 = from_numpy(self.NA21, device=device)
        NA22 = from_numpy(self.NA22, device=device)
        NB = from_numpy(self.NB, device=device)
        NC = from_numpy(self.NC, device=device)
        Dkuw = from_numpy(self.Dkuw, device=device)
        Dkvyhat = from_numpy(self.Dkvyhat, device=device)
        Dkvwhat = from_numpy(self.Dkvwhat, device=device)
        Lambda = from_numpy(self.Lambda, device=device)
        return ControllerThetahatParameters(
            S, R, NA11, NA12, NA21, NA22, NB, NC, Dkuw, Dkvyhat, Dkvwhat, Lambda
        )

    def torch_construct_theta(self, plant_params, eps=1e-3):
        """Construct theta, the parameters of the controller, from thetahat,
        the decision variables for the dissipativity condition."""

        state_size = self.S.shape[0]
        assert state_size == plant_params.Ap.shape[0]

        svdU, svdSigma, svdV_T = torch.linalg.svd(torch.eye(state_size) - self.R @ self.S)
        sqrt_svdSigma = torch.diag(torch.sqrt(svdSigma))
        V = svdU @ sqrt_svdSigma
        U = svdV_T.t() @ sqrt_svdSigma

        # Construct P via P = [I, S; 0, U^T] Y^-1
        # fmt: off
        Y = torch.vstack((
            torch.hstack((self.R, torch.eye(self.R.shape[0], device=self.R.device))),
            torch.hstack((V.t(), V.new_zeros((V.t().shape[0], self.R.shape[0]))))
        ))
        Y2 = torch.vstack((
            torch.hstack((torch.eye(self.S.shape[0], device=self.S.device), self.S)),
            torch.hstack((V.new_zeros((U.t().shape[0], self.S.shape[0])), U.t()))
        ))
        # fmt: on
        P = torch.linalg.solve(Y.t(), Y2.t())
        P = 0.5 * (P + P.t())
        assert torch.allclose(P, P.t()), "R is not symmetric"
        assert (
            torch.linalg.eigvalsh(P).min() > 0
        ), f"P min eigval is {torch.linalg.eigvalsh(P).min()}"

        # Reconstruct Dvy and Dvw from Dvyhat and Dvwhat
        Dvy = torch.linalg.solve(self.Lambda, self.Dkvyhat)
        Dvw = torch.linalg.solve(self.Lambda, self.Dkvwhat)

        Ap = plant_params.Ap
        Bpu = plant_params.Bpu
        Cpy = plant_params.Cpy

        Duy = self.NA22
        By = torch.linalg.solve(U, self.NA12 - self.S @ Bpu @ Duy)
        Cu = torch.linalg.solve(V, self.NA21.t() - self.R @ Cpy.t() @ Duy.t()).t()
        # fmt: off
        AVT = torch.linalg.solve(
            U,
            self.NA11 \
                - U @ By @ Cpy @ self.R \
                - self.S @ Bpu @ (Cu @ V.t() + Duy @ Cpy @ self.R) \
                - self.S @ Ap @ self.R
        )
        # fmt: on
        A = torch.linalg.solve(V, AVT.t()).t()

        # Solve for Bw from NB
        # NB = NB1 + U Bw
        NB1 = torch.mm(self.S, torch.mm(Bpu, self.Dkuw))
        Bw = torch.linalg.solve(U, self.NB - NB1)

        # Solve for Cv from NC
        # NC = NC1 + Lambda Cv V^T
        NC1 = torch.mm(self.Dkvyhat, torch.mm(Cpy, self.R))
        CvVT = torch.linalg.solve(self.Lambda, self.NC - NC1)
        Cv = torch.linalg.solve(V, CvVT.t()).t()

        # Bring together the theta parameters here
        theta = ControllerThetaParameters(A, Bw, By, Cv, Dvw, Dvy, Cu, self.Dkuw, Duy, self.Lambda)

        return theta, P


@dataclass
class ControllerLTIThetaParameters:
    Ak: Any  # NDArrayF32
    Bky: Any  # NDArrayF32
    Cku: Any  # NDArrayF32
    Dkuy: Any  # NDArrayF32

    def torch_to_np(self):
        Ak = to_numpy(self.Ak)
        Bky = to_numpy(self.Bky)
        Cku = to_numpy(self.Cku)
        Dkuy = to_numpy(self.Dkuy)
        return ControllerLTIThetaParameters(Ak, Bky, Cku, Dkuy)

    def np_to_torch(self, device):
        Ak = from_numpy(self.Ak, device=device)
        Bky = from_numpy(self.Bky, device=device)
        Cku = from_numpy(self.Cku, device=device)
        Dkuy = from_numpy(self.Dkuy, device=device)
        return ControllerLTIThetaParameters(Ak, Bky, Cku, Dkuy)


@dataclass
class ControllerLTIThetahatParameters:
    S: Any
    R: Any
    NA11: Any
    NA12: Any
    NA21: Any
    NA22: Any

    def torch_to_np(self):
        S = to_numpy(self.S)
        R = to_numpy(self.R)
        NA11 = to_numpy(self.NA11)
        NA12 = to_numpy(self.NA12)
        NA21 = to_numpy(self.NA21)
        NA22 = to_numpy(self.NA22)
        return ControllerLTIThetahatParameters(S, R, NA11, NA12, NA21, NA22)

    def np_to_torch(self, device):
        S = from_numpy(self.S, device=device)
        R = from_numpy(self.R, device=device)
        NA11 = from_numpy(self.NA11, device=device)
        NA12 = from_numpy(self.NA12, device=device)
        NA21 = from_numpy(self.NA21, device=device)
        NA22 = from_numpy(self.NA22, device=device)
        return ControllerLTIThetahatParameters(S, R, NA11, NA12, NA21, NA22)


# Other reconstruction methods

# Construct V and U by solving VU^T = I - RS
# V = self.R
# U = torch.linalg.solve(
#     V, torch.eye(state_size, device=V.device) - torch.mm(self.R, self.S)
# ).t()

# # Solve for A, By, Cu, and Duy from NA
# # NA = NA1 + NA2 NA3 NA4
# NA = torch.vstack(
#     (torch.hstack((self.NA11, self.NA12)), torch.hstack((self.NA21, self.NA22)))
# )

# NA111 = torch.mm(self.S, torch.mm(self.Ap, self.R))
# NA112 = V.new_zeros((state_size, self.Cpy.shape[0]))
# NA121 = V.new_zeros((self.Bpu.shape[1], state_size))
# NA122 = V.new_zeros((self.Bpu.shape[1], self.Cpy.shape[0]))
# NA1 = torch.vstack((torch.hstack((NA111, NA112)), torch.hstack((NA121, NA122))))

# NA211 = U
# NA212 = torch.mm(self.S, self.Bpu)
# NA221 = V.new_zeros((self.Bpu.shape[1], U.shape[1]))
# NA222 = torch.eye(self.Bpu.shape[1], device=V.device)
# NA2 = torch.vstack((torch.hstack((NA211, NA212)), torch.hstack((NA221, NA222))))

# NA411 = V.t()
# NA412 = V.new_zeros(V.shape[1], self.Cpy.shape[0])
# NA421 = torch.mm(self.Cpy, self.R)
# NA422 = torch.eye(self.Cpy.shape[0], device=V.device)
# NA4 = torch.vstack((torch.hstack((NA411, NA412)), torch.hstack((NA421, NA422))))

# NA3NA4 = torch.linalg.solve(NA2, NA - NA1)
# NA3 = torch.linalg.solve(NA4.t(), NA3NA4.t()).t()

# A = NA3[: state_size, : state_size]
# By = NA3[: state_size, state_size :]
# Cu = NA3[state_size :, : state_size]
# Duy = NA3[state_size :, state_size :]
