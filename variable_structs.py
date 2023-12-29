from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

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
    MDeltapvv: Any = None # NDArrayF32
    MDeltapvw: Any = None # NDArrayF32
    MDeltapww: Any = None # NDArrayF32
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
        return ControllerThetaParameters(
            Ak, Bkw, Bky, Ckv, Dkvw, Dkvy, Cku, Dkuw, Dkuy, Lambda
        )
    
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
        return ControllerThetaParameters(
            Ak, Bkw, Bky, Ckv, Dkvw, Dkvy, Cku, Dkuw, Dkuy, Lambda
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
        return ControllerLTIThetaParameters(
            Ak, Bky, Cku, Dkuy
        )
    
    def np_to_torch(self, device):
        Ak = from_numpy(self.Ak, device=device)
        Bky = from_numpy(self.Bky, device=device)
        Cku = from_numpy(self.Cku, device=device)
        Dkuy = from_numpy(self.Dkuy, device=device)
        return ControllerLTIThetaParameters(
            Ak, Bky, Cku, Dkuy
        )


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
        return ControllerLTIThetahatParameters(
            S, R, NA11, NA12, NA21, NA22
        )
    
    def np_to_torch(self, device):
        S = from_numpy(self.S, device=device)
        R = from_numpy(self.R, device=device)
        NA11 = from_numpy(self.NA11, device=device)
        NA12 = from_numpy(self.NA12, device=device)
        NA21 = from_numpy(self.NA21, device=device)
        NA22 = from_numpy(self.NA22, device=device)
        return ControllerLTIThetahatParameters(
            S, R, NA11, NA12, NA21, NA22
        )

