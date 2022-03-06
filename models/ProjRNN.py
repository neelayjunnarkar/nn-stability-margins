from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn
from models.utils import uniform, to_numpy, from_numpy
from models.RNN import BaseRNN
import numpy as np
from models.ren_projection import ren_project_nonlin, LinProjector

class ProjRNNModel(BaseRNN):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        lmi_eps = 1e-5,
        exp_stability_rate = 0.98,
        plant_cstor = None,
        plant_config = None,
        **custom_args
    ):
        assert plant_cstor is not None, "plant_cstor parameter is None"
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **custom_args)

        self.lmi_eps = lmi_eps
        self.exp_stability_rate = exp_stability_rate

        # Get plant parameters
        plant = plant_cstor(plant_config)
        self.plant_is_nonlin = plant.is_nonlin()
        self.plant_state_size = plant.state_size
        if self.plant_is_nonlin:
            self.plant_nonlin_size = plant.nonlin_size
            C_Delta = plant.C_Delta
            D_Delta = plant.D_Delta
            S_Delta = (C_Delta + D_Delta)/2.0
            L_Delta = (D_Delta - C_Delta)/2.0
            MG3 = np.linalg.inv(np.eye(plant.DG3.shape[0]) - S_Delta * plant.DG3)
            self.AG_t = from_numpy(plant.AG + S_Delta * plant.BG1 @ MG3 @ plant.CG2)
            self.BG1_t = from_numpy(plant.BG1 @ MG3 * L_Delta)
            self.BG2 = from_numpy(plant.BG2)
            self.CG1 = from_numpy(plant.CG1)
            self.CG2_t = from_numpy(plant.CG2 + S_Delta * plant.DG3 @ MG3 @ plant.CG2)
            self.DG3_t = from_numpy(L_Delta * plant.DG3 @ MG3)        
        else:
            self.AG_t = from_numpy(plant.AG)
            self.BG2  = from_numpy(plant.BG)
            self.CG1  = from_numpy(plant.CG)

        #_T for transpose
        #_t for tilde
        #_h for hat
        self.X_cstor  = nn.Parameter(uniform(self.plant_state_size, self.plant_state_size)/2)
        self.Y_cstor  = nn.Parameter(uniform(self.plant_state_size, self.plant_state_size)/2)
        self.N11 = nn.Parameter(uniform(self.plant_state_size, self.plant_state_size))
        self.N12 = nn.Parameter(uniform(self.plant_state_size, self.ob_dim))
        self.N21 = nn.Parameter(uniform(self.ac_dim, self.plant_state_size))
        self.N22 = nn.Parameter(uniform(self.ac_dim, self.ob_dim))
        if self.plant_is_nonlin:
            self.Lambda_p_vec = torch.ones(self.plant_nonlin_size)
        self.Lambda_c_vec = nn.Parameter(torch.rand(self.hidden_size)) # Must be positive
        self.N12_h = nn.Parameter(uniform(self.plant_state_size, self.hidden_size))
        self.N21_h = nn.Parameter(uniform(self.hidden_size, self.plant_state_size))
        self.DK1_t = nn.Parameter(uniform(self.ac_dim, self.hidden_size))
        # In RNN, DK3_h = 0
        self.DK4_h = nn.Parameter(uniform(self.hidden_size, self.ob_dim))
        

        if not self.plant_is_nonlin:
            self.lin_projector = LinProjector(plant.AG, plant.BG, plant.CG, 
                self.lmi_eps, self.exp_stability_rate,
                self.state_size, self.hidden_size, self.ob_dim, self.ac_dim, rnn = True)

        self.project()

    def construct_theta_h(self):
        self.X = self.X_cstor + self.X_cstor.t()
        self.Y = self.Y_cstor + self.Y_cstor.t()
        if self.plant_is_nonlin:
            self.Lambda_p = torch.diag(self.Lambda_p_vec)
        self.Lambda_c = torch.diag(self.Lambda_c_vec)

    def project(self):
        self.construct_theta_h()
        self.project_to_stabilizing_set()
        self.recover_theta_t()

    def project_to_stabilizing_set(self):
        X = to_numpy(self.X)
        Y = to_numpy(self.Y)
        N11 = to_numpy(self.N11)
        N12 = to_numpy(self.N12)
        N21  = to_numpy(self.N21)
        N22 = to_numpy(self.N22)
        if self.plant_is_nonlin:
           Lambda_p = to_numpy(self.Lambda_p)
        Lambda_c = to_numpy(self.Lambda_c)
        N12_h = to_numpy(self.N12_h)
        N21_h = to_numpy(self.N21_h)
        DK1_t = to_numpy(self.DK1_t)
        DK4_h = to_numpy(self.DK4_h)
        if self.plant_is_nonlin:
            AG_t  = to_numpy(self.AG_t)
            BG2   = to_numpy(self.BG2)
            CG1   = to_numpy(self.CG1)
            BG1_t = to_numpy(self.BG1_t)
            CG2_t = to_numpy(self.CG2_t)
            DG3_t = to_numpy(self.DG3_t)

        if self.plant_is_nonlin:
            X, Y, N11, N12, N21, N22, Lambda_c, N12_h, N21_h, DK1_t, _, DK4_h = \
                ren_project_nonlin(X, Y, N11, N12, N21, N22, Lambda_p, Lambda_c, N12_h, N21_h, \
                    DK1_t, None, DK4_h, \
                    AG_t, BG1_t, BG2, CG1, CG2_t, DG3_t, \
                    eps = self.lmi_eps, decay_factor = self.exp_stability_rate, rnn = True)
        else:
            X, Y, N11, N12, N21, N22, Lambda_c, N12_h, N21_h, DK1_t, _, DK4_h = self.lin_projector.project(
                X, Y, N11, N12, N21, N22, Lambda_c, N12_h, N21_h, DK1_t, None, DK4_h
            )

        if X is not None: # If X is None then the parameters after the gradient step already are stabilizing.
            X_cstor = X/2.0
            Y_cstor = Y/2.0
            missing, unexpected = self.load_state_dict({
                'X_cstor': from_numpy(X_cstor),
                'Y_cstor': from_numpy(Y_cstor),
                'N11': from_numpy(N11),
                'N12': from_numpy(N12),
                'N21': from_numpy(N21),
                'N22': from_numpy(N22),
                'Lambda_c_vec': torch.diagonal(from_numpy(Lambda_c)),
                'N12_h': from_numpy(N12_h),
                'N21_h': from_numpy(N21_h),
                'DK1_t': from_numpy(DK1_t),
                'DK4_h': from_numpy(DK4_h)
            }, strict = False)
            assert unexpected == [], 'Loading unexpected key after projection'
            assert missing == ["log_stds", "value.0.weight", "value.0.bias", "value.2.weight", 
                "value.2.bias", "value.4.weight", "value.4.bias"], 'Missing keys after projection'
            self.construct_theta_h()
    
    def recover_theta_t(self):
        """
        Converts theta hat parameters to theta tilde parameters. 
        X and Y must be positive definite symmetric.
        Lambda must be positive definite diagonal.
        """
        U = torch.hstack((self.X, torch.zeros(self.X.shape[0], self.state_size - self.plant_state_size)))
        V = torch.hstack((torch.inverse(self.X) - self.Y, torch.zeros(self.X.shape[0], self.state_size - self.plant_state_size)))

        left = torch.vstack((torch.hstack((U, self.X @ self.BG2)),
                            torch.hstack((torch.zeros(self.BG2.shape[1], U.shape[1]), torch.eye(self.BG2.shape[1])))))
        mid = torch.vstack((torch.hstack((self.N11 - self.X@self.AG_t@self.Y, self.N12)),
                            torch.hstack((self.N21, self.N22))))
        right_T = torch.vstack((torch.hstack((V, self.Y.t() @ self.CG1.t())),
                                torch.hstack((torch.zeros(self.CG1.shape[0], V.shape[1]), torch.eye(self.CG1.shape[0])))))
        ABCD = (right_T.pinverse() @ (left.pinverse() @ mid).t()).t()

        self.AK_tT  = ABCD[:self.state_size, :self.state_size].t()
        self.BK2_tT = ABCD[:self.state_size, self.state_size:].t()
        self.CK1_tT = ABCD[self.state_size:, :self.state_size].t()
        self.DK2_tT = ABCD[self.state_size:, self.state_size:].t()

        self.BK1_tT = (U.pinverse() @ (self.N12_h - self.X @ self.BG2 @ self.DK1_t)).t()
        Lambda_c_inv = self.Lambda_c.inverse()
        self.CK2_tT = V.pinverse() @ (self.N21_h - self.DK4_h @ self.CG1 @ self.Y).t() @ Lambda_c_inv
        self.DK4_tT = torch.t(Lambda_c_inv @ self.DK4_h)
        self.DK1_tT = torch.t(self.DK1_t)

    @override(BaseRNN)
    def phi_t(self, state, obs):
        """Loop transformed phi"""
        # v(k) = CK2_t xi(k) + DK4_t y(k)
        # z(k) = phi_t(v(k))
        v = state @ self.CK2_tT + obs @ self.DK4_tT
        if self._scalar_bounds:
            z = self.L_phi_inv * (self.phi(v) - self.S_phi * v)
        else:
            z = self.L_phi_inv @ (self.phi(v) - self.S_phi @ v)
        return z
