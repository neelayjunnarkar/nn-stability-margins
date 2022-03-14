import numpy as np
import torch
import torch.nn as nn
from models.ren_projection import LinProjector, NonlinProjector
from models.utils import uniform, to_numpy, from_numpy     

class ThetaHatParameterization:
    def __init__(
        self,
        rnn,
        lmi_eps,
        exp_stability_rate,
        plant_cstor,
        plant_config,
        ac_dim, 
        ob_dim,
        state_size,
        hidden_size
    ):
        self.rnn = rnn
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
        
        #_T for transpose, _t for tilde, _h for hat
        X_cstor = uniform(self.plant_state_size, self.plant_state_size)
        X_cstor = X_cstor.t() @ X_cstor
        self.X_cstor  = nn.Parameter(X_cstor/2.0)
        Y_cstor = uniform(self.plant_state_size, self.plant_state_size)
        Y_cstor = Y_cstor.t() @ Y_cstor
        self.Y_cstor  = nn.Parameter(Y_cstor/2.0)
        self.N11 = nn.Parameter(uniform(self.plant_state_size, self.plant_state_size))
        self.N12 = nn.Parameter(uniform(self.plant_state_size, ob_dim))
        self.N21 = nn.Parameter(uniform(ac_dim, self.plant_state_size))
        self.N22 = nn.Parameter(uniform(ac_dim, ob_dim))
        self.Lambda_c_vec = nn.Parameter(torch.rand(hidden_size) + 1e-5)
        self.N12_h = nn.Parameter(uniform(self.plant_state_size, hidden_size))
        self.N21_h = nn.Parameter(uniform(hidden_size, self.plant_state_size))
        self.DK1_t = nn.Parameter(uniform(ac_dim, hidden_size))
        if self.rnn:
            self.DK3_h = torch.zeros(hidden_size, hidden_size)
        else:
            # DK3_h_cstor = uniform(hidden_size, hidden_size)
            # DK3_h_cstor /= 4*torch.max(torch.abs(torch.linalg.eigvals(DK3_h_cstor)))
            DK3_h_cstor = torch.zeros(hidden_size, hidden_size)
            self.DK3_h = nn.Parameter(DK3_h_cstor)
        self.DK4_h = nn.Parameter(uniform(hidden_size, ob_dim))


        if self.plant_is_nonlin:
            self.projector = NonlinProjector(
                to_numpy(self.AG_t), to_numpy(self.BG1_t), to_numpy(self.BG2),
                to_numpy(self.CG1), to_numpy(self.CG2_t), to_numpy(self.DG3_t),
                self.lmi_eps, self.exp_stability_rate,
                state_size, hidden_size, ob_dim, ac_dim,
                rnn = self.rnn, recenter_lambda_p = True
            )
        else:
            self.projector = LinProjector(plant.AG, plant.BG, plant.CG, 
                self.lmi_eps, self.exp_stability_rate,
                state_size, hidden_size, ob_dim, ac_dim, rnn = self.rnn)

        self.project()

    def construct_theta_h(self):
        self.X = self.X_cstor + self.X_cstor.t()
        self.Y = self.Y_cstor + self.Y_cstor.t()
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
        Lambda_c = to_numpy(self.Lambda_c)
        N12_h = to_numpy(self.N12_h)
        N21_h = to_numpy(self.N21_h)
        DK1_t = to_numpy(self.DK1_t)
        DK3_h = to_numpy(self.DK3_h)
        DK4_h = to_numpy(self.DK4_h)

        X, Y, N11, N12, N21, N22, Lambda_c, N12_h, N21_h, DK1_t, DK3_h, DK4_h = self.projector.project(
            X, Y, N11, N12, N21, N22, Lambda_c, N12_h, N21_h, DK1_t, DK3_h, DK4_h
        )

        if X is not None: # If X is None then the parameters after the gradient step already are stabilizing.
            X_cstor = X/2.0
            Y_cstor = Y/2.0
            state_dict = {
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
            }
            if not self.rnn:
                state_dict['DK3_h'] = from_numpy(DK3_h)
            missing, unexpected = self.load_state_dict(state_dict, strict = False)
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
        self.DK3_tT = torch.t(Lambda_c_inv @ self.DK3_h)
        self.DK4_tT = torch.t(Lambda_c_inv @ self.DK4_h)
        self.DK1_tT = torch.t(self.DK1_t)

class RNNThetaHatParameterization(ThetaHatParameterization):
    def __init__(self, *args):
        super().__init__(True, *args)

class RENThetaHatParameterization(ThetaHatParameterization):
    def __init__(self, *args):
        super().__init__(False, *args)