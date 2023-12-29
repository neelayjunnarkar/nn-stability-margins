from models.RNN import RNNModel
import numpy as np
from models.rnn_projection import rnn_project, rnn_project_nonlin
from utils import to_numpy, from_numpy    

class ProjRNNOldModel(RNNModel):
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
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **custom_args)

        self.lmi_eps = lmi_eps
        self.exp_stability_rate = exp_stability_rate
        
        plant = plant_cstor(plant_config)
        self.plant_is_nonlin = plant.is_nonlin()
        if self.plant_is_nonlin:
            self.Ae = plant.Ae
            self.Be1 = plant.Be1
            self.Be2 = plant.Be2
            self.Ce1 = plant.Ce1
            self.De1 = plant.De1
            self.Ce2 = plant.Ce2
            self.M = plant.M
        else:
            self.AG = plant.AG
            self.BG = plant.BG
            self.CG = plant.CG

        self.Q1_bar = None
        self.Q2_bar = np.eye(self.hidden_size)

    def project(self):
        AK_t  = to_numpy(self.AK_tT).T
        BK1_t = to_numpy(self.BK1_tT).T
        BK2_t = to_numpy(self.BK2_tT).T
        CK1_t = to_numpy(self.CK1_tT).T
        DK1_t = to_numpy(self.DK1_tT).T
        DK2_t = to_numpy(self.DK2_tT).T
        CK2_t = to_numpy(self.CK2_tT).T
        DK4_t = to_numpy(self.DK4_tT).T

        if self.plant_is_nonlin:
            AK_t, BK1_t, BK2_t, CK1_t, DK1_t, DK2_t, CK2_t, DK4_t, self.Q1_bar, self.Q2_bar = rnn_project_nonlin(
                AK_t, BK1_t, BK2_t, CK1_t, DK1_t, DK2_t, CK2_t, DK4_t,
                self.Q1_bar, self.Q2_bar,
                self.Ae, self.Be1, self.Be2, self.Ce1, self.De1, self.Ce2, self.M,
                eps = self.lmi_eps,
                decay_factor = self.exp_stability_rate
            )
        else:
            AK_t, BK1_t, BK2_t, CK1_t, DK1_t, DK2_t, CK2_t, DK4_t, self.Q1_bar, self.Q2_bar = rnn_project(
                AK_t, BK1_t, BK2_t, CK1_t, DK1_t, DK2_t, CK2_t, DK4_t, self.Q1_bar, self.Q2_bar,
                self.AG, self.BG, self.CG,
                eps = self.lmi_eps,
                decay_factor = self.exp_stability_rate
            )

        missing, unexpected = self.load_state_dict({
            'AK_tT' : from_numpy(AK_t.T), 
            'BK1_tT': from_numpy(BK1_t.T), 
            'BK2_tT': from_numpy(BK2_t.T), 
            'CK1_tT': from_numpy(CK1_t.T), 
            'DK1_tT': from_numpy(DK1_t.T),
            'DK2_tT': from_numpy(DK2_t.T), 
            'CK2_tT': from_numpy(CK2_t.T), 
            'DK4_tT': from_numpy(DK4_t.T) 
        }, strict = False)
        assert unexpected == [], 'Loading unexpected key after projection'
