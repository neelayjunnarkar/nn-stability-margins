from envs import InvertedPendulumEnv

class LearnedInvertedPendulumEnv(InvertedPendulumEnv):
    """
    Class for loading in learned nonlinear inverted pendulum model.
    Reveals only the learned parameters, but uses the true model for 
    updating state internally.
    """

    def __init__(self, env_config):
        InvertedPendulumEnv.__init__(self, env_config)

        # Need to supply controller with learned params but simulate with real
        params = env_config['model_params']

        self.learned_AG  = params['A_T'].numpy().T
        self.learned_BG1 = params['B1_T'].numpy().T
        self.learned_BG2 = params['B2_T'].numpy().T
        self.learned_CG2 = params['C2_T'].numpy().T
        self.learned_DG3 = params['D3_T'].numpy().T

        # Not actually learned, but the parameters with which the model was trained.
        self.learned_nonlin_size = self.learned_DG3.shape[0]
        self.learned_C_Delta = 0
        self.learned_D_Delta = 1

    def get_params(self):
        return [
            self.learned_AG, self.learned_BG1, self.learned_BG2,
            self.CG1, self.learned_CG2, self.learned_DG3,
            self.learned_C_Delta, self.learned_D_Delta
        ]
