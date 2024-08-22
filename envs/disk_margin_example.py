import envs.disk_margin_base as disk_margin_base
import numpy as np
import gym


class DiskMarginExampleEnv(disk_margin_base.DiskMarginBaseEnv):
    """
    A simple example for testing disk margin code.
    """

    def __init__(self, env_config):
        Ap = np.array([
            [-3, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        Bpd = np.array([[0], [0], [0]], dtype=np.float32)
        Bpu = np.array([[4], [0], [0]], dtype=np.float32)
        Cpe = 0*np.eye(Ap.shape[0], dtype=np.float32)
        Dped = np.zeros((Ap.shape[0], 1), dtype=np.float32)
        Dpeu = np.zeros((Ap.shape[0], 1), dtype=np.float32)
        Cpy = np.array([[0, 2.5, 5]], dtype=np.float32)
        Dpyd = np.zeros((1, 1), dtype=np.float32)

        skew = 0
        alpha = 1.0
        super().__init__(
            env_config,
            Ap=Ap,
            Bpd=Bpd,
            Bpu=Bpu,
            Cpe=Cpe,
            Dped=Dped,
            Dpeu=Dpeu,
            Cpy=Cpy,
            Dpyd=Dpyd,
            skew=skew,
            alpha=alpha,
        )

        u_high = 1.0
        self.action_space = gym.spaces.Box(
            low=-u_high, high=u_high, shape=(self.nu,), dtype=np.float32
        )

        x_high = np.ones(self.nx, dtype=np.float32)
        self.state_space = gym.spaces.Box(low=-x_high, high=x_high, dtype=np.float32)

        observation_high = np.linalg.norm(self.Cpy) * np.ones(self.ny, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-observation_high,
            high=observation_high,
            dtype=np.float32,
        )

        self.time_max = env_config["time_max"] if "time_max"in env_config else 200
        self.max_reward = 1.0

        self.Xdd = np.zeros((self.nd, self.nd), dtype=np.float32)
        self.Xde = np.zeros((self.nd, self.ne), dtype=np.float32)
        self.Xee = np.zeros((self.ne, self.ne), dtype=np.float32)

    def compute_reward(self, x, u):
        return np.exp(-(u[0] ** 2))

    def random_initial_state(self):
        return 0.5 * self.state_space.sample()
        
