import envs.disk_margin_base as disk_margin_base
import numpy as np
import gym
from variable_structs import PlantParameters


class FlexibleArmDiskMarginEnv(disk_margin_base.DiskMarginBaseEnv):
    """
    An environment for a flexible rod on a cart with a rigid rod on a cart design model with disk margins.
    """

    def __init__(self, env_config):

        self.disturbance_model = env_config["disturbance_model"] if "disturbance_model" in env_config else "none"

        self.mb = 1  # Mass of base (Kg)
        self.mt = 0.1  # Mass of tip (Kg)
        self.L = 1  # Length of link (m)
        self.rho = 0.1  # Mass density of Link (N/m)

        self.r = 1e-2  # Radius of rod cross-section (m)
        self.E = 200e9  # Young's modulus for steel, GPa
        self.I = (np.pi / 4) * self.r**4  # Area second moment of inertia (m^4)
        self.EI = self.E * self.I
        self.flexdamp = 0.9
        self.Mr = self.mb + self.mt + self.rho * self.L  # Total mass (Kg)

        Ap = np.array([[0, 1], [0, 0]], dtype=np.float32)
        # TODO: No disturbances in design model?
        Bpd = np.array([[0], [0]], dtype=np.float32)
        Bpu = np.array([[0], [1.0 / self.Mr]], dtype=np.float32)
        Cpe = np.eye(Ap.shape[0], dtype=np.float32)
        Dped = np.zeros((Ap.shape[0], 1), dtype=np.float32)
        Dpeu = np.zeros((Ap.shape[0], 1), dtype=np.float32)
        # Partial observability
        Cpy = np.array([[1, 0]], dtype=np.float32)
        Dpyd = np.zeros((1, 1), dtype=np.float32)

        self.true_model = self._build_true_model()
        

        self.max_force = 20  # 200
        self.max_x = 1.5
        self.max_h = 0.66 * self.L
        self.max_xdot = 25
        self.max_hdot = 200

        # The specified disk margin implies the corresponding gain and phase margins with 0 skew
        # unless it is set to custom, in which case the specified skew and alpha parameters are used.
        disk_margin_type = (
            env_config["disk_margin_type"]
            if "disk_margin_type" in env_config
            else "3dB20deg"
        )
        if disk_margin_type == "12dB60deg":
            skew = 0
            alpha = 2 / np.sqrt(3)
        elif disk_margin_type == "6dB36deg":
            skew = 0
            # alpha approx 0.665
            alpha = 2 * (np.power(10, 3 / 10) - 1) / (np.power(10, 3 / 10) + 1)
        elif disk_margin_type == "3dB20deg":
            skew = 0
            alpha = 0.353
        elif disk_margin_type == "custom":
            skew = env_config["skew"]
            alpha = env_config["alpha"]
        else:
            raise ValueError(f"Unexpected disk_margin_type: {disk_margin_type}")
        print(f"Disk margin: alpha={alpha},  skew={skew}")

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

        u_high = 20.0
        self.action_space = gym.spaces.Box(
            low=-u_high, high=u_high, shape=(self.nu,), dtype=np.float32
        )

        x_high = np.array(
            [self.max_x, self.max_h, self.max_xdot, self.max_hdot], dtype=np.float32
        )
        self.state_space = gym.spaces.Box(low=-x_high, high=x_high, dtype=np.float32)

        if "normed" in env_config and env_config["normed"]:
            self.true_model.Cpy = self.true_model.Cpy / self.state_space.high
        observation_high = self.true_model.Cpy @ x_high
        self.observation_space = gym.spaces.Box(
            low=-observation_high,
            high=observation_high,
            dtype=np.float32,
        )

        self.time_max = (
            env_config["rollout_length"] if "rollout_length" in env_config else 1000
        )
        self.max_reward = 1.0

        # Stability
        self.Xdd = np.zeros((self.nd, self.nd), dtype=np.float32)
        self.Xde = np.zeros((self.nd, self.ne), dtype=np.float32)
        self.Xee = np.zeros((self.ne, self.ne), dtype=np.float32)

        # L2 gain
        # self.Xdd = 0.99**2 * np.eye(self.nd, dtype=np.float32)
        # self.Xde = np.zeros((self.nd, self.ne), dtype=np.float32)
        # self.Xee = -np.eye(self.ne, dtype=np.float32)

    def compute_reward(self, x, u):
        return np.exp(-(np.linalg.norm(u) ** 2))

    def random_initial_state(self):
        high = np.array(
            [
                0.66 * self.max_x,
                0.66 * self.max_h,
                0.01 * self.max_xdot,
                0.01 * self.max_hdot,
            ],
            dtype=np.float32,
        )
        state = self.np_random.uniform(low=-high, high=high).astype(np.float32)
        return state

    # Override base xdot to use true model here
    def xdot(self, state, d, u):
        xdot = (
            self.true_model.Ap @ state
            + self.true_model.Bpd @ d
            + self.true_model.Bpu @ u
        )
        return xdot

    def disturbance(self):
        if self.disturbance_model == "none":
            return np.zeros((self.nd,), dtype=np.float32)
        elif self.disturbance_model == "occasional":
            # Have disturbance ocurr (in expectation) twice a second.
            # TODO: maybe try disturbance like 50 magnitude
            p = 2 * self.dt
            rand = self.np_random.uniform()
            if rand < p / 2:
                return 1 * self.action_space.low.astype(np.float32)
            elif rand < p:
                return 1 * self.action_space.high.astype(np.float32)
            else:
                return np.zeros((self.nd,), dtype=np.float32)
        else:
            raise ValueError(f"Unexpected disturbance model: {self.disturbance_model}.")

    # Override base xdot to use true model here
    def get_obs(self):
        # TODO: Assuming Dpyd = 0
        # return self.true_model.Cpy @ self.state + self.true_model.Dpyd @ d
        return self.true_model.Cpy @ self.state

    def _build_true_model(self):
        nx = 4  # Plant state size
        nw = 1  # Output size of uncertainty Deltap
        nd = None  # Disturbance size. Defined based on disturbance model.
        nu = 1  # Control input size
        nv = 1  # Input size to uncertainty Deltap
        ne = None  # Performance output size. Defined based on supply_rate.
        ny = None  # Measurement output size. Defined based on observation.
        M = np.array(
            [
                [self.Mr, self.mt + self.rho * self.L / 3],
                [self.mt + self.rho * self.L / 3, self.mt + self.rho * self.L / 5],
            ],
            dtype=np.float32,
        )
        K = np.array([[0, 0], [0, 4 * self.EI / (self.L**3)]], dtype=np.float32)
        B = np.diag([0, self.flexdamp]).astype(np.float32)

        # fmt: off
        Ap = np.bmat([
                [np.zeros((2, 2)), np.eye(2)],
                [-np.linalg.solve(M, K), -np.linalg.solve(M, B)],
        ])
        # fmt: on
        Ap = np.asarray(Ap).astype(np.float32)
        Bpw = np.zeros((nx, nw), dtype=np.float32)
        Bpu = np.vstack([np.zeros((2, 1)), np.linalg.solve(M, np.array([[1], [0]]))])
        Bpu = np.asarray(Bpu).astype(np.float32)

        Cpv = np.zeros((nv, nx), dtype=np.float32)
        Dpvw = np.zeros((nv, nw), dtype=np.float32)
        Dpvu = np.zeros((nv, nu), dtype=np.float32)

        ny = 1
        Cpy = np.array([[1, 1, 0, 0]], dtype=np.float32)
        Dpyw = np.zeros((ny, nw), dtype=np.float32)
        # Dpyu is always 0

        # Enable disturbances at plant input in true model so can do some simulation with it.
        nd = nu
        Bpd = Bpu.copy()
        Dpvd = np.zeros((nv, nd), dtype=np.float32)
        Dpyd = np.zeros((ny, nd), dtype=np.float32)

        ne = nx
        Cpe = np.eye(nx, dtype=np.float32)
        Dpew = np.zeros((ne, nw), dtype=np.float32)
        Dped = np.zeros((ne, nd), dtype=np.float32)
        Dpeu = np.zeros((ne, nu), dtype=np.float32)

        # fmt: off
        return PlantParameters(
            Ap, Bpw, Bpd, Bpu, Cpv, Dpvw, Dpvd, Dpvu, Cpe, Dpew, Dped, Dpeu, Cpy, Dpyw, Dpyd,
            None, None, None, None, None, None # Don't need to specify 
        )
        # fmt: on
