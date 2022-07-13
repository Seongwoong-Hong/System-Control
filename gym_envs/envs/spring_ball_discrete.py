import gym
import numpy as np
from scipy.sparse import csc_matrix
from gym_envs.envs import BaseDiscEnv, UncorrDiscreteInfo


class SpringBallDisc(BaseDiscEnv):
    def __init__(self):
        super(SpringBallDisc, self).__init__()
        self.map_size = 0.4
        self.dt = 0.05
        self.st = None
        self.mass = 1
        self.l0 = 0.1
        self.k = 1
        self.viewer = None
        self.target = np.array([0.1])

        self.num_cells = [40, 40]
        self.num_actions = [5]
        self.obs_low = np.array([-self.map_size, -2.0])
        self.obs_high = np.array([self.map_size, 2.0])
        self.acts_low = np.array([-10.0])
        self.acts_high = np.array([10.0])
        self.obs_info = UncorrDiscreteInfo(self.num_cells)
        self.acts_info = UncorrDiscreteInfo(self.num_actions)
        self.obs_info.set_info(self.obs_high, self.obs_low)
        self.acts_info.set_info(self.acts_high, self.acts_low)
        self.observation_space = gym.spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = gym.spaces.Box(low=self.acts_low, high=self.acts_high)

    def step(self, action: np.ndarray):
        assert self.st is not None, "Can't step the environment before calling reset function"
        assert action in self.action_space, f"{action} is Out of action space"
        self.last_a = action
        info = {'obs': self.st.reshape(1, -1), 'acts': action.reshape(1, -1)}
        r = self.get_reward(self.st, action)
        self.st = self._get_next_state(self.st, action)
        return self.st, r, None, info

    def reset(self):
        self.st = self.np_random.uniform(low=self.obs_low, high=self.obs_high)
        return self._get_obs()

    def set_state(self, state):
        assert state in self.observation_space
        self.st = state

    def get_reward(self, state, action):
        x, dx = np.split(state, 2, axis=-1)
        return - (x ** 2 - 2 * self.target * x + 0.1 * dx ** 2 + 1e-2 * action ** 2)
        # return 1/3 * (np.exp(-3 * (self.target - x) ** 2) + 0.5 * np.exp(-2 * dx ** 2) + 0.2 * np.exp(-2 * action ** 2))

    def _get_next_state(self, state, action):
        x, dx = np.split(state, 2, axis=-1)
        ddx = - (self.k / self.mass) * x + action / self.mass + self.k * self.l0 / self.mass
        next_state = state + np.append(dx, ddx, axis=-1) * self.dt
        return np.clip(next_state, a_min=self.obs_low + 1e-8, a_max=self.obs_high - 1e-8)

    def _get_obs(self):
        return self.st

    def get_init_vector(self):
        return self.get_vectorized()

    def get_trans_mat(self):
        s_vec, a_vec = self.get_vectorized()
        P = []
        for a in a_vec:
            next_s_vec = self._get_next_state(s_vec, a)
            tot_idx = self.get_idx_from_obs(next_s_vec)
            P.append(csc_matrix((np.ones(len(s_vec)), (tot_idx, np.arange(len(s_vec)))),
                                shape=[len(s_vec), len(s_vec)]))
        return np.stack(P)

    def get_reward_mat(self):
        s_vec, a_vec = self.get_vectorized()
        R = []
        for a in a_vec:
            R.append(self.get_reward(s_vec, a).flatten())
        return np.stack(R)

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(600, 100)
            self.viewer.set_bounds(-1.2 * self.map_size, 1.2 * self.map_size, -0.2 * self.map_size, 0.2 * self.map_size)

            # for x in range(len(self.disc_states) + 1):
            #     vertical = rendering.Line(start=(x, 0.), end=(x, self.map_size))
            #     horizontal = rendering.Line(start=(0., x), end=(self.map_size, x))
            #     self.viewer.add_geom(vertical)
            #     self.viewer.add_geom(horizontal)

            goal = rendering.make_circle(0.05)
            goal.set_color(0.3, 0.8, 0.3)
            goal_transform = rendering.Transform()
            goal.add_attr(goal_transform)
            goal_transform.set_translation(self.target[0], 0.0)
            self.viewer.add_geom(goal)

            agent = rendering.make_circle(0.05)
            agent.set_color(0.3, 0.3, 0.8)
            self.agent_transform = rendering.Transform()
            agent.add_attr(self.agent_transform)
            self.viewer.add_geom(agent)

        self.agent_transform.set_translation(self.st[0], 0.0)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")


class SpringBallDiscDet(SpringBallDisc):
    def __init__(self, init_states=None):
        super().__init__()
        self.idx = 0
        if init_states is None:
            s_vec, _ = self.get_vectorized()
            self.init_states = s_vec[0:len(s_vec):len(s_vec)//15]
        else:
            self.init_states = np.array(init_states)

    def reset(self):
        self.st = self.init_states[self.idx]
        self.idx = (self.idx + 1) % len(self.init_states)
        return self._get_obs()

    def get_init_vector(self):
        s_vec = self.init_states.copy()
        _, a_vec = self.get_vectorized()
        return s_vec, a_vec
