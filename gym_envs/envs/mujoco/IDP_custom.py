import os

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


class IDPCustom(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self._order = 0
        self.time = 0.0
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "IDP_custom.xml"), 8)
        utils.EzPickle.__init__(self)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2, ))
        self.init_qpos = np.array([0.0, 0.0])
        self.init_qvel = np.array([0.0, 0.0])
        self.time = 0.0

    def step(self, action: np.ndarray):
        prev_ob = self._get_obs()
        r = 1.0 - (prev_ob[0] ** 2 + prev_ob[1] ** 2 + 0.1 * (prev_ob[2] ** 2 + prev_ob[3] ** 2)
                   + 1e-6 * (self.data.qfrc_actuator @ np.eye(2, 2) @ self.data.qfrc_actuator.T))
        self.do_simulation(action, self.frame_skip)
        self.time += self.dt / 4.8
        ob = self._get_obs()
        done = bool(
            0.95 <= ob[0] or ob[0] <= -0.95 or
            0.95 <= ob[1] or ob[1] <= -0.95
        )
        info = {'obs': prev_ob.reshape(1, -1), "acts": action.reshape(1, -1)}
        return ob, r, None, info

    @property
    def order(self):
        return self._order

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos,  # link angles
            self.sim.data.qvel,   # link angular velocities
            np.array([self.time])
        ]).ravel()

    @property
    def current_obs(self):
        return self._get_obs()

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)
        self.time = 0.0

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.3, high=.3, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.3, high=.3, size=self.model.nv),
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.5  # 0.12250000000000005  # v.model.stat.center[2]


class IDPCustomExp(IDPCustom):
    def __init__(self):
        super().__init__()
        self.init_group = np.array([[[+0.10, +0.10], [+0.05, -0.05]],
                                    [[+0.15, +0.10], [-0.05, +0.05]],
                                    [[-0.16, +0.20], [+0.10, -0.10]],
                                    [[-0.10, +0.06], [+0.05, -0.10]],
                                    [[+0.05, +0.15], [-0.20, -0.20]],
                                    [[-0.05, +0.05], [+0.15, +0.15]],
                                    [[+0.12, +0.05], [-0.10, -0.15]],
                                    [[-0.08, +0.15], [+0.05, -0.15]],
                                    [[-0.15, +0.20], [-0.10, +0.05]],
                                    [[+0.20, +0.01], [+0.09, -0.15]],
                                    ])

    def reset_model(self):
        self._order += 1
        q = self.init_group[self._order % len(self.init_group)]
        self.set_state(
            q[0].reshape(self.model.nq),
            q[1].reshape(self.model.nv)
        )
        return self._get_obs()


class IDPCustomEasy(IDPCustom):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float64)

    def step(self, action):
        ob = self._get_obs()
        r = - abs(ob[0]) - abs(ob[1])
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, r, done, {}
