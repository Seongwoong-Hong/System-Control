import os

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


class IDPCustom(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self._order = 0
        self.timesteps = 0
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "HPC_custom.xml"), 1)
        utils.EzPickle.__init__(self)
        self.observation_space = spaces.Box(
            low=np.array([-0.05, -0.2, -0.08, -0.4]),
            high=np.array([0.05, 0.05, 0.3, 0.35])
        )
        self.init_qpos = np.array([0.0, 0.0])
        self.init_qvel = np.array([0.0, 0.0])
        self.timesteps = 0

    def step(self, action: np.ndarray):
        prev_ob = self._get_obs()
        r = - (3.5139 * prev_ob[0] ** 2 + 0.2872182 * prev_ob[1] ** 2
               + 0.24639979 * prev_ob[2] ** 2 + 0.01540204 * prev_ob[3] ** 2
               + 0.016237216 * action[0] ** 2 + 0.002894010177514793 * action[1])
        self.do_simulation(action, self.frame_skip)
        self.timesteps += 1
        ob = self._get_obs()
        done = bool(
            ((ob[:2] < np.array([-0.05, -0.2])).any() or
             (ob[:2] > np.array([0.05, 0.05])).any() or
             (ob[2:4] < np.array([-0.08, -0.4])).any() or
             (ob[2:4] > np.array([0.3, 0.35])).any()
             ) and self.timesteps > 10
        )
        qpos = np.clip(ob[:2], a_min=np.array([-0.05, -0.2]), a_max=np.array([0.05, 0.05]))
        qvel = np.clip(ob[2:4], a_min=np.array([-0.08, -0.4]), a_max=np.array([0.3, 0.35]))
        self.set_state(qpos, qvel)
        ob = self._get_obs()
        info = {'obs': prev_ob.reshape(1, -1), "acts": action.reshape(1, -1)}
        return ob, r, None, info

    @property
    def order(self):
        return self._order

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos,  # link angles
            self.sim.data.qvel,   # link angular velocities
        ]).ravel()

    @property
    def current_obs(self):
        return self._get_obs()

    def set_state(self, qpos, qvel):
        super().set_state(qpos, qvel)

    def reset_model(self):
        init_state = self.observation_space.sample()
        self.set_state(init_state[:2], init_state[2:])
        self.timesteps = 0
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.5  # 0.12250000000000005  # v.model.stat.center[2]


class IDPCustomExp(IDPCustom):
    def __init__(self, init_states):
        super().__init__()
        self.init_group = init_states
        # self.init_group = np.array([[[+0.10, +0.10], [+0.05, -0.05]],
        #                             [[+0.15, +0.10], [-0.05, +0.05]],
        #                             [[-0.16, +0.20], [+0.10, -0.10]],
        #                             [[-0.10, +0.06], [+0.05, -0.10]],
        #                             [[+0.05, +0.15], [-0.20, -0.20]],
        #                             [[-0.05, +0.05], [+0.15, +0.15]],
        #                             [[+0.12, +0.05], [-0.10, -0.15]],
        #                             [[-0.08, +0.15], [+0.05, -0.15]],
        #                             [[-0.15, +0.20], [-0.10, +0.05]],
        #                             [[+0.20, +0.01], [+0.09, -0.15]],
        #                             ])

    def reset_model(self):
        self._order += 1
        q = self.init_group[self._order % len(self.init_group)]
        self.set_state(q[:2], q[2:])
        return self._get_obs()
