import os
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


class IPCustom(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.traj_len = 0
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "IP_custom.xml"), 1)
        self.observation_space = spaces.Box(
            low=np.array([-0.25, -0.3]),
            high=np.array([0.25, 0.3]),
            dtype=np.float64,
        )

    def step(self, action: np.ndarray):
        prev_ob = self._get_obs()
        rew = - prev_ob[0] ** 2
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        qpos = np.clip(ob[:1], a_min=np.array([-0.25]), a_max=np.array([0.25]))
        qvel = np.clip(ob[1:], a_min=np.array([-0.3]), a_max=np.array([0.3]))
        self.set_state(qpos, qvel)
        ob = self._get_obs()
        info = {'obs': prev_ob.reshape(1, -1), "acts": action.reshape(1, -1)}
        self.traj_len += 1
        return ob, rew, False, info

    def reset_model(self):
        q = np.random.uniform(size=2, low=-0.25, high=0.25)
        self.set_state(np.array([q[0]]), np.array([q[1]]))
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    @property
    def current_obs(self):
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = 2 * self.model.stat.extent


class IPCustomExp(IPCustom):
    def __init__(self):
        super().__init__()
        self.init_group = np.array([[+0.30, -0.15],
                                    [-0.30, +0.15],
                                    [+0.15, +0.15],
                                    [-0.15, -0.15],
                                    [+0.12, +0.26],
                                    [-0.12, -0.26],
                                    [+0.28, -0.15],
                                    [-0.28, +0.15],
                                    [-0.20, -0.25],
                                    [+0.20, +0.25]])
        self.i = 0

    def reset_model(self):
        if self.i >= len(self.init_group):
            self.i = 0
        q = self.init_group[self.i]
        self.set_state(np.array([q[0]]), np.array([q[1]]))
        self.i += 1
        return self._get_obs()
