import os

import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box


class CartpoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        filepath = os.path.join(os.path.dirname(__file__), "..", "assets", "cartpole.xml")
        mujoco_env.MujocoEnv.__init__(self, filepath, 1)

    def step(self, a):

        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        reward = 1.0 - ob[1] ** 2 - 0.01 * np.abs(ob[2]) - 0.005 * np.abs(ob[3])

        done = False
        if np.abs(ob[0]) > 3.0:
            reward = -2
            done = True
        elif np.abs(ob[1]) > np.pi/2:
            reward = -2
            done = True

        return ob, reward, done, {}

    def reset_model(self):
        qpos = 0.2*(self.np_random.uniform(size=self.model.nq, low=-1., high=1.) - 0.5)
        qvel = 0.5*(self.np_random.uniform(size=self.model.nv, low=-1., high=1.) - 0.5)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent