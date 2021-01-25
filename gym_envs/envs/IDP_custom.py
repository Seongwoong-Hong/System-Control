import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class IDP_custom(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, n_steps=None):
        self.traj_len = 0
        self.n_steps = n_steps
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "IDP_custom.xml"), 25)
        utils.EzPickle.__init__(self)
        self.init_qpos = np.array([0.1, 0.25])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        r = 0
        done = False
        info = {}
        if self.n_steps is None:
            pass
        elif self.traj_len == self.n_steps:
            done = True
            info = {"terminal observation": ob}
            self.traj_len = 0
        self.traj_len += 1
        return ob, r, done, info

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos, # link angles
            np.clip(self.sim.data.qvel, -10, 10)
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.2, high=.2, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .05
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
