import os
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


class IDPCustomExp(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, n_steps=None):
        self.traj_len = 0
        self.n_steps = n_steps
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "IDP_custom.xml"), 25)
        utils.EzPickle.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2, ))
        self.init_group = np.array([[[+0.10, +0.10], [+0.05, -0.05]],
                                    [[+0.15, +0.10], [-0.05, +0.05]],
                                    [[-0.16, +0.20], [+0.10, -0.10]],
                                    [[-0.10, +0.06], [+0.05, -0.10]],
                                    [[+0.05, +0.15], [-0.20, -0.20]],
                                    [[-0.05, +0.05], [+0.15, +0.15]],
                                    [[+0.12, +0.05], [-0.10, -0.15]],
                                    [[-0.08, +0.15], [+0.05, -0.15]],
                                    [[-0.15, +0.20], [-0.10, +0.05]],
                                    [[+0.20, +0.01], [+0.09, -0.15]]])
        self.i = 0

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
            self.sim.data.qpos,  # link angles
            self.sim.data.qvel
        ]).ravel()

    def reset_model(self):
        if self.i >= len(self.init_group):
            self.i = 0
        q = self.init_group[self.i]
        self.set_state(
            q[0].reshape(self.model.nq),
            q[1].reshape(self.model.nv)
        )
        self.i += 1
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
