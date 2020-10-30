import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class IDP_custom(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'IDP_custom.xml', 1)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        alive_bonus = 1
        action_bonus = -abs(action[0] - (-8*ob[0] - 0.2*ob[2])) - abs(action[1] - (-2*ob[1] - 0.2*ob[3]))
        r = alive_bonus + action_bonus
        done = bool(abs(ob[0]) >= .3) or bool(abs(ob[1]) >= .3)
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos, # link angles
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
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
