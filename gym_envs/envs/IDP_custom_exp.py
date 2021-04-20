import numpy as np
from gym_envs.envs import IDPCustom


class IDPCustomExp(IDPCustom):
    def __init__(self, n_steps=None):
        super().__init__(n_steps=n_steps)
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
