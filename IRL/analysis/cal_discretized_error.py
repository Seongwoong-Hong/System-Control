import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from gym_envs.envs.double_pendulum_discretized import DiscretizedDoublePendulum

irl_path = os.path.abspath("..")


class TestDiscretization(DiscretizedDoublePendulum):
    def __init__(self, N=None, NT=None, dobs=None, dacts=None):
        assert dobs is not None and dacts is not None
        super().__init__(N=N, NT=NT)
        self.max_torques = np.array([100., 100.])
        self.max_speeds = np.array([0.9, 2.5])
        self.max_angles = np.array([0.21, 0.72])
        self.obs_shape = []
        self.obs_high = np.array([*self.max_angles, *self.max_speeds])
        for h, n in zip(self.obs_high, self.num_cells):
            x = (np.logspace(0, np.log10(dobs), n // 2 + 1) - 1) * (h / (dobs - 1))
            self.obs_shape.append(np.append(-np.flip(x[1:]), x))
        self.torque_lists = []
        for h, n in zip(self.max_torques, self.num_actions):
            x = (np.logspace(0, np.log10(dacts), n // 2 + 1) - 1) * (h / (dacts - 1))
            self.torque_lists.append(np.append(-np.flip(x[1:]), x))


def main(dobs, dacts, plot_state=False, plot_act=False):
    disc_env = TestDiscretization(N=[29, 29, 29, 29], NT=[17, 17], dobs=dobs, dacts=dacts)
    states_error_accum, acts_error_accum = [], []
    for actuation in range(1, 7):
        for subj in [f"sub{i:02d}" for i in [1, 2, 4, 5, 6, 7, 9, 10]]:
            target_data_dir = f"{irl_path}/demos/HPC/{subj}_half/{subj}i{actuation}_0.mat"

            states = io.loadmat(target_data_dir)['state'][:, :4]
            acts = io.loadmat(target_data_dir)['tq']

            disc_states = disc_env.get_obs_from_idx(disc_env.get_idx_from_obs(states))
            disc_acts = disc_env.get_torque(disc_env.get_acts_from_idx(disc_env.get_idx_from_acts(acts))).T

            states_error = np.abs(states - disc_states).mean(axis=0) / disc_env.obs_high
            acts_error = np.abs(acts - disc_acts).mean(axis=0) / disc_env.max_torques

            states_error_accum.append(states_error)
            acts_error_accum.append(acts_error)

    print(f"({dacts, dobs})     mean states error    : {np.array(states_error_accum).mean(axis=0)}")
    print(f"({dacts, dobs}) states standard deviation: {np.array(states_error_accum).std(axis=0)}")
    # print(f"({dacts, dobs})     mean acts error    : {np.array(acts_error_accum).mean(axis=0)}")
    # print(f"({dacts, dobs}) acts standard deviation: {np.array(acts_error_accum).std(axis=0)}")

    x = np.arange(len(states_error_accum))
    if plot_state:
        for st_idx in range(4):
            ax = st_fig.add_subplot(2, 2, st_idx + 1)
            ax.plot(x, np.stack(states_error_accum)[:, st_idx])
    if plot_act:
        for act_idx in range(2):
            ax = act_fig.add_subplot(1, 2, act_idx + 1)
            ax.plot(x, np.stack(acts_error_accum)[:, act_idx])


if __name__ == "__main__":
    st_fig = plt.figure(figsize=[27, 18])
    act_fig = plt.figure(figsize=[27, 9])
    # for dobs in [10, 12, 15, 18, 20]:
    main(dobs=15, dacts=10)
    st_fig.tight_layout()
    act_fig.tight_layout()
    plt.show()
