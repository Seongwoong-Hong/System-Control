import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from gym_envs.envs.double_pendulum_discretized import DiscretizedHuman, calc_trans_mat_error

irl_path = os.path.abspath("..")


class TestDiscretization(DiscretizedHuman):
    def __init__(self, N=None, NT=None, dobs=None, dacts=None, bsp=None):
        assert dobs is not None and dacts is not None
        super().__init__(N=N, NT=NT, bsp=bsp)
        self.max_torques = np.array([100., 100.])
        t1, t2, t3, t4 = 0.16, 0.67, 0.8, 2.4
        m1 = (dobs - 1) * t1 / (10 ** (np.log10(dobs) - 2 * np.log10(dobs) / (N[0] // 2)) - 1)
        m2 = (dobs - 1) * t2 / (10 ** (np.log10(dobs) - 2 * np.log10(dobs) / (N[1] // 2)) - 1)
        m3 = (dobs - 1) * t3 / (10 ** (np.log10(dobs) - 2 * np.log10(dobs) / (N[2] // 2)) - 1)
        m4 = (dobs - 1) * t4 / (10 ** (np.log10(dobs) - 2 * np.log10(dobs) / (N[3] // 2)) - 1)
        print(m1, m2, m3, m4)
        # m1, m2, m3, m4 = 0.4, 1.2, 1.6, 4.8
        self.max_speeds = np.array([m3, m4])
        self.max_angles = np.array([m1, m2])
        self.obs_shape = []
        self.obs_high = np.array([*self.max_angles, *self.max_speeds])
        for h, n in zip(self.obs_high, self.num_cells):
            x = (np.logspace(0, np.log10(dobs), n // 2 + 1) - 1) * (h / (dobs - 1))
            self.obs_shape.append(np.append(-np.flip(x[1:]), x))
        self.torque_lists = []
        for h, n in zip(self.max_torques, self.num_actions):
            x = (np.logspace(0, np.log10(dacts), n // 2 + 1) - 1) * (h / (dacts - 1))
            self.torque_lists.append(np.append(-np.flip(x[1:]), x))


def cal_human_data_error(disc_env):
    states_error_accum, acts_error_accum = [], []
    for actuation in range(1, 7):
        for subj in [f"sub{i:02d}" for i in [1, 2, 4, 5, 6, 7, 9, 10]]:
            target_data_dir = f"{irl_path}/demos/HPC/{subj}_half/{subj}i{actuation}_0.mat"

            states = io.loadmat(target_data_dir)['state'][:, :4]
            acts = io.loadmat(target_data_dir)['tq']

            disc_states = disc_env.get_obs_from_idx(disc_env.get_idx_from_obs(states))
            disc_acts = disc_env.get_acts_from_idx(disc_env.get_idx_from_acts(acts))

            states_error = np.abs(states - disc_states).mean(axis=0) / disc_env.obs_high
            acts_error = np.abs(acts - disc_acts).mean(axis=0) / disc_env.max_torques

            states_error_accum.append(states_error)
            acts_error_accum.append(acts_error)
    return states_error_accum, acts_error_accum


def cal_states_error_via_time(disc_env):
    s_vec, a_vec = disc_env.get_vectorized()
    high = np.array([*disc_env.max_angles, *disc_env.max_speeds])
    low = np.array([*disc_env.min_angles, *disc_env.min_speeds])
    err = calc_trans_mat_error(disc_env, s_vec, a_vec, np.random.uniform(low=low, high=high, size=[1000, 4]))
    return err / high[None, :], None


def main(dobs, dacts, plot_state=False, plot_act=False):
    bsp = io.loadmat(f"{irl_path}/demos/HPC/sub06/sub06i1.mat")['bsp']
    disc_env = TestDiscretization(N=[19, 19, 19, 19], NT=[11, 11], dobs=dobs, dacts=dacts, bsp=bsp)
    states_error_accum, acts_error_accum = cal_states_error_via_time(disc_env)
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
    for inp in [2, 5, 10, 15, 20, 25, 30]:
        main(dobs=inp, dacts=17)
    st_fig.tight_layout()
    act_fig.tight_layout()
    plt.show()
