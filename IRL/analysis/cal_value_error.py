import os
from scipy import io

from algos.tabular.viter import *
from common.util import make_env

from imitation.data.rollout import make_sample_until, generate_trajectories

subj = "sub06"
irl_path = os.path.abspath("..")


def get_trajectories_from_approx_dyn(env, agent, init_state, n_episodes, deterministic=False):
    traj_obs, traj_rews = [], []
    P = env.env_method("get_trans_mat")[0]
    for _ in range(n_episodes):
        obs_approx, rews_approx = [], []
        s_vec, _ = env.env_method("get_vectorized")[0]
        current_obs = init_state.reshape(1, -1)
        obs_approx.append(current_obs)
        for _ in range(50):
            act, _ = agent.predict(current_obs, deterministic=deterministic)
            rews_approx.append(env.env_method("get_reward", current_obs, act)[0])
            torque = env.env_method("get_torque", act)[0].T
            a_ind = env.env_method("get_idx_from_acts", torque)[0]
            obs_ind = env.env_method("get_ind_from_state", current_obs.squeeze())[0]
            next_obs = obs_ind @ P[a_ind[0]].T @ s_vec
            obs_approx.append(next_obs)
            current_obs = next_obs
        obs_approx = np.vstack(obs_approx)
        rews_approx = np.array(rews_approx).flatten()
        traj_obs.append(obs_approx)
        traj_rews.append(rews_approx)
    return traj_obs, traj_rews


def cal_trajectory_value():
    bsp = io.loadmat(f"{irl_path}/demos/HPC/{subj}/{subj}i1.mat")['bsp']
    env = make_env("DiscretizedHuman-v2", num_envs=1, N=[21, 23, 21, 29], NT=[19, 19], bsp=bsp)
    init_state = env.observation_space.sample()
    agent = SoftQiter(env, gamma=0.8, alpha=0.0001, device='cuda:0')
    agent.learn(1000)

    eval_env = make_env("DiscretizedHuman-v0", num_envs=1, N=[21, 23, 21, 29], NT=[19, 19], bsp=bsp,
                        init_states=init_state)
    n_episodes = 50

    sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    trajectories = generate_trajectories(agent, eval_env, sample_until, deterministic_policy=False)
    # agent = FiniteViter(env, gamma=0.8, alpha=0.01, device='cpu')
    # agent.learn(0)
    # traj_rews = []
    # for _ in range(50):
    #     obs, _, rews = agent.predict(init_state, deterministic=True)
    #     traj_rews.append(rews)

    obs_approx, rews_approx = get_trajectories_from_approx_dyn(eval_env, agent, init_state, n_episodes,
                                                               deterministic=False)

    gammas = np.array([agent.gamma ** i for i in range(50)])
    value_from_sample, value_from_approx = [], []
    for i in range(n_episodes):
        value_from_sample.append(np.sum(trajectories[i].rews * gammas))
        value_from_approx.append(np.sum(rews_approx[i] * gammas))

    init_state_idx = env.env_method("get_idx_from_obs", init_state[None, :])[0]
    value_from_algo = agent.policy.v_table[init_state_idx].item()
    error = value_from_algo - np.mean(value_from_sample)

    print(f"init_state: {init_state}")
    print(f"mean of values: {np.mean(value_from_sample)}, std of values: {np.std(value_from_sample)}")
    print(f"value error: {error}")

    print(f"mean approx. obs differ: {np.abs(obs_approx[0] - trajectories[0].obs).mean()}")
    print(f"mean approx. value differ: {np.abs(np.array(value_from_approx) - np.array(value_from_sample)).mean()}")


if __name__ == "__main__":
    # for _ in range(10):
    cal_trajectory_value()
