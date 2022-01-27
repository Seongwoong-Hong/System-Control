import os
from scipy import io

from algos.tabular.viter import *
from common.util import make_env
from common.rollouts import get_trajectories_from_approx_dyn

from imitation.data.rollout import make_sample_until, generate_trajectories

subj = "sub06"
irl_path = os.path.abspath("..")


def cal_value_error(n):
    bsp = io.loadmat(f"{irl_path}/demos/HPC/{subj}/{subj}i1.mat")['bsp']
    env = make_env("DiscretizedHuman-v2", num_envs=1, N=[n, n, n, n], NT=[19, 19], bsp=bsp)
    # init_state = np.array([-0.07723441,  0.38298573, -0.70412399,  1.00609718])
    init_state = env.reset()[0]
    agent = SoftQiter(env, gamma=0.95, alpha=0.001, device='cuda:2', verbose=False)
    agent.learn(1000)

    eval_env = make_env("DiscretizedHuman-v0", num_envs=1, bsp=bsp,
                        N=[n, n, n, n], NT=[19, 19], init_states=init_state)
    n_episodes = 100

    sample_until = make_sample_until(n_timesteps=None, n_episodes=n_episodes)
    trajectories = generate_trajectories(agent, eval_env, sample_until, deterministic_policy=False)
    # agent = FiniteViter(env, gamma=0.8, alpha=0.01, device='cpu')
    # agent.learn(0)
    # traj_rews = []
    # for _ in range(50):
    #     obs, _, rews = agent.predict(init_state, deterministic=True)
    #     traj_rews.append(rews)

    approx_trajs = get_trajectories_from_approx_dyn(eval_env, agent, n_episodes, deterministic=False)

    gammas = np.array([agent.gamma ** i for i in range(50)])
    value_from_sample, value_from_approx = [], []
    for i in range(n_episodes):
        value_from_sample.append(np.sum(trajectories[i].rews * gammas))
        value_from_approx.append(np.sum(approx_trajs[i].rews * gammas))

    init_state_idx = env.env_method("get_idx_from_obs", init_state[None, :])[0]
    value_from_algo = agent.policy.v_table[init_state_idx].item()
    error = value_from_algo - np.mean(value_from_sample)

    # print(f"init_state: {init_state}")
    print(f"Expected value: {value_from_algo}")
    print(f"mean of values: {np.mean(value_from_sample)}, std of values: {np.std(value_from_sample)}")
    print(f"value error: {error}, {np.abs(error / value_from_algo) * 100:.2f}%")
    # print(f"mean approx. value differ: {np.abs(np.array(value_from_approx) - np.array(value_from_sample)).mean()}")
    print(
        f"mean approx. obs differ: {np.abs([approx_trajs[i].obs - trajectories[i].obs for i in range(n_episodes)]).mean()}\n")


if __name__ == "__main__":
    for n in [19]:
        print(f"Number of discretized state: {n}")
        cal_value_error(n)
