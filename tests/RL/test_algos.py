import os
import pickle
from algos.torch.ppo import PPO
from algos.tabular.viter import FiniteSoftQiter, FiniteViter, SoftQiter
from common.util import make_env, CPU_Unpickler
from common.wrappers import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def test_finite_algo(idpdiffpolicy, hpc_env):
    env = hpc_env
    agent = idpdiffpolicy(env, gamma=1, alpha=0.002)
    ob = env.reset()
    obs, acts, rws = agent.predict(ob, deterministic=True)
    assert len(obs) == len(acts) + 1
    print(obs[-1])


def test_get_gains_wrt_init_states(idpilqrpolicy, hpc_with_rwrap_env):
    env = hpc_with_rwrap_env
    agent = idpilqrpolicy(env, gamma=1, alpha=0.002)
    ks, kks = [], []
    fig = plt.figure()
    for _ in range(10):
        ob = env.reset()
        obs, _, _ = agent.predict(ob, deterministic=True)
        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1)
            ax.plot(obs[:, i])
        ks.append(agent.ks)
        kks.append(agent.kks)
    plt.show()
    env.close()


def test_toy_disc_env():
    name = "SpringBall"
    env_id = f"{name}_disc"
    env = make_env(f"{env_id}-v2", wrapper=DiscretizeWrapper)
    algo = SoftQiter(env=env, gamma=0.99, alpha=0.01, device='cpu')
    algo.learn(1000)
    fig = plt.figure(figsize=[6.4, 6.4])
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    for _ in range(10):
        obs = []
        acts = []
        ob = env.reset()
        done = False
        # env.render()
        while not done:
            obs.append(ob)
            act, _ = algo.predict(ob, deterministic=False)
            ob, _, done, _ = env.step(act[0])
            acts.append(act[0])
            # env.render()
            # time.sleep(env.dt)
        obs = np.array(obs)
        acts = np.array(acts)
        ax1.plot(obs[:, 0])
        ax2.plot(obs[:, 1])
        ax3.plot(acts)
    fig.tight_layout()
    plt.show()


def test_1d(rl_path):
    name = "1DTarget_disc"
    env_id = f"{name}"
    map_size = 50
    env = make_env(f"{env_id}-v2", map_size=map_size, num_envs=1)
    model_dir = os.path.join(rl_path, name, "tmp", "log", env_id, "softqiter", "policies_1")
    with open(model_dir + "/agent.pkl", "rb") as f:
        algo = pickle.load(f)
    # algo = PPO.load(model_dir + "/agent")
    plt.imshow(algo.policy.policy_table, cmap=cm.rainbow)
    plt.show()
    print('end')


def test_total_reward(rl_path):
    env_type = "HPC"
    name = f"{env_type}_custom"
    model_dir = os.path.join(rl_path, env_type, "tmp", "log", name, "ppo", "policies_4")
    stats_path = None
    if os.path.isfile(model_dir + "normalization.pkl"):
        stats_path = model_dir + "normalization.pkl"
    env = make_env(f"{name}-v1", subpath="../../IRL/demos/HPC/sub01/sub01", wrapper=ActionWrapper, use_norm=stats_path)
    algo = PPO.load(model_dir + f"/agent")
    ob = env.reset()
    done = False
    actions = []
    obs = []
    rewards = []
    while not done:
        act, _ = algo.predict(ob, deterministic=False)
        ob, reward, done, info = env.step(act)
        rewards.append(reward)
        actions.append(info['acts'])
        obs.append(info['obs'])
    # plt.plot(np.array(actions).reshape(-1, 2))
    # plt.show()
    # plt.plot(np.array(obs).reshape(-1, 6)[:, :2])
    # plt.show()
    plt.plot(rewards)
    plt.show()